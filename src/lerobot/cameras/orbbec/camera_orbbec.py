#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Orbbec RGB camera via OrbbecSDK v2 (``pyorbbecsdk2``)."""

from __future__ import annotations

import logging
import time
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any

import cv2  # type: ignore
import numpy as np
from numpy.typing import NDArray

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import is_package_available, require_package

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_orbbec import OrbbecCameraConfig
from .image_utils import color_frame_to_rgb

if TYPE_CHECKING or is_package_available("pyorbbecsdk2", import_name="pyorbbecsdk"):
    from pyorbbecsdk import Config, Context, Device, OBError, OBFormat, OBSensorType, Pipeline, VideoStreamProfile
else:  # pragma: no cover
    Config = Context = Device = OBError = OBFormat = OBSensorType = Pipeline = VideoStreamProfile = Any  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


def _resolve_device(serial_number: str) -> Device:
    ctx = Context()
    device_list = ctx.query_devices()
    count = device_list.get_count()
    if count == 0:
        raise ConnectionError(
            "No Orbbec device found. Check USB connection and udev rules "
            "(see /opt/OrbbecSDK_v2.8.6/shared/install_udev_rules.sh)."
        )

    target = serial_number.strip()
    if not target:
        return device_list.get_device_by_index(0)

    for index in range(count):
        device = device_list.get_device_by_index(index)
        info = device.get_device_info()
        if info.get_serial_number() == target:
            return device

    available = [device_list.get_device_by_index(i).get_device_info().get_serial_number() for i in range(count)]
    raise ValueError(
        f"No Orbbec device with serial_number={target!r}. Available serial numbers: {available}"
    )


def _select_color_profile(
    pipeline: Pipeline,
    width: int | None,
    height: int | None,
    fps: int | None,
) -> VideoStreamProfile:
    profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    # Prefer uncompressed / YUV over MJPG to avoid variable imdecode latency in the reader thread.
    fmt_priority = (OBFormat.RGB, OBFormat.YUYV, OBFormat.BGR, OBFormat.MJPG)

    if width and height and fps:
        for fmt in fmt_priority:
            try:
                profile = profiles.get_video_stream_profile(width, height, fmt, fps)
                logger.info(
                    "Selected Orbbec color profile %dx%d@%d format=%s",
                    width,
                    height,
                    fps,
                    profile.get_format(),
                )
                return profile
            except OBError:
                continue
        logger.warning(
            "Requested Orbbec color profile %dx%d@%d not found; using device default.",
            width,
            height,
            fps,
        )

    return profiles.get_default_video_stream_profile()


class OrbbecCamera(Camera):
    """Capture color frames from Orbbec cameras using OrbbecSDK v2."""

    def __init__(self, config: OrbbecCameraConfig):
        require_package("pyorbbecsdk2", extra="orbbec", import_name="pyorbbecsdk")
        super().__init__(config)
        self.config = config
        self.serial_number = config.serial_number.strip()
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s
        self.read_timeout_ms = config.read_timeout_ms

        self._device: Device | None = None
        self._pipeline: Pipeline | None = None
        self._color_profile: VideoStreamProfile | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()
        self.rotation: int | None = get_cv2_rotation(config.rotation)

        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width

    def __str__(self) -> str:
        label = self.serial_number or "auto"
        return f"{self.__class__.__name__}({label})"

    @property
    def is_connected(self) -> bool:
        return self._pipeline is not None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        require_package("pyorbbecsdk2", extra="orbbec", import_name="pyorbbecsdk")
        ctx = Context()
        device_list = ctx.query_devices()
        found: list[dict[str, Any]] = []

        for index in range(device_list.get_count()):
            device = device_list.get_device_by_index(index)
            info = device.get_device_info()
            camera_info: dict[str, Any] = {
                "type": "Orbbec",
                "id": info.get_serial_number(),
                "name": info.get_name(),
                "serial_number": info.get_serial_number(),
                "firmware_version": info.get_firmware_version(),
                "connection_type": info.get_connection_type(),
            }
            try:
                pipeline = Pipeline(device)
                profile = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_default_video_stream_profile()
                camera_info["default_stream_profile"] = {
                    "format": str(profile.get_format()),
                    "width": profile.get_width(),
                    "height": profile.get_height(),
                    "fps": profile.get_fps(),
                }
            except OBError:
                pass
            found.append(camera_info)

        return found

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        self._device = _resolve_device(self.serial_number)
        if not self.serial_number:
            self.serial_number = self._device.get_device_info().get_serial_number()

        self._pipeline = Pipeline(self._device)
        self._color_profile = _select_color_profile(
            self._pipeline, self.width, self.height, self.fps
        )

        sdk_config = Config()
        sdk_config.enable_stream(self._color_profile)

        try:
            self._pipeline.start(sdk_config)
        except OBError as exc:
            self._pipeline = None
            self._device = None
            raise ConnectionError(f"Failed to start Orbbec pipeline for {self}.") from exc

        self._sync_capture_dimensions()
        self._start_read_thread()

        if warmup:
            warmup_duration_s = max(self.warmup_s, 1.0)
            deadline = time.perf_counter() + warmup_duration_s
            while time.perf_counter() < deadline:
                if self.thread is not None and not self.thread.is_alive():
                    raise ConnectionError(f"{self} read thread stopped during warmup.")
                with self.frame_lock:
                    if self.latest_frame is not None:
                        break
                time.sleep(0.05)
            with self.frame_lock:
                if self.latest_frame is None:
                    raise ConnectionError(
                        f"{self} failed to capture frames during warmup ({warmup_duration_s:.1f}s)."
                    )

        logger.info("%s connected (%dx%d @ %sfps).", self, self.capture_width, self.capture_height, self.fps)

    def _sync_capture_dimensions(self) -> None:
        if self._color_profile is None:
            return
        actual_width = self._color_profile.get_width()
        actual_height = self._color_profile.get_height()
        actual_fps = self._color_profile.get_fps()

        if self.width and self.width != actual_width:
            logger.warning("%s: requested width=%s, using %s from device.", self, self.width, actual_width)
        if self.height and self.height != actual_height:
            logger.warning("%s: requested height=%s, using %s from device.", self, self.height, actual_height)
        if self.fps and self.fps != actual_fps:
            logger.warning("%s: requested fps=%s, using %s from device.", self, self.fps, actual_fps)

        self.width = actual_width
        self.height = actual_height
        self.fps = actual_fps
        self.capture_width, self.capture_height = self.width, self.height
        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            self.capture_width, self.capture_height = self.height, self.width

    def _join_read_thread(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            join_timeout_s = max(self.read_timeout_ms / 1000.0 + 0.5, 2.0)
            self.thread.join(timeout=join_timeout_s)
        self.thread = None
        self.stop_event = None
        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
        self.new_frame_event.clear()

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        self._join_read_thread()

    def _start_read_thread(self) -> None:
        self._stop_read_thread()
        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, daemon=True, name=f"{self}-read")
        self.thread.start()
        time.sleep(0.1)

    def _read_loop(self) -> None:
        if self.stop_event is None or self._pipeline is None:
            raise RuntimeError(f"{self}: read loop started before pipeline initialization.")

        while not self.stop_event.is_set():
            if self._pipeline is None:
                break
            try:
                frames = self._pipeline.wait_for_frames(int(self.read_timeout_ms))
                if frames is None:
                    continue
                color_frame = frames.get_color_frame()
                if color_frame is None:
                    continue
                rgb = self._postprocess_image(color_frame_to_rgb(color_frame))
                with self.frame_lock:
                    self.latest_frame = rgb
                    self.latest_timestamp = time.perf_counter()
                self.new_frame_event.set()
            except DeviceNotConnectedError:
                break
            except Exception:
                if not self.stop_event.is_set():
                    logger.exception("%s read loop error", self)

    def _postprocess_image(self, image: NDArray[Any]) -> NDArray[Any]:
        h, w, c = image.shape
        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(
                f"{self} frame width={w} or height={h} do not match configured "
                f"width={self.capture_width} or height={self.capture_height}."
            )
        if c != 3:
            raise RuntimeError(f"{self} frame channels={c} do not match expected 3 channels.")

        processed = image
        if self.color_mode == ColorMode.BGR:
            processed = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed = cv2.rotate(processed, self.rotation)

        return processed

    @check_if_not_connected
    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        if color_mode is not None:
            logger.warning(
                "%s read() color_mode parameter is deprecated and will be removed in future versions.",
                self,
            )
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")
        self.new_frame_event.clear()
        return self.async_read(timeout_ms=max(int(self.read_timeout_ms), 10_000))

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"{self} timed out waiting for a frame ({timeout_ms} ms).")

        with self.frame_lock:
            if self.latest_frame is None:
                raise RuntimeError(f"{self} has no frame available.")
            return self.latest_frame.copy()

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        with self.frame_lock:
            if self.latest_frame is None:
                raise RuntimeError(f"{self} has not captured any frames yet.")
            if self.latest_timestamp is None:
                raise RuntimeError(f"{self} frame timestamp is missing.")
            age_ms = (time.perf_counter() - self.latest_timestamp) * 1000.0
            if age_ms > max_age_ms:
                raise TimeoutError(f"{self} latest frame is {age_ms:.0f} ms old (max {max_age_ms} ms).")
            return self.latest_frame.copy()

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()

        pipeline = self._pipeline
        if pipeline is not None:
            try:
                pipeline.stop()
            except OBError:
                logger.exception("%s pipeline stop failed", self)

        self._pipeline = None
        self._device = None
        self._color_profile = None
        self._join_read_thread()
        logger.info("%s disconnected.", self)
