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

"""ROS2 ``sensor_msgs/Image`` camera subscriber."""

from __future__ import annotations

import logging
import time
from threading import Lock
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lerobot.utils.ros_spin import RosSpinSession, default_ros_node_name, resolve_ros_topic, rclpy_available
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from .configuration_ros import RosImageCameraConfig
from .image_utils import image_msg_to_ndarray

logger = logging.getLogger(__name__)

try:
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    _RCLPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    rclpy = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment,misc]
    qos_profile_sensor_data = None  # type: ignore[assignment]
    _RCLPY_AVAILABLE = False


class RosImageCamera(Camera):
    """Subscribe to a ROS2 ``sensor_msgs/msg/Image`` topic and expose frames via ``read()``."""

    def __init__(self, config: RosImageCameraConfig):
        super().__init__(config)
        self.config = config
        self._topic = resolve_ros_topic(config.image_topic, config.ros_namespace)

        self._shared_node: Any | None = None
        self._owns_ros_session = False
        self._ros = RosSpinSession()
        self._subscription = None

        self._frame_lock = Lock()
        self._latest_frame: NDArray[Any] | None = None
        self._latest_timestamp: float | None = None
        self._connected = False

    def __str__(self) -> str:
        return f"RosImageCamera({self._topic})"

    @property
    def is_connected(self) -> bool:
        if not self._connected:
            return False
        if self._owns_ros_session and not self._ros.is_alive:
            return False
        return self._subscription is not None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        raise NotImplementedError("ROS image cameras require explicit topic configuration.")

    def attach_ros_node(self, node: Any) -> None:
        """Use an existing ``rclpy`` node (e.g. from ``CRPHumanoid``) instead of spawning another."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        self._shared_node = node

    def _image_callback(self, msg: Image) -> None:
        try:
            frame = image_msg_to_ndarray(msg, color_mode=self.config.color_mode)
        except Exception:
            logger.exception("%s failed to decode Image message on %s", self, self._topic)
            return

        if self.config.width and frame.shape[1] != self.config.width:
            logger.warning(
                "%s width mismatch: expected %s got %s", self, self.config.width, frame.shape[1]
            )
        if self.config.height and frame.shape[0] != self.config.height:
            logger.warning(
                "%s height mismatch: expected %s got %s", self, self.config.height, frame.shape[0]
            )

        with self._frame_lock:
            self._latest_frame = frame
            self._latest_timestamp = time.perf_counter()

    def _create_subscription(self, node: Any) -> None:
        if Image is None or qos_profile_sensor_data is None:
            raise ImportError("rclpy and sensor_msgs are required for RosImageCamera.")
        self._subscription = node.create_subscription(
            Image, self._topic, self._image_callback, qos_profile_sensor_data
        )
        logger.info("%s subscribing to sensor_msgs/Image on %s", self, self._topic)

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        if not rclpy_available():
            raise ImportError("rclpy is required for RosImageCamera but is not installed.")

        if self._shared_node is not None:
            node = self._shared_node
        else:
            node_name = default_ros_node_name(f"ros_image_{self._topic.strip('/').replace('/', '_')}")
            node = self._ros.start(node_name)
            self._owns_ros_session = True

        self._create_subscription(node)
        self._connected = True

        if warmup:
            deadline = time.perf_counter() + float(self.config.warmup_s)
            while time.perf_counter() < deadline:
                with self._frame_lock:
                    if self._latest_frame is not None:
                        break
                time.sleep(0.01)
            with self._frame_lock:
                if self._latest_frame is None:
                    logger.warning("%s: no Image received during warmup on %s", self, self._topic)

    @check_if_not_connected
    def read(self) -> NDArray[Any]:
        return self.read_latest(max_age_ms=int(self.config.read_timeout_ms))

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        deadline = time.perf_counter() + timeout_ms / 1000.0
        while time.perf_counter() < deadline:
            try:
                return self.read_latest(max_age_ms=int(timeout_ms))
            except (TimeoutError, RuntimeError):
                time.sleep(0.005)
        raise TimeoutError(f"{self} async_read timeout after {timeout_ms}ms on {self._topic}")

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        with self._frame_lock:
            frame = self._latest_frame
            timestamp = self._latest_timestamp

        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not received any Image messages yet on {self._topic}")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms)."
            )
        return frame

    def disconnect(self) -> None:
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._subscription = None
        if self._owns_ros_session:
            self._ros.shutdown()
            self._owns_ros_session = False
        self._shared_node = None
        self._connected = False

        with self._frame_lock:
            self._latest_frame = None
            self._latest_timestamp = None

        logger.info("%s disconnected.", self)
