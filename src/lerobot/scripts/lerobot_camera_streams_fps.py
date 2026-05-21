#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Preview camera streams in OpenCV windows with measured frame rate (Hz) overlaid.

Uses the same discovery and connection flow as `lerobot_find_cameras` for OpenCV and
RealSense cameras. Reported **Stream** Hz is the inverse of the time between
consecutive successful `read()` calls (exponential moving average).

Example:

```shell
lerobot-camera-streams-fps
lerobot-camera-streams-fps opencv
lerobot-camera-streams-fps --camera-id /dev/video0
```

Press `q` / `Esc` or Ctrl+C to quit.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from typing import Any

import cv2
import numpy as np

from lerobot.scripts.lerobot_find_cameras import (
    cleanup_cameras,
    create_camera_instance,
    find_and_print_cameras,
)
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)

# EMA smoothing for measured stream rate (higher = more responsive)
_FPS_EMA_ALPHA = 0.12


def _match_camera_id(camera_meta: dict[str, Any], camera_id: str | None) -> bool:
    if camera_id is None:
        return True
    return str(camera_meta.get("id")) == str(camera_id)


def _window_name(meta: dict[str, Any]) -> str:
    cid = str(meta.get("id", "?")).replace("/", "_").replace("\\", "_")
    return f"{meta.get('type', '?')} | {cid}"


def _nominal_fps(meta: dict[str, Any]) -> float | None:
    profile = meta.get("default_stream_profile")
    if not isinstance(profile, dict):
        return None
    fps = profile.get("fps")
    if fps is None:
        return None
    try:
        f = float(fps)
    except (TypeError, ValueError):
        return None
    if f <= 0 or f > 1000:
        return None
    return f


def _overlay_fps_bgr(
    bgr: np.ndarray,
    stream_hz: float,
    nominal: float | None,
) -> None:
    h, _w = bgr.shape[:2]
    scale = max(h / 720.0, 0.5)
    thickness = max(int(round(scale)), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55 * scale

    line1 = f"Stream: {stream_hz:.1f} Hz"
    lines = [line1]
    if nominal is not None:
        lines.append(f"Nominal: {nominal:.1f} Hz")

    y0 = int(28 * scale)
    dy = int(26 * scale)
    for i, text in enumerate(lines):
        y = y0 + i * dy
        cv2.putText(bgr, text, (10, y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)


def _camera_display_loop(
    cam_dict: dict[str, Any],
    stop_event: threading.Event,
) -> None:
    camera = cam_dict["instance"]
    meta = cam_dict["meta"]
    name = _window_name(meta)
    nominal = _nominal_fps(meta)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    ema_hz = 0.0
    last_t = time.perf_counter()

    while not stop_event.is_set():
        try:
            frame_rgb = camera.read()
        except TimeoutError:
            logger.warning("Timeout reading camera %s", meta.get("id"))
            time.sleep(0.05)
            continue
        except Exception as exc:
            logger.error("Error reading camera %s: %s", meta.get("id"), exc)
            time.sleep(0.05)
            continue

        now = time.perf_counter()
        dt = now - last_t
        last_t = now
        if dt > 0:
            inst = 1.0 / dt
            ema_hz = _FPS_EMA_ALPHA * inst + (1.0 - _FPS_EMA_ALPHA) * ema_hz if ema_hz > 0 else inst

        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        _overlay_fps_bgr(bgr, ema_hz, nominal)
        cv2.imshow(name, bgr)

    cv2.destroyWindow(name)


def display_streams_with_fps(
    camera_type: str | None = None,
    camera_id: str | None = None,
) -> None:
    all_meta = find_and_print_cameras(camera_type_filter=camera_type)
    if camera_id is not None:
        all_meta = [m for m in all_meta if _match_camera_id(m, camera_id)]

    if not all_meta:
        logger.warning("No cameras matched the filter.")
        return

    cameras_to_use: list[dict[str, Any]] = []
    for cam_meta in all_meta:
        inst = create_camera_instance(cam_meta)
        if inst:
            cameras_to_use.append(inst)

    if not cameras_to_use:
        logger.warning("Could not connect to any camera.")
        return

    stop_event = threading.Event()

    threads: list[threading.Thread] = []
    for cam_dict in cameras_to_use:
        t = threading.Thread(
            target=_camera_display_loop,
            args=(cam_dict, stop_event),
            daemon=True,
            name=f"cam-{_window_name(cam_dict['meta'])}",
        )
        threads.append(t)
        t.start()

    logger.info("Opened %d preview window(s). Press q / Esc to quit.", len(cameras_to_use))

    try:
        while not stop_event.is_set():
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                stop_event.set()
                break
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        stop_event.set()
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=2.0)
        cleanup_cameras(cameras_to_use)
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview cameras in OpenCV windows with measured stream frequency (Hz).",
    )
    parser.add_argument(
        "camera_type",
        type=str,
        nargs="?",
        default=None,
        choices=["realsense", "opencv"],
        help="Only probe this camera backend; omit to try both OpenCV and RealSense.",
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default=None,
        help="Open only the camera whose id/serial matches this string (as printed by find-cameras).",
    )
    args = parser.parse_args()

    init_logging()
    display_streams_with_fps(camera_type=args.camera_type, camera_id=args.camera_id)


if __name__ == "__main__":
    main()
