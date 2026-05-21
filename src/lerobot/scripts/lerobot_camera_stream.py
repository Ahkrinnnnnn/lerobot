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
Display camera streams with LeRobot's built-in Rerun visualization.

Example:

```shell
lerobot-camera-stream
lerobot-camera-stream opencv --camera-id 0
lerobot-camera-stream realsense --display-ip 127.0.0.1 --display-port 9876
```
"""

import argparse
import logging
import time
from typing import Any

from lerobot.scripts.lerobot_find_cameras import (
    cleanup_cameras,
    create_camera_instance,
    find_and_print_cameras,
)
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

logger = logging.getLogger(__name__)


def _match_camera_id(camera_meta: dict[str, Any], camera_id: str | None) -> bool:
    if camera_id is None:
        return True
    return str(camera_meta.get("id")) == str(camera_id)


def stream_cameras(
    camera_type: str | None = None,
    camera_id: str | None = None,
    fps: float = 30.0,
    session_name: str = "camera_stream",
    display_ip: str | None = None,
    display_port: int | None = None,
    display_compressed_images: bool = False,
) -> None:
    """Stream selected cameras to Rerun until user interrupts (Ctrl+C)."""
    if fps <= 0:
        raise ValueError(f"`fps` must be positive, got {fps}.")

    all_camera_metadata = find_and_print_cameras(camera_type_filter=camera_type)
    if camera_id is not None:
        all_camera_metadata = [meta for meta in all_camera_metadata if _match_camera_id(meta, camera_id)]

    if not all_camera_metadata:
        logger.warning("No cameras matched the requested filter. Nothing to stream.")
        return

    cameras_to_use = []
    for cam_meta in all_camera_metadata:
        camera_instance = create_camera_instance(cam_meta)
        if camera_instance:
            cameras_to_use.append(camera_instance)

    if not cameras_to_use:
        logger.warning("No cameras could be connected. Aborting stream.")
        return

    init_rerun(session_name=session_name, ip=display_ip, port=display_port)
    logger.info(
        "Streaming %d camera(s) to Rerun at up to %.2f FPS. Press Ctrl+C to stop.",
        len(cameras_to_use),
        fps,
    )

    period_s = 1.0 / fps
    try:
        while True:
            loop_start = time.perf_counter()

            observation: dict[str, Any] = {}
            for cam_dict in cameras_to_use:
                camera = cam_dict["instance"]
                camera_meta = cam_dict["meta"]
                camera_key = f"camera.{str(camera_meta.get('type', 'unknown')).lower()}_{camera_meta.get('id')}"

                try:
                    frame = camera.read()
                    observation[camera_key] = frame
                except TimeoutError:
                    logger.warning("Timeout while reading camera %s.", camera_meta.get("id"))
                except Exception as exc:
                    logger.error("Error while reading camera %s: %s", camera_meta.get("id"), exc)

            if observation:
                log_rerun_data(observation=observation, compress_images=display_compressed_images)

            dt = time.perf_counter() - loop_start
            sleep_s = period_s - dt
            if sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        logger.info("Camera stream interrupted by user.")
    finally:
        cleanup_cameras(cameras_to_use)


def main():
    parser = argparse.ArgumentParser(description="Display camera streams using LeRobot's Rerun visualization.")
    parser.add_argument(
        "camera_type",
        type=str,
        nargs="?",
        default=None,
        choices=["realsense", "opencv"],
        help="Optional camera type filter. Streams all cameras when omitted.",
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default=None,
        help="Optional camera id/serial/index filter (exact string match).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Max stream FPS (best effort). Default: 30.0",
    )
    parser.add_argument(
        "--session-name",
        type=str,
        default="camera_stream",
        help="Rerun session name. Default: camera_stream",
    )
    parser.add_argument(
        "--display-ip",
        type=str,
        default=None,
        help="Optional Rerun server IP (connect mode).",
    )
    parser.add_argument(
        "--display-port",
        type=int,
        default=None,
        help="Optional Rerun server port (connect mode).",
    )
    parser.add_argument(
        "--display-compressed-images",
        action="store_true",
        help="Compress images before logging to Rerun.",
    )
    args = parser.parse_args()

    init_logging()
    stream_cameras(**vars(args))


if __name__ == "__main__":
    main()
