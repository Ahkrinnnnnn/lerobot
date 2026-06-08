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

from __future__ import annotations

import numpy as np

from lerobot.cameras.configs import ColorMode

try:
    import cv2  # type: ignore

    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False


def image_msg_to_ndarray(msg: object, *, color_mode: ColorMode = ColorMode.RGB) -> np.ndarray:
    """Convert ``sensor_msgs/Image`` to ``(H, W, 3)`` uint8 RGB/BGR array."""
    try:
        from cv_bridge import CvBridge

        desired = "rgb8" if color_mode == ColorMode.RGB else "bgr8"
        return np.asarray(CvBridge().imgmsg_to_cv2(msg, desired_encoding=desired))
    except ImportError:
        pass

    encoding = str(getattr(msg, "encoding", "")).lower()
    height = int(getattr(msg, "height"))
    width = int(getattr(msg, "width"))
    step = int(getattr(msg, "step"))
    data = np.frombuffer(getattr(msg, "data"), dtype=np.uint8)

    if encoding in {"rgb8"}:
        image = data.reshape(height, width, 3)
    elif encoding in {"bgr8"}:
        image = data.reshape(height, width, 3)
        if color_mode == ColorMode.RGB:
            if not _CV2_AVAILABLE:
                raise ImportError("OpenCV (cv2) is required to convert bgr8 to rgb8 without cv_bridge.")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif encoding in {"rgba8"}:
        image = data.reshape(height, width, 4)[..., :3]
        if color_mode == ColorMode.RGB and _CV2_AVAILABLE:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif encoding in {"bgra8"}:
        image = data.reshape(height, width, 4)[..., :3]
        if color_mode == ColorMode.RGB:
            if not _CV2_AVAILABLE:
                raise ImportError("OpenCV (cv2) is required to convert bgra8 to rgb8 without cv_bridge.")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif encoding in {"mono8"}:
        gray = data.reshape(height, width)
        image = np.stack([gray, gray, gray], axis=-1)
    else:
        channels = max(step // max(width, 1), 1)
        if channels == 1:
            gray = data[: height * step].reshape(height, step)[..., :width]
            image = np.stack([gray, gray, gray], axis=-1)
        else:
            image = data.reshape(height, width, channels)[..., :3]
            if encoding.startswith("bgr") and color_mode == ColorMode.RGB and _CV2_AVAILABLE:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if color_mode == ColorMode.BGR and encoding.startswith("rgb") and _CV2_AVAILABLE:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return np.ascontiguousarray(image)
