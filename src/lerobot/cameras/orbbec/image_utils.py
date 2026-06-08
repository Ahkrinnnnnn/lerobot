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

"""Convert Orbbec SDK color frames to RGB numpy arrays."""

from __future__ import annotations

from typing import Any

import cv2  # type: ignore
import numpy as np
from numpy.typing import NDArray

try:
    from pyorbbecsdk import OBFormat, VideoFrame

    _PYORBBECSDK_AVAILABLE = True
except ImportError:  # pragma: no cover
    OBFormat = None  # type: ignore[assignment,misc]
    VideoFrame = Any  # type: ignore[assignment,misc]
    _PYORBBECSDK_AVAILABLE = False


def _yuyv_to_rgb(data: NDArray[Any], width: int, height: int) -> NDArray[Any]:
    yuyv = data.reshape((height, width, 2))
    bgr = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _uyvy_to_rgb(data: NDArray[Any], width: int, height: int) -> NDArray[Any]:
    uyvy = data.reshape((height, width, 2))
    bgr = cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def color_frame_to_rgb(frame: VideoFrame) -> NDArray[Any]:
    """Convert an Orbbec color ``VideoFrame`` to an ``(H, W, 3)`` RGB uint8 array."""
    if not _PYORBBECSDK_AVAILABLE:
        raise ImportError("pyorbbecsdk is required for Orbbec cameras.")

    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())

    if color_format == OBFormat.RGB:
        rgb = np.resize(data, (height, width, 3)).copy()
        return rgb.astype(np.uint8, copy=False)

    if color_format == OBFormat.BGR:
        bgr = np.resize(data, (height, width, 3))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if color_format == OBFormat.YUYV:
        return _yuyv_to_rgb(np.resize(data, (height, width, 2)), width, height)

    if color_format == OBFormat.UYVY:
        return _uyvy_to_rgb(np.resize(data, (height, width, 2)), width, height)

    if color_format == OBFormat.MJPG:
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("Failed to decode MJPG color frame.")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    raise RuntimeError(f"Unsupported Orbbec color format: {color_format}")
