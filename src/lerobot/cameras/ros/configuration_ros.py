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

from dataclasses import dataclass

from ..configs import CameraConfig, ColorMode


@CameraConfig.register_subclass("ros_image")
@dataclass
class RosImageCameraConfig(CameraConfig):
    """Configuration for a ``sensor_msgs/msg/Image`` ROS2 camera topic."""

    image_topic: str
    ros_namespace: str = ""
    color_mode: ColorMode = ColorMode.RGB
    warmup_s: float = 1.0
    read_timeout_ms: float = 500.0
    max_frame_age_ms: float = 500.0

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
