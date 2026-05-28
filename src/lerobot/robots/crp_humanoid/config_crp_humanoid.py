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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("crp_humanoid")
@dataclass
class CRPHumanoidConfig(RobotConfig):
    """Configuration for the CRP humanoid follower (observation / cameras only during record)."""

    # Unused for ROS-only hardware; kept for CLI compatibility with other robots.
    port: str = "ros"

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # TODO(ros): Set the topic that publishes the humanoid ``sensor_msgs/JointState``.
    joint_states_topic: str = ""
    # TODO(ros): Joint names in ``JointState.name`` order → dataset keys ``{name}.pos``.
    ros_joint_names: tuple[str, ...] = ()

    # TODO(ros): Optional ROS node name; empty → auto ``crp_humanoid_record_<pid>``.
    ros_node_name: str = ""
    # TODO(ros): Namespace prefix for topics (e.g. ``/humanoid``).
    ros_namespace: str = ""

    camera_async_read_retries: int = 2
