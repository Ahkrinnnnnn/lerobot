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

from lerobot.robots.crp_humanoid._ti5_ros import (
    HAND_FINGERS_PER_SIDE_DEFAULT,
    NUM_BODY_MOTORS_DEFAULT,
    default_dataset_action_keys,
)

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("crp_skeleton")
@dataclass
class CRPSkeletonConfig(TeleoperatorConfig):
    """Configuration for CRP skeleton labels during passive humanoid recording.

    Subscribes to robot **command** topics (processed exoskeleton → hardware targets) and
    records them verbatim — no LeRobot processor pipeline.
    """

    # Unused for ROS-only hardware; kept for CLI compatibility.
    port: str = "ros"

    # Body + arm commands: ``ti5_interfaces/msg/MotorCommand`` (position mode, rad).
    motor_command_topic: str = "/motor_command"
    # Dexterous hand commands: ``ti5_interfaces/msg/SkillfulHandCommand`` (L/R on same topic).
    skillful_hand_command_topic: str = "/skillfulHand_command"

    # Optional subset of dataset action keys; empty → auto ``m*.pos`` + ``hand_*.finger_*.pos``.
    ros_action_keys: tuple[str, ...] = ()
    ros_joint_names: tuple[str, ...] = ()
    num_body_motors: int = NUM_BODY_MOTORS_DEFAULT
    hand_fingers_per_side: int = HAND_FINGERS_PER_SIDE_DEFAULT

    ros_node_name: str = ""
    ros_namespace: str = ""

    stub_action_keys: tuple[str, ...] = field(
        default_factory=lambda: default_dataset_action_keys(
            NUM_BODY_MOTORS_DEFAULT, HAND_FINGERS_PER_SIDE_DEFAULT
        )
    )
