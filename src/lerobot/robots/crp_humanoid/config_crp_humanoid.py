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
from ._ti5_ros import HAND_FINGERS_PER_SIDE_DEFAULT, NUM_BODY_MOTORS_DEFAULT


def _default_crp_humanoid_cameras() -> dict[str, CameraConfig]:
    from lerobot.cameras.ros.configuration_ros import RosImageCameraConfig

    common = dict(fps=30, width=640, height=480)
    return {
        "head": RosImageCameraConfig(image_topic="/head/color/image_raw", **common),
        "left_wrist": RosImageCameraConfig(image_topic="/left_wrist/color/image_raw", **common),
        "right_wrist": RosImageCameraConfig(image_topic="/right_wrist/color/image_raw", **common),
    }


@RobotConfig.register_subclass("crp_humanoid")
@dataclass
class CRPHumanoidConfig(RobotConfig):
    """Configuration for the CRP humanoid (ROS feedback + optional command publishing)."""

    # Unused for ROS-only hardware; kept for CLI compatibility with other robots.
    port: str = "ros"

    cameras: dict[str, CameraConfig] = field(default_factory=_default_crp_humanoid_cameras)

    # --- Feedback (observation) ---
    # Body + arm: ``ti5_interfaces/msg/MultiMotorState`` (position in rad).
    multi_motor_state_topic: str = "/multi_motor_state"
    # Dexterous hand: ``ti5_interfaces/msg/JointSkillfulHandState``.
    joint_skillful_hand_state_topic: str = "/joint_skillful_hand_state"

    # --- Commands (``send_action`` / deploy) ---
    motor_command_topic: str = "/motor_command"
    skillful_hand_command_topic: str = "/skillfulHand_command"
    motor_control_mode: int = 1  # MotorCommand.CONTROL_MODE_POSITION
    skillful_hand_multfingercw: int = 0

    # Optional ``motor_id`` → dataset stem; empty → ``m0`` … ``m{num_body_motors-1}``.
    ros_joint_names: tuple[str, ...] = ()
    num_body_motors: int = NUM_BODY_MOTORS_DEFAULT
    hand_fingers_per_side: int = HAND_FINGERS_PER_SIDE_DEFAULT

    ros_node_name: str = ""
    ros_namespace: str = ""

    camera_async_read_retries: int = 2
