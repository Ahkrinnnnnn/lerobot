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

"""Shared ti5_interfaces ROS helpers for CRP humanoid passive recording."""

from __future__ import annotations

from lerobot.utils.ros_spin import (
    RosSpinSession,
    default_ros_node_name,
    init_rclpy,
    rclpy_available,
    resolve_ros_topic,
)

__all__ = [
    "RosSpinSession",
    "default_ros_node_name",
    "init_rclpy",
    "rclpy_available",
    "resolve_ros_topic",
]

NUM_BODY_MOTORS_DEFAULT = 20
HAND_FINGERS_PER_SIDE_DEFAULT = 6

# SkillfulHandCommand.control_skillful_hand_id
SKILLFUL_HAND_R = 1
SKILLFUL_HAND_L = 2

# MotorCommand.control_mode
MOTOR_CONTROL_MODE_POSITION = 1


def default_motor_stems(num_motors: int = NUM_BODY_MOTORS_DEFAULT) -> tuple[str, ...]:
    return tuple(f"m{i}" for i in range(num_motors))


def motor_stem_for_id(motor_id: int, names: tuple[str, ...]) -> str:
    if names and 0 <= motor_id < len(names):
        return names[motor_id].removesuffix(".pos")
    return f"m{motor_id}"


def motor_id_for_key(key: str, names: tuple[str, ...]) -> int | None:
    """Map a dataset motor key (``m3.pos`` or custom stem) to hardware ``motor_id``."""
    if not key.endswith(".pos") or key.startswith("hand_"):
        return None
    stem = key.removesuffix(".pos")
    if names:
        for motor_id, name in enumerate(names):
            if name.removesuffix(".pos") == stem:
                return motor_id
        return None
    if stem.startswith("m") and stem[1:].isdigit():
        return int(stem[1:])
    return None


def motor_targets_from_action(
    action: dict[str, float],
    motor_stems: tuple[str, ...],
    ros_joint_names: tuple[str, ...],
    fallback: dict[str, float],
) -> list[tuple[int, float]]:
    """Build sorted ``(motor_id, target_rad)`` pairs for ``MotorCommand``."""
    targets_by_id: dict[int, float] = {}
    for stem in motor_stems:
        key = f"{stem}.pos"
        motor_id = motor_id_for_key(key, ros_joint_names)
        if motor_id is None:
            continue
        targets_by_id[motor_id] = float(action.get(key, fallback.get(key, 0.0)))
    for key, value in action.items():
        motor_id = motor_id_for_key(key, ros_joint_names)
        if motor_id is not None:
            targets_by_id[motor_id] = float(value)
    return sorted(targets_by_id.items())


def hand_targets_from_action(
    action: dict[str, float],
    side: str,
    fingers_per_side: int,
    fallback: dict[str, float],
) -> list[float]:
    keys = hand_side_keys(side, fingers_per_side)
    return [float(action.get(key, fallback.get(key, 0.0))) for key in keys]


def hand_side_present_in_action(action: dict[str, float], side: str, fingers_per_side: int) -> bool:
    return any(key in action for key in hand_side_keys(side, fingers_per_side))


def hand_side_keys(side: str, fingers_per_side: int) -> tuple[str, ...]:
    return tuple(f"hand_{side}.finger_{i}.pos" for i in range(fingers_per_side))


def default_hand_obs_keys(fingers_per_side: int = HAND_FINGERS_PER_SIDE_DEFAULT) -> tuple[str, ...]:
    return hand_side_keys("l", fingers_per_side) + hand_side_keys("r", fingers_per_side)


def default_motor_obs_keys(num_motors: int = NUM_BODY_MOTORS_DEFAULT) -> tuple[str, ...]:
    return tuple(f"{stem}.pos" for stem in default_motor_stems(num_motors))


def default_dataset_action_keys(
    num_motors: int = NUM_BODY_MOTORS_DEFAULT,
    fingers_per_side: int = HAND_FINGERS_PER_SIDE_DEFAULT,
) -> tuple[str, ...]:
    return default_motor_obs_keys(num_motors) + default_hand_obs_keys(fingers_per_side)


def import_ti5_messages() -> tuple[type, type, type, type]:
    try:
        from ti5_interfaces.msg import (
            JointSkillfulHandState,
            MotorCommand,
            MultiMotorState,
            SkillfulHandCommand,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ti5_interfaces is required for CRP humanoid ROS topics. "
            "Source your ROS2 workspace (e.g. `source install/setup.bash`) before running."
        ) from exc
    return MultiMotorState, JointSkillfulHandState, MotorCommand, SkillfulHandCommand
