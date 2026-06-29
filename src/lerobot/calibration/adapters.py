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

"""Robot adapters for generic hand-eye / scene calibration."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from lerobot.cameras.camera import Camera
from lerobot.robots.robot import Robot

from .scene import RobotPoseFrame
from .transforms import transform_from_xyz_rpy_deg


@runtime_checkable
class HandEyeRobot(Protocol):
    """Minimal robot interface required by ``calibration.runner``."""

    robot_id: str
    robot_type: str
    cameras: dict[str, Camera]

    def connect(self, calibrate: bool = True) -> None: ...

    def disconnect(self) -> None: ...

    def read_robot_to_ee(self, frame: RobotPoseFrame) -> np.ndarray:
        """Return ``T_robot_to_ee`` as 4x4 (mm + deg, robot-specific frame)."""
        ...


class CrpArmHandEyeAdapter:
    """CRP arm: ``read_end_pose_world`` / ``read_end_pose_user`` → 4x4 transform."""

    def __init__(self, robot):
        self._robot = robot

    @property
    def robot_id(self) -> str:
        return self._robot.id

    @property
    def robot_type(self) -> str:
        return self._robot.name

    @property
    def cameras(self) -> dict[str, Camera]:
        return self._robot.cameras

    def connect(self, calibrate: bool = True) -> None:
        self._robot.connect(calibrate=calibrate)

    def disconnect(self) -> None:
        self._robot.disconnect()

    def read_robot_to_ee(self, frame: RobotPoseFrame) -> np.ndarray:
        if frame == RobotPoseFrame.USER:
            x, y, z, roll, pitch, yaw = self._robot.crp_arm_robot.read_end_pose_user()
        else:
            x, y, z, roll, pitch, yaw = self._robot.crp_arm_robot.read_end_pose_world()
        return transform_from_xyz_rpy_deg(x, y, z, roll, pitch, yaw)


def as_hand_eye_robot(robot: Robot) -> HandEyeRobot:
    """Wrap a LeRobot ``Robot`` for hand-eye calibration."""
    from lerobot.robots.crp_arm.crp_arm import CRPArm

    if isinstance(robot, CRPArm):
        return CrpArmHandEyeAdapter(robot)
    raise NotImplementedError(
        f"Hand-eye calibration has no adapter for robot type {robot.name!r}. "
        "Implement HandEyeRobot in lerobot.calibration.adapters."
    )
