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

"""CLI configuration for ChArUco hand-eye / scene calibration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from lerobot.calibration.board_config import CharucoBoardConfig
from lerobot.calibration.hand_eye import CameraMount
from lerobot.calibration.runner import CalibrationRunConfig
from lerobot.calibration.scene import RobotPoseFrame
from lerobot.robots.config import RobotConfig


class CalibrationPhase(str, Enum):
    INTRINSICS = "intrinsics"
    HAND_EYE = "hand_eye"
    LANDMARK_MAP = "landmark_map"
    CAMERA_VIA_LANDMARK = "camera_via_landmark"
    VALIDATE = "validate"


@dataclass
class HandEyeCalibrationConfig:
    """Robot config + optional board config path; pick phase on CLI."""

    robot: RobotConfig
    board_config_path: Path | None = None
    board: CharucoBoardConfig | None = None
    phase: CalibrationPhase = CalibrationPhase.INTRINSICS
    camera: str = "top"
    mount: CameraMount | None = None
    output_path: Path | None = None
    robot_pose_frame: RobotPoseFrame = RobotPoseFrame.WORLD
    min_hand_eye_samples: int = 12
    min_landmark_samples: int = 6
    show_methods_help: bool = False

    def resolve_board(self) -> CharucoBoardConfig:
        if self.board is not None:
            return self.board
        if self.board_config_path is not None:
            return CharucoBoardConfig.from_json(self.board_config_path)
        return CharucoBoardConfig()

    def to_run_config(self) -> CalibrationRunConfig:
        return CalibrationRunConfig(
            target=self.resolve_board().to_target(),
            robot_pose_frame=self.robot_pose_frame,
            min_hand_eye_samples=self.min_hand_eye_samples,
            min_landmark_samples=self.min_landmark_samples,
            output_path=self.output_path,
        )
