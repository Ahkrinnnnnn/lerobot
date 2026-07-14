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

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from lerobot.calibration.board_config import CharucoBoardConfig
from lerobot.calibration.experiment_paths import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_OUTPUTS_BASE,
    ExperimentPaths,
    resolve_experiment_root,
)
from lerobot.calibration.hand_eye import CameraMount
from lerobot.calibration.runner import CalibrationRunConfig
from lerobot.calibration.scene import RobotPoseFrame
from lerobot.robots.config import RobotConfig


class CalibrationPhase(str, Enum):
    INTRINSICS = "intrinsics"
    HAND_EYE = "hand_eye"
    INTRINSICS_AND_HAND_EYE = "intrinsics_and_hand_eye"
    LANDMARK_MAP = "landmark_map"
    TOP_EXTRINSIC_DUAL = "top_extrinsic_dual"
    CAMERA_VIA_LANDMARK = "camera_via_landmark"
    VALIDATE = "validate"


@dataclass
class CalibrationOutputsConfig:
    """One experiment folder holds NPZ, per-camera intrinsics, and phase process files.

    Layout::

        {base_dir}/{robot_id}/{experiment}/
          calibration.npz
          experiment_meta.json
          intrinsics/wrist.json          ← hand_eye reads this when split phases
          intrinsics/top.json
          phases/intrinsics_wrist/       images, report, log, …
          phases/hand_eye_wrist/
          archive/…                      previous phase runs (auto on re-run)
    """

    base_dir: Path | None = None
    experiment: str = DEFAULT_EXPERIMENT_NAME
    experiment_dir: Path | None = None


@dataclass
class HandEyeCalibrationConfig:
    """Robot config + optional board config path; pick phase on CLI."""

    robot: RobotConfig
    board_config_path: Path | None = None
    board: CharucoBoardConfig | None = None
    phase: CalibrationPhase = CalibrationPhase.INTRINSICS
    camera: str = "top"
    mount: CameraMount | None = None
    outputs: CalibrationOutputsConfig = field(default_factory=CalibrationOutputsConfig)
    experiment: str | None = None
    new_experiment: bool = False
    experiment_dir: Path | None = None
    output_path: Path | None = None
    artifacts_dir: Path | None = None
    robot_pose_frame: RobotPoseFrame = RobotPoseFrame.WORLD
    min_hand_eye_samples: int = 12
    min_landmark_samples: int = 6
    also_collect_top: bool = True
    top_camera: str = "top"
    wrist_camera: str = "wrist"
    calibrate_top_intrinsics: bool = False
    debug: bool = True
    show_methods_help: bool = False

    def resolve_board(self) -> CharucoBoardConfig:
        if self.board is not None:
            return self.board
        if self.board_config_path is not None:
            return CharucoBoardConfig.from_json(self.board_config_path)
        return CharucoBoardConfig()

    def resolve_experiment_paths(self, robot_id: str) -> ExperimentPaths:
        """Active experiment folder (all phases share this directory)."""
        if self.experiment_dir is not None:
            return resolve_experiment_root(robot_id, experiment_dir=self.experiment_dir)
        if self.outputs.experiment_dir is not None:
            return resolve_experiment_root(robot_id, experiment_dir=self.outputs.experiment_dir)
        if self.artifacts_dir is not None:
            return resolve_experiment_root(robot_id, experiment_dir=self.artifacts_dir)
        return resolve_experiment_root(
            robot_id,
            base_dir=self.outputs.base_dir or DEFAULT_OUTPUTS_BASE,
            experiment_name=self.experiment or self.outputs.experiment,
            new_experiment=self.new_experiment,
        )

    def resolve_experiment_name(self) -> str:
        if self.new_experiment:
            from datetime import datetime

            return f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return self.experiment or self.outputs.experiment

    def to_run_config(self, robot_id: str) -> CalibrationRunConfig:
        paths = self.resolve_experiment_paths(robot_id)
        npz_path = self.output_path.expanduser() if self.output_path is not None else paths.calibration_npz
        return CalibrationRunConfig(
            target=self.resolve_board().to_target(),
            robot_pose_frame=self.robot_pose_frame,
            min_hand_eye_samples=self.min_hand_eye_samples,
            min_landmark_samples=self.min_landmark_samples,
            also_collect_top=self.also_collect_top,
            top_camera=self.top_camera,
            wrist_camera=self.wrist_camera,
            calibrate_top_intrinsics=self.calibrate_top_intrinsics,
            debug=self.debug,
            experiment_dir=paths.root,
            output_path=npz_path,
        )
