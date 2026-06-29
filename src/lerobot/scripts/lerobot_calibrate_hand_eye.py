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

"""
ChArUco hand-eye / scene calibration.

Configs: ``crp.json`` (robot) + ``charuco_board.json`` (board, via board_config_path).
Output: ``<robot_id>_calibration.npz``
"""

from __future__ import annotations

import json
import logging

from lerobot.calibration.adapters import as_hand_eye_robot
from lerobot.calibration.config import CalibrationPhase, HandEyeCalibrationConfig
from lerobot.calibration.hand_eye import CameraMount
from lerobot.calibration.runner import (
    print_methods_help,
    run_camera_via_landmark_calibration,
    run_hand_eye_calibration,
    run_intrinsics_calibration,
    run_landmark_map_calibration,
    validate_scene_calibration,
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.robots import make_robot_from_config
import lerobot.robots.crp_arm  # noqa: F401 — registers crp_arm
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


def _require_mount(phase: CalibrationPhase, mount: CameraMount | None, camera: str) -> CameraMount | None:
    needs_mount = phase in (
        CalibrationPhase.INTRINSICS,
        CalibrationPhase.HAND_EYE,
        CalibrationPhase.CAMERA_VIA_LANDMARK,
    )
    if needs_mount and mount is None:
        raise ValueError(
            f"--mount is required for phase={phase.value} (eye_in_hand or eye_to_hand). "
            f"Example: --mount=eye_to_hand --camera={camera}"
        )
    return mount


@parser.wrap()
def calibrate_hand_eye(cfg: HandEyeCalibrationConfig) -> None:
    init_logging()

    if cfg.show_methods_help:
        print_methods_help()
        return

    run_cfg = cfg.to_run_config()
    robot = as_hand_eye_robot(make_robot_from_config(cfg.robot))

    robot.connect(calibrate=False)
    try:
        phase = cfg.phase
        mount = _require_mount(phase, cfg.mount, cfg.camera)

        if phase == CalibrationPhase.INTRINSICS:
            run_intrinsics_calibration(robot, cfg.camera, mount, run_cfg)
        elif phase == CalibrationPhase.HAND_EYE:
            run_hand_eye_calibration(robot, cfg.camera, mount, run_cfg)
        elif phase == CalibrationPhase.LANDMARK_MAP:
            run_landmark_map_calibration(robot, cfg.camera, run_cfg)
        elif phase == CalibrationPhase.CAMERA_VIA_LANDMARK:
            run_camera_via_landmark_calibration(robot, cfg.camera, mount, run_cfg)
        elif phase == CalibrationPhase.VALIDATE:
            report = validate_scene_calibration(robot, run_cfg)
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            raise ValueError(f"Unknown phase {phase}")
    finally:
        robot.disconnect()


def main() -> None:
    register_third_party_plugins()
    calibrate_hand_eye()


if __name__ == "__main__":
    main()
