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

"""ChArUco calibration target API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .board_config import ARUCO_DICT_BY_NAME, resolve_aruco_dict
from .charuco import (
    CharucoConfig,
    calibrate_charuco_intrinsics,
    draw_charuco,
    estimate_charuco_to_camera,
    table_grid_points_robot_mm,
)
from .chessboard import CameraIntrinsics


_ARUCO_DICT_NAME_BY_ID = {v: k for k, v in ARUCO_DICT_BY_NAME.items()}


@dataclass
class CalibrationTargetConfig:
    """ChArUco board glued on the table (same board for all phases)."""

    squares_x: int = 8
    squares_y: int = 11
    square_size_mm: float = 15.0
    marker_size_mm: float = 11.0
    aruco_dict: int = cv2.aruco.DICT_4X4_50

    def charuco(self) -> CharucoConfig:
        return CharucoConfig(
            squares_x=self.squares_x,
            squares_y=self.squares_y,
            square_size=self.square_size_mm,
            marker_size=self.marker_size_mm,
            square_size_unit="mm",
            aruco_dict=self.aruco_dict,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "squares_x": self.squares_x,
            "squares_y": self.squares_y,
            "square_size_mm": self.square_size_mm,
            "marker_size_mm": self.marker_size_mm,
            "aruco_dict": _ARUCO_DICT_NAME_BY_ID.get(self.aruco_dict, str(self.aruco_dict)),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationTargetConfig:
        raw_dict = data.get("aruco_dict", "DICT_4X4_50")
        if isinstance(raw_dict, str):
            aruco_dict = resolve_aruco_dict(raw_dict)
        else:
            aruco_dict = int(raw_dict)
        return cls(
            squares_x=int(data.get("squares_x", 8)),
            squares_y=int(data.get("squares_y", 11)),
            square_size_mm=float(data.get("square_size_mm", 15.0)),
            marker_size_mm=float(data.get("marker_size_mm", 11.0)),
            aruco_dict=aruco_dict,
        )


def calibrate_intrinsics(images: list[np.ndarray], target: CalibrationTargetConfig) -> CameraIntrinsics:
    return calibrate_charuco_intrinsics(images, target.charuco())


def estimate_target_to_camera(
    image: np.ndarray,
    intrinsics: CameraIntrinsics,
    target: CalibrationTargetConfig,
) -> np.ndarray | None:
    """Return 4x4 ``T_charuco_to_camera``."""
    return estimate_charuco_to_camera(image, intrinsics, target.charuco())


def detect_target(image: np.ndarray, target: CalibrationTargetConfig) -> bool:
    from .charuco import detect_charuco

    return detect_charuco(image, target.charuco()) is not None


def draw_target(image: np.ndarray, target: CalibrationTargetConfig) -> tuple[np.ndarray, bool]:
    vis, found = draw_charuco(image, target.charuco())
    if vis.ndim == 3 and vis.shape[2] == 3:
        vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    return vis, found


def grid_points_in_robot_frame(T_robot_to_table: np.ndarray, target: CalibrationTargetConfig) -> list[list[float]]:
    return table_grid_points_robot_mm(T_robot_to_table, target.charuco())
