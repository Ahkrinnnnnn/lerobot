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

from .board_config import CharucoBoardConfig
from .charuco import (
    CharucoConfig,
    calibrate_charuco_intrinsics,
    draw_charuco,
    estimate_charuco_to_camera,
    table_grid_points_robot_mm,
)
from .chessboard import CameraIntrinsics


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
        return CharucoBoardConfig.from_target(self).to_dict()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationTargetConfig:
        """Parse board JSON or scene ``charuco`` block (same keys as ``charuco_board.json``)."""
        return CharucoBoardConfig.from_dict(data).to_target()

    @classmethod
    def from_board_config(cls, board: CharucoBoardConfig) -> CalibrationTargetConfig:
        return board.to_target()


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


def draw_target(
    image: np.ndarray,
    target: CalibrationTargetConfig,
    *,
    intrinsics: CameraIntrinsics | None = None,
    T_target_to_camera: np.ndarray | None = None,
    axis_length_mm: float | None = None,
) -> tuple[np.ndarray, bool]:
    vis, found = draw_charuco(
        image,
        target.charuco(),
        intrinsics=intrinsics,
        T_target_to_camera=T_target_to_camera,
        axis_length_mm=axis_length_mm,
    )
    if vis.ndim == 3 and vis.shape[2] == 3:
        vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    return vis, found


def grid_points_in_robot_frame(T_robot_to_table: np.ndarray, target: CalibrationTargetConfig) -> list[list[float]]:
    return table_grid_points_robot_mm(T_robot_to_table, target.charuco())
