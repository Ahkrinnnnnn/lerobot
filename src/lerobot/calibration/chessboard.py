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

"""Chessboard-based camera intrinsics and target pose estimation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .transforms import make_transform


@dataclass(frozen=True)
class ChessboardConfig:
    """Inner corner counts and physical square size."""

    cols: int = 9
    rows: int = 6
    square_size: float = 0.025
    square_size_unit: str = "m"


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    reprojection_error: float | None = None

    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2])


def _object_points(board: ChessboardConfig) -> np.ndarray:
    objp = np.zeros((board.rows * board.cols, 3), dtype=np.float32)
    grid = np.mgrid[0 : board.cols, 0 : board.rows].T.reshape(-1, 2)
    objp[:, :2] = grid.astype(np.float32)
    objp *= float(board.square_size)
    return objp


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def find_chessboard_corners(image: np.ndarray, board: ChessboardConfig) -> np.ndarray | None:
    """Return (N, 1, 2) corner array or ``None`` if detection fails."""
    gray = _to_gray(np.asarray(image))
    pattern_size = (board.cols, board.rows)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners


def calibrate_camera_intrinsics(
    images: list[np.ndarray],
    board: ChessboardConfig,
) -> CameraIntrinsics:
    """Estimate pinhole intrinsics from multiple chessboard views."""
    if len(images) < 3:
        raise ValueError("Need at least 3 images with a visible chessboard for intrinsics calibration.")

    obj_points = _object_points(board)
    img_points_list: list[np.ndarray] = []
    object_points_list: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    for image in images:
        corners = find_chessboard_corners(image, board)
        if corners is None:
            continue
        h, w = image.shape[:2]
        image_size = (w, h)
        img_points_list.append(corners)
        object_points_list.append(objp.copy())

    if image_size is None or len(img_points_list) < 3:
        raise RuntimeError(
            f"Chessboard visible in only {len(img_points_list)} images; need >= 3 valid views."
        )

    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points_list,
        img_points_list,
        image_size,
        None,
        None,
    )
    return CameraIntrinsics(
        width=image_size[0],
        height=image_size[1],
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs.reshape(-1),
        reprojection_error=float(rms),
    )


def estimate_target_to_camera(
    image: np.ndarray,
    intrinsics: CameraIntrinsics,
    board: ChessboardConfig,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Estimate ``T_target_to_camera`` as (R, t) from one image. Returns ``None`` on failure."""
    corners = find_chessboard_corners(image, board)
    if corners is None:
        return None

    objp = _object_points(board)
    ok, rvec, tvec = cv2.solvePnP(
        objp,
        corners,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    translation = tvec.reshape(3)
    return rotation.astype(np.float64), translation.astype(np.float64)


def target_to_camera_transform(
    image: np.ndarray,
    intrinsics: CameraIntrinsics,
    board: ChessboardConfig,
) -> np.ndarray | None:
    """Return 4x4 ``T_target_to_camera`` or ``None``."""
    pose = estimate_target_to_camera(image, intrinsics, board)
    if pose is None:
        return None
    rotation, translation = pose
    return make_transform(rotation, translation)


def draw_detected_board(
    image: np.ndarray,
    board: ChessboardConfig,
) -> tuple[np.ndarray, bool]:
    """Draw chessboard corners; returns (annotated_image, found)."""
    vis = np.asarray(image).copy()
    corners = find_chessboard_corners(vis, board)
    if corners is None:
        return vis, False
    cv2.drawChessboardCorners(vis, (board.cols, board.rows), corners, True)
    return vis, True
