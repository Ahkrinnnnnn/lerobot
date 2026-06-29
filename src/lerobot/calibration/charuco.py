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

"""ChArUco board detection, intrinsics, and pose estimation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .chessboard import CameraIntrinsics, _to_gray
from .transforms import make_transform


@dataclass(frozen=True)
class CharucoConfig:
    """ChArUco grid: number of chessboard squares and marker size."""

    squares_x: int = 7
    squares_y: int = 5
    square_size: float = 40.0
    marker_size: float = 30.0
    square_size_unit: str = "mm"
    aruco_dict: int = cv2.aruco.DICT_4X4_50


def make_charuco_board(config: CharucoConfig) -> cv2.aruco.CharucoBoard:
    dictionary = cv2.aruco.getPredefinedDictionary(config.aruco_dict)
    return cv2.aruco.CharucoBoard(
        (config.squares_x, config.squares_y),
        float(config.square_size),
        float(config.marker_size),
        dictionary,
    )


def make_charuco_detector(config: CharucoConfig) -> cv2.aruco.CharucoDetector:
    board = make_charuco_board(config)
    return cv2.aruco.CharucoDetector(board)


def detect_charuco(image: np.ndarray, config: CharucoConfig) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(charuco_corners, charuco_ids)`` or ``None``."""
    gray = _to_gray(np.asarray(image))
    detector = make_charuco_detector(config)
    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is None or len(ids) < 4:
        return None
    return corners, ids


def estimate_charuco_to_camera(
    image: np.ndarray,
    intrinsics: CameraIntrinsics,
    config: CharucoConfig,
) -> np.ndarray | None:
    """Return 4x4 ``T_charuco_to_camera``."""
    detected = detect_charuco(image, config)
    if detected is None:
        return None
    corners, ids = detected
    board = make_charuco_board(config)
    obj_pts, img_pts = board.matchImagePoints(corners, ids)
    if obj_pts is None or len(obj_pts) < 4:
        return None
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    return make_transform(rotation.astype(np.float64), tvec.reshape(3))


def calibrate_charuco_intrinsics(
    images: list[np.ndarray],
    config: CharucoConfig,
) -> CameraIntrinsics:
    if len(images) < 3:
        raise ValueError("Need >= 3 ChArUco views for intrinsics calibration.")

    board = make_charuco_board(config)
    all_corners: list[np.ndarray] = []
    all_ids: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    for image in images:
        detected = detect_charuco(image, config)
        if detected is None:
            continue
        corners, ids = detected
        h, w = image.shape[:2]
        image_size = (w, h)
        all_corners.append(corners)
        all_ids.append(ids)

    if image_size is None or len(all_corners) < 3:
        raise RuntimeError(f"ChArUco visible in only {len(all_corners)} images; need >= 3.")

    rms, camera_matrix, dist_coeffs, _, _ = cv2.aruco.calibrateCameraCharuco(
        all_corners,
        all_ids,
        board,
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


def draw_charuco(image: np.ndarray, config: CharucoConfig) -> tuple[np.ndarray, bool]:
    vis = np.asarray(image).copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    elif vis.shape[2] == 3:
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    detected = detect_charuco(image, config)
    if detected is None:
        return vis, False
    corners, ids = detected
    cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids)
    return vis, True


def charuco_corner_points_board_mm(config: CharucoConfig) -> list[np.ndarray]:
    """Inner chessboard corners in ChArUco board frame (Z=0, mm)."""
    cols = config.squares_x - 1
    rows = config.squares_y - 1
    size = float(config.square_size)
    return [
        np.array([c * size, r * size, 0.0], dtype=np.float64) for r in range(rows) for c in range(cols)
    ]


def table_grid_points_robot_mm(T_robot_to_table: np.ndarray, config: CharucoConfig) -> list[list[float]]:
    """ChArUco inner corner grid in robot frame (mm)."""
    points: list[list[float]] = []
    for p in charuco_corner_points_board_mm(config):
        p_h = np.append(p, 1.0)
        p_robot = (T_robot_to_table @ p_h)[:3]
        points.append(p_robot.tolist())
    return points
