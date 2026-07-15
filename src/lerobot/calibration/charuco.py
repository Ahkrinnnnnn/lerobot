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
    board = cv2.aruco.CharucoBoard(
        (config.squares_x, config.squares_y),
        float(config.square_size),
        float(config.marker_size),
        dictionary,
    )
    if hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(True)
    return board


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


def _default_axis_length_mm(config: CharucoConfig) -> float:
    return float(config.square_size) * 3.0


def draw_board_frame_axes(
    vis_bgr: np.ndarray,
    intrinsics: CameraIntrinsics,
    T_target_to_camera: np.ndarray,
    *,
    axis_length_mm: float | None = None,
    config: CharucoConfig | None = None,
) -> None:
    """Draw ChArUco board frame on ``vis_bgr`` (BGR): X=red, Y=green, Z=blue (OpenCV)."""
    length = axis_length_mm
    if length is None:
        length = _default_axis_length_mm(config) if config is not None else 60.0
    R = T_target_to_camera[:3, :3]
    t = T_target_to_camera[:3, 3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)
    cv2.drawFrameAxes(
        vis_bgr,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
        rvec,
        t,
        float(length),
        3,
    )
    obj_pts = np.array(
        [[0.0, 0.0, 0.0], [length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]],
        dtype=np.float64,
    )
    img_pts, _ = cv2.projectPoints(
        obj_pts,
        rvec,
        t,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
    )
    labels = ("O", "X", "Y", "Z")
    colors = ((255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0))
    for i, (label, color) in enumerate(zip(labels, colors, strict=True)):
        x, y = int(round(img_pts[i, 0, 0])), int(round(img_pts[i, 0, 1]))
        cv2.putText(vis_bgr, label, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def calibrate_charuco_intrinsics(
    images: list[np.ndarray],
    config: CharucoConfig,
) -> CameraIntrinsics:
    if len(images) < 3:
        raise ValueError("Need >= 3 ChArUco views for intrinsics calibration.")

    board = make_charuco_board(config)

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    for image in images:
        detected = detect_charuco(image, config)
        if detected is None:
            continue

        corners, ids = detected
        h, w = image.shape[:2]
        image_size = (w, h)

        obj_pts, img_pts = board.matchImagePoints(corners, ids)

        if obj_pts is None or img_pts is None or len(obj_pts) < 4:
            continue

        object_points.append(obj_pts.astype(np.float32))
        image_points.append(img_pts.astype(np.float32))

    if image_size is None or len(object_points) < 3:
        raise RuntimeError(f"ChArUco visible in only {len(object_points)} images; need >= 3.")

    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points,
        image_points,
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


def draw_charuco(
    image: np.ndarray,
    config: CharucoConfig,
    *,
    intrinsics: CameraIntrinsics | None = None,
    T_target_to_camera: np.ndarray | None = None,
    axis_length_mm: float | None = None,
) -> tuple[np.ndarray, bool]:
    vis = np.asarray(image).copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    elif vis.shape[2] == 3:
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    detected = detect_charuco(image, config)
    if detected is None:
        return vis, False

    corners, ids = detected

    # OpenCV 4.13/5.0 may reject the returned corner/id shapes during drawing.
    # Drawing is only visualization; calibration can continue without it.
    try:
        cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids)
    except cv2.error as e:
        print(f"[WARN] drawDetectedCornersCharuco failed, skip drawing only: {e}")

    T_board = T_target_to_camera
    if intrinsics is not None and T_board is None:
        T_board = estimate_charuco_to_camera(image, intrinsics, config)
    if intrinsics is not None and T_board is not None:
        draw_board_frame_axes(
            vis,
            intrinsics,
            T_board,
            axis_length_mm=axis_length_mm,
            config=config,
        )

    return vis, True

def charuco_corner_points_board_mm(config: CharucoConfig) -> list[np.ndarray]:
    """Inner chessboard corners in ChArUco board frame (Z=0, mm)."""
    cols = config.squares_x - 1
    rows = config.squares_y - 1
    size = float(config.square_size)
    return [
        np.array([c * size, r * size, 0.0], dtype=np.float64) for r in range(rows) for c in range(cols)
    ]


def table_grid_points_robot_mm(T_table_to_robot: np.ndarray, config: CharucoConfig) -> list[list[float]]:
    """ChArUco inner corner grid in robot frame (mm)."""
    points: list[list[float]] = []
    for p in charuco_corner_points_board_mm(config):
        p_h = np.append(p, 1.0)
        p_robot = (T_table_to_robot @ p_h)[:3]
        points.append(p_robot.tolist())
    return points
