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

"""Visual multi-view table mapping and cross-camera extrinsics."""

from __future__ import annotations

import numpy as np

from .hand_eye import CameraMount, board_to_robot_from_eye_in_hand
from .scene import SceneCalibration, TableCalibration
from .target import CalibrationTargetConfig, grid_points_in_robot_frame
from .transforms import average_rotation_matrices, compose_transforms, invert_transform, make_transform, transform_to_list


def fuse_landmark_to_robot(transforms: list[np.ndarray]) -> tuple[np.ndarray, float]:
    if not transforms:
        raise ValueError("Need at least one transform to fuse.")
    translations = np.stack([t[:3, 3] for t in transforms], axis=0)
    rotations = [t[:3, :3] for t in transforms]
    mean_t = translations.mean(axis=0)
    mean_r = average_rotation_matrices(rotations)
    mean_transform = make_transform(mean_r, mean_t)
    position_std = float(np.linalg.norm(np.std(translations, axis=0)))
    return mean_transform, position_std


def landmark_to_robot_from_eye_in_hand(
    T_ee_to_robot: np.ndarray,
    T_ee_to_camera: np.ndarray,
    T_landmark_to_camera: np.ndarray,
) -> np.ndarray:
    """Alias for :func:`board_to_robot_from_eye_in_hand` (landmark = fixed board)."""
    return board_to_robot_from_eye_in_hand(T_ee_to_robot, T_ee_to_camera, T_landmark_to_camera)


def camera_extrinsic_from_landmark(
    T_board_to_robot: np.ndarray,
    T_landmark_to_camera: np.ndarray,
) -> np.ndarray:
    """``T_camera_to_robot``: ``p_robot = T_board_to_robot @ inv(T_landmark_to_camera) @ p_cam``."""
    return compose_transforms(T_board_to_robot, invert_transform(T_landmark_to_camera))


def get_eye_in_hand_extrinsic(scene: SceneCalibration, camera_name: str) -> np.ndarray:
    cam = scene.cameras[camera_name]
    if cam.mount != CameraMount.EYE_IN_HAND:
        raise ValueError(f"Camera {camera_name!r} must be eye_in_hand, got {cam.mount.value}.")
    if cam.T_ee_to_camera is None:
        raise RuntimeError(f"Run hand_eye for {camera_name!r} (eye_in_hand) first.")
    from .transforms import transform_from_list

    return transform_from_list(cam.T_ee_to_camera)


def intrinsics_from_scene(scene: SceneCalibration, camera_name: str):
    from .chessboard import CameraIntrinsics

    cam = scene.cameras[camera_name]
    return CameraIntrinsics(
        width=cam.width,
        height=cam.height,
        camera_matrix=cam.camera_matrix_np(),
        dist_coeffs=cam.dist_coeffs_np(),
        reprojection_error=cam.reprojection_error,
    )


def landmark_consistency_report(transforms: list[np.ndarray]) -> dict[str, float]:
    _, std = fuse_landmark_to_robot(transforms)
    translations = np.stack([t[:3, 3] for t in transforms], axis=0)
    return {
        "num_samples": float(len(transforms)),
        "position_std_mm": std,
        "max_pairwise_mm": float(
            max(
                (np.linalg.norm(translations[i] - translations[j]) for i in range(len(transforms)) for j in range(i + 1, len(transforms))),
                default=0.0,
            )
        ),
    }


def save_landmark_as_table(
    scene: SceneCalibration,
    T_landmark_to_robot: np.ndarray,
    target: CalibrationTargetConfig,
    notes: str,
) -> None:
    scene.charuco = target.to_dict()
    grid = grid_points_in_robot_frame(T_landmark_to_robot, target)
    scene.table = TableCalibration(
        T_table_to_robot=transform_to_list(T_landmark_to_robot),
        grid_points_robot_mm=grid,
        notes=notes,
    )
