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

from .hand_eye import CameraMount, robot_to_board_from_eye_in_hand
from .scene import SceneCalibration, TableCalibration
from .target import CalibrationTargetConfig, grid_points_in_robot_frame
from .transforms import average_rotation_matrices, compose_transforms, invert_transform, make_transform, transform_to_list


def fuse_robot_to_landmark(transforms: list[np.ndarray]) -> tuple[np.ndarray, float]:
    if not transforms:
        raise ValueError("Need at least one transform to fuse.")
    translations = np.stack([t[:3, 3] for t in transforms], axis=0)
    rotations = [t[:3, :3] for t in transforms]
    mean_t = translations.mean(axis=0)
    mean_r = average_rotation_matrices(rotations)
    mean_transform = make_transform(mean_r, mean_t)
    position_std = float(np.linalg.norm(np.std(translations, axis=0)))
    return mean_transform, position_std


def robot_to_landmark_from_eye_in_hand(
    T_robot_to_ee: np.ndarray,
    T_ee_to_camera: np.ndarray,
    T_landmark_to_camera: np.ndarray,
) -> np.ndarray:
    """Alias for :func:`robot_to_board_from_eye_in_hand` (landmark = fixed board)."""
    return robot_to_board_from_eye_in_hand(T_robot_to_ee, T_ee_to_camera, T_landmark_to_camera)


def camera_extrinsic_from_landmark(
    T_robot_to_landmark: np.ndarray,
    T_landmark_to_camera: np.ndarray,
) -> np.ndarray:
    return compose_transforms(T_robot_to_landmark, invert_transform(T_landmark_to_camera))


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
    _, std = fuse_robot_to_landmark(transforms)
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
    T_robot_to_landmark: np.ndarray,
    target: CalibrationTargetConfig,
    notes: str,
) -> None:
    scene.charuco = target.to_dict()
    grid = grid_points_in_robot_frame(T_robot_to_landmark, target)
    scene.table = TableCalibration(
        T_robot_to_table=transform_to_list(T_robot_to_landmark),
        grid_points_robot_mm=grid,
        notes=notes,
    )
