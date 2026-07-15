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

"""Hand-eye calibration solvers (eye-in-hand and eye-to-hand)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from .transforms import invert_transform, make_transform, rotation_translation_from_transform


class CameraMount(str, Enum):
    EYE_IN_HAND = "eye_in_hand"
    EYE_TO_HAND = "eye_to_hand"


@dataclass
class HandEyeSample:
    """One synchronized robot pose + chessboard observation."""

    T_ee_to_robot: np.ndarray
    """Gripper→base (OpenCV ``gripper2base``): ``p_robot = T @ p_ee``."""

    T_target_to_camera: np.ndarray
    """Board→camera (OpenCV solvePnP): ``p_cam = T @ p_board``."""


def solve_hand_eye(
    samples: list[HandEyeSample],
    mount: CameraMount,
    method: int = cv2.CALIB_HAND_EYE_TSAI,
) -> np.ndarray:
    """Solve for camera extrinsics.

    Returns:
        * ``eye_in_hand``: ``T_ee_to_camera`` (4x4) with ``p_cam = T @ p_ee``
        * ``eye_to_hand``: ``T_camera_to_robot`` (4x4) with ``p_robot = T @ p_cam``
          (fixed camera; OpenCV ``cam2base``)
    """
    if len(samples) < 3:
        raise ValueError(f"Need >= 3 hand-eye samples, got {len(samples)}.")

    r_gripper2base: list[np.ndarray] = []
    t_gripper2base: list[np.ndarray] = []
    r_target2cam: list[np.ndarray] = []
    t_target2cam: list[np.ndarray] = []

    for sample in samples:
        r_ee, t_ee = rotation_translation_from_transform(sample.T_ee_to_robot)
        # OpenCV expects board→camera (solvePnP), NOT camera→board.
        r_tgt, t_tgt = rotation_translation_from_transform(sample.T_target_to_camera)
        r_gripper2base.append(r_ee)
        t_gripper2base.append(t_ee.reshape(3, 1))
        r_target2cam.append(r_tgt)
        t_target2cam.append(t_tgt.reshape(3, 1))

    if mount == CameraMount.EYE_IN_HAND:
        r_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            r_gripper2base,
            t_gripper2base,
            r_target2cam,
            t_target2cam,
            method=method,
        )
        # OpenCV returns camera→gripper; we store gripper→camera.
        T_camera_to_ee = make_transform(r_cam2gripper, t_cam2gripper.reshape(3))
        return invert_transform(T_camera_to_ee)

    r_base2gripper: list[np.ndarray] = []
    t_base2gripper: list[np.ndarray] = []
    for r_g2b, t_g2b in zip(r_gripper2base, t_gripper2base, strict=True):
        r_b2g = r_g2b.T
        t_b2g = (-r_b2g @ t_g2b).reshape(3, 1)
        r_base2gripper.append(r_b2g)
        t_base2gripper.append(t_b2g)

    r_cam2base, t_cam2base = cv2.calibrateHandEye(
        r_base2gripper,
        t_base2gripper,
        r_target2cam,
        t_target2cam,
        method=method,
    )
    return make_transform(r_cam2base, t_cam2base.reshape(3))


def camera_to_robot_transform(
    T_ee_to_robot: np.ndarray,
    T_ee_to_camera: np.ndarray,
) -> np.ndarray:
    """``T_camera_to_robot`` with ``p_robot = T @ p_cam`` for eye-in-hand."""
    return T_ee_to_robot @ invert_transform(T_ee_to_camera)


def board_to_robot_from_eye_in_hand(
    T_ee_to_robot: np.ndarray,
    T_ee_to_camera: np.ndarray,
    T_target_to_camera: np.ndarray,
) -> np.ndarray:
    """Board pose in robot frame (eye-in-hand, fixed board).

    Conventions (point maps)::

        p_robot = T_ee_to_robot @ p_ee
        p_cam   = T_ee_to_camera @ p_ee
        p_cam   = T_target_to_camera @ p_board

    therefore::

        T_board_to_robot = T_ee_to_robot @ inv(T_ee_to_camera) @ T_target_to_camera
    """
    return T_ee_to_robot @ invert_transform(T_ee_to_camera) @ T_target_to_camera


def per_sample_board_origin_mm(
    samples: list[HandEyeSample],
    mount: CameraMount,
    T_solved: np.ndarray,
) -> np.ndarray:
    """Board origin in robot frame for each sample (mm)."""
    origins: list[np.ndarray] = []
    for sample in samples:
        if mount == CameraMount.EYE_IN_HAND:
            T_ref_to_target = board_to_robot_from_eye_in_hand(
                sample.T_ee_to_robot, T_solved, sample.T_target_to_camera
            )
        else:
            # T_solved = T_camera_to_robot with p_robot = T @ p_cam
            T_target_to_robot = T_solved @ sample.T_target_to_camera
            # Board fixed on gripper → report in ee frame
            T_ref_to_target = invert_transform(sample.T_ee_to_robot) @ T_target_to_robot
        origins.append(T_ref_to_target[:3, 3])
    return np.stack(origins, axis=0)


def hand_eye_motion_residual_mm(
    samples: list[HandEyeSample],
    T_ee_to_camera: np.ndarray,
) -> dict[str, float]:
    """Mean/max AX=XB residual (mm) for consecutive sample pairs.

    OpenCV eye-in-hand uses ``X = T_camera_to_ee = inv(T_ee_to_camera)`` with::

        A = inv(G_{i-1}) @ G_i
        B = C_{i-1} @ inv(C_i)
        A X = X B
    """
    if len(samples) < 2:
        return {"mean_mm": 0.0, "max_mm": 0.0}
    X_cam_to_ee = invert_transform(T_ee_to_camera)
    residuals: list[float] = []
    for i in range(1, len(samples)):
        d_gripper = invert_transform(samples[i - 1].T_ee_to_robot) @ samples[i].T_ee_to_robot
        d_target = samples[i - 1].T_target_to_camera @ invert_transform(samples[i].T_target_to_camera)
        lhs = d_gripper @ X_cam_to_ee
        rhs = X_cam_to_ee @ d_target
        rot_err = np.linalg.norm(lhs[:3, :3] - rhs[:3, :3])
        trans_err = np.linalg.norm(lhs[:3, 3] - rhs[:3, 3])
        residuals.append(float(rot_err + trans_err))
    return {"mean_mm": float(np.mean(residuals)), "max_mm": float(np.max(residuals))}


def hand_eye_target_origin_std(
    samples: list[HandEyeSample],
    mount: CameraMount,
    T_solved: np.ndarray,
) -> float:
    """Std-dev of chessboard origin in the reference frame (lower is better).

    * eye-in-hand: board fixed in robot frame → ``T_board_to_robot`` should be constant
    * eye-to-hand: board on gripper → board pose in ee frame should be constant
    """
    stacked = per_sample_board_origin_mm(samples, mount, T_solved)
    return float(np.linalg.norm(np.std(stacked, axis=0)))
