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

    T_robot_to_ee: np.ndarray
    T_target_to_camera: np.ndarray


def solve_hand_eye(
    samples: list[HandEyeSample],
    mount: CameraMount,
    method: int = cv2.CALIB_HAND_EYE_TSAI,
) -> np.ndarray:
    """Solve for camera extrinsics.

    Returns:
        * ``eye_in_hand``: ``T_ee_to_camera`` (4x4)
        * ``eye_to_hand``: ``T_robot_to_camera`` (4x4), i.e. fixed camera in robot frame
    """
    if len(samples) < 3:
        raise ValueError(f"Need >= 3 hand-eye samples, got {len(samples)}.")

    r_gripper2base: list[np.ndarray] = []
    t_gripper2base: list[np.ndarray] = []
    r_cam2target: list[np.ndarray] = []
    t_cam2target: list[np.ndarray] = []

    for sample in samples:
        r_ee, t_ee = rotation_translation_from_transform(sample.T_robot_to_ee)
        # OpenCV ``calibrateHandEye`` expects camera→target (inverse of solvePnP board→camera).
        T_cam_to_target = invert_transform(sample.T_target_to_camera)
        r_tgt, t_tgt = rotation_translation_from_transform(T_cam_to_target)
        r_gripper2base.append(r_ee)
        t_gripper2base.append(t_ee.reshape(3, 1))
        r_cam2target.append(r_tgt)
        t_cam2target.append(t_tgt.reshape(3, 1))

    if mount == CameraMount.EYE_IN_HAND:
        r_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            r_gripper2base,
            t_gripper2base,
            r_cam2target,
            t_cam2target,
            method=method,
        )
        return make_transform(r_cam2gripper, t_cam2gripper.reshape(3))

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
        r_cam2target,
        t_cam2target,
        method=method,
    )
    return make_transform(r_cam2base, t_cam2base.reshape(3))


def camera_to_robot_transform(
    T_robot_to_ee: np.ndarray,
    T_ee_to_camera: np.ndarray,
) -> np.ndarray:
    """Compose eye-in-hand extrinsics to ``T_robot_to_camera`` at a given pose."""
    return T_robot_to_ee @ T_ee_to_camera


def robot_to_board_from_eye_in_hand(
    T_robot_to_ee: np.ndarray,
    T_ee_to_camera: np.ndarray,
    T_target_to_camera: np.ndarray,
) -> np.ndarray:
    """Board pose in robot frame via three rigid transforms (eye-in-hand, fixed board).

    ``T_target_to_camera`` is OpenCV solvePnP output (board → camera). Chain:

        T_robot_to_board = T_robot_to_ee @ T_ee_to_camera @ inv(T_target_to_camera)
    """
    T_camera_to_target = invert_transform(T_target_to_camera)
    return T_robot_to_ee @ T_ee_to_camera @ T_camera_to_target


def per_sample_board_origin_mm(
    samples: list[HandEyeSample],
    mount: CameraMount,
    T_solved: np.ndarray,
) -> np.ndarray:
    """Board origin in robot frame for each sample (mm)."""
    origins: list[np.ndarray] = []
    for sample in samples:
        if mount == CameraMount.EYE_IN_HAND:
            T_ref_to_target = robot_to_board_from_eye_in_hand(
                sample.T_robot_to_ee, T_solved, sample.T_target_to_camera
            )
        else:
            T_cam_to_target = invert_transform(sample.T_target_to_camera)
            T_robot_to_cam = T_solved
            T_robot_to_target = T_robot_to_cam @ T_cam_to_target
            T_ref_to_target = invert_transform(sample.T_robot_to_ee) @ T_robot_to_target
        origins.append(T_ref_to_target[:3, 3])
    return np.stack(origins, axis=0)


def hand_eye_motion_residual_mm(
    samples: list[HandEyeSample],
    T_ee_to_camera: np.ndarray,
) -> dict[str, float]:
    """Mean/max AX=XB residual (mm) for consecutive sample pairs."""
    if len(samples) < 2:
        return {"mean_mm": 0.0, "max_mm": 0.0}
    residuals: list[float] = []
    for i in range(1, len(samples)):
        d_gripper = invert_transform(samples[i - 1].T_robot_to_ee) @ samples[i].T_robot_to_ee
        d_target = invert_transform(samples[i - 1].T_target_to_camera) @ samples[i].T_target_to_camera
        lhs = d_gripper @ T_ee_to_camera
        rhs = T_ee_to_camera @ d_target
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

    * eye-in-hand: board fixed in robot frame → ``T_robot_to_target`` should be constant
    * eye-to-hand: board on gripper → ``T_ee_to_target`` should be constant
    """
    stacked = per_sample_board_origin_mm(samples, mount, T_solved)
    return float(np.linalg.norm(np.std(stacked, axis=0)))
