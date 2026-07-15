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

"""Rigid-body transform utilities for camera / robot calibration."""

from __future__ import annotations

import math
import os

import numpy as np


def _axis_rotation(axis: str, angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)

    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

    raise ValueError(f"Unknown Euler axis: {axis!r}")


def rotation_matrix_from_euler_xyz_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Build rotation matrix from CRP ``[roll, pitch, yaw]`` in degrees.

    The CRP controller's displayed RPY convention may differ from the default
    assumption. Use env vars to test conventions without editing code:

        CRP_RPY_ORDER=xyz / XYZ / zyx / ZYX / xzy / XZY / yxz / YXZ / yzx / YZX / zxy / ZXY
        CRP_RPY_SIGNS=1,1,1 or -1,1,1 etc.
        CRP_RPY_INVERT=1 to use R.T

    Lowercase means fixed-axis/extrinsic composition.
    Uppercase means body-axis/intrinsic composition.
    """

    order = os.getenv("CRP_RPY_ORDER", "xyz").strip()
    if len(order) != 3 or {c.lower() for c in order} != {"x", "y", "z"}:
        raise ValueError(f"Invalid CRP_RPY_ORDER={order!r}; expected permutation like xyz, ZYX, etc.")

    raw_signs = os.getenv("CRP_RPY_SIGNS", "1,1,1").strip()
    signs = [float(v) for v in raw_signs.split(",")]
    if len(signs) != 3:
        raise ValueError(f"Invalid CRP_RPY_SIGNS={raw_signs!r}; expected e.g. 1,1,1 or -1,1,1")

    roll_deg *= signs[0]
    pitch_deg *= signs[1]
    yaw_deg *= signs[2]

    angle_by_axis_deg = {
        "x": roll_deg,
        "y": pitch_deg,
        "z": yaw_deg,
    }

    mats = [
        _axis_rotation(axis.lower(), math.radians(angle_by_axis_deg[axis.lower()]))
        for axis in order
    ]

    if order.islower():
        # Extrinsic/fixed-axis: xyz -> Rz @ Ry @ Rx, matching the old behavior.
        r = mats[0]
        for m in mats[1:]:
            r = m @ r
    elif order.isupper():
        # Intrinsic/body-axis: XYZ -> Rx @ Ry @ Rz.
        r = mats[0]
        for m in mats[1:]:
            r = r @ m
    else:
        raise ValueError(f"Use all-lowercase or all-uppercase CRP_RPY_ORDER, got {order!r}")

    if os.getenv("CRP_RPY_INVERT", "0").lower() in ("1", "true", "yes"):
        r = r.T

    return r


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from R (3x3) and t (3,) or (3,1)."""
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rotation
    out[:3, 3] = t
    return out


def transform_from_xyz_rpy_deg(
    x: float,
    y: float,
    z: float,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> np.ndarray:
    """Build ``T_ee_to_robot`` from CRP pose ``[x, y, z, roll, pitch, yaw]`` (mm + degrees).

    CRP ``read_end_pose_world`` / ``read_end_pose_user`` match the teach pendant:
    XYZ is the TCP origin in the chosen frame, RPY is TCP orientation. Composed as
    ``T = [R(rpy) | xyz]`` (default extrinsic xyz RPY) this is the gripper→base
    transform used by hand-eye (OpenCV ``gripper2base``).
    """
    return make_transform(
        rotation_matrix_from_euler_xyz_deg(roll_deg, pitch_deg, yaw_deg),
        np.array([x, y, z], dtype=np.float64),
    )


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid transform."""
    t = np.asarray(transform, dtype=np.float64)
    r = t[:3, :3]
    p = t[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ p
    return out


def compose_transforms(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return ``left @ right`` for 4x4 transforms."""
    return np.asarray(left, dtype=np.float64) @ np.asarray(right, dtype=np.float64)


def rotation_translation_from_transform(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(transform, dtype=np.float64)
    return t[:3, :3].copy(), t[:3, 3].copy()


def transform_to_list(transform: np.ndarray) -> list[list[float]]:
    return np.asarray(transform, dtype=np.float64).tolist()


def transform_from_list(values: list[list[float]]) -> np.ndarray:
    return np.array(values, dtype=np.float64)


def average_rotation_matrices(rotations: list[np.ndarray]) -> np.ndarray:
    """Average rotations via SVD (Markley et al.)."""
    if not rotations:
        raise ValueError("Need at least one rotation matrix.")
    m = np.zeros((3, 3), dtype=np.float64)
    for r in rotations:
        m += r
    u, _, vt = np.linalg.svd(m)
    r_avg = u @ vt
    if np.linalg.det(r_avg) < 0:
        u[:, -1] *= -1
        r_avg = u @ vt
    return r_avg
