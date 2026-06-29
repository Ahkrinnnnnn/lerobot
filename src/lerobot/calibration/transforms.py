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

import numpy as np


def rotation_matrix_from_euler_xyz_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Build R = Rz(yaw) @ Ry(pitch) @ Rx(roll) from degrees (CRP ``read_end_pose_*`` convention)."""
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


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
    """Build ``T`` from CRP-style pose ``[x, y, z, roll, pitch, yaw]`` (mm + degrees)."""
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
