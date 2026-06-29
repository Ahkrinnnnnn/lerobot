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

import numpy as np

from lerobot.calibration.transforms import (
    compose_transforms,
    invert_transform,
    make_transform,
    rotation_matrix_from_euler_xyz_deg,
    transform_from_xyz_rpy_deg,
)


def test_transform_roundtrip():
    t = transform_from_xyz_rpy_deg(100.0, 200.0, 300.0, 10.0, -20.0, 30.0)
    inv = invert_transform(t)
    identity = compose_transforms(t, inv)
    assert np.allclose(identity, np.eye(4), atol=1e-9)


def test_rotation_matrix_is_proper():
    r = rotation_matrix_from_euler_xyz_deg(45.0, 0.0, 90.0)
    assert np.isclose(np.linalg.det(r), 1.0)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)


def test_make_transform():
    r = np.eye(3)
    t = np.array([1.0, 2.0, 3.0])
    m = make_transform(r, t)
    assert m.shape == (4, 4)
    assert np.allclose(m[:3, 3], t)
