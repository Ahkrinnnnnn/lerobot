#!/usr/bin/env python

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

from __future__ import annotations

import numpy as np
import torch

from lerobot.rl.ee_abs_to_delta import (
    abs_ee_action_to_delta,
    compute_abs_ee_delta_action_stats,
    detect_offline_ee_action_mode,
    ee_delta_action_dim,
)
from lerobot.utils.constants import ACTION


def test_detect_abs_ee_from_names():
    features = {
        ACTION: {
            "names": ["ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw", "gripper.pos"],
            "shape": [7],
        }
    }
    assert detect_offline_ee_action_mode(features) == "abs"


def test_ee_delta_action_dim():
    assert ee_delta_action_dim(include_rpy=True, use_gripper=True) == 7
    assert ee_delta_action_dim(include_rpy=False, use_gripper=True) == 4
    assert ee_delta_action_dim(include_rpy=True, use_gripper=False) == 6


def test_abs_ee_action_to_delta_with_and_without_rpy():
    state = torch.tensor([10.0, 20.0, 30.0, 0.0, 0.0, 179.0, 100.0])
    action = torch.tensor([11.5, 19.0, 32.0, 1.0, -2.0, -179.0, 400.0])
    out7 = abs_ee_action_to_delta(action, state, include_rpy=True)
    out4 = abs_ee_action_to_delta(action, state, include_rpy=False)
    assert out7.shape == (7,)
    assert out4.shape == (4,)
    assert torch.allclose(out4[:3], out7[:3])
    assert float(out4[3]) == float(out7[6]) == 400.0


def test_compute_delta_stats_include_rpy_toggle():
    actions = np.array(
        [
            [1.0, 2.0, 3.0, 0.5, 0.0, 0.0, 10.0],
            [2.0, 2.0, 5.0, 1.0, 0.0, 0.0, 20.0],
        ],
        dtype=np.float32,
    )
    states = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0, 0.5, 0.0, 0.0, 10.0],
        ],
        dtype=np.float32,
    )
    stats7 = compute_abs_ee_delta_action_stats(actions, states, include_rpy=True)
    stats4 = compute_abs_ee_delta_action_stats(actions, states, include_rpy=False)
    assert len(stats7["min"]) == 7
    assert len(stats4["min"]) == 4
    assert stats4["min"][3] == stats7["min"][6]
