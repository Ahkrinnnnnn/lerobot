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

"""Load-time CRP EE abs→delta helpers for HIL-SERL offline demos (disk unchanged)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from lerobot.robots.crp_arm.ee_gp import wrap_angle_delta_deg
from lerobot.utils.constants import ACTION, OBS_STATE

logger = logging.getLogger(__name__)

_DELTA_NAME_TOKENS = ("delta_x", "delta_y", "delta_z")
_ABS_EE_NAME_TOKENS = ("ee.x", "ee.y", "ee.z")


def action_feature_names(features: dict[str, Any] | None) -> list[str]:
    if not features or ACTION not in features:
        return []
    names = features[ACTION].get("names") or []
    return [str(n) for n in names]


def detect_offline_ee_action_mode(
    features: dict[str, Any] | None,
    *,
    override: str | None = None,
) -> str:
    """Return ``"delta"``, ``"abs"``, or ``"other"``.

    ``override`` may be ``"abs"`` / ``"delta"`` to force the mode.
    """
    if override in ("abs", "delta"):
        return override
    names = action_feature_names(features)
    lowered = [n.lower() for n in names]
    if any(tok in n for n in lowered for tok in _DELTA_NAME_TOKENS):
        return "delta"
    if any(tok in n for n in lowered for tok in _ABS_EE_NAME_TOKENS):
        return "abs"
    return "other"


def ee_delta_action_dim(*, include_rpy: bool, use_gripper: bool = True) -> int:
    """3 (xyz) [+3 rpy] [+1 gripper]."""
    dim = 6 if include_rpy else 3
    if use_gripper:
        dim += 1
    return dim


def abs_ee_action_to_delta(
    action: torch.Tensor | np.ndarray,
    state: torch.Tensor | np.ndarray,
    *,
    include_rpy: bool = True,
) -> torch.Tensor:
    """Convert abs EE command to delta.

    - ``include_rpy=True`` → 7D: δxyz, δrpy (shortest-angle deg), absolute gripper
    - ``include_rpy=False`` → 4D: δxyz, absolute gripper (rpy ignored / held at runtime)
    """
    a = torch.as_tensor(action, dtype=torch.float32).reshape(-1)
    s = torch.as_tensor(state, dtype=torch.float32).reshape(-1)
    if a.numel() < 3 or s.numel() < 3:
        raise ValueError(f"abs EE action/state need >=3 dims, got action={a.numel()} state={s.numel()}")
    dxyz = a[:3] - s[:3]
    grip = a[6] if a.numel() >= 7 else (a[-1] if a.numel() >= 4 else torch.tensor(0.0))
    grip = grip.to(dtype=torch.float32).reshape(1)
    if not include_rpy:
        return torch.cat([dxyz, grip])
    if a.numel() < 6 or s.numel() < 6:
        raise ValueError(f"include_rpy requires >=6 pose dims, got action={a.numel()} state={s.numel()}")
    drpy = torch.tensor(
        [wrap_angle_delta_deg(float(a[i]), float(s[i])) for i in range(3, 6)],
        dtype=torch.float32,
    )
    return torch.cat([dxyz, drpy, grip])


# Back-compat aliases
def abs_ee_action_to_delta7(action, state):
    return abs_ee_action_to_delta(action, state, include_rpy=True)


def abs_ee_action_to_delta4(action, state):
    return abs_ee_action_to_delta(action, state, include_rpy=False)


def compute_abs_ee_delta_action_stats(
    actions: np.ndarray,
    states: np.ndarray,
    *,
    include_rpy: bool = True,
) -> dict[str, list[float]]:
    """Min/max/mean/std for converted delta actions (no disk write)."""
    if actions.ndim != 2 or states.ndim != 2:
        raise ValueError("actions/states must be 2D arrays")
    if actions.shape[1] < 3 or states.shape[1] < 3:
        raise ValueError("actions/states need >=3 columns for EE xyz")
    dxyz = actions[:, :3] - states[:, :3]
    grip = actions[:, 6] if actions.shape[1] >= 7 else actions[:, -1]
    if include_rpy:
        if actions.shape[1] < 6 or states.shape[1] < 6:
            raise ValueError("include_rpy requires >=6 pose columns")
        drpy = np.stack(
            [
                [
                    wrap_angle_delta_deg(float(a), float(s))
                    for a, s in zip(actions[:, i], states[:, i], strict=True)
                ]
                for i in range(3, 6)
            ],
            axis=1,
        )
        arr = np.concatenate([dxyz, drpy, grip.reshape(-1, 1)], axis=1).astype(np.float64)
    else:
        arr = np.concatenate([dxyz, grip.reshape(-1, 1)], axis=1).astype(np.float64)
    return {
        "min": arr.min(axis=0).astype(float).tolist(),
        "max": arr.max(axis=0).astype(float).tolist(),
        "mean": arr.mean(axis=0).astype(float).tolist(),
        "std": (arr.std(axis=0) + 1e-8).astype(float).tolist(),
    }


def maybe_override_policy_action_stats_from_abs_ee_dataset(
    dataset_stats: dict[str, dict[str, list[float]]] | None,
    *,
    features: dict[str, Any] | None,
    hf_dataset: Any | None,
    override: str | None = None,
    include_rpy: bool = True,
) -> dict[str, dict[str, list[float]]] | None:
    """If demo is abs EE, replace ``action`` stats with load-time delta stats."""
    mode = detect_offline_ee_action_mode(features, override=override)
    if mode != "abs":
        if mode == "delta":
            logger.info("offline EE action already delta — keeping dataset action stats")
        return dataset_stats
    if hf_dataset is None:
        logger.warning("abs EE demo detected but hf_dataset missing; action stats not recomputed")
        return dataset_stats

    cols = hf_dataset.select_columns([ACTION, OBS_STATE])
    actions = np.asarray(cols[ACTION], dtype=np.float32)
    states = np.asarray(cols[OBS_STATE], dtype=np.float32)
    delta_stats = compute_abs_ee_delta_action_stats(actions, states, include_rpy=include_rpy)
    out = dict(dataset_stats or {})
    out[ACTION] = delta_stats
    logger.info(
        "offline EE abs→delta: recomputed action stats include_rpy=%s dim=%s min=%s max=%s "
        "(disk unchanged)",
        include_rpy,
        len(delta_stats["min"]),
        [round(x, 4) for x in delta_stats["min"]],
        [round(x, 4) for x in delta_stats["max"]],
    )
    return out
