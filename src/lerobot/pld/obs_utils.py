# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.configs.types import FeatureType
from lerobot.policies.residual_gaussian.modeling_residual_gaussian import BASE_ACTION_KEY
from lerobot.utils.constants import ACTION, OBS_STATE


def camera_shape_to_policy_shape(
    shape: tuple,
    resize_size: tuple[int, int] | None = None,
) -> tuple[int, int, int]:
    """Convert robot camera (H, W, C) to policy tensor shape (C, H, W)."""
    if len(shape) == 3 and shape[-1] in (1, 3):
        h, w, c = shape
        chw = (int(c), int(h), int(w))
    elif len(shape) == 3:
        chw = (int(shape[0]), int(shape[1]), int(shape[2]))
    else:
        raise ValueError(f"Unexpected camera shape {shape}")
    if resize_size is not None:
        return (chw[0], resize_size[0], resize_size[1])
    return chw


def apply_residual_preprocessor(
    rl_state: dict[str, Tensor],
    preprocessor,
) -> dict[str, Tensor]:
    if preprocessor is None:
        return rl_state
    processed = preprocessor(rl_state)
    # Preprocessor output is a LeRobot batch (action/reward/done keys included); keep tensors only.
    return {
        k: v
        for k, v in processed.items()
        if k != ACTION and isinstance(v, torch.Tensor) and v is not None
    }


def fetch_obs_for_rl(ctx, obs_policy: dict | None, obs_processed: dict) -> dict:
    """Return policy-facing observations with cameras (RTC control ticks are proprio-only)."""
    if obs_policy is not None:
        return obs_policy
    robot = ctx.hardware.robot_wrapper
    try:
        obs_raw = robot.get_observation(include_images=True)
    except TypeError:
        obs_raw = robot.get_observation()
    return ctx.processors.robot_observation_processor(obs_raw)


def align_rl_state_shapes(
    state: dict[str, Tensor],
    input_features: dict,
) -> dict[str, Tensor]:
    """Resize visual tensors to ``input_features`` shapes (replay buffer must not store full-res cameras)."""
    out = dict(state)
    for key, ft in input_features.items():
        if ft.type is not FeatureType.VISUAL or key not in out:
            continue
        val = out[key]
        if not isinstance(val, torch.Tensor) or len(ft.shape) != 3:
            continue
        _, target_h, target_w = (int(x) for x in ft.shape)
        if val.ndim == 3:
            val = val.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        _, _, h, w = val.shape
        if (h, w) != (target_h, target_w):
            val = F.interpolate(val, size=(target_h, target_w), mode="bilinear", align_corners=False)
        out[key] = val.squeeze(0) if squeeze else val
    return out


def prepare_rl_state_for_buffer(
    obs_processed: dict[str, Any],
    input_features: dict,
    preprocessor,
    device: str = "cpu",
) -> dict[str, Tensor]:
    """Build RL state tensors sized for replay-buffer storage (typically 128×128 images)."""
    state = build_rl_state_from_obs(obs_processed, input_features, device=device)
    state = apply_residual_preprocessor(state, preprocessor)
    state = align_rl_state_shapes(state, input_features)
    return filter_rl_state_tensors(state, input_features)


def filter_rl_state_tensors(
    state: dict[str, Any],
    input_features: dict,
) -> dict[str, Tensor]:
    """Keep only visual/state tensors required by the residual policy."""
    allowed = {
        k
        for k, ft in input_features.items()
        if ft.type in (FeatureType.VISUAL, FeatureType.STATE)
    }
    out: dict[str, Tensor] = {}
    missing: list[str] = []
    for key in allowed:
        val = state.get(key)
        if val is None or not isinstance(val, torch.Tensor):
            missing.append(key)
            continue
        ft = input_features[key]
        if ft.type is FeatureType.VISUAL and len(ft.shape) == 3:
            _, eh, ew = (int(x) for x in ft.shape)
            shape = val.shape[-2:] if val.ndim >= 2 else ()
            if shape != (eh, ew):
                raise RuntimeError(
                    f"RL state {key!r} has spatial shape {tuple(shape)}, expected ({eh}, {ew}). "
                    "Replay buffer cannot store full-resolution camera frames."
                )
        out[key] = val
    if missing:
        raise RuntimeError(
            f"RL state missing required keys {missing}. "
            "Ensure camera observations are loaded each control tick (RTC mode)."
        )
    return out


def action_dict_to_tensor(action_dict: dict[str, float], ordered_keys: list[str]) -> Tensor:
    values = [float(action_dict[k]) for k in ordered_keys if k in action_dict]
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0)


def build_rl_state_from_obs(
    obs_processed: dict[str, Any],
    input_features: dict,
    device: str = "cpu",
) -> dict[str, Tensor]:
    """Map processed robot observations to residual-policy RL state tensors."""
    state: dict[str, Tensor] = {}
    for key, ft in input_features.items():
        if ft.type is FeatureType.VISUAL:
            short = key.replace("observation.images.", "")
            candidates = [key, short, f"observation.images.{short}"]
            for cand in candidates:
                if cand in obs_processed:
                    arr = obs_processed[cand]
                    if arr is None:
                        continue
                    if isinstance(arr, torch.Tensor):
                        t = arr
                        if t.dtype == torch.uint8:
                            t = t.float() / 255.0
                        if t.ndim == 3:
                            if t.shape[-1] in (1, 3):
                                t = t.permute(2, 0, 1)
                            t = t.unsqueeze(0)
                        elif t.ndim == 4 and t.shape[-1] in (1, 3):
                            t = t.permute(0, 3, 1, 2)
                        state[key] = t.to(device)
                    elif isinstance(arr, np.ndarray):
                        t = torch.from_numpy(arr.copy())
                        if t.dtype == torch.uint8:
                            t = t.float() / 255.0
                        if t.ndim == 3:
                            if t.shape[-1] in (1, 3):
                                t = t.permute(2, 0, 1)
                            t = t.unsqueeze(0)
                        state[key] = t.to(device)
                    break
        elif key == OBS_STATE or key.endswith(".state"):
            state_dim = int(ft.shape[0]) if getattr(ft, "shape", None) else None
            # Match robot proprio convention: arm joints only in observation.state (no gripper).
            joint_keys = sorted(
                k for k in obs_processed if k.endswith(".pos") and not k.startswith("gripper")
            )
            if state_dim is not None:
                joint_keys = joint_keys[:state_dim]
            if joint_keys:
                vals = [float(obs_processed[k]) for k in joint_keys]
                state[key] = torch.tensor(vals, dtype=torch.float32).unsqueeze(0).to(device)
    return state


def make_complementary_info(base_action: Tensor, next_base_action: Tensor | None = None) -> dict[str, Tensor]:
    info = {BASE_ACTION_KEY: base_action}
    if next_base_action is not None:
        info["next_base_action"] = next_base_action
    return info
