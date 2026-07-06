#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Framework-agnostic RLT Stage-2 (TD3 + BC) loss math (paper §IV-B, Eq. 3 & 5).

This module is the **shared mathematical core** between ``lerobot`` (real-robot
online RL) and external training frameworks such as ``RLinf`` (large-scale
sim). It contains only pure free functions that operate on:

* ``nn.Module`` instances conforming to the RLT chunked-actor / chunked-critic
  call signatures (see :class:`~lerobot.policies.rlt_actor.ChunkActor` and
  :class:`~lerobot.policies.rlt_actor.ChunkCriticEnsemble`), and
* a plain ``forward_batch`` dict (the ``fb`` contract documented below).

Nothing here imports lerobot's ``RLAlgorithm`` / ``PreTrainedPolicy`` base
classes, hardware, datasets, or Ray — so it can be imported and unit-tested in
any environment that has ``torch`` and the RLT networks.

``fb`` (forward-batch) contract — a plain dict with these keys::

    {
        ACTION:        Tensor[B, C*d],          # executed action chunk (flat)
        "reward":      Tensor[B],
        "done":        Tensor[B],
        "state":       {"z_rl": Tensor[B, token_dim], "proprio": Tensor[B, proprio_dim]},
        "next_state":  {"z_rl": Tensor[B, token_dim], "proprio": Tensor[B, proprio_dim]},
        "ref_chunk":     Tensor[B, C*d],        # VLA reference chunk at s
        "ref_chunk_next":Tensor[B, C*d],        # VLA reference chunk at s'
        "source":        Tensor[B] (long),      # SOURCE_* label per window
        "bc_target":     Tensor[B, C*d],        # per-source BC target chunk
    }

Actor call contract (used by both losses)::

    actor(z_rl, proprio, ref_chunk, dropout_mask=None) -> (action, log_prob, mean)

Critic call contract::

    critic_ensemble(z_rl, proprio, action_chunk) -> Tensor[num_critics, B]
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.utils.constants import ACTION

# Control-source labels stored per replay window
# (mirror lerobot/rlt/replay_windows.py — kept duplicated as plain ints here so
# this shared math module has zero dependency on the ``lerobot.rlt`` package,
# avoiding a circular import when ``lerobot.rl.algorithms.rlt_td3`` is imported
# before ``lerobot.rlt``).
SOURCE_BASE = 0  # VLA-only chunk (warmup / no actor)
SOURCE_RL = 1  # actor-refined chunk
SOURCE_HUMAN = 2  # human-intervention chunk
SOURCE_MIXED = 3  # blended human + actor within the chunk

__all__ = [
    "SOURCE_BASE",
    "SOURCE_RL",
    "SOURCE_HUMAN",
    "SOURCE_MIXED",
    "bc_weight_schedule",
    "polyak_update",
    "ref_dropout_mask",
    "td3_critic_loss",
    "rlt_actor_loss",
]


def bc_weight_schedule(
    step: int,
    bc_weight_max: float,
    bc_weight_min: float,
    bc_decay_steps: int,
) -> float:
    """Linear BC-weight schedule (paper Eq. 5, ``lambda``).

    Decays linearly from ``bc_weight_max`` to ``bc_weight_min`` over
    ``bc_decay_steps`` optimizer steps, then stays at ``bc_weight_min``.
    ``bc_decay_steps <= 0`` keeps a constant ``bc_weight_max``.
    """
    if bc_decay_steps <= 0:
        return bc_weight_max
    frac = min(1.0, step / bc_decay_steps)
    return bc_weight_max + (bc_weight_min - bc_weight_max) * frac


def polyak_update(target_module: nn.Module, source_module: nn.Module, tau: float) -> None:
    """In-place soft target update ``θ_target ← (1-τ)·θ_target + τ·θ_source``.

    Runs under ``torch.no_grad()``. Parameter lists are zipped in registration
    order, so ``target`` and ``source`` must have identical architecture.
    """
    with torch.no_grad():
        for tp, p in zip(target_module.parameters(), source_module.parameters(), strict=True):
            tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)


def ref_dropout_mask(
    batch_size: int,
    prob: float,
    device: torch.device | str = "cpu",
    *,
    generator: torch.Generator | None = None,
) -> Tensor | None:
    """Reference-action dropout mask of shape ``[B, 1]`` (1 = keep, 0 = zero).

    Returns ``None`` when ``prob <= 0`` (no dropout), matching the actor's
    ``dropout_mask=None`` semantics. When a ``generator`` is supplied the draw
    is hermetic (used by conformance tests).
    """
    if prob <= 0.0:
        return None
    keep = (torch.rand(batch_size, 1, device=device, generator=generator) > prob).float()
    return keep


def td3_critic_loss(
    critic_ensemble: nn.Module,
    critic_target: nn.Module,
    actor: nn.Module,
    fb: dict[str, Any],
    discount: float,
) -> Tensor:
    """TD3 chunked Bellman critic loss (paper Eq. 3).

    Next action is the actor's deterministic mean ``μ(z'_rl, s'_p, ã')``; the
    target is the clipped double-Q ``min_e Q_target(z'_rl, s'_p, μ')``; the
    loss is MSE summed over the critic ensemble.
    """
    z_rl = fb["state"]["z_rl"]
    proprio = fb["state"]["proprio"]
    action = fb[ACTION]
    reward = fb["reward"]
    done = fb["done"].float()
    z_rl_next = fb["next_state"]["z_rl"]
    proprio_next = fb["next_state"]["proprio"]
    ref_next = fb["ref_chunk_next"]

    with torch.no_grad():
        next_mean = actor(z_rl_next, proprio_next, ref_next)[2]
        next_q = critic_target(z_rl_next, proprio_next, next_mean)  # [E, B]
        min_next_q = next_q.min(dim=0).values
        td_target = reward + (1.0 - done) * discount * min_next_q

    q_preds = critic_ensemble(z_rl, proprio, action)  # [E, B]
    td_target_exp = td_target.unsqueeze(0).expand_as(q_preds)
    return F.mse_loss(q_preds, td_target_exp, reduction="none").mean(dim=1).sum()


def rlt_actor_loss(
    actor: nn.Module,
    critic_ensemble: nn.Module,
    fb: dict[str, Any],
    *,
    chunk_len: int,
    action_dim: int,
    bc_weight: float,
    delta_penalty_weight: float,
    ref_dropout_prob: float = 0.0,
    in_warmup: bool = False,
    dropout_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """TD3 policy-improvement + BC-regularized actor loss (paper Eq. 5).

    Args:
        actor: callable with signature ``actor(z_rl, proprio, ref, mask) ->
            (action, log_prob, mean)``. Only the mean is used.
        critic_ensemble: callable ``critic(z_rl, proprio, a) -> [E, B]``.
        fb: forward-batch dict (see module docstring).
        chunk_len / action_dim: reshape dims for the delta-penalty term.
        bc_weight: current BC schedule weight (see :func:`bc_weight_schedule`).
        delta_penalty_weight: weight on the across-chunk action smoothness term.
        ref_dropout_prob: if ``> 0`` and ``dropout_mask`` is ``None``, a fresh
            dropout mask is sampled internally (non-deterministic; pass
            ``dropout_mask=`` explicitly for deterministic tests).
        in_warmup: when ``True`` the RL (min-Q) term is zeroed — the actor is
            trained with BC only (paper warmup phase).
        dropout_mask: optional precomputed dropout mask ``[B, 1]``. Takes
            precedence over internal sampling.

    Returns:
        ``(actor_loss, info)`` where ``info`` holds detached floats
        ``rl_loss`` / ``bc_loss`` / ``delta_penalty``.
    """
    z_rl = fb["state"]["z_rl"]
    proprio = fb["state"]["proprio"]
    ref = fb["ref_chunk"]
    source = fb["source"]
    bc_target = fb["bc_target"]

    if dropout_mask is None and ref_dropout_prob > 0.0:
        dropout_mask = ref_dropout_mask(z_rl.shape[0], ref_dropout_prob, z_rl.device)

    mean_flat = actor(z_rl, proprio, ref, dropout_mask)[2]

    # Policy improvement: maximize min-Q (TD3 clipped double Q).
    q_preds = critic_ensemble(z_rl, proprio, mean_flat)  # [E, B]
    min_q = q_preds.min(dim=0).values
    if in_warmup:
        rl_loss = torch.zeros((), device=min_q.device)
    else:
        rl_loss = -min_q.mean()

    # BC regularization toward the per-source BC target (RL windows contribute
    # no BC term — masked out via ``source != SOURCE_RL``).
    bc_mask = (source != SOURCE_RL).float()
    bc_per = F.mse_loss(mean_flat, bc_target, reduction="none").mean(dim=-1)
    denom = bc_mask.sum().clamp_min(1.0)
    bc_loss = (bc_per * bc_mask).sum() / denom

    # Action smoothness across the chunk (delta penalty).
    chunk = mean_flat.reshape(-1, chunk_len, action_dim)
    delta = (chunk[:, 1:] - chunk[:, :-1]).pow(2).mean(dim=[1, 2])
    delta_penalty = delta.mean()

    actor_loss = rl_loss + bc_weight * bc_loss + delta_penalty_weight * delta_penalty
    info = {
        "rl_loss": float(rl_loss.detach().item()),
        "bc_loss": float(bc_loss.detach().item()),
        "delta_penalty": float(delta_penalty.detach().item()),
    }
    return actor_loss, info
