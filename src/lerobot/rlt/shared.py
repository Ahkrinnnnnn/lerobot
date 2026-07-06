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

"""Public shared-core surface for RLT Stage-2 (TD3 + BC).

This module is the single import point that external training frameworks
(e.g. ``RLinf``) use to *reuse* lerobot's RLT algorithmic core instead of
re-implementing it. It re-exports, from their canonical homes:

* **Networks** — :class:`~lerobot.policies.rlt_actor.ChunkActor`,
  :class:`~lerobot.policies.rlt_actor.ChunkCriticEnsemble` (pure ``nn.Module``).
* **Replay logic** — :class:`~lerobot.rlt.replay_windows.StepRecord`,
  :class:`~lerobot.rlt.replay_windows.ReplayWindow`,
  :func:`~lerobot.rlt.replay_windows.build_replay_windows` and the
  ``SOURCE_*`` control-source labels.
* **Loss / target math** — :func:`td3_critic_loss`, :func:`rlt_actor_loss`,
  :func:`polyak_update`, :func:`bc_weight_schedule`, :func:`ref_dropout_mask`.

It also exposes a deterministic **conformance fixture**
:func:`make_rlt_tiny_fixture` and a :func:`golden_loss_values` helper used by
the cross-framework identity test (see ``tests/rl/test_rlt_shared_core.py``
in lerobot and ``tests/unit_tests/test_rlt_shared_core_conformance.py`` in
RLinf). The fixture is hermetic: it saves and restores the global torch RNG
state so callers' randomness is unaffected.

Design rationale: keeping the math here as free functions operating on
``nn.Module`` + plain dicts (rather than on ``RLAlgorithm`` / ``PreTrainedPolicy``
subclasses) means both frameworks execute *the same* code path. Networks are
weight-compatible by construction (same class, same init), so a checkpoint
trained in RLinf (simulation) loads directly into lerobot (real robot) and
vice-versa — enabling sim-to-real / dual-framework cooperation without
re-implementation risk.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from lerobot.policies.rlt_actor import ChunkActor, ChunkCriticEnsemble, RLTActorConfig
from lerobot.policies.rlt_actor.configuration_rlt_actor import ChunkActorNetworkConfig
from lerobot.rl.algorithms.rlt_td3.losses import (
    SOURCE_BASE,
    SOURCE_HUMAN,
    SOURCE_MIXED,
    SOURCE_RL,
    bc_weight_schedule,
    polyak_update,
    ref_dropout_mask,
    rlt_actor_loss,
    td3_critic_loss,
)
from lerobot.rlt.replay_windows import (
    ReplayWindow,
    StepRecord,
    build_replay_windows,
)
from lerobot.utils.constants import ACTION

__all__ = [
    # Networks
    "ChunkActor",
    "ChunkCriticEnsemble",
    "RLTActorConfig",
    # Replay
    "StepRecord",
    "ReplayWindow",
    "build_replay_windows",
    "SOURCE_BASE",
    "SOURCE_RL",
    "SOURCE_HUMAN",
    "SOURCE_MIXED",
    # Loss / target math
    "td3_critic_loss",
    "rlt_actor_loss",
    "polyak_update",
    "bc_weight_schedule",
    "ref_dropout_mask",
    # Conformance
    "make_rlt_tiny_fixture",
    "golden_loss_values",
]


# ---------------------------------------------------------------------------
# Conformance fixture
# ---------------------------------------------------------------------------

# Fixed tiny dims — small enough to run on CPU in milliseconds, large enough
# to exercise every branch of the losses (ensemble min, BC masking, delta
# penalty, next-state target).
_TINY = {
    "token_dim": 8,
    "proprio_dim": 4,
    "chunk_len": 3,
    "action_dim": 2,
    "hidden_dims": [4, 4],
    "num_critics": 2,
    "fixed_std": 0.1,
    "batch_size": 5,
    "discount": 0.99,
    "bc_weight": 0.7,
    "delta_penalty_weight": 0.05,
}


def _build_tiny_actor_config() -> RLTActorConfig:
    # ref_dropout_prob=0 → deterministic loss path (no RNG in dropout).
    return RLTActorConfig(
        chunk_len=_TINY["chunk_len"],
        action_dim=_TINY["action_dim"],
        token_dim=_TINY["token_dim"],
        proprio_dim=_TINY["proprio_dim"],
        fixed_std=_TINY["fixed_std"],
        ref_dropout_prob=0.0,
        actor_network_kwargs=ChunkActorNetworkConfig(
            hidden_dims=list(_TINY["hidden_dims"]),
            activate_final=True,
            activations="SiLU",
        ),
    )


def make_rlt_tiny_fixture(seed: int = 0, device: str = "cpu") -> dict[str, Any]:
    """Build a deterministic tiny RLT actor + critic ensemble + forward batch.

    The fixture is hermetic: the global torch RNG state is saved and restored,
    so calling this function does not perturb the caller's randomness.

    Returns a dict with keys:
        ``actor``, ``critic_ensemble``, ``critic_target``, ``fb``, ``config``,
        ``meta`` (chunk_len, action_dim, discount, bc_weight,
        delta_penalty_weight, ref_dropout_prob, in_warmup, num_critics, dims).
    """
    state = torch.get_rng_state()
    try:
        torch.manual_seed(seed)

        cfg = _build_tiny_actor_config()
        actor = ChunkActor(cfg)
        chunk_dim = cfg.chunk_len * cfg.action_dim
        critic_ensemble = ChunkCriticEnsemble(
            token_dim=cfg.token_dim,
            proprio_dim=cfg.proprio_dim,
            chunk_dim=chunk_dim,
            num_critics=_TINY["num_critics"],
            hidden_dims=list(_TINY["hidden_dims"]),
            activations="SiLU",
            activate_final=True,
        )
        critic_target = ChunkCriticEnsemble(
            token_dim=cfg.token_dim,
            proprio_dim=cfg.proprio_dim,
            chunk_dim=chunk_dim,
            num_critics=_TINY["num_critics"],
            hidden_dims=list(_TINY["hidden_dims"]),
            activations="SiLU",
            activate_final=True,
        )
        critic_target.load_state_dict(critic_ensemble.state_dict())

        b = _TINY["batch_size"]
        fb = {
            ACTION: torch.randn(b, chunk_dim),
            "reward": torch.randn(b),
            "done": torch.zeros(b),
            "state": {
                "z_rl": torch.randn(b, cfg.token_dim),
                "proprio": torch.randn(b, cfg.proprio_dim),
            },
            "next_state": {
                "z_rl": torch.randn(b, cfg.token_dim),
                "proprio": torch.randn(b, cfg.proprio_dim),
            },
            "ref_chunk": torch.randn(b, chunk_dim),
            "ref_chunk_next": torch.randn(b, chunk_dim),
            # Source mix: BASE, RL, HUMAN, MIXED, BASE → exercises BC masking.
            "source": torch.tensor([SOURCE_BASE, SOURCE_RL, SOURCE_HUMAN, SOURCE_MIXED, SOURCE_BASE]),
            "bc_target": torch.randn(b, chunk_dim),
        }

        actor.to(device)
        critic_ensemble.to(device)
        critic_target.to(device)
        fb = _to_device(fb, device)

        meta = {
            "chunk_len": cfg.chunk_len,
            "action_dim": cfg.action_dim,
            "chunk_dim": chunk_dim,
            "token_dim": cfg.token_dim,
            "proprio_dim": cfg.proprio_dim,
            "num_critics": _TINY["num_critics"],
            "discount": _TINY["discount"],
            "bc_weight": _TINY["bc_weight"],
            "delta_penalty_weight": _TINY["delta_penalty_weight"],
            "ref_dropout_prob": 0.0,
            "in_warmup": False,
            "batch_size": b,
            "seed": seed,
            "device": device,
        }
        return {
            "actor": actor,
            "critic_ensemble": critic_ensemble,
            "critic_target": critic_target,
            "fb": fb,
            "config": cfg,
            "meta": meta,
        }
    finally:
        torch.set_rng_state(state)


def _to_device(fb: dict[str, Any], device: str) -> dict[str, Any]:
    def _move(x):
        if isinstance(x, Tensor):
            return x.to(device)
        if isinstance(x, dict):
            return {k: _move(v) for k, v in x.items()}
        return x

    return {k: _move(v) for k, v in fb.items()}


def golden_loss_values(seed: int = 0, device: str = "cpu") -> dict[str, float]:
    """Compute reference loss values from the tiny fixture (recomputed each call).

    Both lerobot's own test and RLinf's conformance test call this; RLinf also
    re-runs the losses on its own fixture copy and must match these values
    bit-for-bit on CPU. Because the fixture is deterministic and the loss path
    contains no RNG (``ref_dropout_prob=0``), the values are stable across
    processes / environments for a given torch version.
    """
    fx = make_rlt_tiny_fixture(seed=seed, device=device)
    meta = fx["meta"]
    loss_critic = td3_critic_loss(
        fx["critic_ensemble"], fx["critic_target"], fx["actor"], fx["fb"], meta["discount"]
    )
    loss_actor, info = rlt_actor_loss(
        fx["actor"],
        fx["critic_ensemble"],
        fx["fb"],
        chunk_len=meta["chunk_len"],
        action_dim=meta["action_dim"],
        bc_weight=meta["bc_weight"],
        delta_penalty_weight=meta["delta_penalty_weight"],
        ref_dropout_prob=meta["ref_dropout_prob"],
        in_warmup=meta["in_warmup"],
    )
    return {
        "loss_critic": float(loss_critic.detach().item()),
        "loss_actor": float(loss_actor.detach().item()),
        "rl_loss": info["rl_loss"],
        "bc_loss": info["bc_loss"],
        "delta_penalty": info["delta_penalty"],
    }
