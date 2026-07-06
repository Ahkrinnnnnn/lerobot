#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""TD3 + BC algorithm for RLT Stage-2 (paper §IV-B, Eq. 3 & 5)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer

from lerobot.policies.rlt_actor import ChunkCriticEnsemble, RLTActorPolicy
from lerobot.types import BatchType
from lerobot.utils.constants import ACTION
from lerobot.utils.transition import move_state_dict_to_device

from ..base import RLAlgorithm
from ..configs import TrainingStats
from .configuration_rlt_td3 import RLTTD3AlgorithmConfig
from .losses import (
    SOURCE_BASE,  # noqa: F401  (re-exported for downstream callers)
    SOURCE_HUMAN,  # noqa: F401
    SOURCE_MIXED,  # noqa: F401
    SOURCE_RL,  # noqa: F401
    bc_weight_schedule,
    polyak_update,
    rlt_actor_loss,
    td3_critic_loss,
)


class RLTTD3Algorithm(RLAlgorithm):
    """TD3 with chunked Bellman backup and BC-regularized actor (RLT Stage-2)."""

    config_class = RLTTD3AlgorithmConfig
    name = "rlt_td3"

    def __init__(self, policy: RLTActorPolicy, config: RLTTD3AlgorithmConfig):
        self.config = config
        self.policy_config = config.policy_config
        self.policy = policy
        self.optimizers: dict[str, Optimizer] = {}
        self._optimization_step: int = 0

        pcfg = policy.config
        self._token_dim = pcfg.token_dim
        self._proprio_dim = pcfg.proprio_dim
        self._chunk_dim = pcfg.chunk_dim
        self._chunk_len = pcfg.chunk_len
        self._action_dim = pcfg.action_dim

        self._ref_dropout_prob = (
            config.ref_dropout_prob
            if config.ref_dropout_prob is not None
            else pcfg.ref_dropout_prob
        )

        critic_kw = asdict(config.critic_network_kwargs)
        self.critic_ensemble = ChunkCriticEnsemble(
            token_dim=self._token_dim,
            proprio_dim=self._proprio_dim,
            chunk_dim=self._chunk_dim,
            num_critics=config.num_critics,
            hidden_dims=list(critic_kw["hidden_dims"]),
            activations=critic_kw["activations"],
            activate_final=critic_kw["activate_final"],
        )
        self.critic_target = ChunkCriticEnsemble(
            token_dim=self._token_dim,
            proprio_dim=self._proprio_dim,
            chunk_dim=self._chunk_dim,
            num_critics=config.num_critics,
            hidden_dims=list(critic_kw["hidden_dims"]),
            activations=critic_kw["activations"],
            activate_final=critic_kw["activate_final"],
        )
        self.critic_target.load_state_dict(self.critic_ensemble.state_dict())

        self._device = torch.device(pcfg.device or "cpu")
        self._move_to_device()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _move_to_device(self) -> None:
        self.policy.to(self._device)
        self.critic_ensemble.to(self._device)
        self.critic_target.to(self._device)

    def make_optimizers_and_scheduler(self) -> dict[str, Optimizer]:
        optim_cls = torch.optim.AdamW if self.config.use_adamw else torch.optim.Adam
        kw = {}
        if self.config.use_adamw:
            kw["weight_decay"] = self.config.optimizer_weight_decay
        actor_params = self.policy.get_optim_params()["actor"]
        self.optimizers = {
            "actor": optim_cls(actor_params, lr=self.config.actor_lr, **kw),
            "critic": optim_cls(self.critic_ensemble.parameters(), lr=self.config.critic_lr, **kw),
        }
        return self.optimizers

    def get_optimizers(self) -> dict[str, Optimizer]:
        return self.optimizers

    # ------------------------------------------------------------------
    # Weight schedule
    # ------------------------------------------------------------------

    def _current_bc_weight(self) -> float:
        return bc_weight_schedule(
            self._optimization_step,
            self.config.bc_weight_max,
            self.config.bc_weight_min,
            self.config.bc_decay_steps,
        )

    def _in_warmup(self) -> bool:
        return self._optimization_step < self.config.warmup_pretraining_updates

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, batch_iterator: Iterator[BatchType]) -> TrainingStats:
        clip = self.config.grad_clip_norm

        # Extra critic updates (UTD > 1); targets updated after each.
        for _ in range(self.config.utd_ratio - 1):
            batch = next(batch_iterator)
            fb = self._prepare_forward_batch(batch)
            loss_critic = self._compute_loss_critic(fb)
            self.optimizers["critic"].zero_grad()
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_ensemble.parameters(), max_norm=clip)
            self.optimizers["critic"].step()
            self._update_target_networks()

        batch = next(batch_iterator)
        fb = self._prepare_forward_batch(batch)
        loss_critic = self._compute_loss_critic(fb)
        self.optimizers["critic"].zero_grad()
        loss_critic.backward()
        critic_grad = torch.nn.utils.clip_grad_norm_(
            self.critic_ensemble.parameters(), max_norm=clip
        ).item()
        self.optimizers["critic"].step()

        stats = TrainingStats(
            losses={"loss_critic": loss_critic.item()},
            grad_norms={"critic": critic_grad},
        )

        if self._optimization_step % self.config.policy_update_freq == 0:
            loss_actor, actor_info = self._compute_loss_actor(fb)
            self.optimizers["actor"].zero_grad()
            loss_actor.backward()
            actor_grad = torch.nn.utils.clip_grad_norm_(
                self.policy.actor.parameters(), max_norm=clip
            ).item()
            self.optimizers["actor"].step()
            stats.losses["loss_actor"] = loss_actor.item()
            stats.grad_norms["actor"] = actor_grad
            stats.extra["bc_weight"] = self._current_bc_weight()
            stats.extra["rl_loss"] = actor_info["rl_loss"]
            stats.extra["bc_loss"] = actor_info["bc_loss"]
            stats.extra["delta_penalty"] = actor_info["delta_penalty"]

        polyak_update(self.critic_target, self.critic_ensemble, self.config.critic_target_update_weight)
        self._optimization_step += 1
        return stats

    # ------------------------------------------------------------------
    # Losses (delegated to the shared core in .losses)
    # ------------------------------------------------------------------

    def _compute_loss_critic(self, fb: dict[str, Any]) -> Tensor:
        return td3_critic_loss(
            self.critic_ensemble,
            self.critic_target,
            self.policy.actor,
            fb,
            self.config.discount,
        )

    def _compute_loss_actor(self, fb: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        return rlt_actor_loss(
            self.policy.actor,
            self.critic_ensemble,
            fb,
            chunk_len=self._chunk_len,
            action_dim=self._action_dim,
            bc_weight=self._current_bc_weight(),
            delta_penalty_weight=self.config.delta_penalty_weight,
            ref_dropout_prob=self._ref_dropout_prob,
            in_warmup=self._in_warmup(),
        )

    def _update_target_networks(self) -> None:
        polyak_update(self.critic_target, self.critic_ensemble, self.config.critic_target_update_weight)

    # ------------------------------------------------------------------
    # Batch packing
    # ------------------------------------------------------------------

    def _prepare_forward_batch(self, batch: BatchType) -> dict[str, Any]:
        info = batch.get("complementary_info", {}) or {}
        fb: dict[str, Any] = {
            ACTION: batch[ACTION],
            "reward": batch["reward"],
            "state": batch["state"],
            "next_state": batch["next_state"],
            "done": batch["done"],
            "ref_chunk": info["ref_chunk"],
            "ref_chunk_next": info["ref_chunk_next"],
            "source": info["source"],
            "bc_target": info["bc_target"],
        }
        return self._to_device(fb)

    def _to_device(self, fb: dict[str, Any]) -> dict[str, Any]:
        def _move(x):
            if isinstance(x, Tensor):
                return x.to(self._device, non_blocking=True)
            if isinstance(x, dict):
                return {k: _move(v) for k, v in x.items()}
            return x

        return {k: _move(v) for k, v in fb.items()}

    # ------------------------------------------------------------------
    # Checkpointing / weight sync
    # ------------------------------------------------------------------

    def get_weights(self) -> dict[str, Any]:
        return {
            "actor": move_state_dict_to_device(self.policy.actor.state_dict(), device="cpu"),
        }

    def load_weights(self, weights: dict[str, Any], device: str | torch.device = "cpu") -> None:
        self.policy.actor.load_state_dict(move_state_dict_to_device(weights["actor"], device=device))

    def state_dict(self) -> dict[str, Tensor]:
        bundle: dict[str, Tensor] = {}
        for k, v in self.critic_ensemble.state_dict().items():
            bundle[f"critic_ensemble.{k}"] = v.detach()
        for k, v in self.critic_target.state_dict().items():
            bundle[f"critic_target.{k}"] = v.detach()
        return bundle

    def load_state_dict(self, state_dict: dict[str, Tensor], device: str | torch.device = "cpu") -> None:
        ce = {k.removeprefix("critic_ensemble."): v for k, v in state_dict.items() if k.startswith("critic_ensemble.")}
        ct = {k.removeprefix("critic_target."): v for k, v in state_dict.items() if k.startswith("critic_target.")}
        if ce:
            self.critic_ensemble.load_state_dict(ce, strict=False)
        if ct:
            self.critic_target.load_state_dict(ct, strict=False)
