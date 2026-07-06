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

"""Configuration for the RLT Stage-2 chunked actor-critic policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig


@dataclass
class ChunkActorNetworkConfig:
    """MLP architecture for the chunked Gaussian actor."""

    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    activate_final: bool = True
    activations: str = "SiLU"


@PreTrainedConfig.register_subclass("rlt_actor")
@dataclass
class RLTActorConfig(PreTrainedConfig):
    """Lightweight chunked actor-critic on top of a frozen RL Token + base VLA.

    The actor refines the frozen VLA reference action chunk ``ã_{1:C}`` into an
    executed chunk ``a_{1:C}`` (RLT paper Eq. 4). The critic estimates
    ``Q(z_rl, s_p, a_{1:C})`` (Eq. 3). Both are small MLPs that consume the
    compact RL token ``z_rl`` produced by the frozen Stage-1 module, plus
    proprioceptive state ``s_p`` and the (optionally dropout-masked) reference
    chunk.

    The heavy base VLA + RLT encoder are referenced via ``rlt_checkpoint_path``
    (a Stage-1 RLT checkpoint directory) and are *not* stored in this policy's
    checkpoints.
    """

    # Path to a Stage-1 RLT checkpoint (contains config.json + RL token weights,
    # and references the frozen base VLA via base_policy.path).
    rlt_checkpoint_path: str | None = None

    # Chunk length C (paper: 10) and per-step action dimension d (paper: 14).
    chunk_len: int = 10
    action_dim: int = 14

    # RL token feature dimension (flatten of num_rl_tokens x embed_dim; paper: 2048).
    token_dim: int = 2048
    # Proprioceptive state dimension s_p (e.g. arm joint count).
    proprio_dim: int = 14

    # Actor MLP architecture.
    actor_network_kwargs: ChunkActorNetworkConfig = field(default_factory=ChunkActorNetworkConfig)

    # Gaussian actor with a small fixed standard deviation (paper Appendix B).
    fixed_std: float = 0.1
    # Reference-action dropout probability during training (paper: 0.5).
    ref_dropout_prob: float = 0.5

    # Action representation: "delta_chunk" (delta actions, openpi-RLT alignment)
    # or "abs_chunk" (absolute actions).
    action_representation: str = "delta_chunk"

    # Optimization (algorithm-side overrides these via make_optimizers_and_scheduler).
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    def __post_init__(self):
        super().__post_init__()
        if self.rlt_checkpoint_path is not None:
            self.pretrained_path = Path(self.rlt_checkpoint_path)
        if self.chunk_len < 1:
            raise ValueError(f"chunk_len must be >= 1, got {self.chunk_len}")
        if self.action_representation not in ("delta_chunk", "abs_chunk"):
            raise ValueError(
                f"action_representation must be 'delta_chunk' or 'abs_chunk', "
                f"got {self.action_representation!r}"
            )

    @property
    def chunk_dim(self) -> int:
        """Flattened action chunk dimension C*d."""
        return self.chunk_len * self.action_dim

    @property
    def critic_action_dim(self) -> int:
        return self.chunk_dim

    def validate_features(self) -> None:
        # RLT actor/critic operate on z_rl + proprio, not on raw dataset features.
        # Feature validation is delegated to the frozen base VLA at construction.
        return None

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self):
        return None

    @property
    def reward_delta_indices(self):
        return None

    def get_optimizer_preset(self):
        from lerobot.optim import MultiAdamConfig

        return MultiAdamConfig(
            weight_decay=0.0,
            optimizer_groups={
                "actor": {"lr": self.actor_lr},
                "critic": {"lr": self.critic_lr},
            },
        )

    def get_scheduler_preset(self):
        return None

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["rlt_checkpoint_path"]
