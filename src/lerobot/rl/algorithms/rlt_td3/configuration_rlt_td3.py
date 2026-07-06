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

"""RLT Stage-2 algorithm: TD3 + BC regularization on chunked actions (paper §IV-B)."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.gaussian_actor.configuration_gaussian_actor import CriticNetworkConfig

from ..configs import RLAlgorithmConfig


@RLAlgorithmConfig.register_subclass("rlt_td3")
@dataclass
class RLTTD3AlgorithmConfig(RLAlgorithmConfig):
    """TD3-style off-policy algorithm for RLT Stage-2.

    Differences from SAC (per the RLT paper):
      * No entropy / temperature term — the actor uses a fixed-std Gaussian
        (deterministic mean at inference).
      * Clipped double-Q (min over the critic ensemble) for both Bellman backup
        and the policy-improvement loss.
      * Delayed policy updates (``policy_update_freq``) and soft target EMA.
      * Behavioral-cloning regularization toward a per-source BC target
        (VLA reference chunk for ``BASE`` data, human chunk for ``HUMAN``/
        ``MIXED`` data; ``RL`` data contributes no BC term), with a warmup→online
        weight schedule.
      * Optional action-smoothness ``delta_penalty`` across the chunk.
    """

    # Optimizer learning rates
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    # Bellman update
    discount: float = 0.99
    # Soft target EMA weight (TD3 default 0.005).
    critic_target_update_weight: float = 0.005

    # Critic ensemble
    num_critics: int = 2
    critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)

    # Update loop
    utd_ratio: int = 1
    policy_update_freq: int = 2  # TD3: delayed policy updates
    grad_clip_norm: float = 40.0
    use_adamw: bool = False
    optimizer_weight_decay: float = 0.01

    # BC regularization
    # Maximum BC weight applied during warmup (paper Eq. 5: lambda).
    bc_weight_max: float = 1.0
    # Number of optimizer steps over which bc_weight decays from bc_weight_max
    # to bc_weight_min (linear). 0 = constant bc_weight_max.
    bc_decay_steps: int = 10000
    bc_weight_min: float = 0.0
    # Number of pre-training (warmup) updates before the actor is released for
    # online collection. During warmup the actor is trained with BC only and
    # the rollout runs the frozen VLA reference (no actor sampling).
    warmup_pretraining_updates: int = 0

    # Action-smoothness penalty weight (openpi-RLT delta_penalty).
    delta_penalty_weight: float = 0.0

    # Frozen-VLA reference dropout applied to the actor input during training
    # (paper: 0.5). Read from the policy config when None.
    ref_dropout_prob: float | None = None

    # Policy config (populated by the pipeline).
    policy_config: PreTrainedConfig | None = None

    @classmethod
    def from_policy_config(cls, policy_cfg) -> RLTTD3AlgorithmConfig:
        return cls(
            policy_config=policy_cfg,
            ref_dropout_prob=getattr(policy_cfg, "ref_dropout_prob", None),
        )
