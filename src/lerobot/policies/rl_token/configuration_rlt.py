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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig

from .rl_token import RLTokenConfig


@PreTrainedConfig.register_subclass("rlt")
@dataclass
class RLTConfig(PreTrainedConfig):
    """RL Token wrapper around any registered base VLA policy.

    Set ``rlt_alpha=0`` to train only the RL token module (frozen base policy).
    Set ``rlt_alpha>0`` to jointly fine-tune the base policy with its native SFT loss.
    """

    # NOTE: Must NOT annotate as PreTrainedConfig (or any ChoiceRegistry subclass).
    # RLTConfig itself is a PreTrainedConfig; draccus would recurse forever when
    # building CLI wrappers for nested base-policy choice fields.
    # RLT checkpoints are usually saved locally alongside a frozen base VLA path.
    push_to_hub: bool = False
    base_policy: Any = field(default=None)

    rlt_num_tokens: int = 1
    rlt_num_layers: int = 2
    rlt_embed_dim: int = 2048
    rlt_input_dim: int = 2048
    rlt_mlp_ratio: float = 4.0
    rlt_num_heads: int = 8
    rlt_dropout: float = 0.0
    # Paper Algorithm 1: L = L_ro + alpha * L_vla.
    # Mode switch (authoritative): when false, rlt_alpha is forced to 0.
    rlt_finetune_vla: bool = False
    # VLA SFT loss weight (paper alpha). Only used when rlt_finetune_vla=true; defaults to 1.0.
    rlt_alpha: float = 0.0
    rlt_optimizer_lr: float = 1e-4
    rlt_scheduler_warmup_steps: int = 500
    rlt_scheduler_decay_steps: int = 10_000

    # When True (default), checkpoints store only the RL token encoder-decoder weights.
    # Base VLA weights stay at ``base_policy.pretrained_path`` and are referenced in the saved config.
    rlt_save_token_only: bool = True

    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        PreTrainedConfig.__post_init__(self)
        if self.rlt_num_tokens < 1:
            raise ValueError(f"rlt_num_tokens must be >= 1, got {self.rlt_num_tokens}")
        if self.rlt_alpha < 0:
            raise ValueError(f"rlt_alpha must be >= 0, got {self.rlt_alpha}")
        if self.rlt_finetune_vla:
            if self.rlt_alpha <= 0:
                self.rlt_alpha = 1.0
        else:
            self.rlt_alpha = 0.0

        self.base_policy = self._resolve_base_policy()
        self._configure_base_policy_for_rlt()

        if self.device is not None:
            self.base_policy.device = self.device
        if not self.input_features:
            self.input_features = self.base_policy.input_features
        if not self.output_features:
            self.output_features = self.base_policy.output_features
        if self.n_obs_steps == 1 and self.base_policy.n_obs_steps != 1:
            self.n_obs_steps = self.base_policy.n_obs_steps

    def _resolve_base_policy(self) -> PreTrainedConfig:
        loaded = parser.load_pretrained_config_from_path_field("policy.base_policy")
        if loaded is not None:
            return loaded

        if isinstance(self.base_policy, PreTrainedConfig):
            cfg = self.base_policy
        elif isinstance(self.base_policy, dict):
            cfg = self._base_policy_from_dict(self.base_policy)
        else:
            cfg = None

        if cfg is None or cfg.pretrained_path is None:
            raise ValueError(
                "RLTConfig requires `policy.base_policy.path` (or `base_policy.pretrained_path`) "
                "pointing to a pretrained base VLA checkpoint."
            )
        cfg.pretrained_path = Path(cfg.pretrained_path)
        return cfg

    @staticmethod
    def _base_policy_from_dict(raw: dict[str, Any]) -> PreTrainedConfig:
        from lerobot.policies.factory import make_policy_config

        payload = dict(raw)
        policy_type = payload.pop("type", "pi05")
        if "path" in payload:
            raise ValueError(
                "Use `policy.base_policy.path` at the JSON top level of base_policy; "
                "draccus cannot decode a nested `path` field on PI05Config."
            )
        if "pretrained_path" in payload:
            payload["pretrained_path"] = Path(payload["pretrained_path"])
        return make_policy_config(policy_type, **payload)

    def _configure_base_policy_for_rlt(self) -> None:
        """Apply RLT-specific overrides to the nested base VLA config."""
        if hasattr(self.base_policy, "compile_model"):
            self.base_policy.compile_model = False

    def to_rl_token_config(self) -> RLTokenConfig:
        return RLTokenConfig(
            num_rl_tokens=self.rlt_num_tokens,
            num_layers=self.rlt_num_layers,
            embed_dim=self.rlt_embed_dim,
            input_dim=self.rlt_input_dim,
            mlp_ratio=self.rlt_mlp_ratio,
            num_heads=self.rlt_num_heads,
            dropout=self.rlt_dropout,
        )

    def validate_features(self) -> None:
        self.base_policy.validate_features()

    def get_optimizer_preset(self):
        if not self.rlt_finetune_vla:
            from lerobot.optim import AdamWConfig

            return AdamWConfig(
                lr=self.rlt_optimizer_lr,
                betas=self.optimizer_betas,
                eps=self.optimizer_eps,
                weight_decay=self.optimizer_weight_decay,
                grad_clip_norm=self.optimizer_grad_clip_norm,
            )
        return self.base_policy.get_optimizer_preset()

    def get_scheduler_preset(self):
        if not self.rlt_finetune_vla:
            from lerobot.optim import CosineDecayWithWarmupSchedulerConfig

            return CosineDecayWithWarmupSchedulerConfig(
                peak_lr=self.rlt_optimizer_lr,
                decay_lr=self.scheduler_decay_lr,
                num_warmup_steps=self.rlt_scheduler_warmup_steps,
                num_decay_steps=self.rlt_scheduler_decay_steps,
            )
        return self.base_policy.get_scheduler_preset()

    @property
    def observation_delta_indices(self):
        return self.base_policy.observation_delta_indices

    @property
    def action_delta_indices(self):
        return self.base_policy.action_delta_indices

    @property
    def reward_delta_indices(self):
        return self.base_policy.reward_delta_indices
