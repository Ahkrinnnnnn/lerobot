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

"""Policy wrapper that composes a base VLA with a standalone RL token module."""

from __future__ import annotations

import builtins
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig

from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from safetensors.torch import load_model as load_model_as_safetensor
from safetensors.torch import save_model as save_model_as_safetensor

from ..pretrained import PreTrainedPolicy, T
from .backbone import create_rlt_backbone
from .backbones import pi05 as _register_pi05_backbone  # noqa: F401
from .configuration_rlt import RLTConfig
from .stage1 import RLTStage1Module

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.envs import EnvConfig


class RLTPolicy(PreTrainedPolicy):
    """Wraps any registered base policy with an RL token Stage-1 adapter."""

    config_class = RLTConfig
    name = "rlt"

    def __init__(
        self,
        config: RLTConfig,
        *,
        ds_meta: LeRobotDatasetMetadata | None = None,
        dataset_meta: LeRobotDatasetMetadata | None = None,
        env_cfg: EnvConfig | None = None,
        rename_map: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(config)
        config.validate_features()

        from lerobot.policies.factory import make_policy

        ds_meta = ds_meta or dataset_meta
        if ds_meta is None and env_cfg is None:
            raise ValueError("RLTPolicy requires ds_meta or env_cfg to instantiate the base policy.")

        self.base_policy = make_policy(
            cfg=config.base_policy,
            ds_meta=ds_meta,
            env_cfg=env_cfg,
            rename_map=rename_map,
        )
        self.backbone = create_rlt_backbone(self.base_policy)
        self.rlt_module = RLTStage1Module(config.to_rl_token_config())
        if config.device is not None:
            self.rlt_module.to(config.device)
        self._configure_rlt_training()

    def _configure_rlt_training(self) -> None:
        if not self.config.rlt_finetune_vla:
            for param in self.base_policy.parameters():
                param.requires_grad = False
        for param in self.rlt_module.parameters():
            param.requires_grad = True

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save config + RL token module only (not the full base VLA)."""
        if self.config.base_policy.pretrained_path is None:
            raise ValueError(
                "RLT checkpoint requires base_policy.pretrained_path to be set so the frozen "
                "VLA can be reloaded at inference time."
            )
        self.config._save_pretrained(save_directory)
        if self.config.rlt_save_token_only:
            save_model_as_safetensor(self.rlt_module, str(save_directory / SAFETENSORS_SINGLE_FILE))
        else:
            model_to_save = self.module if hasattr(self, "module") else self
            save_model_as_safetensor(model_to_save, str(save_directory / SAFETENSORS_SINGLE_FILE))

    @classmethod
    def _load_rlt_module_weights(cls, policy: "RLTPolicy", model_file: str, *, strict: bool) -> None:
        """Load weights into ``rlt_module`` (RLT Stage-1 checkpoint format)."""
        from safetensors.torch import load_file

        state_dict = load_file(model_file)
        if any(key.startswith("base_policy.") for key in state_dict):
            load_model_as_safetensor(policy, model_file, strict=strict)
            return

        load_model_as_safetensor(policy.rlt_module, model_file, strict=strict)

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        strict: bool = False,
        ds_meta: LeRobotDatasetMetadata | None = None,
        env_cfg: EnvConfig | None = None,
        rename_map: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> T:
        if config is None:
            config = PreTrainedConfig.from_pretrained(pretrained_name_or_path, **kwargs)
        if not isinstance(config, RLTConfig):
            raise TypeError(f"Expected RLTConfig, got {type(config).__name__}")

        policy = cls(
            config,
            ds_meta=ds_meta,
            env_cfg=env_cfg,
            rename_map=rename_map,
            **kwargs,
        )

        model_path = Path(pretrained_name_or_path) / SAFETENSORS_SINGLE_FILE
        if model_path.is_file():
            cls._load_rlt_module_weights(policy, str(model_path), strict=strict)
            policy._configure_rlt_training()

        policy.eval()
        return policy

    def get_optim_params(self):
        if not self.config.rlt_finetune_vla:
            return self.rlt_module.parameters()
        from itertools import chain

        return chain(self.base_policy.parameters(), self.rlt_module.parameters())

    def reset(self):
        self.base_policy.reset()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        return self.base_policy.select_action(batch)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        return self.base_policy.predict_action_chunk(batch, **kwargs)

    @torch.no_grad()
    def extract_rl_token(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        prefix = self.backbone.extract_prefix_embeddings(batch)
        return self.rlt_module.encode_prefix(prefix)

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        return self.rlt_module.compute_stage1_loss(
            self.backbone,
            batch,
            alpha=self.config.rlt_alpha,
            reduction=reduction,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.config.rlt_finetune_vla:
            self.base_policy.eval()
        return self
