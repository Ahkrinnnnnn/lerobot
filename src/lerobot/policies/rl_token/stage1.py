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

"""Shared Stage-1 training logic: L_ro + alpha * L_base."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .backbone import (
    PrefixEmbeddings,
    RLTokenBackbone,
    RLTokenJointBackbone,
    supports_joint_rlt_training,
)
from .rl_token import RLTokenConfig, RLTokenModel


class RLTStage1Module(nn.Module):
    """Standalone RL token module with Stage-1 loss computation."""

    def __init__(self, rl_token_config: RLTokenConfig):
        super().__init__()
        self.rl_token = RLTokenModel(rl_token_config)

    @torch.no_grad()
    def encode_prefix(self, prefix: PrefixEmbeddings) -> Tensor:
        embs = prefix.embeddings.to(dtype=torch.float32)
        return self.rl_token.encode(embs, prefix.mask)

    def compute_stage1_loss(
        self,
        backbone: RLTokenBackbone,
        batch: dict[str, Tensor],
        *,
        alpha: float,
        reduction: str = "mean",
    ) -> tuple[Tensor, dict]:
        """Compute ``L_ro + alpha * L_base`` for a batch."""
        joint_training = alpha > 0.0

        if joint_training:
            if not supports_joint_rlt_training(backbone):
                raise ValueError(
                    f"Joint RLT training (rlt_alpha={alpha}) requires an RLTokenJointBackbone, "
                    f"but got {type(backbone).__name__}."
                )
            base_losses, prefix = backbone.forward_with_prefix_for_rlt(batch)  # type: ignore[union-attr]
        else:
            base_losses = None
            prefix = backbone.extract_prefix_embeddings(batch)

        prefix_embs = prefix.embeddings.to(dtype=torch.float32)
        rlt_loss, rlt_info = self.rl_token.reconstruction_loss(
            prefix_embs,
            prefix.mask,
            reduction=reduction if not joint_training else "none",
        )

        rlt_loss_scalar = rlt_loss.mean() if rlt_loss.ndim > 0 else rlt_loss
        loss_dict = {
            "rlt_loss": rlt_loss_scalar.item(),
            "rlt_mse": rlt_info["rlt_mse"],
        }

        if joint_training and base_losses is not None:
            if reduction == "none":
                base_loss_per_sample = base_losses.mean(dim=tuple(range(1, base_losses.ndim)))
                per_sample_loss = rlt_loss + alpha * base_loss_per_sample
                loss_dict["vla_loss"] = base_loss_per_sample.mean().item()
                loss_dict["loss"] = per_sample_loss.mean().item()
                return per_sample_loss, loss_dict

            base_loss = base_losses.mean()
            loss = rlt_loss_scalar + alpha * base_loss
            loss_dict["vla_loss"] = base_loss.item()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

        loss_dict["loss"] = rlt_loss_scalar.item()
        if reduction == "none":
            batch_size = prefix_embs.shape[0]
            if rlt_loss.ndim == 0:
                return rlt_loss.expand(batch_size), loss_dict
            return rlt_loss, loss_dict
        return rlt_loss_scalar, loss_dict
