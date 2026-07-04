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

from dataclasses import dataclass

from torch import Tensor

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from ...pretrained import PreTrainedPolicy
from ..backbone import PrefixEmbeddings, RLTokenJointBackbone, register_rlt_backbone


@register_rlt_backbone("pi05")
def _create_pi05_backbone(policy: PreTrainedPolicy) -> "Pi05RLTokenBackbone":
    return Pi05RLTokenBackbone(policy)


class Pi05RLTokenBackbone(RLTokenJointBackbone):
    """RL token backbone adapter for PI0.5 flow-matching VLAs."""

    def __init__(self, policy: PreTrainedPolicy):
        self.policy = policy

    def extract_prefix_embeddings(self, batch: dict[str, Tensor]) -> PrefixEmbeddings:
        images, img_masks = self.policy._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        prefix_embs, prefix_mask = self.policy.model.extract_prefix_embeddings(
            images, img_masks, tokens, masks
        )
        return PrefixEmbeddings(embeddings=prefix_embs, mask=prefix_mask)

    def forward_with_prefix_for_rlt(self, batch: dict[str, Tensor]) -> tuple[Tensor, PrefixEmbeddings]:
        images, img_masks = self.policy._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.policy.prepare_action(batch)
        noise = self.policy.model.sample_noise(actions.shape, actions.device)
        time = self.policy.model.sample_time(actions.shape[0], actions.device)

        losses, prefix_embs, prefix_mask = self.policy.model.forward_with_prefix_output(
            images, img_masks, tokens, masks, actions, noise, time
        )
        original_action_dim = self.policy.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]
        return losses, PrefixEmbeddings(embeddings=prefix_embs, mask=prefix_mask)
