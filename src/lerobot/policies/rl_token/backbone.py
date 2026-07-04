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

"""Backbone adapter protocol for RL Token Stage-1 training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import torch
from torch import Tensor

from ..pretrained import PreTrainedPolicy

BackboneFactory = Callable[[PreTrainedPolicy], "RLTokenBackbone"]

_BACKBONE_FACTORIES: dict[str, BackboneFactory] = {}


@dataclass
class PrefixEmbeddings:
    """Final-layer token embeddings extracted from a base VLA policy."""

    embeddings: Tensor
    mask: Tensor


@runtime_checkable
class RLTokenBackbone(Protocol):
    """Minimal interface a base policy must expose for RL token training."""

    def extract_prefix_embeddings(self, batch: dict[str, Tensor]) -> PrefixEmbeddings:
        """Return prefix hidden states ``z`` and a valid-token mask."""
        ...


@runtime_checkable
class RLTokenJointBackbone(RLTokenBackbone, Protocol):
    """Backbone that supports joint RL-token + base-policy SFT in one forward pass."""

    def forward_with_prefix_for_rlt(self, batch: dict[str, Tensor]) -> tuple[Tensor, PrefixEmbeddings]:
        """Return per-element base policy losses and prefix embeddings."""
        ...


def register_rlt_backbone(policy_type: str) -> Callable[[BackboneFactory], BackboneFactory]:
    """Register a backbone factory for a base policy type (e.g. ``pi05``)."""

    def decorator(factory: BackboneFactory) -> BackboneFactory:
        _BACKBONE_FACTORIES[policy_type] = factory
        return factory

    return decorator


def get_registered_rlt_backbone_types() -> list[str]:
    return sorted(_BACKBONE_FACTORIES.keys())


def create_rlt_backbone(base_policy: PreTrainedPolicy) -> RLTokenBackbone:
    """Instantiate the registered backbone adapter for ``base_policy``."""
    policy_type = base_policy.config.type
    if policy_type not in _BACKBONE_FACTORIES:
        registered = ", ".join(get_registered_rlt_backbone_types()) or "(none)"
        raise ValueError(
            f"Base policy type '{policy_type}' has no RL token backbone registered. "
            f"Registered types: {registered}. "
            f"Implement an adapter in policies/rl_token/backbones/ and register it."
        )
    return _BACKBONE_FACTORIES[policy_type](base_policy)


def supports_joint_rlt_training(backbone: RLTokenBackbone) -> bool:
    return isinstance(backbone, RLTokenJointBackbone)
