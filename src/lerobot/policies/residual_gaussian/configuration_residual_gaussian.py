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

from dataclasses import dataclass, field

from lerobot.configs import PreTrainedConfig
from lerobot.configs.types import PolicyFeature
from lerobot.utils.constants import OBS_IMAGE

from ..gaussian_actor.configuration_gaussian_actor import (
    ActorLearnerConfig,
    ActorNetworkConfig,
    ConcurrencyConfig,
    CriticNetworkConfig,
    GaussianActorConfig,
    PolicyConfig,
    is_image_feature,
)


@dataclass
class ResidualImagePreprocessingConfig:
    """Optional crop/resize for ResNet10-style residual policies (HIL-SERL uses 128×128)."""

    crop_params_dict: dict[str, tuple[int, int, int, int]] | None = None
    resize_size: tuple[int, int] | None = None


@PreTrainedConfig.register_subclass("residual_gaussian")
@dataclass
class ResidualGaussianActorConfig(GaussianActorConfig):
    """Residual Gaussian actor for PLD (π_δ conditioned on s and base action a_b).

    CLI: ``--residual_policy.type=residual_gaussian``.
    """

    # Scale delta actions to [-xi, xi] after tanh squashing (paper §3.2).
    xi: float = 0.1
    # Override HIL-SERL default: PLD treats gripper as a continuous dim in ā = a_b + a_δ.
    num_discrete_actions: int | None = None
    # Optional override; default applies residual to every action dim (joints + gripper).
    residual_action_dim: int | None = None
    # Applied in the residual preprocessor before normalization (robot cameras are often 640×480).
    image_preprocessing: ResidualImagePreprocessingConfig | None = None

    # Re-declare inherited fields for clarity in draccus help.
    actor_network_kwargs: ActorNetworkConfig = field(default_factory=ActorNetworkConfig)
    policy_kwargs: PolicyConfig = field(default_factory=PolicyConfig)
    discrete_critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)
    actor_learner_config: ActorLearnerConfig = field(default_factory=ActorLearnerConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)

    def __post_init__(self):
        if self.num_discrete_actions is not None:
            raise ValueError(
                "ResidualGaussianActorConfig (PLD) uses continuous gripper in ā = a_b + a_δ. "
                "Remove num_discrete_actions; discrete critics are HIL-SERL-only."
            )
        super().__post_init__()
        if self.vision_encoder_name is not None:
            if self.image_preprocessing is None:
                self.image_preprocessing = ResidualImagePreprocessingConfig(resize_size=(128, 128))
            elif self.image_preprocessing.resize_size is None:
                self.image_preprocessing.resize_size = (128, 128)
            resize_size = self.image_preprocessing.resize_size
            for key, feat in self.input_features.items():
                if key.startswith(OBS_IMAGE) and len(feat.shape) == 3:
                    self.input_features[key] = PolicyFeature(
                        type=feat.type,
                        shape=(feat.shape[0], resize_size[0], resize_size[1]),
                    )

    @property
    def effective_residual_dim(self) -> int:
        if self.residual_action_dim is not None:
            return self.residual_action_dim
        return self.critic_action_dim

    @property
    def critic_action_dim(self) -> int:
        """Full composite action dim (continuous joints + continuous gripper), per PLD §3.1."""
        return int(self.output_features["action"].shape[0])


__all__ = ["ResidualGaussianActorConfig", "ResidualImagePreprocessingConfig", "is_image_feature"]
