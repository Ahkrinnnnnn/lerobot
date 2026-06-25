# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.policies.gaussian_actor.configuration_gaussian_actor import CriticNetworkConfig
from lerobot.policies.residual_gaussian.configuration_residual_gaussian import ResidualGaussianActorConfig

from ..configs import RLAlgorithmConfig
from ..sac.configuration_sac import SACAlgorithmConfig


@RLAlgorithmConfig.register_subclass("residual_sac")
@dataclass
class ResidualSACAlgorithmConfig(SACAlgorithmConfig):
    """SAC for PLD residual policy with optional Cal-QL critic pre-training."""

    use_calql: bool = False
    calql_alpha: float = 1.0
    critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)

    @classmethod
    def from_policy_config(cls, policy_cfg: ResidualGaussianActorConfig) -> ResidualSACAlgorithmConfig:
        return cls(
            policy_config=policy_cfg,
            discrete_critic_network_kwargs=policy_cfg.discrete_critic_network_kwargs,
        )
