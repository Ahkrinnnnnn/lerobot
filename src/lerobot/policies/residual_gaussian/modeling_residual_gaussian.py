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

from dataclasses import asdict

import torch
import torch.nn as nn
from torch import Tensor

from lerobot.utils.constants import ACTION

from ..gaussian_actor.modeling_gaussian_actor import (
    GaussianActorObservationEncoder,
    MLP,
    TanhMultivariateNormalDiag,
    orthogonal_init,
)
from ..pretrained import PreTrainedPolicy
from .configuration_residual_gaussian import ResidualGaussianActorConfig

BASE_ACTION_KEY = "base_action"


class ResidualPolicy(nn.Module):
    """Gaussian residual policy π_δ(·|s, a_b) with tanh squashing scaled by xi."""

    def __init__(
        self,
        encoder: GaussianActorObservationEncoder,
        config: ResidualGaussianActorConfig,
        continuous_action_dim: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.continuous_action_dim = continuous_action_dim
        self.residual_dim = config.effective_residual_dim
        self.xi = config.xi
        self.encoder_is_shared = config.shared_encoder

        network_input_dim = encoder.output_dim + continuous_action_dim
        self.network = MLP(
            input_dim=network_input_dim,
            **asdict(config.actor_network_kwargs),
        )
        policy_kwargs = asdict(config.policy_kwargs)
        out_features = config.actor_network_kwargs.hidden_dims[-1]
        self.mean_layer = nn.Linear(out_features, self.residual_dim)
        init_final = policy_kwargs.get("init_final")
        if init_final is not None:
            nn.init.uniform_(self.mean_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.mean_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.mean_layer.weight)

        self.std_min = policy_kwargs.get("std_min", 1e-5)
        self.std_max = policy_kwargs.get("std_max", 10.0)
        self.fixed_std = None
        self.std_layer = nn.Linear(out_features, self.residual_dim)
        if init_final is not None:
            nn.init.uniform_(self.std_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.std_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.std_layer.weight)

    def forward(
        self,
        observations: dict[str, Tensor],
        base_action: Tensor,
        observation_features: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        obs_enc = self.encoder(
            observations, cache=observation_features, detach=self.encoder_is_shared
        )
        if base_action.shape[-1] > self.residual_dim:
            base_cont = base_action[..., : self.residual_dim]
        else:
            base_cont = base_action
        features = torch.cat([obs_enc, base_cont], dim=-1)
        outputs = self.network(features)
        means = self.mean_layer(outputs)

        if self.fixed_std is None:
            log_std = self.std_layer(outputs)
            std = torch.exp(log_std)
            std = torch.clamp(std, self.std_min, self.std_max)
        else:
            std = self.fixed_std.expand_as(means)

        dist = TanhMultivariateNormalDiag(loc=means, scale_diag=std)
        delta_raw = dist.rsample()
        delta = delta_raw * self.xi
        log_probs = dist.log_prob(delta_raw) - self.residual_dim * torch.log(
            torch.tensor(self.xi, device=delta_raw.device, dtype=delta_raw.dtype)
        )
        return delta, log_probs, means


class ResidualGaussianActorPolicy(PreTrainedPolicy):
    """Residual SAC actor policy for PLD Stage 1."""

    config_class = ResidualGaussianActorConfig
    name = "residual_gaussian"

    def __init__(self, config: ResidualGaussianActorConfig | None = None):
        super().__init__(config)
        config.validate_features()
        self.config = config

        continuous_action_dim = config.output_features[ACTION].shape[0]

        self.shared_encoder = config.shared_encoder
        self.encoder_critic = GaussianActorObservationEncoder(config)
        self.encoder_actor = (
            self.encoder_critic if self.shared_encoder else GaussianActorObservationEncoder(config)
        )
        self.actor = ResidualPolicy(
            encoder=self.encoder_actor,
            config=config,
            continuous_action_dim=continuous_action_dim,
        )

    def get_optim_params(self) -> dict:
        return {
            "actor": [
                p
                for n, p in self.actor.named_parameters()
                if not n.startswith("encoder") or not self.shared_encoder
            ],
        }

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError("ResidualGaussianActorPolicy returns single-step actions only.")

    def _get_base_action(self, batch: dict[str, Tensor]) -> Tensor:
        if BASE_ACTION_KEY in batch:
            return batch[BASE_ACTION_KEY]
        if "complementary_info" in batch and batch["complementary_info"] is not None:
            return batch["complementary_info"][BASE_ACTION_KEY]
        raise KeyError(f"Residual policy requires '{BASE_ACTION_KEY}' in batch or complementary_info.")

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        observations = batch
        base_action = self._get_base_action(batch)
        observation_features = None
        if self.shared_encoder and self.encoder_actor.has_images:
            observation_features = self.encoder_actor.get_cached_image_features(observations)

        delta, _, _ = self.actor(observations, base_action, observation_features)
        return self.composite_action(base_action, delta)

    def sample_delta(
        self,
        observations: dict[str, Tensor],
        base_action: Tensor,
        observation_features: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Sample residual delta and log-prob (for SAC actor loss)."""
        return self.actor(observations, base_action, observation_features)[:2]

    def composite_action(self, base_action: Tensor, delta: Tensor) -> Tensor:
        """Combine base action and scaled residual."""
        composite = base_action.clone()
        composite[..., : delta.shape[-1]] = composite[..., : delta.shape[-1]] + delta
        return composite

    def forward(self, batch: dict[str, Tensor | dict[str, Tensor]]) -> dict[str, Tensor]:
        observations = batch.get("state", batch)
        base_action = self._get_base_action(batch)  # type: ignore[arg-type]
        observation_features = batch.get("observation_feature") if isinstance(batch, dict) else None
        delta, log_probs, means = self.actor(observations, base_action, observation_features)
        composite = self.composite_action(base_action, delta)
        return {"action": composite, "delta": delta, "log_prob": log_probs, "action_mean": means}

    def set_xi(self, xi: float) -> None:
        self.config.xi = xi
        self.actor.xi = xi
