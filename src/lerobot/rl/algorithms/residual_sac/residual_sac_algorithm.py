# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

from typing import Any

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.policies.residual_gaussian.modeling_residual_gaussian import (
    BASE_ACTION_KEY,
    ResidualGaussianActorPolicy,
)
from lerobot.rl.algorithms.configs import TrainingStats
from lerobot.rl.algorithms.sac.sac_algorithm import SACAlgorithm
from lerobot.types import BatchType
from lerobot.utils.constants import ACTION

from .configuration_residual_sac import ResidualSACAlgorithmConfig


class ResidualSACAlgorithm(SACAlgorithm):
    """SAC for composite policy π̄ = π_b + π_δ with Cal-QL critic option (PLD §3.1)."""

    config_class = ResidualSACAlgorithmConfig
    name = "residual_sac"

    def __init__(
        self,
        policy: ResidualGaussianActorPolicy,
        config: ResidualSACAlgorithmConfig,
    ):
        super().__init__(policy=policy, config=config)

    @property
    def residual_policy(self) -> ResidualGaussianActorPolicy:
        return self.policy  # type: ignore[return-value]

    def _get_base_action(self, batch: dict[str, Any]) -> Tensor:
        comp = batch.get("complementary_info")
        if comp is not None and BASE_ACTION_KEY in comp:
            return comp[BASE_ACTION_KEY]
        if ACTION in batch:
            return batch[ACTION]
        raise KeyError(f"Batch must contain complementary_info['{BASE_ACTION_KEY}'] for residual SAC.")

    def _sample_composite_action(
        self,
        observations: dict[str, Tensor],
        base_action: Tensor,
        observation_features: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        delta, log_probs = self.residual_policy.sample_delta(
            observations, base_action, observation_features
        )
        composite = self.residual_policy.composite_action(base_action, delta)
        return composite, log_probs

    def _compute_loss_critic(self, batch: dict[str, Any]) -> Tensor:
        observations = batch["state"]
        actions = batch[ACTION]
        observation_features = batch.get("observation_feature")
        rewards = batch["reward"]
        next_observations = batch["next_state"]
        done = batch["done"]
        next_observation_features = batch.get("next_observation_feature")
        base_action = self._get_base_action(batch)

        with torch.no_grad():
            next_base = base_action
            if "complementary_info" in batch and "next_base_action" in batch["complementary_info"]:
                next_base = batch["complementary_info"]["next_base_action"]

            next_composite, _ = self._sample_composite_action(
                next_observations, next_base, next_observation_features
            )
            q_targets = self._critic_forward(
                observations=next_observations,
                actions=next_composite,
                use_target=True,
                observation_features=next_observation_features,
            )
            if self.config.num_subsample_critics is not None:
                indices = torch.randperm(self.config.num_critics)
                indices = indices[: self.config.num_subsample_critics]
                q_targets = q_targets[indices]

            min_q, _ = q_targets.min(dim=0)
            td_target = rewards + (1 - done) * self.config.discount * min_q

        q_preds = self._critic_forward(
            observations=observations,
            actions=actions,
            use_target=False,
            observation_features=observation_features,
        )

        td_target_duplicate = einops.repeat(td_target, "b -> e b", e=q_preds.shape[0])
        td_loss = (
            F.mse_loss(input=q_preds, target=td_target_duplicate, reduction="none").mean(dim=1)
        ).sum()

        if not getattr(self.config, "use_calql", False):
            return td_loss

        calql_alpha = getattr(self.config, "calql_alpha", 1.0)
        with torch.no_grad():
            v_mu = q_preds.mean(dim=0)
        policy_actions, _ = self._sample_composite_action(observations, base_action, observation_features)
        q_policy = self._critic_forward(
            observations=observations,
            actions=policy_actions,
            use_target=False,
            observation_features=observation_features,
        ).mean(dim=0)
        calql_penalty = calql_alpha * torch.relu(q_policy - v_mu).mean()
        return 0.5 * td_loss + calql_penalty

    def _compute_loss_actor(self, batch: dict[str, Any]) -> Tensor:
        observations = batch["state"]
        observation_features = batch.get("observation_feature")
        base_action = self._get_base_action(batch)

        delta, log_probs = self.residual_policy.sample_delta(
            observations, base_action, observation_features
        )
        actions_pi = self.residual_policy.composite_action(base_action, delta)

        q_preds = self._critic_forward(
            observations=observations,
            actions=actions_pi,
            use_target=False,
            observation_features=observation_features,
        )
        min_q_preds = q_preds.min(dim=0)[0]
        return ((self.temperature * log_probs) - min_q_preds).mean()

    def _compute_loss_temperature(self, batch: dict[str, Any]) -> Tensor:
        observations = batch["state"]
        observation_features = batch.get("observation_feature")
        base_action = self._get_base_action(batch)
        with torch.no_grad():
            _, log_probs = self.residual_policy.sample_delta(
                observations, base_action, observation_features
            )
        return (-self.log_alpha.exp() * (log_probs + self.target_entropy)).mean()

    def update_critic_only(self, batch: BatchType) -> TrainingStats:
        """Single critic gradient step (Cal-QL pre-training)."""
        fb = self._prepare_forward_batch(batch, include_complementary_info=True)
        loss_critic = self._compute_loss_critic(fb)
        self.optimizers["critic"].zero_grad()
        loss_critic.backward()
        critic_grad = torch.nn.utils.clip_grad_norm_(
            self.critic_ensemble.parameters(), max_norm=self.config.grad_clip_norm
        ).item()
        self.optimizers["critic"].step()
        self._update_target_networks()
        return TrainingStats(
            losses={"loss_critic": loss_critic.item()},
            grad_norms={"critic": critic_grad},
        )
