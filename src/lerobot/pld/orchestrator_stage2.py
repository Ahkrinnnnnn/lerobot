# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

import logging
from pathlib import Path
from threading import Event

from lerobot.pld.configs import PLDStage2Config
from lerobot.pld.residual_setup import init_residual_policy_from_robot, resolve_stage1_residual_weights
from lerobot.policies.residual_gaussian.configuration_residual_gaussian import ResidualGaussianActorConfig
from lerobot.rewards.classifier import RewardClassifierDetector
from lerobot.rollout.context import build_rollout_context
from lerobot.rollout.strategies.factory import create_strategy
from lerobot.rollout.strategies.pld_hybrid_collect import PLDHybridCollectStrategy

logger = logging.getLogger(__name__)


class PLDStage2Orchestrator:
    """PLD Stage 2: hybrid base+residual rollouts → LeRobotDataset for base-policy SFT."""

    def __init__(self, cfg: PLDStage2Config, shutdown_event: Event | None = None):
        self.cfg = cfg
        self.shutdown_event = shutdown_event or Event()
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_weights_path(self) -> str:
        path = resolve_stage1_residual_weights(
            self.cfg.pld.stage1_output_dir,
            self.cfg.pld.resume_residual_weights,
        )
        if path is None:
            raise ValueError(
                "Stage 2 requires frozen residual actor weights. Set --pld.resume_residual_weights "
                "or --pld.stage1_output_dir pointing to a completed Stage 1 run "
                "(with residual_policy_weights.pt)."
            )
        return path

    def collect_sft_dataset(self) -> None:
        if not isinstance(self.cfg.residual_policy, ResidualGaussianActorConfig):
            raise TypeError("residual_policy must be ResidualGaussianActorConfig")

        device = self.cfg.device or self.cfg.residual_policy.device or "cuda"
        weights_path = self._resolve_weights_path()

        residual_policy, residual_preprocessor, _state_keys = init_residual_policy_from_robot(
            robot_cfg=self.cfg.robot,
            policy_cfg=self.cfg.residual_policy,
            device=device,
            xi=self.cfg.pld.xi,
            base_policy=self.cfg.base_policy,
            weights_path=weights_path,
        )

        reward_detector = RewardClassifierDetector(self.cfg.reward_classifier)
        rollout_cfg = self.cfg.to_rollout_config()
        ctx = build_rollout_context(rollout_cfg, self.shutdown_event)
        strategy = create_strategy(rollout_cfg.strategy)

        if not isinstance(strategy, PLDHybridCollectStrategy):
            raise TypeError("Expected PLDHybridCollectStrategy for Stage 2 collection.")

        strategy.set_hybrid_runtime(
            residual_policy=residual_policy,
            residual_preprocessor=residual_preprocessor,
            reward_detector=reward_detector,
            output_dir=self.output_dir,
        )

        try:
            strategy.setup(ctx)
            strategy.run(ctx)
        finally:
            strategy.teardown(ctx)
            reward_detector.stop()

        dataset = ctx.data.dataset
        if dataset is not None:
            logger.info(
                "SFT dataset ready: %s (%d episodes at %s)",
                dataset.repo_id,
                dataset.num_episodes,
                dataset.root,
            )

    def run(self) -> None:
        if not self.cfg.skip_collection:
            logger.info("=== PLD Stage 2: hybrid SFT collection ===")
            self.collect_sft_dataset()
        else:
            logger.info("Skipping PLD Stage 2 collection (skip_collection=true)")
