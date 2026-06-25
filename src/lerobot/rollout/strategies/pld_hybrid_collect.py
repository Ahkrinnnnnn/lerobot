# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import torch

from lerobot.datasets import VideoEncodingManager
from lerobot.pld.obs_utils import (
    action_dict_to_tensor,
    fetch_obs_for_rl,
    prepare_rl_state_for_buffer,
)
from lerobot.rewards.classifier import RewardClassifierDetector
from lerobot.policies.residual_gaussian.modeling_residual_gaussian import ResidualGaussianActorPolicy
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame

from ..configs import PLDHybridCollectStrategyConfig
from ..context import RolloutContext
from ..collect_intervention import CollectInterventionController
from ..control.multiprocess_loop import multiprocess_control_loop
from .core import RolloutStrategy

if TYPE_CHECKING:
    from lerobot.processor import PolicyProcessorPipeline

logger = logging.getLogger(__name__)


class PLDHybridCollectStrategy(RolloutStrategy):
    """PLD Stage 2: hybrid base+residual rollouts written to a LeRobotDataset.

    Each episode samples ``T_base ~ Uniform[0, probing_alpha * max_episode_steps]``.
    Steps ``t < T_base`` execute the frozen base policy only; afterwards the frozen
    residual policy composes actions. Only successful episodes are saved.
    """

    config: PLDHybridCollectStrategyConfig

    def __init__(self, config: PLDHybridCollectStrategyConfig):
        super().__init__(config)
        self._residual_policy: ResidualGaussianActorPolicy | None = None
        self._residual_preprocessor: PolicyProcessorPipeline | None = None
        self._reward_detector: RewardClassifierDetector | None = None
        self._residual_input_features: dict | None = None
        self._output_dir: Path | None = None

        self._episode_steps = 0
        self._episode_reward = 0.0
        self._t_base = 0
        self._successful_episodes = 0
        self._failed_episodes = 0
        self._t_base_samples: list[int] = []
        self._collect_stop = False
        self._collect_intervention: CollectInterventionController | None = None

    def set_hybrid_runtime(
        self,
        *,
        residual_policy: ResidualGaussianActorPolicy,
        residual_preprocessor: PolicyProcessorPipeline | None = None,
        reward_detector: RewardClassifierDetector | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self._residual_policy = residual_policy
        self._residual_preprocessor = residual_preprocessor
        self._reward_detector = reward_detector
        self._residual_input_features = residual_policy.config.input_features
        self._output_dir = output_dir

    def setup(self, ctx: RolloutContext) -> None:
        self._collect_stop = False
        if self.config.seed is not None:
            random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
        self._init_engine(ctx)
        self._collect_intervention = CollectInterventionController(self.config.manual_scene_reset_pause_key)
        self._collect_intervention.start()
        self._sample_t_base()
        logger.info(
            "PLD hybrid collect ready (target=%d episodes, probing_alpha=%.2f, T_base=%d, "
            "%.1fs reset between episodes)",
            self.config.n_successful_episodes,
            self.config.probing_alpha,
            self._t_base,
            self.config.episode_reset_time_s,
        )

    def run(self, ctx: RolloutContext) -> None:
        dataset = ctx.data.dataset
        if dataset is None:
            raise RuntimeError("PLD hybrid collect requires --dataset.repo_id in rollout config.")

        stats: dict = {}
        callback = self._make_step_callback(ctx)
        modify_fn = self._make_modify_action_fn(ctx)

        with VideoEncodingManager(dataset):
            multiprocess_control_loop(
                ctx,
                self,
                control_loop_stats=stats,
                on_step_callback=callback,
                modify_action_fn=modify_fn,
            )

        logger.info(
            "Hybrid collection finished: %d successful / %d failed (target %d)",
            self._successful_episodes,
            self._failed_episodes,
            self.config.n_successful_episodes,
        )
        self._write_collection_stats()

    def teardown(self, ctx: RolloutContext) -> None:
        if self._collect_intervention is not None:
            self._collect_intervention.stop()
            self._collect_intervention = None
        self._teardown_hardware(
            ctx.hardware,
            return_to_initial_position=ctx.runtime.cfg.return_to_initial_position,
        )

    def _should_stop(self) -> bool:
        return self._successful_episodes >= self.config.n_successful_episodes

    def _sample_t_base(self) -> None:
        max_t = int(self.config.probing_alpha * self.config.max_episode_steps)
        self._t_base = random.randint(0, max_t)
        self._t_base_samples.append(self._t_base)

    def _make_modify_action_fn(self, ctx: RolloutContext) -> Callable[..., dict[str, float] | None]:
        if self._residual_policy is None:
            return lambda **kwargs: None

        ordered_keys = ctx.data.ordered_action_keys
        device = ctx.runtime.cfg.device or "cpu"

        def modify_action_fn(
            *,
            ctx: RolloutContext,
            base_action_dict: dict[str, float],
            obs_processed: dict,
            obs_policy: dict | None,
        ) -> dict[str, float] | None:
            if self._episode_steps < self._t_base:
                return None
            obs_for_rl = fetch_obs_for_rl(ctx, obs_policy, obs_processed)
            rl_state = prepare_rl_state_for_buffer(
                obs_for_rl,
                self._residual_input_features or {},
                self._residual_preprocessor,
                device=device,
            )
            base_action = action_dict_to_tensor(base_action_dict, ordered_keys).to(device)
            batch = {**rl_state, "base_action": base_action}
            with torch.inference_mode():
                composite = self._residual_policy.select_action(batch)
            values = composite.squeeze(0).cpu().tolist()
            return {k: float(values[i]) for i, k in enumerate(ordered_keys) if i < len(values)}

        return modify_action_fn

    def _make_step_callback(self, ctx: RolloutContext) -> Callable[..., None]:
        ordered_keys = ctx.data.ordered_action_keys
        device = ctx.runtime.cfg.device or "cpu"
        dataset = ctx.data.dataset
        features = ctx.data.dataset_features
        task_str = (
            ctx.runtime.cfg.dataset.single_task
            if ctx.runtime.cfg.dataset and ctx.runtime.cfg.dataset.single_task
            else ctx.runtime.cfg.task
        )

        def on_step(
            *,
            obs_processed: dict,
            obs_policy: dict | None,
            base_action_dict: dict[str, float],
            executed_action_dict: dict[str, float],
        ) -> None:
            if self._should_stop():
                self._collect_stop = True
                return

            obs_for_rl = fetch_obs_for_rl(ctx, obs_policy, obs_processed)

            reward, done = 0.0, False
            if self._reward_detector is not None:
                reward, done = self._reward_detector.predict(obs_for_rl)
                if self._reward_detector.consume_manual_success():
                    reward, done = self._reward_detector.config.success_reward, True
                if self._reward_detector.consume_manual_failure():
                    reward, done = 0.0, True

            obs_frame = build_dataset_frame(features, obs_for_rl, prefix=OBS_STR)
            action_frame = build_dataset_frame(features, executed_action_dict, prefix=ACTION)
            dataset.add_frame({**obs_frame, **action_frame, "task": task_str})

            self._episode_steps += 1
            self._episode_reward = max(self._episode_reward, reward)
            if self._episode_steps >= self.config.max_episode_steps:
                done = True

            if done:
                success = self._episode_reward >= 0.5
                if success:
                    dataset.save_episode()
                    self._successful_episodes += 1
                    logger.info(
                        "Saved successful hybrid episode %d/%d (T_base=%d, steps=%d)",
                        self._successful_episodes,
                        self.config.n_successful_episodes,
                        self._t_base,
                        self._episode_steps,
                    )
                elif self.config.discard_failed_episodes:
                    dataset.clear_episode_buffer()
                    self._failed_episodes += 1
                    logger.info(
                        "Discarded failed episode (reward=%.2f, T_base=%d)",
                        self._episode_reward,
                        self._t_base,
                    )
                else:
                    dataset.save_episode()
                    self._failed_episodes += 1

                if not self._should_stop():
                    self._wait_between_episodes(
                        ctx,
                        episode_index=self._successful_episodes + self._failed_episodes,
                        episode_reset_time_s=self.config.episode_reset_time_s,
                        manual_scene_reset_pause_key=self.config.manual_scene_reset_pause_key,
                        collect_intervention=self._collect_intervention,
                    )
                self._reset_episode()
                if self._should_stop():
                    self._collect_stop = True

        return on_step

    def _reset_episode(self) -> None:
        self._episode_steps = 0
        self._episode_reward = 0.0
        self._sample_t_base()
        if self._reward_detector is not None:
            self._reward_detector.reset_manual()

    def _write_collection_stats(self) -> None:
        if self._output_dir is None:
            return
        stats_path = self._output_dir / "collection_stats.json"
        payload = {
            "successful_episodes": self._successful_episodes,
            "failed_episodes": self._failed_episodes,
            "target_episodes": self.config.n_successful_episodes,
            "probing_alpha": self.config.probing_alpha,
            "t_base_samples": self._t_base_samples,
        }
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Wrote collection stats to %s", stats_path)
