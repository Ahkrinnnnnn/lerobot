# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

import torch

from lerobot.pld.obs_utils import (
    action_dict_to_tensor,
    fetch_obs_for_rl,
    make_complementary_info,
    prepare_rl_state_for_buffer,
)
from lerobot.rewards.classifier import RewardClassifierDetector
from lerobot.policies.residual_gaussian.modeling_residual_gaussian import ResidualGaussianActorPolicy
from lerobot.rl.buffer import ReplayBuffer
from lerobot.utils.constants import ACTION

from ..configs import PLDCollectStrategyConfig
from ..context import RolloutContext
from ..collect_intervention import CollectInterventionController
from ..control.multiprocess_loop import multiprocess_control_loop
from .core import RolloutStrategy

if TYPE_CHECKING:
    from lerobot.processor import PolicyProcessorPipeline

logger = logging.getLogger(__name__)


class PLDCollectStrategy(RolloutStrategy):
    """Collect PLD transitions using the deploy control loop."""

    config: PLDCollectStrategyConfig

    def __init__(self, config: PLDCollectStrategyConfig):
        super().__init__(config)
        self._offline_buffer: ReplayBuffer | None = None
        self._online_buffer: ReplayBuffer | None = None
        self._residual_policy: ResidualGaussianActorPolicy | None = None
        self._residual_preprocessor: PolicyProcessorPipeline | None = None
        self._reward_detector: RewardClassifierDetector | None = None
        self._residual_input_features: dict | None = None
        self._state_keys: list[str] | None = None

        self._prev_state: dict[str, torch.Tensor] | None = None
        self._prev_base_action: torch.Tensor | None = None
        self._prev_executed_action: torch.Tensor | None = None
        self._episode_transitions: list[dict] = []
        self._episode_steps = 0
        self._successful_trials = 0
        self._failed_trials = 0
        self._episode_index = 0
        self._env_steps = 0
        self._episode_reward = 0.0
        self._collect_stop = False
        self._warmup_done_logged = False
        self._round_episode_rewards: list[float] = []
        self._round_success_count = 0
        self._finish_episode_before_round_stop = False
        self._collect_intervention: CollectInterventionController | None = None

    def set_pld_runtime(
        self,
        *,
        offline_buffer: ReplayBuffer | None = None,
        online_buffer: ReplayBuffer | None = None,
        residual_policy: ResidualGaussianActorPolicy | None = None,
        residual_preprocessor: PolicyProcessorPipeline | None = None,
        reward_detector: RewardClassifierDetector | None = None,
        state_keys: list[str] | None = None,
    ) -> None:
        self._offline_buffer = offline_buffer
        self._online_buffer = online_buffer
        self._residual_policy = residual_policy
        self._residual_preprocessor = residual_preprocessor
        self._reward_detector = reward_detector
        self._state_keys = state_keys
        if residual_policy is not None:
            self._residual_input_features = residual_policy.config.input_features

    def reset_collect_state(self) -> None:
        """Reset per-round counters without tearing down hardware or inference."""
        self._collect_stop = False
        self._prev_state = None
        self._prev_base_action = None
        self._prev_executed_action = None
        self._episode_transitions = []
        self._episode_steps = 0
        self._episode_reward = 0.0
        self._env_steps = 0
        self._round_episode_rewards = []
        self._round_success_count = 0
        self._finish_episode_before_round_stop = False

    def online_round_reward_summary(self) -> dict[str, float | int]:
        """Episode reward stats for the current online collect round."""
        rewards = self._round_episode_rewards
        if not rewards:
            return {"episodes": 0, "successes": 0}
        return {
            "episodes": len(rewards),
            "successes": self._round_success_count,
            "mean_max_reward": sum(rewards) / len(rewards),
            "last_episode_reward": rewards[-1],
        }

    def begin_online_collect_round(
        self,
        *,
        collect_round: int,
        warmup_env_steps: int,
        max_env_steps: int,
        max_episode_steps: int,
        episode_reset_time_s: float,
        finish_episode_before_round_stop: bool = True,
        manual_scene_reset_pause_key: str = "space",
    ) -> None:
        """Configure and reset state for one online collect round inside ``rl_train``."""
        self.config = PLDCollectStrategyConfig(
            mode="online",
            warmup_env_steps=warmup_env_steps,
            max_env_steps=max_env_steps,
            max_episode_steps=max_episode_steps,
            episode_reset_time_s=episode_reset_time_s,
            finish_episode_before_round_stop=finish_episode_before_round_stop,
            manual_scene_reset_pause_key=manual_scene_reset_pause_key,
        )
        self.reset_collect_state()
        if self._reward_detector is not None:
            self._reward_detector.reset_manual()
        if finish_episode_before_round_stop:
            logger.info(
                "=== Online collect round %d === collect ≥%d env steps, then finish current episode before SAC "
                "(%.1fs between episodes)",
                collect_round,
                max_env_steps,
                episode_reset_time_s,
            )
        else:
            logger.info(
                "=== Online collect round %d === hard stop at %d env steps (%.1fs between episodes)",
                collect_round,
                max_env_steps,
                episode_reset_time_s,
            )

    def setup(self, ctx: RolloutContext) -> None:
        self._collect_stop = False
        self._init_engine(ctx)
        self._collect_intervention = CollectInterventionController(self.config.manual_scene_reset_pause_key)
        self._collect_intervention.start()
        if self.config.mode == "offline":
            logger.info(
                "PLD offline collect started — target %d successful trials "
                "(max %d steps/episode, %.1fs reset) | operator keys: s=success f=failure %s=pause/resume",
                self.config.n_successful_trials,
                self.config.max_episode_steps,
                self.config.episode_reset_time_s,
                self.config.manual_scene_reset_pause_key,
            )
        else:
            logger.info(
                "=== PLD online collect ready === %.1fs reset between episodes | "
                "operator keys: s=success f=failure %s=pause/resume",
                self.config.episode_reset_time_s,
                self.config.manual_scene_reset_pause_key,
            )

    def run(self, ctx: RolloutContext) -> None:
        stats: dict = {}
        callback = self._make_step_callback(ctx)
        modify_fn = self._make_modify_action_fn(ctx)
        multiprocess_control_loop(
            ctx,
            self,
            control_loop_stats=stats,
            on_step_callback=callback,
            modify_action_fn=modify_fn,
        )
        if stats:
            frames = stats.get("frames", 0)
            overruns = stats.get("overruns", 0)
            sends = stats.get("sends", 0)
            pct = (100.0 * overruns / frames) if frames else 0.0
            logger.info(
                "PLD collect: %d control frames, %d send_action calls, %d overruns (%.1f%%).",
                frames,
                sends,
                overruns,
                pct,
            )
        if self.config.mode == "offline":
            logger.info(
                "Offline collection finished: %d/%d successful trials (%d failed attempts, %d env steps)",
                self._successful_trials,
                self.config.n_successful_trials,
                self._failed_trials,
                self._env_steps,
            )
        else:
            logger.info("=== Online collect finished === %d env steps this round", self._env_steps)

    def teardown(self, ctx: RolloutContext) -> None:
        if self._collect_intervention is not None:
            self._collect_intervention.stop()
            self._collect_intervention = None
        self._teardown_hardware(
            ctx.hardware,
            return_to_initial_position=ctx.runtime.cfg.return_to_initial_position,
        )

    def _should_stop(self) -> bool:
        if self.config.mode == "offline":
            return self._successful_trials >= self.config.n_successful_trials
        return False

    def _round_budget_reached(self) -> bool:
        return (
            self.config.mode == "online"
            and self.config.max_env_steps > 0
            and self._env_steps >= self.config.max_env_steps
        )

    def _maybe_request_round_stop_after_budget(self) -> None:
        """Online: defer SAC until the current episode ends (if configured)."""
        if not self._round_budget_reached() or self._finish_episode_before_round_stop:
            return
        if not self.config.finish_episode_before_round_stop:
            logger.info(
                "=== Online collect round limit === %d/%d env steps (hard stop)",
                self._env_steps,
                self.config.max_env_steps,
            )
            self._collect_stop = True
            return
        if self._episode_steps > 0:
            self._finish_episode_before_round_stop = True
            logger.info(
                "=== Round budget reached (%d/%d) === finishing episode %d (%d steps) before SAC",
                self._env_steps,
                self.config.max_env_steps,
                self._episode_index,
                self._episode_steps,
            )
        else:
            logger.info(
                "=== Round budget reached (%d/%d) === stopping before new episode",
                self._env_steps,
                self.config.max_env_steps,
            )
            self._collect_stop = True

    def _make_modify_action_fn(self, ctx: RolloutContext) -> Callable[..., dict[str, float] | None]:
        if self.config.mode != "online" or self._residual_policy is None:
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
            if self._env_steps < self.config.warmup_env_steps:
                return None
            if not self._warmup_done_logged:
                self._warmup_done_logged = True
                logger.info(
                    "=== Online warmup complete === env step %d — enabling residual policy (π_b + π_δ)",
                    self._env_steps,
                )
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

        def on_step(
            *,
            obs_processed: dict,
            obs_policy: dict | None,
            base_action_dict: dict[str, float],
            executed_action_dict: dict[str, float],
        ) -> None:
            if self._collect_stop:
                return

            obs_for_rl = fetch_obs_for_rl(ctx, obs_policy, obs_processed)

            reward, done = 0.0, False
            manual_success = False
            manual_failure = False
            if self._reward_detector is not None:
                reward, done = self._reward_detector.predict(obs_for_rl)

            self._episode_steps += 1
            if self._episode_steps == 1:
                self._episode_index += 1
                if self.config.mode == "offline":
                    logger.info(
                        "Offline trial %d started (%d/%d successful so far)",
                        self._episode_index,
                        self._successful_trials,
                        self.config.n_successful_trials,
                    )
                else:
                    logger.info("Online episode %d started (env step %d)", self._episode_index, self._env_steps)

            if self._reward_detector is not None:
                if self._reward_detector.consume_manual_success():
                    manual_success = True
                    reward, done = self._reward_detector.config.success_reward, True
                    logger.info("Manual success key pressed — marking trial %d as success", self._episode_index)
                if self._reward_detector.consume_manual_failure():
                    manual_failure = True
                    reward, done = 0.0, True
                    logger.info("Manual failure key pressed — ending trial %d", self._episode_index)

            self._episode_reward = max(self._episode_reward, reward)
            episode_end_reason = None
            if self._episode_steps >= self.config.max_episode_steps:
                done = True
                if not manual_success and not manual_failure:
                    episode_end_reason = "timeout"

            if self._residual_input_features is None:
                raise RuntimeError("Residual input features not configured for PLD collect.")

            rl_state = prepare_rl_state_for_buffer(
                obs_for_rl,
                self._residual_input_features,
                self._residual_preprocessor,
                device=device,
            )
            base_action = action_dict_to_tensor(base_action_dict, ordered_keys).to(device)
            executed_action = action_dict_to_tensor(executed_action_dict, ordered_keys).to(device)

            if self._prev_state is not None and self._prev_base_action is not None:
                prev_action = (
                    self._prev_executed_action
                    if self._prev_executed_action is not None
                    else self._prev_base_action
                )
                transition = {
                    "state": self._prev_state,
                    "action": prev_action,
                    "reward": reward,
                    "next_state": rl_state,
                    "done": done,
                    "truncated": False,
                    "complementary_info": make_complementary_info(self._prev_base_action, base_action),
                }
                self._episode_transitions.append(transition)

                if self.config.mode == "online" and self._online_buffer is not None:
                    self._flush_transition(self._online_buffer, transition)

            self._prev_state = rl_state
            self._prev_base_action = base_action
            self._prev_executed_action = executed_action
            self._env_steps += 1
            self._maybe_request_round_stop_after_budget()

            if done:
                if self._episode_reward >= 0.5 or reward >= 0.5:
                    prob = None
                    if manual_success:
                        prob_label = "manual"
                    elif self._reward_detector is not None and self._reward_detector._runtime is not None:
                        p = self._reward_detector._runtime.last_success_prob
                        prob_label = f"{p:.3f}" if p is not None else "n/a"
                    else:
                        prob_label = "n/a"
                    logger.info(
                        "=== Episode %d END (success) === %d steps, reward=%.2f, classifier P=%s",
                        self._episode_index,
                        self._episode_steps,
                        self._episode_reward,
                        prob_label,
                    )
                elif manual_failure:
                    logger.info(
                        "=== Episode %d END (manual failure) === %d steps, reward=%.2f",
                        self._episode_index,
                        self._episode_steps,
                        self._episode_reward,
                    )
                elif episode_end_reason == "timeout":
                    logger.info(
                        "=== Episode %d END (timeout) === %d steps (max=%d), reward=%.2f",
                        self._episode_index,
                        self._episode_steps,
                        self.config.max_episode_steps,
                        self._episode_reward,
                    )
                else:
                    logger.info(
                        "=== Episode %d END (no success) === %d steps, reward=%.2f",
                        self._episode_index,
                        self._episode_steps,
                        self._episode_reward,
                    )
                self._finalize_episode()
                if self.config.mode == "online":
                    self._round_episode_rewards.append(self._episode_reward)
                    if self._episode_reward >= 0.5:
                        self._round_success_count += 1
                self._wait_between_episodes(
                    ctx,
                    episode_index=self._episode_index,
                    episode_reset_time_s=self.config.episode_reset_time_s,
                    manual_scene_reset_pause_key=self.config.manual_scene_reset_pause_key,
                    collect_intervention=self._collect_intervention,
                )
                self._reset_episode()
                if self.config.mode == "offline" and self._should_stop():
                    self._collect_stop = True
                elif self._finish_episode_before_round_stop:
                    logger.info(
                        "=== Online collect round complete === %d env steps — episode finished at boundary, "
                        "starting SAC",
                        self._env_steps,
                    )
                    self._collect_stop = True
                    self._finish_episode_before_round_stop = False

        return on_step

    def _flush_transition(self, buffer: ReplayBuffer, transition: dict) -> None:
        buffer.add(
            state=transition["state"],
            action=transition[ACTION],
            reward=float(transition["reward"]),
            next_state=transition["next_state"],
            done=bool(transition["done"]),
            truncated=transition["truncated"],
            complementary_info=transition.get("complementary_info"),
        )

    def _finalize_episode(self) -> None:
        success = self._episode_reward >= 0.5
        n_transitions = len(self._episode_transitions)
        if self.config.mode == "offline" and success and self._offline_buffer is not None:
            for tr in self._episode_transitions:
                self._flush_transition(self._offline_buffer, tr)
            self._successful_trials += 1
            logger.info(
                "Offline trial %d succeeded — stored %d transitions (%d/%d successful, buffer size %d)",
                self._episode_index,
                n_transitions,
                self._successful_trials,
                self.config.n_successful_trials,
                len(self._offline_buffer),
            )
        elif self.config.mode == "offline" and not success:
            self._failed_trials += 1
            logger.info(
                "Offline trial %d failed (reward=%.2f, %d steps) — discarded (%d failed so far)",
                self._episode_index,
                self._episode_reward,
                self._episode_steps,
                self._failed_trials,
            )
        elif self.config.mode == "online" and success:
            logger.debug(
                "Online episode %d succeeded (reward=%.2f, %d steps)",
                self._episode_index,
                self._episode_reward,
                self._episode_steps,
            )
        elif self.config.mode == "online" and not success:
            logger.debug(
                "Online episode %d ended without success (reward=%.2f, %d steps)",
                self._episode_index,
                self._episode_reward,
                self._episode_steps,
            )

    def _reset_episode(self) -> None:
        self._episode_transitions = []
        self._episode_steps = 0
        self._episode_reward = 0.0
        self._prev_state = None
        self._prev_base_action = None
        self._prev_executed_action = None
        if self._reward_detector is not None:
            self._reward_detector.reset_manual()
