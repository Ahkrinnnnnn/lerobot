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

"""RLT Stage-2 rollout strategy: chunked open-loop execution + step-trace replay.

The frozen base VLA + RL token module run in-process at chunk boundaries
(no RTC inference engine): each boundary produces ``(z_rl, ã_{1:C})``; the
trainable actor refines the chunk (or the VLA reference is executed directly
during warmup). The chunk is executed open-loop for ``C`` ticks while a raw
per-tick step trace is recorded; at episode end the trace is sliced into
replay windows (see :mod:`lerobot.rlt.replay_windows`).

Human intervention is supported via an optional teleoperator: while the
operator holds the correction toggle, the teleop action overrides the policy
and the step is labelled ``HUMAN`` (becomes the BC target for that chunk).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn  # noqa: F401  (type hint for rollout_actor)
from torch import Tensor

from lerobot.rl.buffer import ReplayBuffer
from lerobot.rlt.replay_windows import (
    SOURCE_BASE,
    SOURCE_HUMAN,
    SOURCE_MIXED,
    SOURCE_RL,
    StepRecord,
    build_replay_windows,
)
from lerobot.utils.action_interpolator import ActionInterpolator
from lerobot.utils.robot_utils import precise_sleep

from ..collect_intervention import CollectInterventionController
from .core import RolloutStrategy

if TYPE_CHECKING:
    from lerobot.policies.rlt_actor import RLTActorPolicy
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.rewards.classifier import RewardClassifierDetector

    from ..configs import RLTCollectStrategyConfig, RolloutContext

logger = logging.getLogger(__name__)


class RLTCollectStrategy(RolloutStrategy):
    """Chunked open-loop rollout with step-trace replay for RLT Stage-2."""

    def __init__(self, config: RLTCollectStrategyConfig) -> None:
        super().__init__(config)
        # Runtime (set via set_rlt_runtime before setup/run).
        self._rlt_policy: RLTActorPolicy | None = None
        self._preprocessor: PolicyProcessorPipeline | None = None
        self._postprocessor: PolicyProcessorPipeline | None = None
        self._online_buffer: ReplayBuffer | None = None
        self._reward_detector: RewardClassifierDetector | None = None
        self._proprio_keys: list[str] = []
        self._ordered_action_keys: list[str] = []
        self._dataset_features: dict = {}
        self._device: str = "cpu"
        self._actor_weights_fn: Callable[[], dict] | None = None
        self._rollout_actor = None  # dedicated inference actor (async mode)
        self._chunk_return_discount: float = self.config.chunk_return_discount

        # Episode / round state.
        self._collect_stop = False
        self._collect_intervention: CollectInterventionController | None = None
        self._episode_index = 0
        self._episode_steps = 0
        self._env_steps = 0
        self._episode_reward = 0.0
        self._successful_trials = 0
        self._failed_trials = 0
        self._trace: list[StepRecord] = []
        self._round_episode_rewards: list[float] = []
        self._round_success_count = 0
        self._finish_episode_before_round_stop = False
        self._online_active = False  # actor driving the robot?

    # ------------------------------------------------------------------
    # Runtime wiring
    # ------------------------------------------------------------------

    def set_rlt_runtime(
        self,
        *,
        rlt_policy: RLTActorPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        online_buffer: ReplayBuffer | None = None,
        reward_detector: RewardClassifierDetector | None = None,
        proprio_keys: list[str] | None = None,
        ordered_action_keys: list[str] | None = None,
        dataset_features: dict | None = None,
        device: str = "cpu",
        actor_weights_fn: Callable[[], dict] | None = None,
        rollout_actor: nn.Module | None = None,
    ) -> None:
        self._rlt_policy = rlt_policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._online_buffer = online_buffer
        self._reward_detector = reward_detector
        self._proprio_keys = list(proprio_keys or [])
        self._ordered_action_keys = list(ordered_action_keys or [])
        self._dataset_features = dataset_features or {}
        self._device = device
        self._actor_weights_fn = actor_weights_fn
        # Optional dedicated inference actor (async mode): kept in sync with the
        # learner's actor via the weight slot, so the learner's in-place updates
        # and the rollout's reads never race on the same module.
        self._rollout_actor = rollout_actor

    def reset_collect_state(self) -> None:
        self._collect_stop = False
        self._episode_index = 0
        self._episode_steps = 0
        self._env_steps = 0
        self._episode_reward = 0.0
        self._successful_trials = 0
        self._failed_trials = 0
        self._trace = []
        self._round_episode_rewards = []
        self._round_success_count = 0
        self._online_active = False

    def begin_online_collect_round(
        self,
        *,
        collect_round: int,
        warmup_active: bool,
        max_env_steps: int,
        max_episode_steps: int,
        episode_reset_time_s: float,
        finish_episode_before_round_stop: bool = True,
        manual_scene_reset_pause_key: str = "space",
    ) -> None:
        self.config.max_env_steps = max_env_steps
        self.config.max_episode_steps = max_episode_steps
        self.config.episode_reset_time_s = episode_reset_time_s
        self.config.finish_episode_before_round_stop = finish_episode_before_round_stop
        self.config.manual_scene_reset_pause_key = manual_scene_reset_pause_key
        self._finish_episode_before_round_stop = False
        self._online_active = not warmup_active
        if self._collect_intervention is not None:
            self._collect_intervention.pause_key  # noqa: B018
        self.reset_collect_state()
        if self._reward_detector is not None:
            self._reward_detector.reset_manual()
        logger.info(
            "=== RLT online collect round %d === warmup=%s actor_active=%s | ≥%d env steps | %.0fs reset",
            collect_round,
            warmup_active,
            self._online_active,
            max_env_steps,
            episode_reset_time_s,
        )

    def online_round_reward_summary(self) -> dict[str, float]:
        return {
            "episodes": len(self._round_episode_rewards),
            "successes": self._round_success_count,
            "mean_max_reward": (
                sum(self._round_episode_rewards) / len(self._round_episode_rewards)
                if self._round_episode_rewards
                else 0.0
            ),
            "last_episode_reward": self._round_episode_rewards[-1] if self._round_episode_rewards else 0.0,
        }

    # ------------------------------------------------------------------
    # Strategy lifecycle
    # ------------------------------------------------------------------

    def setup(self, ctx: RolloutContext) -> None:
        self._collect_stop = False
        self._collect_intervention = CollectInterventionController(
            self.config.manual_scene_reset_pause_key
        )
        self._collect_intervention.start()
        logger.info(
            "RLT collect started — mode=%s | chunk_len=%d | operator key '%s' = pause/resume",
            self.config.mode,
            self._rlt_policy.config.chunk_len if self._rlt_policy else 0,
            self.config.manual_scene_reset_pause_key,
        )

    def run(self, ctx: RolloutContext) -> None:
        while not self._should_stop() and not ctx.runtime.shutdown_event.is_set():
            self._run_episode(ctx)

    def teardown(self, ctx: RolloutContext) -> None:
        if self._collect_intervention is not None:
            self._collect_intervention.stop()
            self._collect_intervention = None
        self._teardown_hardware(
            ctx.hardware,
            return_to_initial_position=ctx.runtime.cfg.return_to_initial_position,
        )

    def _should_stop(self) -> bool:
        if self.config.mode == "eval":
            return self._successful_trials >= max(1, self.config.n_successful_trials)
        if self.config.n_successful_trials > 0:
            return self._successful_trials >= self.config.n_successful_trials
        return self._collect_stop

    def _round_budget_reached(self) -> bool:
        return (
            self.config.mode == "online"
            and self.config.max_env_steps > 0
            and self._env_steps >= self.config.max_env_steps
        )

    # ------------------------------------------------------------------
    # Episode loop
    # ------------------------------------------------------------------

    def _run_episode(self, ctx: RolloutContext) -> None:
        cfg = self.config
        chunk_len = self._rlt_policy.config.chunk_len
        action_dim = self._rlt_policy.config.action_dim
        fps = ctx.runtime.cfg.fps
        control_interval = 1.0 / fps if fps > 0 else 0.0
        robot = ctx.hardware.robot_wrapper
        processors = ctx.processors
        shutdown = ctx.runtime.shutdown_event

        self._episode_index += 1
        self._episode_steps = 0
        self._episode_reward = 0.0
        self._trace = []
        interpolator = ActionInterpolator(multiplier=ctx.runtime.cfg.interpolation_multiplier)

        chunk_id = 0
        executed_chunk: Tensor | None = None  # [C, d] normalized
        z_rl: Tensor | None = None  # [token_dim]
        ref_chunk: Tensor | None = None  # [C, d] normalized
        chunk_source = SOURCE_BASE

        while not shutdown.is_set():
            if self._collect_intervention is not None and self._collect_intervention.paused:
                precise_sleep(0.05)
                continue

            loop_start = time.perf_counter()
            obs_raw = self._get_observation(robot)
            obs_processed = processors.robot_observation_processor(obs_raw)

            # Chunk boundary: query frozen VLA + decide executed chunk.
            if executed_chunk is None:
                z_rl, ref_flat, ref_chunk = self._query_chunk(obs_processed)
                proprio = self._extract_proprio(obs_processed)
                executed_chunk, chunk_source = self._decide_executed_chunk(
                    z_rl, ref_flat, ref_chunk, proprio, obs_processed
                )
                interpolator.reset()

            # Per-tick execution.
            t_in_chunk = self._episode_steps - (self._episode_steps // chunk_len) * chunk_len
            # Advance through the chunk based on interpolator.
            if interpolator.needs_new_action():
                step_idx = min(t_in_chunk, chunk_len - 1)
                step_action_norm = executed_chunk[step_idx].unsqueeze(0)  # [1, d]
                interpolator.add(step_action_norm.cpu())

            interp = interpolator.get()
            executed_step_dict = None
            human_action = None
            step_source = chunk_source
            if interp is not None:
                action_dict = self._action_tensor_to_dict(interp, self._ordered_action_keys)
                # Human intervention override (teleop).
                human = self._read_intervention_action(ctx, obs_raw)
                if human is not None:
                    action_dict = human
                    human_action = torch.tensor(
                        [human[k] for k in self._ordered_action_keys], dtype=torch.float32
                    )
                    step_source = SOURCE_HUMAN if chunk_source in (SOURCE_BASE, SOURCE_HUMAN) else SOURCE_MIXED
                processed = processors.robot_action_processor((action_dict, obs_raw))
                robot.send_action(processed)
                # Executed normalized action stored for replay (in normalized space).
                executed_step_norm = interp
                if human_action is not None:
                    # Store the human action as the executed normalized action so BC target aligns.
                    executed_step_norm = human_action
            else:
                executed_step_norm = executed_chunk[min(t_in_chunk, chunk_len - 1)]

            # Reward / done.
            reward, done = self._compute_reward_done(obs_processed)
            manual_success = False
            manual_failure = False
            if self._reward_detector is not None:
                if self._reward_detector.consume_manual_success():
                    manual_success = True
                    reward, done = self._reward_detector.config.success_reward, True
                if self._reward_detector.consume_manual_failure():
                    manual_failure = True
                    reward, done = 0.0, True
            self._episode_reward = max(self._episode_reward, reward)

            self._episode_steps += 1
            self._env_steps += 1

            # Record step trace.
            proprio_now = self._extract_proprio(obs_processed)
            self._trace.append(
                StepRecord(
                    t=self._episode_steps - 1,
                    chunk_id=chunk_id,
                    t_in_chunk=t_in_chunk,
                    z_rl=z_rl.detach().cpu(),
                    ref_chunk=ref_chunk.detach().cpu(),
                    proprio=proprio_now.detach().cpu(),
                    executed=executed_step_norm.detach().cpu(),
                    reward=float(reward),
                    done=bool(done),
                    source=step_source,
                    human_action=human_action.detach().cpu() if human_action is not None else None,
                )
            )

            # End-of-chunk boundary reached → reset for next chunk.
            if t_in_chunk == chunk_len - 1:
                chunk_id += 1
                executed_chunk = None
                z_rl = None
                ref_chunk = None

            if self._episode_steps >= cfg.max_episode_steps:
                done = True

            if done:
                self._finalize_episode(manual_success, manual_failure)
                self._round_episode_rewards.append(self._episode_reward)
                if self._episode_reward >= 0.5:
                    self._round_success_count += 1
                self._wait_between_episodes(
                    ctx,
                    episode_index=self._episode_index,
                    episode_reset_time_s=cfg.episode_reset_time_s,
                    manual_scene_reset_pause_key=cfg.manual_scene_reset_pause_key,
                    collect_intervention=self._collect_intervention,
                )
                self._reset_episode()
                self._maybe_request_round_stop()
                return

            # Timing.
            dt = time.perf_counter() - loop_start
            if control_interval > 0:
                precise_sleep(max(0.0, control_interval - dt))

    # ------------------------------------------------------------------
    # Chunk-boundary helpers
    # ------------------------------------------------------------------

    def _query_chunk(self, obs_processed: dict) -> tuple[Tensor, Tensor, Tensor]:
        """Run the frozen VLA + RL token at a chunk boundary.

        Returns ``(z_rl [token_dim], ref_flat [C*d], ref_chunk [C, d])`` in the
        VLA's normalized action space.
        """
        batch = self._build_policy_batch(obs_processed)
        z_rl, ref_flat = self._rlt_policy.machine_a_query(batch)
        ref_chunk = ref_flat.reshape(self._rlt_policy.config.chunk_len, self._rlt_policy.config.action_dim)
        return z_rl.squeeze(0), ref_flat.squeeze(0), ref_chunk.squeeze(0)

    def _decide_executed_chunk(
        self,
        z_rl: Tensor,
        ref_flat: Tensor,
        ref_chunk: Tensor,
        proprio: Tensor,
        obs_processed: dict,
    ) -> tuple[Tensor, int]:
        """Choose the executed chunk (warmup → VLA ref; online → actor).

        Uses the dedicated ``_rollout_actor`` when set (async mode) so the
        learner's in-place actor updates never race with this read; otherwise
        falls back to the shared ``_rlt_policy.actor`` (sequential mode).
        Reuses the ``z_rl`` already extracted at the chunk boundary instead of
        re-running the frozen VLA.
        """
        if not self._online_active or self._rlt_policy is None:
            return ref_chunk, SOURCE_BASE
        actor = self._rollout_actor if self._rollout_actor is not None else self._rlt_policy.actor
        if self._actor_weights_fn is not None:
            try:
                weights = self._actor_weights_fn()
                actor.load_state_dict(weights["actor"])
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not sync actor weights: %s — executing VLA reference", e)
                return ref_chunk, SOURCE_BASE
        with torch.inference_mode():
            z_rl_b = z_rl.unsqueeze(0).to(self._device)
            proprio_b = proprio.unsqueeze(0).to(self._device)
            ref_b = ref_flat.unsqueeze(0).to(self._device)
            out = actor(z_rl_b, proprio_b, ref_b)
            action_flat = out[2] if self.config.mode == "eval" else out[0]
            chunk_len = self._rlt_policy.config.chunk_len
            action_dim = self._rlt_policy.config.action_dim
            chunk = action_flat.reshape(chunk_len, action_dim)
        return chunk.squeeze(0), SOURCE_RL

    def _build_policy_batch(self, obs_processed: dict) -> dict[str, Tensor]:
        """Run the frozen VLA preprocessor on processed robot obs."""
        from lerobot.utils.feature_utils import build_dataset_frame
        from lerobot.utils.constants import OBS_STR

        obs_frame = build_dataset_frame(self._dataset_features, obs_processed, prefix=OBS_STR)
        with torch.inference_mode():
            batch = self._preprocessor(obs_frame)
        return batch

    def _extract_proprio(self, obs_processed: dict) -> Tensor:
        keys = self._proprio_keys or [k for k in obs_processed if k.endswith(".pos") and not k.startswith("gripper")]
        vals = [float(obs_processed[k]) for k in keys]
        return torch.tensor(vals, dtype=torch.float32, device=self._device)

    # ------------------------------------------------------------------
    # Per-tick helpers
    # ------------------------------------------------------------------

    def _get_observation(self, robot) -> dict:
        try:
            return robot.get_observation(include_images=True)
        except TypeError:
            return robot.get_observation()

    def _action_tensor_to_dict(self, tensor: Tensor, ordered_keys: list[str]) -> dict[str, float]:
        vals = tensor.flatten().tolist()
        return {k: float(vals[i]) for i, k in enumerate(ordered_keys) if i < len(vals)}

    def _compute_reward_done(self, obs_processed: dict) -> tuple[float, bool]:
        if self._reward_detector is None:
            return 0.0, False
        return self._reward_detector.predict(obs_processed)

    def _read_intervention_action(self, ctx: RolloutContext, obs_raw: dict) -> dict[str, float] | None:
        """Return a human action dict when the operator is intervening, else None."""
        teleop = ctx.hardware.teleop
        if teleop is None or not teleop.is_connected:
            return None
        # No active correction key in this minimal implementation: when a teleop
        # is connected, intervention is opt-in via the operator manually moving
        # the leader. Subclasses / future work can wire a correction key.
        return None

    # ------------------------------------------------------------------
    # Episode finalization
    # ------------------------------------------------------------------

    def _finalize_episode(self, manual_success: bool, manual_failure: bool) -> None:
        cfg = self.config
        chunk_len = self._rlt_policy.config.chunk_len
        success = self._episode_reward >= 0.5
        windows = build_replay_windows(
            self._trace,
            chunk_len=chunk_len,
            stride=cfg.replay_stride,
            discount=self._chunk_return_discount,
        )
        if self._online_buffer is not None and cfg.mode == "online":
            for w in windows:
                self._online_buffer.add(
                    state={k: v.unsqueeze(0) for k, v in w.state.items()},
                    action=w.action.unsqueeze(0),
                    reward=w.reward,
                    next_state={k: v.unsqueeze(0) for k, v in w.next_state.items()},
                    done=w.done,
                    truncated=False,
                    complementary_info={
                        k: (v.unsqueeze(0) if isinstance(v, Tensor) and v.ndim == 1 else v)
                        for k, v in w.complementary_info.items()
                    },
                )
        if cfg.mode == "eval":
            tag = "success" if success else ("manual_fail" if manual_failure else "no_success")
            logger.info(
                "=== RLT eval episode %d END (%s) === %d steps, reward=%.2f, windows=%d",
                self._episode_index,
                tag,
                self._episode_steps,
                self._episode_reward,
                len(windows),
            )
            if success:
                self._successful_trials += 1
            return

        if success:
            self._successful_trials += 1
        elif manual_failure:
            self._failed_trials += 1
        logger.info(
            "=== RLT episode %d END === success=%s steps=%d reward=%.2f windows=%d buffer=%d",
            self._episode_index,
            success,
            self._episode_steps,
            self._episode_reward,
            len(windows),
            len(self._online_buffer) if self._online_buffer else 0,
        )

    def _reset_episode(self) -> None:
        self._trace = []
        self._episode_steps = 0
        self._episode_reward = 0.0
        if self._reward_detector is not None:
            self._reward_detector.reset_manual()

    def _maybe_request_round_stop(self) -> None:
        if not self._round_budget_reached():
            return
        if self.config.finish_episode_before_round_stop:
            if self._episode_steps > 0:
                self._finish_episode_before_round_stop = True
                logger.info(
                    "=== RLT round budget reached (%d/%d) === finishing episode before training",
                    self._env_steps,
                    self.config.max_env_steps,
                )
            else:
                self._collect_stop = True
        else:
            self._collect_stop = True
