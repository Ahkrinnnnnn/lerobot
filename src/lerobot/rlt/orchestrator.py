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

"""RLT Stage-2 orchestrator: warmup → online collect/train alternation → eval.

Mirrors :class:`lerobot.pld.orchestrator.PLDStage1Orchestrator` but for the RLT
chunked actor-critic. The frozen base VLA + RL token run in-process at chunk
boundaries; the trainable actor/critic are updated by
:class:`~lerobot.rl.algorithms.rlt_td3.RLTTD3Algorithm`.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import torch

from lerobot.datasets import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.processor import make_default_processors
from lerobot.rewards.classifier import RewardClassifierDetector
from lerobot.rl.algorithms.factory import make_algorithm
from lerobot.rl.algorithms.rlt_td3 import RLTTD3AlgorithmConfig
from lerobot.rl.async_runner import AsyncActorLearnerRunner, WeightSlot
from lerobot.rl.buffer import ReplayBuffer
from lerobot.rl.data_sources.data_mixer import OnlineOfflineMixer
from lerobot.rl.trainer import RLTrainer
from lerobot.robots import make_robot_from_config
from lerobot.rollout.configs import RLTCollectStrategyConfig
from lerobot.rollout.context import (
    DatasetContext,
    HardwareContext,
    PolicyContext,
    ProcessorContext,
    RolloutContext,
    RuntimeContext,
)
from lerobot.rollout.robot_wrapper import ThreadSafeRobot
from lerobot.rollout.strategies.factory import create_strategy
from lerobot.rollout.strategies.rlt_collect import RLTCollectStrategy
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.feature_utils import combine_feature_dicts, hw_to_dataset_features

from .configs import RLTStage2Config
from .policy_setup import build_rlt_actor_policy

logger = logging.getLogger(__name__)


@dataclass
class _RLTRuntimeCfg:
    """Minimal runtime knobs read by the rollout strategy (duck-typed RolloutConfig)."""

    fps: float = 30.0
    interpolation_multiplier: int = 1
    return_to_initial_position: bool = True
    device: str | None = None
    display_data: bool = False
    task: str = ""


class RLTStage2Orchestrator:
    """Run RLT Stage-2: warmup → online TD3+BC → optional eval."""

    def __init__(self, cfg: RLTStage2Config, shutdown_event: Event | None = None):
        self.cfg = cfg
        self.shutdown_event = shutdown_event or Event()
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._weight_slot: WeightSlot | None = None
        self._algorithm = None
        self._reward_detector = None

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _make_algo_config(self, actor_policy) -> RLTTD3AlgorithmConfig:
        t = self.cfg.training
        algo_cfg = RLTTD3AlgorithmConfig.from_policy_config(actor_policy.config)
        algo_cfg.discount = t.discount
        algo_cfg.grad_clip_norm = t.grad_clip_norm
        algo_cfg.actor_lr = t.actor_lr
        algo_cfg.critic_lr = t.critic_lr
        algo_cfg.critic_target_update_weight = t.critic_target_update_weight
        algo_cfg.num_critics = t.num_critics
        algo_cfg.utd_ratio = t.utd_ratio
        algo_cfg.policy_update_freq = t.policy_update_freq
        algo_cfg.use_adamw = t.use_adamw
        algo_cfg.optimizer_weight_decay = t.optimizer_weight_decay
        algo_cfg.bc_weight_max = t.bc_weight_max
        algo_cfg.bc_weight_min = t.bc_weight_min
        algo_cfg.bc_decay_steps = t.bc_decay_steps
        algo_cfg.warmup_pretraining_updates = t.warmup_pretraining_updates
        algo_cfg.delta_penalty_weight = t.delta_penalty_weight
        algo_cfg.ref_dropout_prob = t.ref_dropout_prob
        return algo_cfg

    def _make_online_buffer(self, device: str) -> ReplayBuffer:
        t = self.cfg.training
        path = t.resume_online_buffer
        if path and Path(path).exists():
            return ReplayBuffer.load(path, device=device)
        return ReplayBuffer(
            capacity=t.online_buffer_capacity,
            device=device,
            state_keys=["z_rl", "proprio"],
            optimize_memory=True,
        )

    def _build_rollout_context(self, device: str) -> RolloutContext:
        """Build hardware + processors + features (no inference engine)."""
        cfg = self.cfg
        robot = make_robot_from_config(cfg.robot)
        robot.connect()
        initial_obs = robot.get_observation()
        initial_position = {k: v for k, v in initial_obs.items() if k.endswith(".pos")}
        robot_wrapper = ThreadSafeRobot(robot)

        teleop = None
        if cfg.teleop is not None:
            teleop = make_teleoperator_from_config(cfg.teleop)
            teleop.connect()

        teleop_action_proc, robot_action_proc, robot_obs_proc = make_default_processors()

        # Feature aggregation (mirrors build_rollout_context steps 3-4).
        all_obs_features = robot.observation_features
        obs_features_hw = {
            k: v
            for k, v in all_obs_features.items()
            if isinstance(v, tuple) or (v is float and k.endswith(".pos"))
        }
        action_features_hw = {k: v for k, v in robot.action_features.items() if k.endswith(".pos")}
        action_dataset_features = aggregate_pipeline_dataset_features(
            pipeline=teleop_action_proc,
            initial_features=create_initial_features(action=action_features_hw),
            use_videos=True,
        )
        obs_dataset_features = aggregate_pipeline_dataset_features(
            pipeline=robot_obs_proc,
            initial_features=create_initial_features(observation=obs_features_hw),
            use_videos=True,
        )
        dataset_features = combine_feature_dicts(action_dataset_features, obs_dataset_features)
        hw_features = hw_to_dataset_features(obs_features_hw, "observation")
        ordered_action_keys = sorted(action_features_hw.keys())

        runtime_cfg = _RLTRuntimeCfg(
            fps=cfg.fps,
            interpolation_multiplier=cfg.interpolation_multiplier,
            return_to_initial_position=cfg.return_to_initial_position,
            device=device,
            display_data=False,
            task=cfg.task,
        )

        # Dummy policy context (the strategy uses set_rlt_runtime instead).
        return RolloutContext(
            runtime=RuntimeContext(cfg=runtime_cfg, shutdown_event=self.shutdown_event),
            hardware=HardwareContext(
                robot_wrapper=robot_wrapper, teleop=teleop, initial_position=initial_position
            ),
            policy=PolicyContext(policy=None, preprocessor=None, postprocessor=None, inference=None),  # type: ignore[arg-type]
            processors=ProcessorContext(
                teleop_action_processor=teleop_action_proc,
                robot_action_processor=robot_action_proc,
                robot_observation_processor=robot_obs_proc,
            ),
            data=DatasetContext(
                dataset=None,
                dataset_features=dataset_features,
                hw_features=hw_features,
                ordered_action_keys=ordered_action_keys,
            ),
        )

    def _build_strategy(self, ctx, actor_policy, preprocessor, postprocessor, proprio_keys, online_buffer, device, rollout_actor=None) -> RLTCollectStrategy:
        strat_cfg = RLTCollectStrategyConfig(
            mode="eval" if self.cfg.eval_only else "online",
            max_episode_steps=self.cfg.training.max_episode_steps,
            warmup_min_buffer_size=self.cfg.training.warmup_min_buffer_size,
            warmup_pretraining_updates=self.cfg.training.warmup_pretraining_updates,
            replay_stride=self.cfg.training.replay_stride,
            chunk_return_discount=self.cfg.training.chunk_return_discount,
            episode_reset_time_s=self.cfg.training.episode_reset_time_s,
            manual_scene_reset_pause_key=self.cfg.training.manual_scene_reset_pause_key,
            finish_episode_before_round_stop=self.cfg.training.finish_episode_before_round_stop,
        )
        strategy: RLTCollectStrategy = create_strategy(strat_cfg)
        reward_detector = RewardClassifierDetector(self.cfg.reward_classifier)
        self._reward_detector = reward_detector
        strategy.set_rlt_runtime(
            rlt_policy=actor_policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            online_buffer=online_buffer,
            reward_detector=reward_detector,
            proprio_keys=proprio_keys,
            ordered_action_keys=ctx.data.ordered_action_keys,
            dataset_features=ctx.data.dataset_features,
            device=device,
            actor_weights_fn=self._get_actor_weights,
            rollout_actor=rollout_actor,
        )
        return strategy

    def _get_actor_weights(self) -> dict:
        """Latest actor weights for the rollout thread.

        In async mode reads from the shared weight slot (decoupled from the
        learner's in-progress updates); in sequential mode reads directly from
        the algorithm. Returns ``{}`` if no weights are available yet, which
        makes the strategy fall back to the VLA reference chunk.
        """
        if self._weight_slot is not None:
            w = self._weight_slot.get()
            return w if w is not None else {}
        if self._algorithm is not None:
            return self._algorithm.get_weights()
        return {}

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def run(self) -> None:
        device = self.cfg.device or "cuda"
        actor_policy, rlt_policy, preprocessor, postprocessor, proprio_keys = build_rlt_actor_policy(
            rlt_checkpoint_path=self.cfg.rlt_checkpoint_path,
            robot_cfg=self.cfg.robot,
            chunk_len=self.cfg.chunk_len,
            proprio_keys=self.cfg.proprio_keys,
            device=device,
            rename_map=self.cfg.rename_map,
        )

        online_buffer = self._make_online_buffer(device)
        algo_cfg = self._make_algo_config(actor_policy)
        self._algorithm = make_algorithm(algo_cfg, actor_policy)

        # Resume actor / algo checkpoints.
        algo_ckpt = self.cfg.training.resume_algo_checkpoint
        actor_ckpt = self.cfg.training.resume_actor_checkpoint
        if algo_ckpt and Path(algo_ckpt).exists():
            self._algorithm.load_state_dict(
                torch.load(algo_ckpt, weights_only=False), device=device
            )
            logger.info("Loaded RLT-TD3 algorithm checkpoint from %s", algo_ckpt)
        if actor_ckpt and Path(actor_ckpt).exists():
            payload = torch.load(actor_ckpt, weights_only=False)
            if "actor" in payload:
                actor_policy.actor.load_state_dict(payload["actor"])
                logger.info("Loaded actor weights from %s", actor_ckpt)

        ctx = self._build_rollout_context(device)
        # In async mode, give the rollout thread a dedicated inference actor so
        # the learner's in-place updates never race with the rollout's reads.
        rollout_actor = None
        if self.cfg.training.async_training and not self.cfg.eval_only:
            import copy

            rollout_actor = copy.deepcopy(actor_policy.actor)
        strategy = self._build_strategy(
            ctx, actor_policy, preprocessor, postprocessor, proprio_keys, online_buffer, device, rollout_actor
        )

        try:
            strategy.setup(ctx)
            if self.cfg.eval_only:
                self._run_eval(strategy, ctx)
                return

            trainer = RLTrainer(
                algorithm=self._algorithm,
                data_mixer=OnlineOfflineMixer(online_buffer=online_buffer, offline_buffer=None, online_ratio=1.0),
                batch_size=self.cfg.training.batch_size,
            )

            # Warmup: collect BASE-only data until the buffer reaches the min size.
            if not self.cfg.skip_warmup:
                self._run_warmup(strategy, ctx, trainer, online_buffer)

            if not self.cfg.skip_online_train:
                if self.cfg.training.async_training:
                    self._run_online_async(strategy, ctx, trainer, online_buffer, actor_policy)
                else:
                    self._run_online(strategy, ctx, trainer, online_buffer)
        finally:
            strategy.teardown(ctx)
            if self._reward_detector is not None:
                self._reward_detector.stop()

    def _run_warmup(self, strategy: RLTCollectStrategy, ctx, trainer, online_buffer: ReplayBuffer) -> None:
        t = self.cfg.training
        if len(online_buffer) >= t.warmup_min_buffer_size:
            logger.info("Warmup skipped — buffer already has %d windows", len(online_buffer))
            return
        round_id = 0
        logger.info(
            "=== RLT warmup === collecting BASE-only data until buffer >= %d windows",
            t.warmup_min_buffer_size,
        )
        while (
            len(online_buffer) < t.warmup_min_buffer_size
            and not self.shutdown_event.is_set()
            and (t.warmup_collect_rounds == 0 or round_id < t.warmup_collect_rounds)
        ):
            round_id += 1
            strategy.begin_online_collect_round(
                collect_round=round_id,
                warmup_active=True,
                max_env_steps=t.collect_env_steps_per_iter,
                max_episode_steps=t.max_episode_steps,
                episode_reset_time_s=t.episode_reset_time_s,
                finish_episode_before_round_stop=t.finish_episode_before_round_stop,
                manual_scene_reset_pause_key=t.manual_scene_reset_pause_key,
            )
            strategy.run(ctx)
            logger.info(
                "Warmup round %d done — buffer=%d/%d",
                round_id,
                len(online_buffer),
                t.warmup_min_buffer_size,
            )

        # Optional BC-only pretraining updates during warmup (algorithm zeroes
        # the RL term while ``_optimization_step < warmup_pretraining_updates``).
        if t.warmup_pretraining_updates > 0 and len(online_buffer) >= t.online_step_before_learning:
            for _ in range(t.warmup_pretraining_updates):
                trainer.training_step()
            logger.info("Warmup BC pretraining: %d updates", t.warmup_pretraining_updates)

    def _run_online(self, strategy, ctx, trainer, online_buffer) -> None:
        t = self.cfg.training
        progress_path = self.output_dir / "rlt_stage2_progress.json"
        train_steps = 0
        collect_round = 0
        if progress_path.exists():
            try:
                progress = json.loads(progress_path.read_text())
                train_steps = int(progress.get("train_steps", 0))
                collect_round = int(progress.get("collect_round", 0))
            except (json.JSONDecodeError, OSError, ValueError):
                pass

        logger.info(
            "=== RLT online RL === %d gradient steps | %d updates/round | ≥%d env steps/round",
            t.rl_steps,
            t.train_iters_per_collect,
            t.collect_env_steps_per_iter,
        )

        while train_steps < t.rl_steps and not self.shutdown_event.is_set():
            collect_round += 1
            strategy.begin_online_collect_round(
                collect_round=collect_round,
                warmup_active=False,
                max_env_steps=t.collect_env_steps_per_iter,
                max_episode_steps=t.max_episode_steps,
                episode_reset_time_s=t.episode_reset_time_s,
                finish_episode_before_round_stop=t.finish_episode_before_round_stop,
                manual_scene_reset_pause_key=t.manual_scene_reset_pause_key,
            )
            strategy.run(ctx)
            summary = strategy.online_round_reward_summary()
            logger.info(
                "Online round %d done — env_steps=%d buffer=%d episodes=%s successes=%s mean_r=%.3f",
                collect_round,
                strategy._env_steps,
                len(online_buffer),
                summary.get("episodes", 0),
                summary.get("successes", 0),
                summary.get("mean_max_reward", 0.0),
            )

            round_steps = 0
            last_stats = None
            for _ in range(t.train_iters_per_collect):
                if len(online_buffer) < t.online_step_before_learning:
                    break
                last_stats = trainer.training_step()
                train_steps += 1
                round_steps += 1
            if last_stats is not None:
                logger.info(
                    "Online round %d summary — grad_steps=%d (total %d/%d) losses=%s extra=%s",
                    collect_round,
                    round_steps,
                    train_steps,
                    t.rl_steps,
                    last_stats.losses,
                    last_stats.extra,
                )

            online_buffer.save(str(self.output_dir / "online_buffer.pt"))
            torch.save(self._algorithm.state_dict(), str(self.output_dir / "rlt_td3.pt"))
            torch.save(self._algorithm.get_weights(), str(self.output_dir / "actor_weights.pt"))
            actor_policy_save = self.output_dir / "rlt_actor"
            self._algorithm.policy.save_pretrained(str(actor_policy_save))
            progress_path.write_text(
                json.dumps(
                    {
                        "train_steps": train_steps,
                        "collect_round": collect_round,
                        "online_buffer_size": len(online_buffer),
                    },
                    indent=2,
                )
            )

    def _run_eval(self, strategy: RLTCollectStrategy, ctx) -> None:
        strat: RLTCollectStrategy = strategy
        strat.config.n_successful_trials = self.cfg.training.eval_episodes
        strat.config.mode = "eval"
        strat._online_active = True
        logger.info("=== RLT eval === running %d episodes with the trained actor", self.cfg.training.eval_episodes)
        strat.run(ctx)
        logger.info(
            "Eval complete — successes=%d/%d",
            strat._successful_trials,
            self.cfg.training.eval_episodes,
        )

    # ------------------------------------------------------------------
    # Asynchronous rollout + learning (paper §IV-B: "rollouts and learning
    # asynchronously"). Delegates the dual-thread lifecycle to the reusable
    # :class:`~lerobot.rl.async_runner.AsyncActorLearnerRunner`; the RLT-specific
    # pieces are the learner step (TD3 update + weight publish), the rollout
    # loop (continuous episode collection), and the checkpoint callback.
    # ------------------------------------------------------------------

    def _run_online_async(self, strategy, ctx, trainer, online_buffer, actor_policy) -> None:
        t = self.cfg.training
        progress_path = self.output_dir / "rlt_stage2_progress.json"
        train_steps = 0
        if progress_path.exists():
            try:
                train_steps = int(json.loads(progress_path.read_text()).get("train_steps", 0))
            except (json.JSONDecodeError, OSError, ValueError):
                pass

        self._weight_slot = WeightSlot()
        self._weight_slot.publish(self._algorithm.get_weights())

        min_buffer = t.online_step_before_learning
        algo = self._algorithm

        def learner_fn(_steps: int):
            if len(online_buffer) < min_buffer:
                time.sleep(0.05)
                return None  # skipped step — runner won't increment the counter
            stats = trainer.training_step()
            if algo._optimization_step % t.policy_update_freq == 0:
                self._weight_slot.publish(algo.get_weights())
            return stats

        def rollout_fn() -> None:
            strategy.begin_online_collect_round(
                collect_round=1,
                warmup_active=False,
                max_env_steps=0,  # 0 → no round budget; runs until _collect_stop / shutdown
                max_episode_steps=t.max_episode_steps,
                episode_reset_time_s=t.episode_reset_time_s,
                finish_episode_before_round_stop=t.finish_episode_before_round_stop,
                manual_scene_reset_pause_key=t.manual_scene_reset_pause_key,
            )
            strategy.run(ctx)

        def stop_rollout_fn() -> None:
            strategy._collect_stop = True

        def checkpoint_fn(steps: int) -> None:
            self._save_async_checkpoint(online_buffer, steps, actor_policy, progress_path)

        runner = AsyncActorLearnerRunner(self.shutdown_event, self._weight_slot, name="rlt")
        train_steps = runner.run(
            learner_fn=learner_fn,
            rollout_fn=rollout_fn,
            stop_rollout_fn=stop_rollout_fn,
            target_steps=t.rl_steps,
            checkpoint_every_steps=t.checkpoint_every_steps,
            log_every=50,
            checkpoint_fn=checkpoint_fn,
            log=logger,
        )
        logger.info(
            "=== RLT async online RL finished === total grad_steps=%d buffer=%d",
            train_steps,
            len(online_buffer),
        )

    def _save_async_checkpoint(
        self, online_buffer, train_steps: int, actor_policy, progress_path: Path
    ) -> None:
        """Persist buffer + algorithm + actor + progress under the buffer lock.

        Takes ``online_buffer._lock`` so the save reads a consistent snapshot
        while the rollout thread's adds (and the prefetch producer's samples)
        are briefly blocked.
        """
        try:
            with online_buffer._lock:
                online_buffer.save(str(self.output_dir / "online_buffer.pt"))
            torch.save(self._algorithm.state_dict(), str(self.output_dir / "rlt_td3.pt"))
            torch.save(self._algorithm.get_weights(), str(self.output_dir / "actor_weights.pt"))
            actor_policy.save_pretrained(str(self.output_dir / "rlt_actor"))
            progress_path.write_text(
                json.dumps(
                    {
                        "train_steps": train_steps,
                        "online_buffer_size": len(online_buffer),
                        "async": True,
                    },
                    indent=2,
                )
            )
        except Exception:
            logger.exception("Async checkpoint save failed")
