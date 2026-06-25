# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from threading import Event

import torch

from lerobot.configs.types import FeatureType
from lerobot.pld.configs import PLDStage1Config
from lerobot.policies.gaussian_actor.configuration_gaussian_actor import CriticNetworkConfig
from lerobot.pld.residual_setup import init_residual_policy
from lerobot.rewards.classifier import RewardClassifierDetector
from lerobot.policies.residual_gaussian.configuration_residual_gaussian import ResidualGaussianActorConfig
from lerobot.policies.residual_gaussian.modeling_residual_gaussian import ResidualGaussianActorPolicy
from lerobot.rl.algorithms.calql.calql_pretrainer import CalQLPretrainer
from lerobot.rl.algorithms.factory import make_algorithm
from lerobot.rl.algorithms.residual_sac.configuration_residual_sac import ResidualSACAlgorithmConfig
from lerobot.rl.buffer import ReplayBuffer
from lerobot.rl.data_sources.data_mixer import OnlineOfflineMixer
from lerobot.rl.trainer import RLTrainer
from lerobot.rollout.context import build_rollout_context
from lerobot.rollout.strategies.factory import create_strategy
from lerobot.rollout.configs import PLDCollectStrategyConfig
from lerobot.rollout.strategies.pld_collect import PLDCollectStrategy
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE

logger = logging.getLogger(__name__)


class PLDStage1Orchestrator:
    """Run PLD Algorithm 1 Stage 1: offline collect → Cal-QL → online SAC."""

    def __init__(self, cfg: PLDStage1Config, shutdown_event: Event | None = None):
        self.cfg = cfg
        self.shutdown_event = shutdown_event or Event()
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _state_keys(self, residual_cfg: ResidualGaussianActorConfig) -> list[str]:
        keys = []
        for key, ft in residual_cfg.input_features.items():
            if ft.type in (FeatureType.VISUAL, FeatureType.STATE):
                keys.append(key)
        return keys or [OBS_STATE]

    def _init_residual_policy(self) -> tuple[ResidualGaussianActorPolicy, Any, list[str]]:
        if not isinstance(self.cfg.residual_policy, ResidualGaussianActorConfig):
            raise TypeError("residual_policy must be ResidualGaussianActorConfig")
        device = self.cfg.device or self.cfg.residual_policy.device or "cuda"
        return init_residual_policy(
            robot_cfg=self.cfg.robot,
            policy_cfg=self.cfg.residual_policy,
            device=device,
            xi=self.cfg.pld.xi,
            base_policy=self.cfg.base_policy,
        )

    def _make_buffers(self, state_keys: list[str]) -> tuple[ReplayBuffer, ReplayBuffer]:
        device = self.cfg.device or "cuda"
        offline_path = self.cfg.pld.resume_offline_buffer
        online_path = self.cfg.pld.resume_online_buffer
        if offline_path and Path(offline_path).exists():
            offline = ReplayBuffer.load(offline_path, device=device)
        else:
            offline = ReplayBuffer(
                capacity=self.cfg.pld.offline_buffer_capacity,
                device=device,
                state_keys=state_keys,
                optimize_memory=self.cfg.pld.buffer_optimize_memory,
            )
        if online_path and Path(online_path).exists():
            online = ReplayBuffer.load(online_path, device=device)
        else:
            online = ReplayBuffer(
                capacity=self.cfg.pld.online_buffer_capacity,
                device=device,
                state_keys=state_keys,
                optimize_memory=self.cfg.pld.buffer_optimize_memory,
            )
        return offline, online

    def _prepare_buffers_for_sac(
        self,
        offline_buffer: ReplayBuffer,
        online_buffer: ReplayBuffer,
    ) -> tuple[ReplayBuffer, ReplayBuffer]:
        offline = offline_buffer.ensure_base_action_complementary_info()
        online = (
            online_buffer.ensure_base_action_complementary_info()
            if len(online_buffer) > 0
            else online_buffer
        )
        return offline, online

    def _effective_warmup_env_steps(self) -> int:
        """Resolve warmup length (paper: 100 base-only episodes)."""
        pld = self.cfg.pld
        if pld.warmup_episodes is not None:
            avg_steps = pld.avg_episode_steps
            if avg_steps is None:
                avg_steps = max(1, pld.max_episode_steps // 2)
                logger.info(
                    "warmup_episodes=%d without avg_episode_steps; using max_episode_steps/2=%d",
                    pld.warmup_episodes,
                    avg_steps,
                )
            return pld.warmup_episodes * avg_steps
        return pld.warmup_env_steps

    def _make_residual_sac_config(self, residual_policy) -> ResidualSACAlgorithmConfig:
        """Build ResidualSAC config from PLD training hyper-parameters (paper Table 5)."""
        pld = self.cfg.pld
        policy_cfg = residual_policy.config
        algo_cfg = ResidualSACAlgorithmConfig.from_policy_config(policy_cfg)
        actor_kw = policy_cfg.actor_network_kwargs
        algo_cfg.critic_network_kwargs = CriticNetworkConfig(
            hidden_dims=list(actor_kw.hidden_dims),
            activate_final=actor_kw.activate_final,
            activations=actor_kw.activations,
        )
        algo_cfg.discount = pld.discount
        algo_cfg.grad_clip_norm = pld.grad_clip_norm
        algo_cfg.actor_lr = pld.actor_lr
        algo_cfg.critic_lr = pld.critic_lr
        algo_cfg.temperature_lr = pld.temperature_lr
        algo_cfg.temperature_init = pld.temperature_init
        algo_cfg.critic_target_update_weight = pld.critic_target_update_weight
        algo_cfg.num_critics = pld.num_critics
        algo_cfg.utd_ratio = pld.utd_ratio
        algo_cfg.policy_update_freq = pld.policy_update_freq
        algo_cfg.use_adamw = pld.use_adamw
        algo_cfg.optimizer_weight_decay = pld.optimizer_weight_decay
        return algo_cfg

    def collect_offline(
        self,
        offline_buffer: ReplayBuffer,
        reward_detector: RewardClassifierDetector,
        state_keys: list[str],
        residual_preprocessor,
        residual_policy,
    ) -> None:
        rollout_cfg = self.cfg.to_rollout_config()
        ctx = build_rollout_context(rollout_cfg, self.shutdown_event)
        logger.info("Debug: build_rollout_context returned, creating PLDCollectStrategy")
        strategy = create_strategy(rollout_cfg.strategy)
        if not isinstance(strategy, PLDCollectStrategy):
            raise TypeError("Expected PLDCollectStrategy for offline collection.")
        strategy.set_pld_runtime(
            offline_buffer=offline_buffer,
            reward_detector=reward_detector,
            residual_preprocessor=residual_preprocessor,
            residual_policy=residual_policy,
            state_keys=state_keys,
        )
        try:
            strategy.setup(ctx)
            strategy.run(ctx)
        finally:
            strategy.teardown(ctx)
        save_path = self.output_dir / "offline_buffer.pt"
        offline_buffer.save(str(save_path))
        logger.info(
            "Offline buffer saved: %s (%d transitions)",
            save_path,
            len(offline_buffer),
        )

    def pretrain_critic(
        self,
        offline_buffer: ReplayBuffer,
        residual_policy,
        state_keys: list[str],
    ) -> None:
        offline_buffer = offline_buffer.ensure_base_action_complementary_info()
        algo_cfg = self._make_residual_sac_config(residual_policy)
        algorithm = make_algorithm(algo_cfg, residual_policy)
        mixer = OnlineOfflineMixer(online_buffer=offline_buffer, offline_buffer=None, online_ratio=1.0)
        batch_iter = mixer.get_iterator(batch_size=self.cfg.pld.batch_size)
        pretrainer = CalQLPretrainer(algorithm, calql_alpha=self.cfg.pld.calql_alpha)
        algorithm.make_optimizers_and_scheduler()
        pretrainer.pretrain(batch_iter, num_steps=self.cfg.pld.calql_steps)
        torch.save(
            algorithm.state_dict(),
            self.output_dir / "calql_critic.pt",
        )

    def rl_train(
        self,
        offline_buffer: ReplayBuffer,
        online_buffer: ReplayBuffer,
        residual_policy,
        reward_detector: RewardClassifierDetector,
        state_keys: list[str],
        residual_preprocessor,
    ) -> None:
        offline_buffer, online_buffer = self._prepare_buffers_for_sac(offline_buffer, online_buffer)
        algo_cfg = self._make_residual_sac_config(residual_policy)
        algorithm = make_algorithm(algo_cfg, residual_policy)

        mixer = OnlineOfflineMixer(
            online_buffer=online_buffer,
            offline_buffer=offline_buffer,
            online_ratio=self.cfg.pld.online_ratio,
        )
        trainer = RLTrainer(
            algorithm=algorithm,
            data_mixer=mixer,
            batch_size=self.cfg.pld.batch_size,
        )
        sac_path = self.output_dir / "residual_sac.pt"
        calql_path = self.output_dir / "calql_critic.pt"
        if sac_path.exists():
            sac_payload = torch.load(sac_path, weights_only=False)
            algorithm.load_state_dict(sac_payload, device=residual_policy.config.device)
            logger.info("Loaded SAC checkpoint from %s (resume online RL)", sac_path)
        elif calql_path.exists():
            calql_payload = torch.load(calql_path, weights_only=False)
            if "policy" in calql_payload:
                algorithm.load_weights(calql_payload, device=residual_policy.config.device)
            else:
                algorithm.load_state_dict(calql_payload, device=residual_policy.config.device)
            logger.info("Loaded Cal-QL critic weights from %s", calql_path)

        rollout_cfg = self.cfg.to_rollout_config()
        total_online_env_steps = len(online_buffer)
        target_warmup_env_steps = self._effective_warmup_env_steps()
        pld = self.cfg.pld

        progress_path = self.output_dir / "rl_train_progress.json"
        train_steps = 0
        collect_round = 0
        if progress_path.exists() and (pld.resume_online_buffer or len(online_buffer) > 0):
            try:
                progress = json.loads(progress_path.read_text())
                train_steps = int(progress.get("train_steps", 0))
                collect_round = int(progress.get("collect_round", 0))
                logger.info(
                    "Resuming RL progress from %s: train_steps=%d collect_round=%d",
                    progress_path,
                    train_steps,
                    collect_round,
                )
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
                logger.warning("Could not load %s: %s", progress_path, e)
        elif pld.resume_online_buffer or sac_path.exists() or len(online_buffer) > 0:
            logger.info(
                "No %s yet — gradient step counter starts at 0. "
                "This file is written automatically after each SAC round completes. "
                "Weights/buffer still resume from residual_sac.pt and online_buffer.pt.",
                progress_path.name,
            )

        rollout_cfg.strategy = PLDCollectStrategyConfig(
            mode="online",
            warmup_env_steps=max(0, target_warmup_env_steps - total_online_env_steps),
            max_env_steps=pld.collect_env_steps_per_iter,
            max_episode_steps=pld.max_episode_steps,
            episode_reset_time_s=pld.episode_reset_time_s,
            manual_scene_reset_pause_key=pld.manual_scene_reset_pause_key,
        )
        ctx = build_rollout_context(rollout_cfg, self.shutdown_event)
        strategy = create_strategy(rollout_cfg.strategy)
        assert isinstance(strategy, PLDCollectStrategy)
        strategy.set_pld_runtime(
            online_buffer=online_buffer,
            residual_policy=residual_policy,
            residual_preprocessor=residual_preprocessor,
            reward_detector=reward_detector,
            state_keys=state_keys,
        )

        remaining_warmup_at_start = max(0, target_warmup_env_steps - total_online_env_steps)
        sac_rounds_planned = math.ceil(max(0, pld.rl_steps - train_steps) / max(1, pld.train_iters_per_collect))
        logger.info(
            "=== RL train plan === %d gradient steps | %d SAC updates/collect round | "
            "~%d collect rounds | ≥%d env steps/round | %.0fs manual reset after homing "
            "(pause key='%s')",
            pld.rl_steps,
            pld.train_iters_per_collect,
            sac_rounds_planned,
            pld.collect_env_steps_per_iter,
            pld.episode_reset_time_s,
            pld.manual_scene_reset_pause_key,
        )
        logger.info(
            "=== Online RL warmup plan === %d env steps with base policy only (π_b), then π_b + π_δ "
            "(online buffer=%d, target warmup=%d)",
            remaining_warmup_at_start,
            len(online_buffer),
            target_warmup_env_steps,
        )
        if remaining_warmup_at_start == 0 and target_warmup_env_steps > 0:
            logger.info(
                "=== Online warmup skipped === buffer already has ≥%d steps — residual enabled immediately",
                target_warmup_env_steps,
            )
        try:
            strategy.setup(ctx)
            while train_steps < pld.rl_steps and not self.shutdown_event.is_set():
                collect_round += 1
                remaining_warmup = max(0, target_warmup_env_steps - total_online_env_steps)
                strategy.begin_online_collect_round(
                    collect_round=collect_round,
                    warmup_env_steps=remaining_warmup,
                    max_env_steps=pld.collect_env_steps_per_iter,
                    max_episode_steps=pld.max_episode_steps,
                    episode_reset_time_s=pld.episode_reset_time_s,
                    finish_episode_before_round_stop=pld.finish_episode_before_round_stop,
                    manual_scene_reset_pause_key=pld.manual_scene_reset_pause_key,
                )
                strategy.run(ctx)
                round_steps = strategy._env_steps
                total_online_env_steps += round_steps
                reward_summary = strategy.online_round_reward_summary()
                logger.info(
                    "=== Online collect round %d DONE === %d env steps this round | "
                    "cumulative %d steps | buffer %d | episodes=%s successes=%s "
                    "mean_max_reward=%.3f last_episode_reward=%.3f",
                    collect_round,
                    round_steps,
                    total_online_env_steps,
                    len(online_buffer),
                    reward_summary.get("episodes", 0),
                    reward_summary.get("successes", 0),
                    reward_summary.get("mean_max_reward", 0.0),
                    reward_summary.get("last_episode_reward", 0.0),
                )

                logger.info(
                    "=== SAC update round %d START === inference paused — robot will not move during GPU training",
                    collect_round,
                )
                if strategy._engine is not None:
                    strategy._engine.pause()

                round_train_steps = 0
                last_stats = None
                for _ in range(pld.train_iters_per_collect):
                    if len(online_buffer) < pld.online_step_before_learning:
                        logger.info(
                            "=== SAC round %d SKIP === online buffer %d/%d steps before learning",
                            collect_round,
                            len(online_buffer),
                            pld.online_step_before_learning,
                        )
                        break
                    last_stats = trainer.training_step()
                    train_steps += 1
                    round_train_steps += 1

                if last_stats is not None:
                    logger.info(
                        "=== SAC round %d SUMMARY === gradient_steps=%d (total %d/%d) | "
                        "losses=%s | grad_norms=%s | collect_reward: episodes=%s successes=%s "
                        "mean_max_reward=%.3f last_episode_reward=%.3f",
                        collect_round,
                        round_train_steps,
                        train_steps,
                        pld.rl_steps,
                        last_stats.losses,
                        last_stats.grad_norms,
                        reward_summary.get("episodes", 0),
                        reward_summary.get("successes", 0),
                        reward_summary.get("mean_max_reward", 0.0),
                        reward_summary.get("last_episode_reward", 0.0),
                    )
                else:
                    logger.info(
                        "=== SAC round %d SUMMARY === no gradient steps (buffer or early stop)",
                        collect_round,
                    )

                online_buffer.save(str(self.output_dir / "online_buffer.pt"))
                offline_buffer.save(str(self.output_dir / "offline_buffer.pt"))
                torch.save(algorithm.state_dict(), self.output_dir / "residual_sac.pt")
                torch.save(algorithm.get_weights(), self.output_dir / "residual_policy_weights.pt")
                progress_path.write_text(
                    json.dumps(
                        {
                            "train_steps": train_steps,
                            "collect_round": collect_round,
                            "total_online_env_steps": total_online_env_steps,
                            "online_buffer_size": len(online_buffer),
                        },
                        indent=2,
                    )
                )

                if self.shutdown_event.is_set():
                    logger.warning("RL train stopping: shutdown_event set (Ctrl+C or inference fatal error)")
                    break
                if train_steps >= pld.rl_steps:
                    break
                logger.info("=== Online collect round %d RESUME === resuming inference", collect_round + 1)
                if strategy._engine is not None:
                    strategy._engine.resume()
        finally:
            strategy.teardown(ctx)

        if self.shutdown_event.is_set() and train_steps < pld.rl_steps:
            logger.warning(
                "RL training ended early at %d/%d gradient steps (shutdown_event set)",
                train_steps,
                pld.rl_steps,
            )
        else:
            logger.info("RL training finished (%d steps)", train_steps)

    def run(self) -> None:
        residual_policy, residual_preprocessor, state_keys = self._init_residual_policy()
        offline_buffer, online_buffer = self._make_buffers(state_keys)
        reward_detector = RewardClassifierDetector(self.cfg.reward_classifier)
        logger.info("Debug: reward classifier detector ready (path=%s)", self.cfg.reward_classifier.path)

        try:
            if not self.cfg.skip_offline_collect:
                logger.info("=== PLD Stage 1: offline collection ===")
                logger.info("Debug: entering collect_offline → build_rollout_context (loads base policy again)")
                self.collect_offline(
                    offline_buffer,
                    reward_detector,
                    state_keys,
                    residual_preprocessor,
                    residual_policy,
                )
            elif self.cfg.pld.resume_offline_buffer:
                offline_buffer = ReplayBuffer.load(
                    self.cfg.pld.resume_offline_buffer, device=self.cfg.device or "cuda"
                )

            if not self.cfg.skip_calql_pretrain:
                logger.info("=== PLD Stage 1: Cal-QL critic pretrain ===")
                self.pretrain_critic(offline_buffer, residual_policy, state_keys)

            if not self.cfg.skip_rl_train:
                logger.info("=== PLD Stage 1: online SAC training ===")
                self.rl_train(
                    offline_buffer,
                    online_buffer,
                    residual_policy,
                    reward_detector,
                    state_keys,
                    residual_preprocessor,
                )
        finally:
            reward_detector.stop()
