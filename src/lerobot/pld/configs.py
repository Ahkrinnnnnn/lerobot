# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field

import draccus

from lerobot.configs import parser
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.residual_gaussian.configuration_residual_gaussian import ResidualGaussianActorConfig
from lerobot.robots.config import RobotConfig
from lerobot.rollout.configs import RolloutConfig
from lerobot.rollout.inference import InferenceEngineConfig, RTCInferenceConfig


from lerobot.rewards.classifier.pipeline_config import RewardClassifierRuntimeConfig as RewardClassifierConfig


@dataclass
class PLDTrainingConfig:
    """Hyper-parameters for PLD Stage 1 RL (defaults aligned with paper Table 5 / Appendix B.2)."""

    n_offline_success_trials: int = 20
    calql_steps: int = 5000
    calql_alpha: float = 1.0
    rl_steps: int = 10000
    # Paper Table 5: 100 episodes of base-only online rollouts before enabling residual.
    # If set, overrides warmup_env_steps via warmup_episodes * avg_episode_steps.
    warmup_episodes: int | None = None
    avg_episode_steps: int | None = None
    warmup_env_steps: int = 500
    online_buffer_capacity: int = 250_000
    offline_buffer_capacity: int = 250_000
    buffer_optimize_memory: bool = True
    batch_size: int = 256
    online_ratio: float = 0.5
    online_step_before_learning: int = 100
    max_episode_steps: int = 500
    xi: float = 0.1
    # Full manual scene-reset window after homing (homing time is not subtracted).
    episode_reset_time_s: float = 15.0
    manual_scene_reset_pause_key: str = "space"
    collect_env_steps_per_iter: int = 50
    train_iters_per_collect: int = 1
    # After this many env steps, finish the current episode then run SAC (avoids mid-episode cut).
    finish_episode_before_round_stop: bool = True
    resume_offline_buffer: str | None = None
    resume_online_buffer: str | None = None

    # SAC / Cal-QL (Table 5)
    discount: float = 0.99
    grad_clip_norm: float = 1.0
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temperature_lr: float = 3e-4
    temperature_init: float = 1.0
    critic_target_update_weight: float = 0.005
    num_critics: int = 2
    utd_ratio: int = 2
    policy_update_freq: int = 1
    use_adamw: bool = True
    optimizer_weight_decay: float = 0.01


@dataclass
class PLDStage1Config:
    """Top-level config for ``lerobot-pld-stage1``."""

    robot: RobotConfig
    # Loaded from --base_policy.path via __post_init__ (same pattern as RolloutConfig.policy).
    base_policy: PreTrainedConfig | None = None
    # Use PreTrainedConfig so draccus accepts ``type`` in JSON/CLI (e.g. residual_gaussian).
    residual_policy: PreTrainedConfig = field(default_factory=ResidualGaussianActorConfig)
    reward_classifier: RewardClassifierConfig = field(default_factory=RewardClassifierConfig)
    pld: PLDTrainingConfig = field(default_factory=PLDTrainingConfig)

    # Rollout / deploy settings
    inference: InferenceEngineConfig = field(default_factory=RTCInferenceConfig)
    fps: float = 30.0
    task: str = ""
    device: str | None = None
    rename_map: dict[str, str] = field(default_factory=dict)
    interpolation_multiplier: int = 1
    use_torch_compile: bool = False
    multiprocess_sync_inference: bool = True
    return_to_initial_position: bool = True
    display_data: bool = False

    # Pipeline stage control
    skip_offline_collect: bool = False
    skip_calql_pretrain: bool = False
    skip_rl_train: bool = False

    output_dir: str = "outputs/pld_stage1"
    job_name: str = "pld_stage1"

    def __post_init__(self):
        loaded = parser.load_pretrained_config_from_path_field("base_policy")
        if loaded is not None:
            self.base_policy = loaded
        if self.base_policy is None:
            raise ValueError("--base_policy.path is required for PLD Stage 1")

        residual_path = parser.get_path_arg("residual_policy")
        if residual_path:
            yaml_overrides = parser.get_yaml_overrides("residual_policy")
            cli_overrides = parser.get_cli_overrides("residual_policy") or []
            self.residual_policy = ResidualGaussianActorConfig.from_pretrained(
                residual_path, cli_overrides=yaml_overrides + cli_overrides
            )
            self.residual_policy.pretrained_path = residual_path
        if not isinstance(self.residual_policy, ResidualGaussianActorConfig):
            raise ValueError(
                f"residual_policy must be residual_gaussian, got {self.residual_policy.type!r}"
            )

        self.residual_policy.xi = self.pld.xi
        if self.device:
            self.base_policy.device = self.device
            self.residual_policy.device = self.device
            self.reward_classifier.device = self.device or self.reward_classifier.device

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["base_policy", "residual_policy"]

    def to_rollout_config(self) -> RolloutConfig:
        """Build a :class:`RolloutConfig` for base-policy rollout collection."""
        from lerobot.rollout.configs import PLDCollectStrategyConfig

        cfg = RolloutConfig(
            robot=self.robot,
            policy=self.base_policy,
            strategy=PLDCollectStrategyConfig(
                mode="offline",
                n_successful_trials=self.pld.n_offline_success_trials,
                max_episode_steps=self.pld.max_episode_steps,
                episode_reset_time_s=self.pld.episode_reset_time_s,
                manual_scene_reset_pause_key=self.pld.manual_scene_reset_pause_key,
            ),
            inference=self.inference,
            fps=self.fps,
            task=self.task,
            device=self.device,
            rename_map=self.rename_map,
            interpolation_multiplier=self.interpolation_multiplier,
            use_torch_compile=self.use_torch_compile,
            multiprocess_sync_inference=self.multiprocess_sync_inference,
            return_to_initial_position=self.return_to_initial_position,
            display_data=self.display_data,
        )
        return cfg


@dataclass
class PLDStage2DataConfig:
    """Hyper-parameters for PLD Stage 2 hybrid SFT data collection."""

    stage1_output_dir: str | None = None
    resume_residual_weights: str | None = None
    n_successful_episodes: int = 200
    probing_alpha: float = 0.6
    max_episode_steps: int = 1000
    xi: float = 0.05
    discard_failed_episodes: bool = True
    seed: int | None = None
    episode_reset_time_s: float = 15.0
    manual_scene_reset_pause_key: str = "space"


@dataclass
class PLDStage2Config:
    """Top-level config for ``lerobot-pld-stage2`` (hybrid rollout → LeRobotDataset)."""

    robot: RobotConfig
    base_policy: PreTrainedConfig | None = None
    residual_policy: PreTrainedConfig = field(default_factory=ResidualGaussianActorConfig)
    reward_classifier: RewardClassifierConfig = field(default_factory=RewardClassifierConfig)
    dataset: DatasetRecordConfig = field(default_factory=DatasetRecordConfig)
    pld: PLDStage2DataConfig = field(default_factory=PLDStage2DataConfig)

    inference: InferenceEngineConfig = field(default_factory=RTCInferenceConfig)
    fps: float = 30.0
    task: str = ""
    device: str | None = None
    rename_map: dict[str, str] = field(default_factory=dict)
    interpolation_multiplier: int = 1
    use_torch_compile: bool = False
    multiprocess_sync_inference: bool = True
    return_to_initial_position: bool = True
    display_data: bool = False

    skip_collection: bool = False
    output_dir: str = "outputs/pld_stage2"
    job_name: str = "pld_stage2"

    def __post_init__(self):
        loaded = parser.load_pretrained_config_from_path_field("base_policy")
        if loaded is not None:
            self.base_policy = loaded
        if self.base_policy is None:
            raise ValueError("--base_policy.path is required for PLD Stage 2")

        residual_path = parser.get_path_arg("residual_policy")
        if residual_path:
            yaml_overrides = parser.get_yaml_overrides("residual_policy")
            cli_overrides = parser.get_cli_overrides("residual_policy") or []
            self.residual_policy = ResidualGaussianActorConfig.from_pretrained(
                residual_path, cli_overrides=yaml_overrides + cli_overrides
            )
            self.residual_policy.pretrained_path = residual_path
        if not isinstance(self.residual_policy, ResidualGaussianActorConfig):
            raise ValueError(
                f"residual_policy must be residual_gaussian, got {self.residual_policy.type!r}"
            )

        if not self.dataset.repo_id:
            raise ValueError("--dataset.repo_id is required for PLD Stage 2")

        self.residual_policy.xi = self.pld.xi
        if self.device:
            self.base_policy.device = self.device
            self.residual_policy.device = self.device
            self.reward_classifier.device = self.device or self.reward_classifier.device

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["base_policy", "residual_policy"]

    def to_rollout_config(self) -> RolloutConfig:
        from lerobot.pld.residual_setup import resolve_stage1_residual_weights
        from lerobot.rollout.configs import PLDHybridCollectStrategyConfig

        if not self.pld.resume_residual_weights:
            resolved = resolve_stage1_residual_weights(
                self.pld.stage1_output_dir, self.pld.resume_residual_weights
            )
            if resolved:
                self.pld.resume_residual_weights = resolved

        cfg = RolloutConfig(
            robot=self.robot,
            policy=self.base_policy,
            dataset=self.dataset,
            strategy=PLDHybridCollectStrategyConfig(
                n_successful_episodes=self.pld.n_successful_episodes,
                probing_alpha=self.pld.probing_alpha,
                max_episode_steps=self.pld.max_episode_steps,
                discard_failed_episodes=self.pld.discard_failed_episodes,
                seed=self.pld.seed,
                episode_reset_time_s=self.pld.episode_reset_time_s,
                manual_scene_reset_pause_key=self.pld.manual_scene_reset_pause_key,
            ),
            inference=self.inference,
            fps=self.fps,
            task=self.task,
            device=self.device,
            rename_map=self.rename_map,
            interpolation_multiplier=self.interpolation_multiplier,
            use_torch_compile=self.use_torch_compile,
            multiprocess_sync_inference=self.multiprocess_sync_inference,
            return_to_initial_position=self.return_to_initial_position,
            display_data=self.display_data,
        )
        return cfg
