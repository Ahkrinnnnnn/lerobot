# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.pld.obs_utils import camera_shape_to_policy_shape
from lerobot.policies import make_pre_post_processors
from lerobot.policies.residual_gaussian.configuration_residual_gaussian import ResidualGaussianActorConfig
from lerobot.policies.residual_gaussian.modeling_residual_gaussian import ResidualGaussianActorPolicy
from lerobot.robots import make_robot_from_config
from lerobot.utils.constants import ACTION, OBS_STATE

if TYPE_CHECKING:
    from lerobot.robots.config import RobotConfig

logger = logging.getLogger(__name__)


def load_normalizer_stats_from_pretrained(pretrained_path: str | Path) -> dict[str, dict[str, Any]] | None:
    """Load normalizer statistics bundled in a base policy ``pretrained_model/`` folder."""
    root = Path(pretrained_path)
    if not root.is_dir():
        return None
    candidates = sorted(root.glob("*normalizer_processor.safetensors"))
    if not candidates:
        return None

    from safetensors import safe_open

    stats: dict[str, dict[str, Any]] = {}
    with safe_open(str(candidates[0]), framework="pt") as handle:
        for flat_key in handle.keys():
            if flat_key.endswith(".count"):
                continue
            feature_key, stat_name = flat_key.rsplit(".", 1)
            stats.setdefault(feature_key, {})[stat_name] = handle.get_tensor(flat_key)
    return stats or None


def state_keys_from_config(residual_cfg: ResidualGaussianActorConfig) -> list[str]:
    keys = []
    for key, ft in residual_cfg.input_features.items():
        if ft.type in (FeatureType.VISUAL, FeatureType.STATE):
            keys.append(key)
    return keys or [OBS_STATE]


def build_residual_policy_features(
    robot,
    policy_cfg: ResidualGaussianActorConfig,
) -> None:
    """Infer input/output features from robot config (no hardware connect required)."""
    joint_keys = sorted(
        k for k in robot.observation_features if k.endswith(".pos") and not k.startswith("gripper")
    )
    action_keys = sorted(k for k in robot.action_features if k.endswith(".pos"))
    if getattr(robot, "config", None) and getattr(robot.config, "use_gripper_feature", False):
        if "gripper.pos" in robot.action_features and "gripper.pos" not in action_keys:
            action_keys.append("gripper.pos")

    resize_size = None
    if policy_cfg.image_preprocessing is not None:
        resize_size = policy_cfg.image_preprocessing.resize_size

    input_features: dict[str, PolicyFeature] = {}
    for cam_name, shape in robot.observation_features.items():
        if isinstance(shape, tuple) and len(shape) == 3:
            visual_shape = camera_shape_to_policy_shape(shape, resize_size=resize_size)
            input_features[f"observation.images.{cam_name}"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=visual_shape
            )
    if joint_keys:
        input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(len(joint_keys),))

    policy_cfg.input_features = input_features
    policy_cfg.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(len(action_keys),))
    }


def load_residual_actor_weights(
    policy: ResidualGaussianActorPolicy,
    weights_path: str | Path,
    device: str = "cpu",
) -> ResidualGaussianActorPolicy:
    """Load Stage 1 actor weights for frozen Stage 2 rollout."""
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"Residual policy weights not found: {path}")

    if path.is_dir():
        loaded = ResidualGaussianActorPolicy.from_pretrained(str(path), config=policy.config)
        logger.info("Loaded residual policy from pretrained directory: %s", path)
        return loaded

    weights = torch.load(path, map_location=device, weights_only=False)
    if isinstance(weights, dict) and "policy" in weights:
        policy.actor.load_state_dict(weights["policy"], strict=False)
        logger.info("Loaded residual actor from get_weights() checkpoint: %s", path)
        return policy

    if isinstance(weights, dict) and any(k.startswith("critic_ensemble.") for k in weights):
        raise ValueError(
            f"{path} contains algorithm critic tensors only. "
            "Run Stage 1 RL training to produce residual_policy_weights.pt, or pass "
            "--pld.resume_residual_weights=... with actor weights."
        )

    policy.actor.load_state_dict(weights, strict=False)
    logger.info("Loaded residual actor state dict from %s", path)
    return policy


def init_residual_policy(
    *,
    robot_cfg: RobotConfig,
    policy_cfg: ResidualGaussianActorConfig,
    device: str,
    xi: float,
    base_policy=None,
    weights_path: str | Path | None = None,
    eval_mode: bool = False,
) -> tuple[ResidualGaussianActorPolicy, Any, list[str]]:
    """Instantiate residual policy + preprocessor from robot **config** only.

    Does not call ``robot.connect()``. CRP ``CrpRobotPy`` is effectively process-global;
    an early connect/disconnect here breaks the later ``build_rollout_context`` session.
    """
    robot = make_robot_from_config(robot_cfg)
    policy_cfg.device = device
    policy_cfg.xi = xi
    build_residual_policy_features(robot, policy_cfg)

    stats = None
    if base_policy is not None:
        stats = getattr(base_policy, "dataset_stats", None)
        pretrained_path = getattr(base_policy, "pretrained_path", None)
        if stats is None and pretrained_path:
            stats = load_normalizer_stats_from_pretrained(pretrained_path)
            if stats:
                logger.info("Loaded residual normalization stats from base policy: %s", pretrained_path)
    if stats:
        policy_cfg.dataset_stats = stats
    else:
        # GaussianActorConfig ships placeholder stats (OBS_STATE length 2) — disable normalization.
        policy_cfg.dataset_stats = None

    if policy_cfg.pretrained_path:
        policy = ResidualGaussianActorPolicy.from_pretrained(
            policy_cfg.pretrained_path, config=policy_cfg
        )
    else:
        policy = ResidualGaussianActorPolicy(config=policy_cfg)

    if weights_path is not None:
        policy = load_residual_actor_weights(policy, weights_path, device=device)

    policy.to(device)
    if eval_mode:
        policy.eval()
    preprocessor, _ = make_pre_post_processors(policy_cfg, dataset_stats=policy_cfg.dataset_stats)
    return policy, preprocessor, state_keys_from_config(policy_cfg)


def init_residual_policy_from_robot(
    *,
    robot_cfg: RobotConfig,
    policy_cfg: ResidualGaussianActorConfig,
    device: str,
    xi: float,
    base_policy=None,
    weights_path: str | Path | None = None,
) -> tuple[ResidualGaussianActorPolicy, Any, list[str]]:
    """Alias for :func:`init_residual_policy` (kept for backward compatibility)."""
    return init_residual_policy(
        robot_cfg=robot_cfg,
        policy_cfg=policy_cfg,
        device=device,
        xi=xi,
        base_policy=base_policy,
        weights_path=weights_path,
        eval_mode=True,
    )


def resolve_stage1_residual_weights(
    stage1_output_dir: str | None,
    resume_residual_weights: str | None,
) -> str | None:
    if resume_residual_weights:
        return resume_residual_weights
    if not stage1_output_dir:
        return None
    candidate = Path(stage1_output_dir) / "residual_policy_weights.pt"
    if candidate.exists():
        return str(candidate)
    return None
