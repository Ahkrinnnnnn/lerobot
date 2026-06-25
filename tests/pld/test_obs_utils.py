# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from unittest.mock import MagicMock

import numpy as np
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.pld.obs_utils import (
    align_rl_state_shapes,
    apply_residual_preprocessor,
    build_rl_state_from_obs,
    camera_shape_to_policy_shape,
    filter_rl_state_tensors,
    prepare_rl_state_for_buffer,
)
from lerobot.policies.residual_gaussian.configuration_residual_gaussian import ResidualGaussianActorConfig
from lerobot.policies.residual_gaussian.modeling_residual_gaussian import ResidualGaussianActorPolicy
from lerobot.utils.constants import ACTION, OBS_STATE


def test_camera_shape_to_policy_shape_hwc():
    assert camera_shape_to_policy_shape((480, 640, 3)) == (3, 480, 640)
    assert camera_shape_to_policy_shape((480, 640, 3), resize_size=(128, 128)) == (3, 128, 128)


def test_residual_policy_resnet10_init_with_resized_images():
    shape = (3, 128, 128)
    cfg = ResidualGaussianActorConfig(
        vision_encoder_name="lerobot/resnet10",
        freeze_vision_encoder=True,
        shared_encoder=True,
        device="cpu",
        input_features={
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=shape),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=shape),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    cfg.validate_features()
    policy = ResidualGaussianActorPolicy(config=cfg)
    assert policy.config.input_features["observation.images.top"].shape == shape


def test_build_rl_state_excludes_gripper_from_observation_state():
    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    }
    obs = {
        "j1.pos": 1.0,
        "j2.pos": 2.0,
        "j3.pos": 3.0,
        "j4.pos": 4.0,
        "j5.pos": 5.0,
        "j6.pos": 6.0,
        "gripper.pos": 1000.0,
    }
    state = build_rl_state_from_obs(obs, input_features, device="cpu")
    assert state[OBS_STATE].shape == (1, 6)
    assert state[OBS_STATE].tolist() == [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]


def test_apply_residual_preprocessor_strips_non_observation_keys():
    rl_state = {OBS_STATE: torch.zeros(1, 6)}
    preprocessor = MagicMock(
        return_value={
            OBS_STATE: torch.ones(1, 6),
            ACTION: None,
            "reward": 0.0,
        }
    )
    out = apply_residual_preprocessor(rl_state, preprocessor)
    assert ACTION not in out
    assert out[OBS_STATE].shape == (1, 6)


def test_align_rl_state_shapes_downscales_full_res():
    input_features = {
        "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
    }
    state = {"observation.images.top": torch.zeros(1, 3, 480, 640)}
    out = align_rl_state_shapes(state, input_features)
    assert out["observation.images.top"].shape == (1, 3, 128, 128)


def test_prepare_rl_state_for_buffer_resizes_without_preprocessor():
    input_features = {
        "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    }
    obs = {
        "top": np.zeros((480, 640, 3), dtype=np.uint8),
        **{f"j{i}.pos": float(i) for i in range(1, 7)},
    }
    state = prepare_rl_state_for_buffer(obs, input_features, preprocessor=None, device="cpu")
    assert state["observation.images.top"].shape == (1, 3, 128, 128)
    assert state[OBS_STATE].shape == (1, 6)


def test_filter_rl_state_tensors_requires_all_features():
    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
    }
    state = {OBS_STATE: torch.zeros(1, 6)}
    try:
        filter_rl_state_tensors(state, input_features)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "observation.images.top" in str(exc)
