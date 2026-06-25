# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.residual_gaussian.configuration_residual_gaussian import ResidualGaussianActorConfig
from lerobot.policies.residual_gaussian.modeling_residual_gaussian import (
    BASE_ACTION_KEY,
    ResidualGaussianActorPolicy,
)
from lerobot.rl.algorithms.residual_sac.configuration_residual_sac import ResidualSACAlgorithmConfig
from lerobot.rl.algorithms.residual_sac.residual_sac_algorithm import ResidualSACAlgorithm
from lerobot.rl.buffer import ReplayBuffer
from lerobot.utils.constants import ACTION, OBS_STATE


def _make_policy(
    action_dim: int = 4,
    has_image: bool = False,
    state_dim: int | None = None,
) -> ResidualGaussianActorPolicy:
    if state_dim is None:
        state_dim = action_dim
    input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))}
    if has_image:
        input_features["observation.images.top"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 64, 64)
        )
    cfg = ResidualGaussianActorConfig(
        input_features=input_features,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        device="cpu",
        xi=0.1,
        vision_encoder_name=None,
        freeze_vision_encoder=False,
        shared_encoder=True,
    )
    return ResidualGaussianActorPolicy(config=cfg)


def test_residual_policy_forward_shape():
    policy = _make_policy(action_dim=4)
    obs = {OBS_STATE: torch.randn(2, 4)}
    base = torch.randn(2, 4)
    out = policy.forward({"state": obs, BASE_ACTION_KEY: base})
    assert out["action"].shape == (2, 4)
    assert out["delta"].shape == (2, 4)


def test_residual_policy_composite_action():
    policy = _make_policy(action_dim=3)
    base = torch.tensor([[1.0, 2.0, 3.0]])
    delta = torch.tensor([[0.1, -0.1, 0.0]])
    composite = policy.composite_action(base, delta)
    assert torch.allclose(composite, base + delta)


def test_replay_buffer_base_action_roundtrip():
    state = {OBS_STATE: torch.randn(1, 4)}
    next_state = {OBS_STATE: torch.randn(1, 4)}
    action = torch.randn(1, 4)
    base = torch.randn(1, 4)
    buffer = ReplayBuffer(capacity=10, device="cpu", state_keys=[OBS_STATE])
    buffer.add(
        state=state,
        action=action,
        reward=1.0,
        next_state=next_state,
        done=True,
        truncated=False,
        complementary_info={BASE_ACTION_KEY: base},
    )
    batch = buffer.sample(1)
    assert BASE_ACTION_KEY in batch["complementary_info"]
    assert batch["complementary_info"][BASE_ACTION_KEY].shape == (1, 4)


def test_replay_buffer_save_load():
    state = {OBS_STATE: torch.randn(1, 3)}
    buffer = ReplayBuffer(capacity=5, device="cpu", state_keys=[OBS_STATE])
    buffer.add(
        state=state,
        action=torch.randn(1, 3),
        reward=0.0,
        next_state=state,
        done=False,
        truncated=False,
        complementary_info={BASE_ACTION_KEY: torch.randn(1, 3)},
    )
    path = "/tmp/test_pld_buffer.pt"
    buffer.save(path)
    loaded = ReplayBuffer.load(path, device="cpu")
    assert len(loaded) == 1
    batch = loaded.sample(1)
    assert batch[ACTION].shape == (1, 3)


def test_calql_critic_step():
    policy = _make_policy(action_dim=4)
    algo_cfg = ResidualSACAlgorithmConfig.from_policy_config(policy.config)
    algo = ResidualSACAlgorithm(policy=policy, config=algo_cfg)
    algo.make_optimizers_and_scheduler()

    state = {OBS_STATE: torch.randn(8, 4)}
    next_state = {OBS_STATE: torch.randn(8, 4)}
    batch = {
        "state": state,
        "next_state": next_state,
        ACTION: torch.randn(8, 4),
        "reward": torch.zeros(8),
        "done": torch.zeros(8),
        "truncated": torch.zeros(8),
        "complementary_info": {BASE_ACTION_KEY: torch.randn(8, 4)},
    }
    algo.config.use_calql = True
    stats = algo.update_critic_only(batch)
    assert "loss_critic" in stats.losses


def test_calql_state_dict_roundtrip_for_online_resume():
    """calql_critic.pt uses algorithm.state_dict(); online SAC must load via load_state_dict."""
    policy = _make_policy(action_dim=7, state_dim=6)
    algo_cfg = ResidualSACAlgorithmConfig.from_policy_config(policy.config)
    algo = ResidualSACAlgorithm(policy=policy, config=algo_cfg)
    algo.make_optimizers_and_scheduler()

    before = algo.critic_ensemble.critics[0].output_layer.weight.clone()
    payload = algo.state_dict()

    algo_dst = ResidualSACAlgorithm(
        policy=_make_policy(action_dim=7, state_dim=6),
        config=algo_cfg,
    )
    assert "policy" not in payload
    algo_dst.load_state_dict(payload, device="cpu")
    after = algo_dst.critic_ensemble.critics[0].output_layer.weight
    assert torch.allclose(before, after.cpu())


def test_calql_critic_step_with_continuous_gripper():
    """CRP-style 6 joints + continuous gripper: critic uses full 7-dim composite actions (PLD §3.1)."""
    action_dim = 7
    state_dim = 6
    policy = _make_policy(action_dim=action_dim, state_dim=state_dim)
    algo_cfg = ResidualSACAlgorithmConfig.from_policy_config(policy.config)
    algo = ResidualSACAlgorithm(policy=policy, config=algo_cfg)
    algo.make_optimizers_and_scheduler()

    assert policy.config.critic_action_dim == action_dim
    first_linear = algo.critic_ensemble.critics[0].net.net[0]
    assert isinstance(first_linear, torch.nn.Linear)
    assert first_linear.in_features == policy.config.critic_action_dim + policy.encoder_critic.output_dim

    state = {OBS_STATE: torch.randn(4, state_dim)}
    batch = {
        "state": state,
        "next_state": state,
        ACTION: torch.randn(4, action_dim),
        "reward": torch.zeros(4),
        "done": torch.zeros(4),
        "truncated": torch.zeros(4),
        "complementary_info": {BASE_ACTION_KEY: torch.randn(4, action_dim)},
    }
    algo.config.use_calql = True
    stats = algo.update_critic_only(batch)
    assert "loss_critic" in stats.losses


def test_residual_gripper_delta_in_composite_action():
    """PLD: ā = a_b + a_δ includes the gripper dimension."""
    policy = _make_policy(action_dim=7, state_dim=6)
    base = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 500.0]])
    delta = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01]])
    composite = policy.composite_action(base, delta)
    assert composite[0, -1].item() == pytest.approx(500.01)
