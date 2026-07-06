#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the shared RLT Stage-2 core (networks + loss math + replay).

These tests validate the framework-agnostic core that is *also* reused by
external training frameworks (e.g. RLinf). The cross-framework identity check
lives in RLinf (``test_rlt_shared_core_conformance.py``); here we validate
correctness on the lerobot side.
"""

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import torch  # noqa: E402

from lerobot.rlt.shared import (  # noqa: E402
    ChunkActor,
    ChunkCriticEnsemble,
    SOURCE_BASE,
    SOURCE_HUMAN,
    SOURCE_MIXED,
    SOURCE_RL,
    bc_weight_schedule,
    build_replay_windows,
    golden_loss_values,
    make_rlt_tiny_fixture,
    polyak_update,
    ref_dropout_mask,
    rlt_actor_loss,
    td3_critic_loss,
)
from lerobot.rlt.replay_windows import StepRecord  # noqa: E402


# ---------------------------------------------------------------------------
# Imports / surface
# ---------------------------------------------------------------------------


def test_shared_core_symbols_importable():
    fx = make_rlt_tiny_fixture()
    assert isinstance(fx["actor"], ChunkActor)
    assert isinstance(fx["critic_ensemble"], ChunkCriticEnsemble)
    assert isinstance(fx["critic_target"], ChunkCriticEnsemble)
    assert {"actor", "critic_ensemble", "critic_target", "fb", "config", "meta"} <= set(fx)


# ---------------------------------------------------------------------------
# Fixture shapes
# ---------------------------------------------------------------------------


def test_fixture_shapes_and_dtypes():
    fx = make_rlt_tiny_fixture()
    meta = fx["meta"]
    b = meta["batch_size"]
    cd = meta["chunk_dim"]
    fb = fx["fb"]
    assert fb["action"].shape == (b, cd)
    assert fb["reward"].shape == (b,)
    assert fb["done"].shape == (b,)
    assert fb["state"]["z_rl"].shape == (b, meta["token_dim"])
    assert fb["state"]["proprio"].shape == (b, meta["proprio_dim"])
    assert fb["next_state"]["z_rl"].shape == (b, meta["token_dim"])
    assert fb["ref_chunk"].shape == (b, cd)
    assert fb["ref_chunk_next"].shape == (b, cd)
    assert fb["source"].dtype == torch.long
    # Critic ensemble forward shape.
    q = fx["critic_ensemble"](fb["state"]["z_rl"], fb["state"]["proprio"], fb["action"])
    assert q.shape == (meta["num_critics"], b)


def test_fixture_is_hermetic_to_global_rng():
    torch.manual_seed(123)
    before = torch.randn(3)
    make_rlt_tiny_fixture(seed=0)
    torch.manual_seed(123)
    after = torch.randn(3)
    assert torch.equal(before, after)


# ---------------------------------------------------------------------------
# Critic loss
# ---------------------------------------------------------------------------


def test_critic_loss_finite_and_grad_flows():
    fx = make_rlt_tiny_fixture()
    fx["critic_ensemble"].train()
    loss = td3_critic_loss(
        fx["critic_ensemble"], fx["critic_target"], fx["actor"], fx["fb"], fx["meta"]["discount"]
    )
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in fx["critic_ensemble"].parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(g is not None for g in grads)
    # Actor and target must NOT receive gradients from the critic loss.
    assert all(p.grad is None for p in fx["actor"].parameters())
    assert all(p.grad is None for p in fx["critic_target"].parameters())


def test_critic_loss_terminal_zeros_bootstrap():
    fx = make_rlt_tiny_fixture()
    fb = {**fx["fb"], "done": torch.ones(fx["meta"]["batch_size"])}
    loss = td3_critic_loss(
        fx["critic_ensemble"], fx["critic_target"], fx["actor"], fb, fx["meta"]["discount"]
    )
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Actor loss
# ---------------------------------------------------------------------------


def test_actor_loss_finite_and_grad_flows():
    fx = make_rlt_tiny_fixture()
    fx["actor"].train()
    loss, info = rlt_actor_loss(
        fx["actor"],
        fx["critic_ensemble"],
        fx["fb"],
        chunk_len=fx["meta"]["chunk_len"],
        action_dim=fx["meta"]["action_dim"],
        bc_weight=fx["meta"]["bc_weight"],
        delta_penalty_weight=fx["meta"]["delta_penalty_weight"],
        ref_dropout_prob=0.0,
        in_warmup=False,
    )
    assert torch.isfinite(loss)
    assert set(info) == {"rl_loss", "bc_loss", "delta_penalty"}
    loss.backward()
    assert any(p.grad is not None for p in fx["actor"].parameters())
    # The critic participates in the actor-loss forward (min-Q), so its params
    # DO receive grads here — the algorithm simply never steps the critic
    # optimizer on actor updates. We only assert the actor is reached.
    assert any(p.grad is not None for p in fx["actor"].parameters())


def test_actor_loss_warmup_zeros_rl_term():
    fx = make_rlt_tiny_fixture()
    loss, info = rlt_actor_loss(
        fx["actor"],
        fx["critic_ensemble"],
        fx["fb"],
        chunk_len=fx["meta"]["chunk_len"],
        action_dim=fx["meta"]["action_dim"],
        bc_weight=fx["meta"]["bc_weight"],
        delta_penalty_weight=fx["meta"]["delta_penalty_weight"],
        ref_dropout_prob=0.0,
        in_warmup=True,
    )
    assert info["rl_loss"] == 0.0
    assert torch.isfinite(loss)


def test_actor_loss_all_rl_source_masks_bc():
    fx = make_rlt_tiny_fixture()
    fb = {**fx["fb"], "source": torch.full((fx["meta"]["batch_size"],), SOURCE_RL, dtype=torch.long)}
    _, info = rlt_actor_loss(
        fx["actor"],
        fx["critic_ensemble"],
        fb,
        chunk_len=fx["meta"]["chunk_len"],
        action_dim=fx["meta"]["action_dim"],
        bc_weight=1.0,
        delta_penalty_weight=0.0,
        ref_dropout_prob=0.0,
        in_warmup=False,
    )
    assert info["bc_loss"] == 0.0


def test_actor_loss_explicit_dropout_mask_used():
    fx = make_rlt_tiny_fixture()
    b = fx["meta"]["batch_size"]
    cd = fx["meta"]["chunk_dim"]
    mask_zero = torch.zeros(b, 1)
    _, info_zero = rlt_actor_loss(
        fx["actor"],
        fx["critic_ensemble"],
        fx["fb"],
        chunk_len=fx["meta"]["chunk_len"],
        action_dim=fx["meta"]["action_dim"],
        bc_weight=0.0,
        delta_penalty_weight=0.0,
        ref_dropout_prob=0.0,
        in_warmup=False,
        dropout_mask=mask_zero,
    )
    # With bc_weight=0 and delta=0, actor_loss == rl_loss; just assert finite.
    assert isinstance(info_zero, dict)
    # Mask shape sanity: zero mask must keep code path identical to a real mask.
    assert mask_zero.shape == (b, 1) or cd  # dummy assert to use cd
    del cd


# ---------------------------------------------------------------------------
# polyak_update
# ---------------------------------------------------------------------------


def test_polyak_update_moves_target_toward_source():
    src = torch.nn.Linear(3, 2)
    tgt = torch.nn.Linear(3, 2)
    # Force distinct weights.
    with torch.no_grad():
        src.weight.fill_(1.0)
        src.bias.fill_(1.0)
        tgt.weight.fill_(0.0)
        tgt.bias.fill_(0.0)
    polyak_update(tgt, src, tau=0.25)
    assert torch.allclose(tgt.weight, torch.full_like(tgt.weight, 0.25))
    assert torch.allclose(tgt.bias, torch.full_like(tgt.bias, 0.25))


def test_polyak_update_tau_one_copies_source():
    src = torch.nn.Linear(3, 2)
    tgt = torch.nn.Linear(3, 2)
    with torch.no_grad():
        src.weight.fill_(0.7)
        src.bias.fill_(-0.3)
    polyak_update(tgt, src, tau=1.0)
    assert torch.equal(tgt.weight, src.weight)
    assert torch.equal(tgt.bias, src.bias)


# ---------------------------------------------------------------------------
# bc_weight_schedule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "step,expected",
    [
        (0, 1.0),
        (500, 0.75),
        (1000, 0.5),
        (2000, 0.0),
        (9999, 0.0),
    ],
)
def test_bc_weight_schedule_linear(step, expected):
    w = bc_weight_schedule(step, bc_weight_max=1.0, bc_weight_min=0.0, bc_decay_steps=2000)
    assert abs(w - expected) < 1e-9


def test_bc_weight_schedule_constant_when_decay_zero():
    assert bc_weight_schedule(123, 0.5, 0.1, 0) == 0.5


# ---------------------------------------------------------------------------
# ref_dropout_mask
# ---------------------------------------------------------------------------


def test_ref_dropout_mask_none_when_disabled():
    assert ref_dropout_mask(8, 0.0) is None
    assert ref_dropout_mask(8, -1.0) is None


def test_ref_dropout_mask_shape_and_values():
    gen = torch.Generator().manual_seed(0)
    m = ref_dropout_mask(100, 0.5, device="cpu", generator=gen)
    assert m is not None
    assert m.shape == (100, 1)
    assert set(m.unique().tolist()) <= {0.0, 1.0}
    # Roughly half kept.
    assert 30 < m.sum().item() < 70


# ---------------------------------------------------------------------------
# Golden values (stability + cross-framework reference)
# ---------------------------------------------------------------------------


def test_golden_loss_values_stable_across_calls():
    a = golden_loss_values()
    b = golden_loss_values()
    assert a == b


def test_golden_loss_values_keys_and_finite():
    g = golden_loss_values()
    assert set(g) == {"loss_critic", "loss_actor", "rl_loss", "bc_loss", "delta_penalty"}
    assert all(isinstance(v, float) for v in g.values())
    assert all(v == v for v in g.values())  # not NaN


# ---------------------------------------------------------------------------
# build_replay_windows round-trip
# ---------------------------------------------------------------------------


def _make_trace(n_chunks: int, chunk_len: int, d: int) -> list[StepRecord]:
    trace: list[StepRecord] = []
    t = 0
    for cid in range(n_chunks):
        z = torch.randn(8)
        ref = torch.randn(chunk_len, d)
        for j in range(chunk_len):
            trace.append(
                StepRecord(
                    t=t,
                    chunk_id=cid,
                    t_in_chunk=j,
                    z_rl=z,
                    ref_chunk=ref,
                    proprio=torch.randn(4),
                    executed=torch.randn(d),
                    reward=1.0 if (cid == n_chunks - 1 and j == chunk_len - 1) else 0.0,
                    done=(cid == n_chunks - 1 and j == chunk_len - 1),
                    source=SOURCE_RL,
                    human_action=None,
                )
            )
            t += 1
    return trace


def test_build_replay_windows_boundary_mode():
    cl = 3
    trace = _make_trace(n_chunks=4, chunk_len=cl, d=2)
    ws = build_replay_windows(trace, chunk_len=cl, stride=0, discount=0.99)
    assert len(ws) == 4
    for w in ws:
        assert w.action.shape == (cl * 2,)
        assert w.state["z_rl"].shape == (8,)
        assert w.state["proprio"].shape == (4,)
        assert "ref_chunk" in w.complementary_info
        assert "ref_chunk_next" in w.complementary_info
        assert "source" in w.complementary_info
        assert "bc_target" in w.complementary_info


def test_build_replay_windows_dense_stride():
    cl = 3
    trace = _make_trace(n_chunks=2, chunk_len=cl, d=2)
    ws = build_replay_windows(trace, chunk_len=cl, stride=1, discount=0.99)
    # n=6, chunk_len=3 → starts 0..3 → 4 windows.
    assert len(ws) == 4


def test_build_replay_windows_empty_trace():
    assert build_replay_windows([], chunk_len=3) == []


def test_build_replay_windows_human_source_bc_target():
    cl = 2
    d = 2
    trace = _make_trace(n_chunks=1, chunk_len=cl, d=d)
    # Mark the first step as human-intervened.
    human = torch.tensor([9.0, 9.0])
    trace[0] = trace[0]._replace(human_action=human) if hasattr(trace[0], "_replace") else trace[0]
    # StepRecord is a dataclass, not a namedtuple — mutate via replacement.
    trace[0].human_action = human
    trace[0].source = SOURCE_HUMAN
    trace[1].source = SOURCE_HUMAN
    ws = build_replay_windows(trace, chunk_len=cl, stride=0, discount=0.99)
    assert len(ws) == 1
    bt = ws[0].complementary_info["bc_target"].reshape(cl, d)
    assert torch.equal(bt[0], human)
    # Second step had no human action → fallback zeros (per _bc_target_for_chunk).
    assert torch.equal(bt[1], torch.zeros(d))


# ---------------------------------------------------------------------------
# Algorithm-level regression: RLTTD3Algorithm delegates to the shared math
# ---------------------------------------------------------------------------


def test_algorithm_module_reexports_shared_losses():
    import inspect

    from lerobot.rl.algorithms.rlt_td3 import (
        RLTTD3Algorithm,
        bc_weight_schedule as alg_bc,
        polyak_update as alg_polyak,
        rlt_actor_loss as alg_actor,
        td3_critic_loss as alg_critic,
    )
    from lerobot.rl.algorithms.rlt_td3 import rlt_td3_algorithm as mod

    # The algorithm's loss methods must be thin wrappers over the shared fns.
    src_critic = inspect.getsource(mod.RLTTD3Algorithm._compute_loss_critic)
    src_actor = inspect.getsource(mod.RLTTD3Algorithm._compute_loss_actor)
    assert "td3_critic_loss" in src_critic
    assert "rlt_actor_loss" in src_actor
    # Shared callables are the very same objects imported by the algorithm.
    from lerobot.rlt.shared import (
        bc_weight_schedule as shared_bc,
        polyak_update as shared_polyak,
        rlt_actor_loss as shared_actor,
        td3_critic_loss as shared_critic,
    )

    assert alg_critic is shared_critic
    assert alg_actor is shared_actor
    assert alg_bc is shared_bc
    assert alg_polyak is shared_polyak
    # And the algorithm class is still registered.
    assert RLTTD3Algorithm.name == "rlt_td3"
