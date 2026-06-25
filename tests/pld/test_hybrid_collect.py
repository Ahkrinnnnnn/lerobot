# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

import random

from lerobot.rollout.configs import PLDHybridCollectStrategyConfig


def test_pld_hybrid_config_probing_alpha_bounds():
    cfg = PLDHybridCollectStrategyConfig(probing_alpha=0.6)
    assert cfg.probing_alpha == 0.6


def test_t_base_sampling_range():
    random.seed(0)
    cfg = PLDHybridCollectStrategyConfig(
        probing_alpha=0.6,
        max_episode_steps=500,
        seed=0,
    )
    max_t = int(cfg.probing_alpha * cfg.max_episode_steps)
    samples = [random.randint(0, max_t) for _ in range(100)]
    assert all(0 <= s <= max_t for s in samples)
    assert max_t == 300
