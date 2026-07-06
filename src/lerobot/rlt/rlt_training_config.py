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

"""RLT Stage-2 training hyper-parameters (paper §IV-B / Appendix B)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RLTTrainingConfig:
    """Hyper-parameters for the RLT Stage-2 TD3 + BC loop."""

    # Online RL loop
    rl_steps: int = 10000
    train_iters_per_collect: int = 1
    collect_env_steps_per_iter: int = 50
    max_episode_steps: int = 500
    online_buffer_capacity: int = 250_000
    batch_size: int = 256
    online_step_before_learning: int = 100

    # Asynchronous rollout/learning (paper §IV-B: "we perform the rollouts and
    # learning asynchronously"). When True, a learner thread runs continuous
    # TD3 updates while a rollout thread collects episodes on the robot in
    # parallel. When False, falls back to the sequential collect→train
    # alternation (PLD-style, useful for debugging).
    async_training: bool = True
    checkpoint_every_steps: int = 500

    # Warmup gating
    warmup_min_buffer_size: int = 200  # min replay windows before actor goes online
    warmup_pretraining_updates: int = 0  # BC-only actor updates during warmup
    warmup_collect_rounds: int = 0  # extra base-only collect rounds before online

    # Replay windowing (paper §IV-B "Subsampling Action Chunks": stride=2 dense).
    replay_stride: int = 2  # 0 = chunk-boundary; >0 = dense stride (paper: 2)
    chunk_return_discount: float = 0.99

    # Episode reset
    episode_reset_time_s: float = 15.0
    manual_scene_reset_pause_key: str = "space"
    finish_episode_before_round_stop: bool = True

    # TD3 + BC (paper §IV-B: UTD=5, two critic updates per actor update).
    discount: float = 0.99
    grad_clip_norm: float = 40.0
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    critic_target_update_weight: float = 0.005
    num_critics: int = 2
    utd_ratio: int = 5  # paper: high update-to-data ratio of 5
    policy_update_freq: int = 2  # paper: two critic updates per actor update
    use_adamw: bool = False
    optimizer_weight_decay: float = 0.01

    # BC regularization schedule
    bc_weight_max: float = 1.0
    bc_weight_min: float = 0.0
    bc_decay_steps: int = 10000
    ref_dropout_prob: float = 0.5

    # Action smoothness
    delta_penalty_weight: float = 0.0

    # Eval
    eval_episodes: int = 10

    # Resume
    resume_online_buffer: str | None = None
    resume_actor_checkpoint: str | None = None
    resume_algo_checkpoint: str | None = None
