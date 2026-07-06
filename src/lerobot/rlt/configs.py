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

"""Top-level configuration for RLT Stage-2 online RL (``lerobot-rlt-stage2``)."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.gaussian_actor.configuration_gaussian_actor import CriticNetworkConfig
from lerobot.rewards.classifier.pipeline_config import (
    RewardClassifierRuntimeConfig as RewardClassifierConfig,
)
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig

from .rlt_training_config import RLTTrainingConfig


@dataclass
class RLTStage2Config:
    """Top-level config for ``lerobot-rlt-stage2``.

    Stage 2 wraps a frozen Stage-1 RLT checkpoint (base VLA + RL token) and
    trains a lightweight chunked actor-critic (TD3 + BC) online on the real
    robot. The base VLA runs in-process at chunk boundaries; no RTC inference
    engine is used.
    """

    robot: RobotConfig
    teleop: TeleoperatorConfig | None = None
    # Path to a Stage-1 RLT checkpoint (config.json + RL token weights).
    rlt_checkpoint_path: str | None = None

    reward_classifier: RewardClassifierConfig = field(default_factory=RewardClassifierConfig)
    training: RLTTrainingConfig = field(default_factory=RLTTrainingConfig)

    # Chunk length C (paper: 10). Must match the base VLA's action horizon if
    # you want the reference chunk to cover exactly one VLA chunk.
    chunk_len: int = 10

    # Proprioceptive joint keys for s_p (arm joints). If empty, inferred from the
    # robot's `.pos` observation features (excluding the gripper).
    proprio_keys: list[str] = field(default_factory=list)

    fps: float = 30.0
    task: str = ""
    device: str | None = None
    rename_map: dict[str, str] = field(default_factory=dict)
    interpolation_multiplier: int = 1
    return_to_initial_position: bool = True

    # Pipeline stage control
    skip_warmup: bool = False
    skip_online_train: bool = False
    eval_only: bool = False

    output_dir: str = "outputs/rlt_stage2"
    job_name: str = "rlt_stage2"

    def __post_init__(self):
        if self.rlt_checkpoint_path is None:
            loaded = parser.get_path_arg("rlt_checkpoint_path")
            if loaded:
                self.rlt_checkpoint_path = loaded
        if self.rlt_checkpoint_path is None:
            raise ValueError("--rlt_checkpoint_path is required for RLT Stage 2")
        if self.chunk_len < 1:
            raise ValueError(f"chunk_len must be >= 1, got {self.chunk_len}")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["rlt_checkpoint_path"]
