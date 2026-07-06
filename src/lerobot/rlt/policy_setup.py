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

"""Build the frozen RLT policy + trainable RLT actor from a Stage-1 checkpoint.

This mirrors :func:`lerobot.pld.residual_setup.init_residual_policy` but for the
RLT Stage-2 actor: the heavy base VLA + RL token are constructed once from the
Stage-1 checkpoint (without a dataset / env config) and wrapped by a lightweight
trainable :class:`~lerobot.policies.rlt_actor.RLTActorPolicy`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rlt_actor import RLTActorConfig, RLTActorPolicy
from lerobot.policies.rl_token import RLTConfig
from lerobot.policies.rl_token.modeling_rlt import RLTPolicy
from lerobot.robots import make_robot_from_config
from lerobot.utils.constants import ACTION

if TYPE_CHECKING:
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.robots.config import RobotConfig

logger = logging.getLogger(__name__)


def _infer_proprio_keys(robot) -> list[str]:
    return sorted(
        k for k in robot.observation_features if k.endswith(".pos") and not k.startswith("gripper")
    )


def build_rlt_actor_policy(
    *,
    rlt_checkpoint_path: str | Path,
    robot_cfg: RobotConfig,
    chunk_len: int,
    proprio_keys: list[str] | None = None,
    device: str = "cpu",
    rename_map: dict[str, str] | None = None,
) -> tuple[RLTActorPolicy, RLTPolicy, PolicyProcessorPipeline, PolicyProcessorPipeline, list[str]]:
    """Construct the frozen RLT policy + trainable actor + processors.

    Returns ``(actor_policy, rlt_policy, preprocessor, postprocessor, proprio_keys)``.
    """
    rlt_config = RLTConfig.from_pretrained(rlt_checkpoint_path)
    rlt_config.device = device

    base_cfg = rlt_config.base_policy
    base_cfg.device = device
    if getattr(base_cfg, "compile_model", None) is not None:
        base_cfg.compile_model = False

    # Construct the base VLA directly from its checkpoint (features + stats are
    # already populated in the saved RLT config; normalization lives in the
    # pre/post processors loaded below, so no ds_meta/env_cfg is needed here).
    base_policy_cls = get_policy_class(base_cfg.type)
    base_policy = base_policy_cls.from_pretrained(base_cfg.pretrained_path, config=base_cfg)
    base_policy.to(device).eval()
    for p in base_policy.parameters():
        p.requires_grad = False

    rlt_policy = RLTPolicy(rlt_config, base_policy=base_policy)
    rlt_policy.to(device).eval()
    for p in rlt_policy.parameters():
        p.requires_grad = False

    # Robot-derived proprio keys (no connect).
    robot = make_robot_from_config(robot_cfg)
    proprio_keys = list(proprio_keys or _infer_proprio_keys(robot))

    # Derive actor config dims.
    action_dim = int(rlt_config.output_features[ACTION].shape[0])
    token_dim = rlt_config.rlt_num_tokens * rlt_config.rlt_embed_dim
    proprio_dim = len(proprio_keys)

    actor_cfg = RLTActorConfig(
        rlt_checkpoint_path=str(rlt_checkpoint_path),
        chunk_len=chunk_len,
        action_dim=action_dim,
        token_dim=token_dim,
        proprio_dim=proprio_dim,
        device=device,
    )
    actor_policy = RLTActorPolicy(actor_cfg, rlt_policy=rlt_policy)
    actor_policy.to(device)

    # Pre/post processors: RLT delegates to the base VLA. Load from the base
    # VLA's pretrained dir (where its processor configs live).
    preprocessor, postprocessor = make_pre_post_processors(
        rlt_config,
        pretrained_path=base_cfg.pretrained_path,
        rename_map=rename_map,
    )
    logger.info(
        "Built RLT actor policy: chunk_len=%d action_dim=%d token_dim=%d proprio_dim=%d device=%s",
        chunk_len,
        action_dim,
        token_dim,
        proprio_dim,
        device,
    )
    return actor_policy, rlt_policy, preprocessor, postprocessor, proprio_keys
