#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import pytest
import torch
from pathlib import Path

from lerobot.policies.factory import make_policy_config
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.rl_token import (
    RLTokenConfig,
    RLTokenModel,
    RLTConfig,
    get_registered_rlt_backbone_types,
)


def test_rl_token_reconstruction_loss():
    config = RLTokenConfig(num_rl_tokens=1, num_layers=2, embed_dim=64, input_dim=128, num_heads=4)
    model = RLTokenModel(config)
    batch_size, seq_len = 2, 16
    prefix_embs = torch.randn(batch_size, seq_len, config.input_dim)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[0, -2:] = False

    loss, info = model.reconstruction_loss(prefix_embs, mask)
    assert loss.ndim == 0
    assert loss.item() >= 0
    assert "rlt_mse" in info

    per_sample_loss, _ = model.reconstruction_loss(prefix_embs, mask, reduction="none")
    assert per_sample_loss.shape == (batch_size,)

    rl_tokens = model.encode(prefix_embs, mask)
    assert rl_tokens.shape == (batch_size, config.num_rl_tokens, config.embed_dim)

    reconstructed = model.decode(rl_tokens, target_seq_len=seq_len)
    assert reconstructed.shape == prefix_embs.shape


def test_rlt_config_requires_base_policy():
    with pytest.raises(ValueError, match="base_policy"):
        make_policy_config("rlt", rlt_alpha=0.0)


def test_rlt_config_with_base_policy():
    base = PI05Config(device="cpu")
    base.pretrained_path = Path("/tmp/fake_pi05/pretrained_model")
    cfg = make_policy_config(
        "rlt",
        rlt_alpha=0.0,
        base_policy=base,
    )
    assert isinstance(cfg, RLTConfig)
    assert cfg.rlt_alpha == 0.0
    assert cfg.base_policy.type == "pi05"
    assert cfg.base_policy.compile_model is False


def test_rlt_backbone_registry():
    assert "pi05" in get_registered_rlt_backbone_types()


def test_rlt_finetune_vla_sync():
    base = PI05Config(device="cpu")
    base.pretrained_path = Path("/tmp/fake_pi05/pretrained_model")
    cfg = RLTConfig(base_policy=base, rlt_finetune_vla=False, rlt_alpha=1.0)
    assert cfg.rlt_alpha == 0.0
    cfg = RLTConfig(base_policy=base, rlt_finetune_vla=True, rlt_alpha=0.0)
    assert cfg.rlt_alpha == 1.0
    cfg = RLTConfig(base_policy=base, rlt_finetune_vla=True, rlt_alpha=0.5)
    assert cfg.rlt_alpha == 0.5


def test_rlt_save_pretrained_only_rlt_module(tmp_path):
    """Checkpoint must contain RL token weights only, not base VLA parameters."""
    from safetensors.torch import load_file

    from lerobot.policies.rl_token.configuration_rlt import RLTConfig
    from lerobot.policies.rl_token.modeling_rlt import RLTPolicy
    from lerobot.policies.rl_token.stage1 import RLTStage1Module
    from lerobot.policies.pretrained import PreTrainedPolicy

    base_cfg = PI05Config(device="cpu", dtype="float32")
    base_cfg.pretrained_path = Path("/tmp/fake_pi05/pretrained_model")
    base_cfg.validate_features()

    rlt_cfg = RLTConfig(base_policy=base_cfg, rlt_alpha=0.0, device="cpu", rlt_embed_dim=64, rlt_input_dim=128)
    policy = RLTPolicy.__new__(RLTPolicy)
    PreTrainedPolicy.__init__(policy, rlt_cfg)
    policy.rlt_module = RLTStage1Module(rlt_cfg.to_rl_token_config())

    fake_vla = torch.nn.Linear(1000, 1000)
    policy.base_policy = torch.nn.Module()
    policy.base_policy.vla = fake_vla

    save_dir = tmp_path / "pretrained_model"
    save_dir.mkdir()
    policy._save_pretrained(save_dir)

    keys = list(load_file(save_dir / "model.safetensors").keys())
    assert keys, "expected non-empty checkpoint"
    assert not any("base_policy" in key or "vla" in key for key in keys)
    assert all("rl_token" in key for key in keys)

    # RL token checkpoint should be much smaller than a fake 1000x1000 VLA layer would be.
    vla_only_bytes = sum(p.numel() * p.element_size() for p in fake_vla.parameters())
    checkpoint_bytes = (save_dir / "model.safetensors").stat().st_size
    assert checkpoint_bytes < vla_only_bytes // 2


def test_rlt_train_config_parses_nested_base_policy_path(tmp_path):
    """JSON ``policy.base_policy.path`` must be extracted before draccus decode (PLD pattern)."""
    import draccus
    import json

    from lerobot.configs import parser as cfg_parser
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.rl_token.configuration_rlt import RLTConfig

    pi05_path = (
        tmp_path / "pi05" / "pretrained_model"
    )
    pi05_path.mkdir(parents=True)
    (pi05_path / "config.json").write_text(
        json.dumps({"type": "pi05", "compile_model": True, "dtype": "float32"})
    )

    config_json = tmp_path / "rlt.json"
    config_json.write_text(
        json.dumps(
            {
                "dataset": {"repo_id": "local/test", "root": str(tmp_path / "data")},
                "policy": {
                    "type": "rlt",
                    "rlt_finetune_vla": False,
                    "base_policy": {
                        "type": "pi05",
                        "path": str(pi05_path),
                        "dtype": "float32",
                    },
                },
            }
        )
    )

    cfg_parser._config_path_args.clear()
    cfg_parser._config_yaml_overrides.clear()
    cleaned = cfg_parser.extract_path_fields_from_config(
        str(config_json), TrainPipelineConfig.__get_path_fields__()
    )

    with draccus.config_type("json"):
        cfg = draccus.parse(TrainPipelineConfig, cleaned, args=[])

    assert isinstance(cfg.policy, RLTConfig)
    assert cfg.policy.base_policy.pretrained_path == pi05_path
    assert cfg.policy.base_policy.compile_model is False
    assert cfg.policy.base_policy.dtype == "float32"
