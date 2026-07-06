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

"""RLT Stage-2 chunked actor-critic policy (paper §IV-B).

Wraps a frozen :class:`~lerobot.policies.rl_token.RLTPolicy` (base VLA + RL
token encoder) and adds lightweight trainable actor / critic MLPs that operate
on the compact RL token ``z_rl``, proprioceptive state ``s_p`` and the VLA
reference action chunk ``ã``.
"""

from __future__ import annotations

import builtins
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch import Tensor

from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

from ..gaussian_actor.modeling_gaussian_actor import MLP, TanhMultivariateNormalDiag, orthogonal_init
from ..pretrained import PreTrainedPolicy, T
from .configuration_rlt_actor import RLTActorConfig

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.envs import EnvConfig


class ChunkActor(nn.Module):
    """Gaussian actor over flattened action chunks, conditioned on (z_rl, s_p, ã)."""

    def __init__(self, config: RLTActorConfig):
        super().__init__()
        self.config = config
        in_dim = config.token_dim + config.proprio_dim + config.chunk_dim
        self.net = MLP(
            input_dim=in_dim,
            hidden_dims=list(config.actor_network_kwargs.hidden_dims),
            activate_final=config.actor_network_kwargs.activate_final,
            activations=config.actor_network_kwargs.activations,
        )
        out_features = config.actor_network_kwargs.hidden_dims[-1]
        self.mean_layer = nn.Linear(out_features, config.chunk_dim)
        orthogonal_init()(self.mean_layer.weight)
        self.fixed_std = float(config.fixed_std)

    def forward(
        self,
        z_rl: Tensor,
        proprio: Tensor,
        ref_chunk: Tensor,
        dropout_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return (action_chunk_flat, log_prob, mean_flat).

        ``action_chunk_flat`` is a tanh-squashed Gaussian sample of shape
        ``[B, C*d]``. ``ref_chunk`` is the flattened VLA reference chunk; when
        ``dropout_mask`` is provided it is multiplied element-wise so that a
        fraction of the batch sees a zeroed reference (paper: reference-action
        dropout).
        """
        if dropout_mask is not None:
            ref_chunk = ref_chunk * dropout_mask
        x = torch.cat([z_rl, proprio, ref_chunk], dim=-1)
        mean = self.mean_layer(self.net(x))
        std = self.fixed_std * torch.ones_like(mean)
        dist = TanhMultivariateNormalDiag(loc=mean, scale_diag=std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        return action, log_prob, mean


class _ChunkCriticHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], activations: str, activate_final: bool):
        super().__init__()
        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=list(hidden_dims),
            activate_final=activate_final,
            activations=activations,
        )
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        orthogonal_init()(self.output_layer.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.output_layer(self.net(x)).squeeze(-1)


class ChunkCriticEnsemble(nn.Module):
    """Ensemble of ``num_critics`` Q-heads sharing the same input signature.

    Unlike :class:`lerobot.rl.algorithms.sac.sac_algorithm.CriticEnsemble` this
    does not wrap a vision encoder: the RLT actor-critic consumes the already
    compact ``z_rl`` token, so each critic is a plain MLP over
    ``[z_rl, s_p, a_{1:C}]``.
    """

    def __init__(
        self,
        token_dim: int,
        proprio_dim: int,
        chunk_dim: int,
        num_critics: int,
        hidden_dims: list[int],
        activations: str = "SiLU",
        activate_final: bool = True,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.proprio_dim = proprio_dim
        self.chunk_dim = chunk_dim
        in_dim = token_dim + proprio_dim + chunk_dim
        self.critics = nn.ModuleList(
            [
                _ChunkCriticHead(
                    input_dim=in_dim,
                    hidden_dims=list(hidden_dims),
                    activations=activations,
                    activate_final=activate_final,
                )
                for _ in range(num_critics)
            ]
        )

    def forward(self, z_rl: Tensor, proprio: Tensor, action_chunk: Tensor) -> Tensor:
        x = torch.cat([z_rl, proprio, action_chunk], dim=-1)
        return torch.stack([c(x) for c in self.critics], dim=0)  # [num_critics, B]


class RLTActorPolicy(PreTrainedPolicy):
    """Stage-2 RLT policy: frozen base VLA + RL token, with a trainable chunk actor.

    The critic ensemble is owned by the :class:`~lerobot.rl.algorithms.rlt_td3`
    algorithm (it is not needed at rollout time), mirroring the SAC split where
    the policy owns the actor and the algorithm owns the critics.
    """

    config_class = RLTActorConfig
    name = "rlt_actor"

    def __init__(
        self,
        config: RLTActorConfig,
        *,
        rlt_policy: Any,
        ds_meta: LeRobotDatasetMetadata | None = None,
        env_cfg: EnvConfig | None = None,
        rename_map: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(config)
        if rlt_policy is None:
            raise ValueError("RLTActorPolicy requires a frozen RLTPolicy instance (rlt_policy=...).")
        self.rlt_policy = rlt_policy
        # Freeze the base VLA + RL token module.
        for param in self.rlt_policy.parameters():
            param.requires_grad = False
        self.rlt_policy.eval()
        self.actor = ChunkActor(config)
        if config.device is not None:
            self.actor.to(config.device)
            self.rlt_policy.to(config.device)

    # ------------------------------------------------------------------
    # Frozen "Machine A" interface: z_rl + VLA reference chunk
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_rl_token(self, batch: dict[str, Tensor]) -> Tensor:
        """Return the flattened RL token ``z_rl`` of shape ``[B, token_dim]``."""
        token = self.rlt_policy.extract_rl_token(batch)  # [B, num_rl_tokens, embed_dim]
        return token.reshape(token.shape[0], -1)

    @torch.no_grad()
    def predict_reference_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Return the VLA reference chunk ``ã`` of shape ``[B, C, d]`` (first C of H)."""
        chunk = self.rlt_policy.predict_action_chunk(batch)  # [B, H, d]
        return chunk[:, : self.config.chunk_len]

    @torch.no_grad()
    def machine_a_query(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Single chunk-boundary query: ``(z_rl_flat [B, token_dim], ref_flat [B, C*d])``."""
        z_rl = self.extract_rl_token(batch)
        ref = self.predict_reference_chunk(batch)
        return z_rl, ref.reshape(ref.shape[0], -1)

    # ------------------------------------------------------------------
    # Trainable actor / critic
    # ------------------------------------------------------------------

    def actor_forward(
        self,
        z_rl: Tensor,
        proprio: Tensor,
        ref_chunk: Tensor,
        dropout_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self.actor(z_rl, proprio, ref_chunk, dropout_mask)

    @torch.no_grad()
    def select_action_chunk(
        self,
        batch: dict[str, Tensor],
        proprio: Tensor,
        ref_chunk_flat: Tensor,
        *,
        deterministic: bool = False,
    ) -> Tensor:
        """Refine the VLA reference into an executed chunk ``[B, C, d]``.

        At inference the reference is always provided (no dropout). When
        ``deterministic`` the actor mean is used (eval mode).
        """
        z_rl = self.extract_rl_token(batch)
        if deterministic:
            mean_flat = self.actor_forward(z_rl, proprio, ref_chunk_flat)[2]
            action_flat = mean_flat
        else:
            action_flat = self.actor_forward(z_rl, proprio, ref_chunk_flat)[0]
        return action_flat.reshape(action_flat.shape[0], self.config.chunk_len, self.config.action_dim)

    def get_optim_params(self) -> dict[str, list[Tensor]]:
        return {
            "actor": list(self.actor.parameters()),
        }

    def reset(self) -> None:
        self.rlt_policy.reset()

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError(
            "RLTActorPolicy is a chunked policy; use select_action_chunk() instead."
        )

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        raise NotImplementedError(
            "RLTActorPolicy.predict_action_chunk is not used; the rollout driver calls "
            "machine_a_query() + select_action_chunk() at chunk boundaries."
        )

    def forward(self, batch: dict[str, Tensor | dict[str, Tensor]]) -> dict[str, Tensor]:
        raise NotImplementedError("RLTActorPolicy is trained via the RLTTD3 algorithm, not via forward().")

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep the frozen base VLA + RL token in eval mode.
        self.rlt_policy.eval()
        return self

    # ------------------------------------------------------------------
    # Checkpointing: persist only the trainable actor.
    # ------------------------------------------------------------------

    def _save_pretrained(self, save_directory: Path) -> None:
        save_directory.mkdir(parents=True, exist_ok=True)
        self.config._save_pretrained(save_directory)
        tensors = {
            f"actor.{k}": v.detach().cpu().contiguous()
            for k, v in self.actor.state_dict().items()
        }
        save_safetensors(tensors, str(save_directory / SAFETENSORS_SINGLE_FILE))

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: RLTActorConfig | None = None,
        strict: bool = False,
        ds_meta: LeRobotDatasetMetadata | None = None,
        env_cfg: EnvConfig | None = None,
        rename_map: dict[str, str] | None = None,
        rlt_policy: Any = None,
        **kwargs: Any,
    ) -> T:
        from ..pretrained import PreTrainedConfig
        from ..rl_token.modeling_rlt import RLTPolicy

        if config is None:
            config = PreTrainedConfig.from_pretrained(pretrained_name_or_path, **kwargs)
        if not isinstance(config, RLTActorConfig):
            raise TypeError(f"Expected RLTActorConfig, got {type(config).__name__}")

        rlt_ckpt = config.rlt_checkpoint_path
        if rlt_ckpt is None:
            raise ValueError(
                "RLTActorConfig.rlt_checkpoint_path must point to a Stage-1 RLT checkpoint."
            )
        if rlt_policy is None:
            rlt_policy = RLTPolicy.from_pretrained(
                rlt_ckpt,
                ds_meta=ds_meta,
                env_cfg=env_cfg,
                rename_map=rename_map,
            )

        policy = cls(
            config,
            rlt_policy=rlt_policy,
            ds_meta=ds_meta,
            env_cfg=env_cfg,
            rename_map=rename_map,
        )

        model_file = Path(pretrained_name_or_path) / SAFETENSORS_SINGLE_FILE
        if model_file.is_file():
            state_dict = load_safetensors(str(model_file))
            actor_state = {
                k.removeprefix("actor."): v for k, v in state_dict.items() if k.startswith("actor.")
            }
            policy.actor.load_state_dict(actor_state, strict=strict)
        policy.eval()
        return policy
