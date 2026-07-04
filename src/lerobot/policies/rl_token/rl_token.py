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

"""RL Token encoder-decoder for compressing VLA prefix embeddings (RLT paper Stage 1)."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


def _build_sinusoidal_pos_embedding(num_positions: int, embed_dim: int) -> Tensor:
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim ({embed_dim}) must be divisible by 2")
    position = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / embed_dim)
    )
    pe = torch.zeros(num_positions, embed_dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class RLTokenCrossAttentionLayer(nn.Module):
    """Single cross-attention block with pre-norm and MLP."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        attn_out, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask)
        query = query + attn_out
        query = query + self.mlp(self.norm_ff(query))
        return query


@dataclass
class RLTokenConfig:
    """Configuration for the RL token encoder-decoder."""

    num_rl_tokens: int = 1
    num_layers: int = 2
    embed_dim: int = 512
    input_dim: int = 2048
    mlp_ratio: float = 4.0
    num_heads: int = 8
    dropout: float = 0.0


class RLTokenEncoder(nn.Module):
    """Compress VLA prefix embeddings into compact RL token(s)."""

    def __init__(self, config: RLTokenConfig):
        super().__init__()
        self.config = config
        self.input_proj = (
            nn.Identity()
            if config.input_dim == config.embed_dim
            else nn.Linear(config.input_dim, config.embed_dim)
        )
        self.query_pos = nn.Parameter(_build_sinusoidal_pos_embedding(config.num_rl_tokens, config.embed_dim))
        self.layers = nn.ModuleList(
            [
                RLTokenCrossAttentionLayer(
                    config.embed_dim,
                    config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

    def forward(self, prefix_embs: Tensor, mask: Tensor | None = None) -> Tensor:
        prefix_embs = self.input_proj(prefix_embs)
        batch_size = prefix_embs.shape[0]
        kv_pos = _build_sinusoidal_pos_embedding(prefix_embs.shape[1], self.config.embed_dim).to(prefix_embs.device)
        kv = prefix_embs + kv_pos.unsqueeze(0)
        query = self.query_pos.unsqueeze(0).expand(batch_size, -1, -1)

        key_padding_mask = ~mask if mask is not None else None

        for layer in self.layers:
            query = layer(query, kv, key_padding_mask=key_padding_mask)
        return query


class RLTokenDecoder(nn.Module):
    """Reconstruct prefix embeddings from RL token(s)."""

    def __init__(self, config: RLTokenConfig):
        super().__init__()
        self.config = config
        self.output_proj = (
            nn.Identity()
            if config.input_dim == config.embed_dim
            else nn.Linear(config.embed_dim, config.input_dim)
        )
        self.kv_pos = nn.Parameter(_build_sinusoidal_pos_embedding(config.num_rl_tokens, config.embed_dim))
        self.layers = nn.ModuleList(
            [
                RLTokenCrossAttentionLayer(
                    config.embed_dim,
                    config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

    def forward(self, rl_tokens: Tensor, target_seq_len: int) -> Tensor:
        batch_size = rl_tokens.shape[0]
        query_pos = _build_sinusoidal_pos_embedding(target_seq_len, self.config.embed_dim).to(rl_tokens.device)
        query = query_pos.unsqueeze(0).expand(batch_size, -1, -1)
        kv = rl_tokens + self.kv_pos.unsqueeze(0)

        for layer in self.layers:
            query = layer(query, kv, key_padding_mask=None)
        return self.output_proj(query)


class RLTokenModel(nn.Module):
    """RL token encoder-decoder with reconstruction loss."""

    def __init__(self, config: RLTokenConfig):
        super().__init__()
        self.config = config
        self.encoder = RLTokenEncoder(config)
        self.decoder = RLTokenDecoder(config)

    def encode(self, prefix_embs: Tensor, mask: Tensor | None = None) -> Tensor:
        return self.encoder(prefix_embs, mask)

    def decode(self, rl_tokens: Tensor, target_seq_len: int) -> Tensor:
        return self.decoder(rl_tokens, target_seq_len)

    def reconstruction_loss(
        self,
        prefix_embs: Tensor,
        mask: Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute masked MSE reconstruction loss with stop-gradient targets."""
        target = prefix_embs.detach()
        rl_tokens = self.encode(target, mask)
        reconstructed = self.decode(rl_tokens, target_seq_len=target.shape[1])

        sq_error = (reconstructed - target).square()
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).to(dtype=sq_error.dtype)
            per_sample_denom = (mask.sum(dim=1, keepdim=True).clamp_min(1) * target.shape[-1]).to(
                dtype=sq_error.dtype
            )
            per_sample_loss = (sq_error * mask_expanded).sum(dim=(1, 2)) / per_sample_denom.squeeze(1)
        else:
            per_sample_loss = sq_error.mean(dim=(1, 2))

        if reduction == "none":
            loss = per_sample_loss
        else:
            loss = per_sample_loss.mean()

        return loss, {"rlt_mse": per_sample_loss.mean().item()}
