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

"""Step-trace → replay-window construction for RLT Stage-2.

During a rollout episode the :class:`~lerobot.rollout.strategies.rlt_collect.
RLTCollectStrategy` records a raw per-tick *step trace*. At episode end the
trace is sliced into fixed-length *replay windows* (one per chunk boundary, or
densely strided) that the off-policy TD3 learner consumes.

This module is pure-Python / pure-tensor and has no hardware dependency, so it
can be unit-tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# Control-source labels. Duplicated in lerobot/rl/algorithms/rlt_td3/losses.py
# (kept as plain ints in both places so the loss math has no dependency on this
# package, avoiding a circular import). Update both together.
SOURCE_BASE = 0
SOURCE_RL = 1
SOURCE_HUMAN = 2
SOURCE_MIXED = 3


@dataclass
class StepRecord:
    """A single per-tick record collected during a rollout episode."""

    t: int  # step index within the episode
    chunk_id: int  # index of the chunk boundary that produced this step's chunk
    t_in_chunk: int  # step index within the chunk [0, C)
    # Frozen "Machine A" outputs at the chunk boundary that owns this step
    # (constant across the chunk).
    z_rl: Tensor  # [token_dim]
    ref_chunk: Tensor  # [C, d]
    # Proprioceptive state at THIS tick (used as the window's s_p start).
    proprio: Tensor  # [proprio_dim]
    # Action actually executed at this tick (already unnormalized to robot cmd).
    executed: Tensor  # [d]
    # Per-step reward and terminal flag.
    reward: float
    done: bool
    # Control source for this tick (BASE / RL / HUMAN / MIXED).
    source: int
    # Human action recorded at this tick (if any). [d] or None.
    human_action: Tensor | None = None


@dataclass
class ReplayWindow:
    """One off-policy transition consumed by the TD3 learner."""

    state: dict[str, Tensor]  # {"z_rl", "proprio"}
    action: Tensor  # [C*d]
    reward: float
    next_state: dict[str, Tensor]  # {"z_rl", "proprio"}
    done: bool
    complementary_info: dict[str, Tensor]


def _chunk_source(step_sources: list[int]) -> int:
    """Aggregate per-step sources into a single chunk source label."""
    has_human = any(s in (SOURCE_HUMAN, SOURCE_MIXED) for s in step_sources)
    has_rl = any(s == SOURCE_RL for s in step_sources)
    if has_human and has_rl:
        return SOURCE_MIXED
    if has_human:
        return SOURCE_HUMAN
    if has_rl:
        return SOURCE_RL
    return SOURCE_BASE


def _bc_target_for_chunk(
    ref_chunk: Tensor,
    step_human: list[Tensor | None],
    chunk_source: int,
) -> Tensor:
    """Select the BC target chunk for a chunk window.

    * BASE  → VLA reference chunk ``ã``.
    * HUMAN / MIXED → the human action chunk (per-tick human actions, falling
      back to the executed action where the human did not intervene).
    * RL → the reference chunk (BC is skipped for RL windows via the source
      mask in the learner, so the value here is a placeholder).
    """
    if chunk_source in (SOURCE_HUMAN, SOURCE_MIXED):
        # Build a human chunk from per-tick human actions, using executed as fallback.
        rows = []
        for h in step_human:
            rows.append(h if h is not None else torch.zeros_like(ref_chunk[0]))
        human_chunk = torch.stack(rows, dim=0)
        return human_chunk.reshape(-1)
    return ref_chunk.reshape(-1)


def _window_from_slice(
    trace: list[StepRecord],
    start: int,
    chunk_len: int,
    discount: float,
    next_boundary: StepRecord | None,
) -> ReplayWindow | None:
    """Build a window covering ``chunk_len`` steps starting at ``trace[start]``."""
    end = start + chunk_len
    if end > len(trace):
        return None
    steps = trace[start:end]
    boundary = trace[steps[0].chunk_id_boundary] if hasattr(steps[0], "chunk_id_boundary") else None
    # The chunk-boundary record for this window is the boundary that owns step[start].
    # We locate it via the chunk_id of the first step.
    first = steps[0]
    # Find the boundary step (t_in_chunk == 0) for first.chunk_id.
    boundary_rec = next(
        (r for r in trace if r.chunk_id == first.chunk_id and r.t_in_chunk == 0),
        first,
    )

    z_rl = boundary_rec.z_rl
    ref_chunk = boundary_rec.ref_chunk
    proprio = first.proprio

    executed_chunk = torch.stack([s.executed for s in steps], dim=0)  # [C, d]
    action_flat = executed_chunk.reshape(-1)

    # Discounted chunk return (gamma^t within the chunk).
    rewards = [s.reward for s in steps]
    dones = [s.done for s in steps]
    chunk_return = 0.0
    gamma = 1.0
    done_in_window = False
    for r, d in zip(rewards, dones):
        chunk_return += gamma * r
        gamma *= discount
        if d:
            done_in_window = True
            break

    # Next state = the chunk boundary after this window's chunk, if available.
    if next_boundary is not None and not done_in_window:
        next_z_rl = next_boundary.z_rl
        next_proprio = next_boundary.proprio
        next_ref = next_boundary.ref_chunk
    else:
        # Terminal: reuse the last step's state (the learner masks with done).
        last = steps[-1]
        next_z_rl = z_rl
        next_proprio = last.proprio
        next_ref = ref_chunk
        done_in_window = True

    step_sources = [s.source for s in steps]
    step_human = [s.human_action for s in steps]
    chunk_source = _chunk_source(step_sources)
    bc_target = _bc_target_for_chunk(ref_chunk, step_human, chunk_source)

    return ReplayWindow(
        state={"z_rl": z_rl, "proprio": proprio},
        action=action_flat,
        reward=float(chunk_return),
        next_state={"z_rl": next_z_rl, "proprio": next_proprio},
        done=bool(done_in_window),
        complementary_info={
            "ref_chunk": ref_chunk.reshape(-1),
            "ref_chunk_next": next_ref.reshape(-1),
            "source": torch.tensor(chunk_source, dtype=torch.long),
            "bc_target": bc_target,
        },
    )


def build_replay_windows(
    trace: list[StepRecord],
    chunk_len: int,
    stride: int = 0,
    discount: float = 0.99,
) -> list[ReplayWindow]:
    """Slice an episode step trace into replay windows.

    Args:
        trace: per-tick records for one episode, ordered by ``t``.
        chunk_len: chunk length ``C``.
        stride: window stride in env steps. ``0`` (default) emits one window
            per chunk boundary (only at ``t_in_chunk == 0`` positions). ``>0``
            emits a window every ``stride`` steps (dense, openpi-RLT style),
            carrying the owning chunk boundary's ``z_rl`` / ``ref_chunk``.
        discount: per-step discount used for the chunk-return Bellman target.

    Returns:
        A list of :class:`ReplayWindow` ready to push into the replay buffer.
    """
    if not trace:
        return []
    if chunk_len < 1:
        raise ValueError(f"chunk_len must be >= 1, got {chunk_len}")

    # Collect chunk-boundary records (t_in_chunk == 0), ordered by chunk_id.
    boundaries = [r for r in trace if r.t_in_chunk == 0]
    # Map chunk_id -> next chunk boundary (for next_state).
    next_boundary_map: dict[int, StepRecord | None] = {}
    for i, b in enumerate(boundaries):
        next_boundary_map[b.chunk_id] = boundaries[i + 1] if i + 1 < len(boundaries) else None

    windows: list[ReplayWindow] = []
    if stride <= 0:
        # Chunk-boundary mode: one window per boundary, aligned to the boundary tick.
        for b in boundaries:
            start = b.t
            next_b = next_boundary_map[b.chunk_id]
            w = _window_from_slice(trace, start, chunk_len, discount, next_b)
            if w is not None:
                windows.append(w)
        return windows

    # Dense mode: a window every `stride` steps, clamped to episode length.
    n = len(trace)
    for start in range(0, n - chunk_len + 1, stride):
        first = trace[start]
        next_b = next_boundary_map.get(first.chunk_id)
        # If this window's chunk_id has no next boundary and it's the last chunk,
        # the next_state falls back to terminal handling inside _window_from_slice.
        w = _window_from_slice(trace, start, chunk_len, discount, next_b)
        if w is not None:
            windows.append(w)
    return windows
