# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

import torch

from lerobot.rl.buffer import ReplayBuffer
from lerobot.utils.constants import ACTION, OBS_STATE


def _make_buffer_with_episodes(episode_lengths: list[int]) -> ReplayBuffer:
    buffer = ReplayBuffer(capacity=1000, device="cpu", state_keys=[OBS_STATE], use_drq=False)
    for ep_len in episode_lengths:
        for step in range(ep_len):
            state = {OBS_STATE: torch.tensor([[float(step)]])}
            action = torch.tensor([[1.0]])
            done = step == ep_len - 1
            reward = 1.0 if done else 0.0
            buffer.add(
                state=state,
                action=action,
                reward=reward,
                next_state=state,
                done=done,
                truncated=False,
            )
    return buffer


def test_summarize_and_filter_episodes():
    buffer = _make_buffer_with_episodes([3, 2, 4])
    summaries = buffer.summarize_episodes()
    assert len(summaries) == 3
    assert [row["length"] for row in summaries] == [3, 2, 4]

    filtered = buffer.filter_episodes(exclude_episode_indices={1})
    kept = filtered.summarize_episodes()
    assert len(kept) == 2
    assert [row["length"] for row in kept] == [3, 4]
    assert len(filtered) == 7
