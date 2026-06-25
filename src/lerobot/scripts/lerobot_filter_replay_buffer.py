#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""List, preview, and filter episodes inside a saved PLD replay buffer (.pt)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lerobot.rl.buffer import ReplayBuffer
from lerobot.utils.constants import OBS_IMAGE


def _parse_int_set(text: str | None) -> set[int] | None:
    if text is None:
        return None
    return {int(x.strip()) for x in text.split(",") if x.strip()}


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    t = tensor.detach().cpu().float()
    if t.ndim == 4:
        t = t[0]
    if t.ndim == 3 and t.shape[0] in (1, 3):
        t = t.permute(1, 2, 0)
    arr = t.numpy()
    if arr.max() <= 1.0:
        arr = (arr * 255.0).clip(0, 255)
    arr = arr.astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = arr[..., 0]
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr)


def _first_image_key(buffer: ReplayBuffer) -> str | None:
    for key in buffer.states:
        if key.startswith(OBS_IMAGE):
            return key
    return None


def cmd_list(args: argparse.Namespace) -> None:
    buffer = ReplayBuffer.load(args.path, device="cpu")
    summaries = buffer.summarize_episodes()
    print(f"Buffer: {args.path}")
    print(f"Total transitions: {len(buffer)}  |  Episodes: {len(summaries)}")
    print("-" * 72)
    print(f"{'ep':>4}  {'len':>6}  {'max_r':>6}  {'final_r':>8}  {'done':>5}")
    for row in summaries:
        print(
            f"{row['episode_index']:4d}  {row['length']:6d}  "
            f"{row['max_reward']:6.2f}  {row['final_reward']:8.2f}  "
            f"{str(row['done']):>5}"
        )


def cmd_preview(args: argparse.Namespace) -> None:
    buffer = ReplayBuffer.load(args.path, device="cpu")
    image_key = args.image_key or _first_image_key(buffer)
    if image_key is None:
        raise SystemExit("No observation image keys found in buffer.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for episode_index, (start_idx, end_idx) in enumerate(buffer.episode_buffer_index_ranges()):
        for label, buf_idx in (("start", start_idx), ("end", end_idx)):
            img = _tensor_to_pil(buffer.states[image_key][buf_idx])
            img.save(out_dir / f"episode_{episode_index:03d}_{label}.png")
    print(f"Saved preview frames to {out_dir} (key={image_key})")


def cmd_compact(args: argparse.Namespace) -> None:
    buffer = ReplayBuffer.load(args.path, device="cpu")
    capacity = int(args.capacity) if args.capacity is not None else None
    optimize_memory = None
    if args.optimize_memory:
        optimize_memory = True
    elif args.no_optimize_memory:
        optimize_memory = False

    compacted = buffer.compact(capacity=capacity, optimize_memory=optimize_memory)
    output = Path(args.output)
    compacted.save(str(output))
    print(
        f"Compacted {len(buffer)} transitions: capacity {buffer.capacity} -> {compacted.capacity} "
        f"(optimize_memory={compacted.optimize_memory})"
    )
    print(f"Saved to {output}")


def cmd_filter(args: argparse.Namespace) -> None:
    exclude = _parse_int_set(args.exclude)
    include = _parse_int_set(args.include)
    if exclude is None and include is None:
        raise SystemExit("Specify --exclude or --include episode indices.")

    buffer = ReplayBuffer.load(args.path, device="cpu")
    before = len(buffer.summarize_episodes())
    filtered = buffer.filter_episodes(
        exclude_episode_indices=exclude,
        include_episode_indices=include,
    )
    after = len(filtered.summarize_episodes())

    output = Path(args.output)
    filtered.save(str(output))
    print(f"Filtered {before} -> {after} episodes ({len(filtered)} transitions)")
    print(f"Saved to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and filter PLD replay buffers.")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="List episodes in a saved buffer.")
    list_parser.add_argument("--path", required=True, help="Path to offline_buffer.pt")
    list_parser.set_defaults(func=cmd_list)

    preview_parser = sub.add_parser("preview", help="Export start/end frames per episode.")
    preview_parser.add_argument("--path", required=True, help="Path to offline_buffer.pt")
    preview_parser.add_argument(
        "--output-dir",
        default="outputs/buffer_preview",
        help="Directory for preview PNGs",
    )
    preview_parser.add_argument(
        "--image-key",
        default=None,
        help="State key for preview images (default: first observation.images.* key)",
    )
    preview_parser.set_defaults(func=cmd_preview)

    compact_parser = sub.add_parser(
        "compact",
        help="Shrink buffer capacity to stored transitions and save a smaller .pt file.",
    )
    compact_parser.add_argument("--path", required=True, help="Path to offline_buffer.pt")
    compact_parser.add_argument(
        "--capacity",
        default=None,
        help="Target capacity (default: exact transition count). Use e.g. 5000 for 3462 transitions.",
    )
    compact_parser.add_argument(
        "--optimize-memory",
        action="store_true",
        help="Enable optimize_memory in the compacted buffer (skip duplicate next_state storage).",
    )
    compact_parser.add_argument(
        "--no-optimize-memory",
        action="store_true",
        help="Disable optimize_memory in the compacted buffer.",
    )
    compact_parser.add_argument(
        "--output",
        required=True,
        help="Output path, e.g. offline_buffer_compact.pt",
    )
    compact_parser.set_defaults(func=cmd_compact)

    filter_parser = sub.add_parser("filter", help="Remove episodes and save a new buffer.")
    filter_parser.add_argument("--path", required=True, help="Path to offline_buffer.pt")
    filter_parser.add_argument(
        "--exclude",
        default=None,
        help="Comma-separated episode indices to drop, e.g. '1,3,5'",
    )
    filter_parser.add_argument(
        "--include",
        default=None,
        help="Comma-separated episode indices to keep (alternative to --exclude)",
    )
    filter_parser.add_argument(
        "--output",
        required=True,
        help="Output path for filtered buffer, e.g. offline_buffer_clean.pt",
    )
    filter_parser.set_defaults(func=cmd_filter)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
