# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""JSON schema and helpers for manual reward-classifier segment annotations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EpisodeRewardAnnotation:
    """Per-episode success/failure segments in episode-local frame indices (inclusive)."""

    success_segments: list[list[int]] = field(default_factory=list)
    failure_segments: list[list[int]] = field(default_factory=list)

    def add_success_segment(self, start: int, end: int) -> None:
        if start > end:
            start, end = end, start
        self.success_segments.append([start, end])

    def add_failure_segment(self, start: int, end: int) -> None:
        if start > end:
            start, end = end, start
        self.failure_segments.append([start, end])

    def undo_last(self) -> None:
        if self.success_segments:
            self.success_segments.pop()
        elif self.failure_segments:
            self.failure_segments.pop()

    def is_empty(self) -> bool:
        return not self.success_segments and not self.failure_segments


@dataclass
class RewardClassifierAnnotations:
    """Manual labels exported from the interactive video annotator."""

    version: int = 1
    source_repo_id: str = ""
    source_root: str = ""
    fps: int = 30
    camera_keys: list[str] = field(default_factory=list)
    episodes: dict[str, EpisodeRewardAnnotation] = field(default_factory=dict)

    def get_episode(self, ep_idx: int) -> EpisodeRewardAnnotation:
        key = str(ep_idx)
        if key not in self.episodes:
            self.episodes[key] = EpisodeRewardAnnotation()
        return self.episodes[key]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "source_repo_id": self.source_repo_id,
            "source_root": self.source_root,
            "fps": self.fps,
            "camera_keys": self.camera_keys,
            "episodes": {
                ep_key: asdict(ep_ann) for ep_key, ep_ann in self.episodes.items() if not ep_ann.is_empty()
            },
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> RewardClassifierAnnotations:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        episodes = {
            ep_key: EpisodeRewardAnnotation(**ep_data) for ep_key, ep_data in payload.get("episodes", {}).items()
        }
        return cls(
            version=payload.get("version", 1),
            source_repo_id=payload.get("source_repo_id", ""),
            source_root=payload.get("source_root", ""),
            fps=payload.get("fps", 30),
            camera_keys=list(payload.get("camera_keys", [])),
            episodes=episodes,
        )

    @classmethod
    def load_or_create(
        cls,
        path: str | Path,
        *,
        source_repo_id: str,
        source_root: str,
        fps: int,
        camera_keys: list[str],
    ) -> RewardClassifierAnnotations:
        path = Path(path)
        if path.is_file():
            ann = cls.load(path)
            ann.source_repo_id = source_repo_id
            ann.source_root = source_root
            ann.fps = fps
            ann.camera_keys = camera_keys
            return ann
        return cls(
            source_repo_id=source_repo_id,
            source_root=source_root,
            fps=fps,
            camera_keys=camera_keys,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source_repo_id": self.source_repo_id,
            "source_root": self.source_root,
            "fps": self.fps,
            "camera_keys": self.camera_keys,
            "num_episodes_labeled": sum(1 for ep in self.episodes.values() if not ep.is_empty()),
            "total_success_segments": sum(len(ep.success_segments) for ep in self.episodes.values()),
            "total_failure_segments": sum(len(ep.failure_segments) for ep in self.episodes.values()),
        }
