# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Configuration for reward-classifier data prep and real-robot deployment."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.rewards.classifier.configuration_classifier import RewardClassifierImagePreprocessingConfig


@dataclass
class RewardClassifierSourceConfig:
    """Read-only source demonstration dataset (never modified by this module)."""

    repo_id: str = "local/stack_the_base_all"
    root: str | None = None


@dataclass
class RewardClassifierOutputConfig:
    """All generated artifacts are written here — separate from ``source``."""

    # JSON segment labels from interactive annotation
    annotation_path: str = "outputs/reward_classifier/annotations/stack_the_base.json"
    # Exported labeled LeRobot dataset for lerobot-train
    dataset_repo_id: str = "local/stack_the_base_reward_classifier"
    dataset_root: str = "outputs/reward_classifier/datasets/stack_the_base_labeled"
    # Held-out evaluation set (HIL-SERL / PLD paper style)
    eval_dataset_repo_id: str = "local/stack_the_base_reward_classifier_eval"
    eval_dataset_root: str = "outputs/reward_classifier/datasets/stack_the_base_labeled_eval"
    eval_split_ratio: float = 0.2
    # Optional inspect report (JSON)
    report_path: str | None = None
    # build only: replace existing train/eval export directories
    overwrite: bool = False


@dataclass
class RewardClassifierPipelineConfig:
    """Top-level config for ``lerobot-reward-classifier`` CLI."""

    # inspect | annotate | build
    mode: str = "inspect"
    # build only: annotations | heuristic
    label_source: str = "annotations"

    source: RewardClassifierSourceConfig = field(default_factory=RewardClassifierSourceConfig)
    output: RewardClassifierOutputConfig = field(default_factory=RewardClassifierOutputConfig)

    episode_start: int = 0
    episode_indices: list[int] | None = None
    playback_fps: float | None = None
    max_frames_per_segment: int | None = None
    balance: bool = True
    # Positive:negative frame count ratio when exporting labeled data, e.g. [1, 5].
    # Overrides ``balance`` when set. Set to null with ``balance: false`` to keep all frames.
    class_ratio: tuple[int, int] | None = None

    # Export-time preprocessing (recommended: crop/resize here to keep labeled datasets small).
    image_preprocessing: RewardClassifierImagePreprocessingConfig | None = None
    # Cameras to export/train on. None = all cameras in the source dataset.
    camera_keys: list[str] | None = None
    # Reward-classifier exports are single-frame episodes; PNG storage avoids per-frame AV1 encoding.
    use_videos: bool = False

    # heuristic-only (fallback when manual labels are unavailable)
    success_tail_frames: int = 15
    negative_prefix_ratio: float = 0.5
    negatives_per_episode: int = 2
    max_episodes: int | None = None
    seed: int = 42


@dataclass
class RewardClassifierRuntimeConfig:
    """Runtime settings when using a trained classifier on the real robot (PLD, HIL-SERL, etc.)."""

    path: str | None = None
    device: str = "cuda"
    success_threshold: float = 0.5
    success_reward: float = 1.0
    terminate_on_success: bool = True
    manual_fallback: bool = True
    manual_success_key: str = "s"
    manual_failure_key: str = "f"
