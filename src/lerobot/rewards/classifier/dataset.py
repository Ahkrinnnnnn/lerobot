# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Inspect and export reward-classifier training datasets (source datasets are read-only)."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as F

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import REWARD

from .annotations import RewardClassifierAnnotations
from .configuration_classifier import RewardClassifierImagePreprocessingConfig
from .paths import assert_source_dataset_readonly

logger = logging.getLogger(__name__)


@dataclass
class RewardDatasetInspection:
    """Summary of whether a dataset can train a binary reward classifier."""

    repo_id: str
    root: str
    total_episodes: int
    total_frames: int
    fps: int
    robot_type: str | None
    camera_keys: list[str]
    has_reward: bool
    reward_key: str | None
    task: str | None
    ready_for_training: bool
    issues: list[str]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "root": self.root,
            "total_episodes": self.total_episodes,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "robot_type": self.robot_type,
            "camera_keys": self.camera_keys,
            "has_reward": self.has_reward,
            "reward_key": self.reward_key,
            "task": self.task,
            "ready_for_training": self.ready_for_training,
            "issues": self.issues,
            "recommendations": self.recommendations,
        }


def inspect_reward_classifier_dataset(
    repo_id: str,
    root: str | Path | None = None,
) -> RewardDatasetInspection:
    """Check if a LeRobot dataset contains the fields needed for reward-classifier training."""
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    issues: list[str] = []
    recommendations: list[str] = []

    camera_keys = list(meta.camera_keys)
    if len(camera_keys) == 0:
        issues.append("No camera features found (expected observation.images.*).")

    reward_key = REWARD if REWARD in meta.features else None
    if reward_key is None and "reward" in meta.features:
        reward_key = "reward"

    has_reward = reward_key is not None
    if not has_reward:
        issues.append(
            f"Missing per-frame success label field '{REWARD}'. "
            "Binary reward classifiers require success/failure labels on each frame."
        )
        recommendations.append(
            "Run `lerobot-reward-classifier --config_path=examples/reward_classifier/<task>_annotate.json` "
            "to mark success/failure segments, then export with `<task>_export.json`."
        )

    task = None
    if meta.tasks is not None and len(meta.tasks) > 0:
        task = str(meta.tasks.index[0])

    ready = len(camera_keys) > 0 and has_reward

    return RewardDatasetInspection(
        repo_id=repo_id,
        root=str(meta.root),
        total_episodes=meta.total_episodes,
        total_frames=meta.total_frames,
        fps=meta.fps,
        robot_type=meta.robot_type,
        camera_keys=camera_keys,
        has_reward=has_reward,
        reward_key=reward_key,
        task=task,
        ready_for_training=ready,
        issues=issues,
        recommendations=recommendations,
    )


def _subsample_segment_frames(start: int, end: int, max_frames: int | None, rng: np.random.Generator) -> list[int]:
    frames = list(range(start, end + 1))
    if max_frames is None or len(frames) <= max_frames:
        return frames
    picks = rng.choice(len(frames), size=max_frames, replace=False)
    return sorted(frames[i] for i in picks)


def _frames_from_episode_segments(
    ep_start: int,
    segments: list[list[int]],
    *,
    label: float,
    max_frames_per_segment: int | None,
    rng: np.random.Generator,
) -> list[tuple[int, float]]:
    samples: list[tuple[int, float]] = []
    for seg in segments:
        if len(seg) != 2:
            continue
        seg_start, seg_end = int(seg[0]), int(seg[1])
        if seg_start > seg_end:
            seg_start, seg_end = seg_end, seg_start
        for local_idx in _subsample_segment_frames(seg_start, seg_end, max_frames_per_segment, rng):
            samples.append((ep_start + local_idx, label))
    return samples


def _subsample_to_class_ratio(
    positives: list[tuple[int, float]],
    negatives: list[tuple[int, float]],
    pos_ratio: int,
    neg_ratio: int,
    rng: np.random.Generator,
) -> list[tuple[int, float]]:
    """Subsample frame labels to a target positive:negative count ratio (e.g. 1:5)."""
    pos_ratio = int(pos_ratio)
    neg_ratio = int(neg_ratio)
    if pos_ratio <= 0 or neg_ratio <= 0:
        raise ValueError(f"class_ratio components must be positive, got ({pos_ratio}, {neg_ratio})")
    if not positives or not negatives:
        raise ValueError("Need both positive and negative samples to apply class_ratio.")

    n_pos = min(len(positives), (len(negatives) * pos_ratio) // neg_ratio)
    if n_pos == 0:
        raise ValueError(
            f"Not enough negative samples for class_ratio {pos_ratio}:{neg_ratio} "
            f"({len(positives)} pos / {len(negatives)} neg)."
        )
    n_neg = min(len(negatives), (n_pos * neg_ratio) // pos_ratio)
    if n_neg == 0:
        raise ValueError(
            f"Not enough positive samples for class_ratio {pos_ratio}:{neg_ratio} "
            f"({len(positives)} pos / {len(negatives)} neg)."
        )

    pos_idx = rng.choice(len(positives), size=n_pos, replace=False)
    neg_idx = rng.choice(len(negatives), size=n_neg, replace=False)
    samples = [positives[i] for i in pos_idx] + [negatives[i] for i in neg_idx]
    rng.shuffle(samples)
    logger.info(
        "Subsampled reward classifier frames to %d:%d (pos:neg) -> %d pos / %d neg",
        pos_ratio,
        neg_ratio,
        n_pos,
        n_neg,
    )
    return samples


def _resolve_class_ratio(
    *,
    balance: bool,
    class_ratio: tuple[int, int] | list[int] | None,
) -> tuple[int, int] | None:
    """Return ``(pos_ratio, neg_ratio)`` or ``None`` to keep all labeled frames."""
    if class_ratio is not None:
        if len(class_ratio) != 2:
            raise ValueError(f"class_ratio must have length 2 [positive, negative], got {class_ratio}")
        return int(class_ratio[0]), int(class_ratio[1])
    if balance:
        return 1, 1
    return None


def select_manual_annotation_frame_labels(
    annotations: RewardClassifierAnnotations,
    episodes,
    *,
    max_frames_per_segment: int | None = None,
    balance: bool = True,
    class_ratio: tuple[int, int] | list[int] | None = None,
    seed: int = 42,
) -> list[tuple[int, float]]:
    """Convert manual segment annotations into global frame indices with binary labels."""
    rng = np.random.default_rng(seed)
    positives: list[tuple[int, float]] = []
    negatives: list[tuple[int, float]] = []

    for ep_key, ep_ann in annotations.episodes.items():
        ep_idx = int(ep_key)
        if ep_idx < 0 or ep_idx >= len(episodes):
            logger.warning("Skipping unknown episode index %s in annotations.", ep_key)
            continue
        ep = episodes[ep_idx]
        ep_start = int(ep["dataset_from_index"])
        positives.extend(
            _frames_from_episode_segments(
                ep_start,
                ep_ann.success_segments,
                label=1.0,
                max_frames_per_segment=max_frames_per_segment,
                rng=rng,
            )
        )
        negatives.extend(
            _frames_from_episode_segments(
                ep_start,
                ep_ann.failure_segments,
                label=0.0,
                max_frames_per_segment=max_frames_per_segment,
                rng=rng,
            )
        )

    if not positives:
        raise ValueError(
            "No success segments found in annotations. "
            "Use annotate mode and mark success segments with `[` and `]`."
        )
    if not negatives:
        raise ValueError(
            "No failure segments found in annotations. "
            "Mark failure segments with `f` and `g`."
        )

    ratio = _resolve_class_ratio(balance=balance, class_ratio=class_ratio)
    if ratio is not None:
        return _subsample_to_class_ratio(positives, negatives, ratio[0], ratio[1], rng)

    samples = positives + negatives
    rng.shuffle(samples)
    return samples


def select_demo_heuristic_frame_labels(
    episodes,
    *,
    success_tail_frames: int = 15,
    negative_prefix_ratio: float = 0.5,
    negatives_per_episode: int = 2,
    max_episodes: int | None = None,
    class_ratio: tuple[int, int] | list[int] | None = (1, 1),
    seed: int = 42,
) -> list[tuple[int, float]]:
    """Pick success/failure frame indices from successful demonstrations (fallback)."""
    rng = np.random.default_rng(seed)
    positives: list[tuple[int, float]] = []
    negatives: list[tuple[int, float]] = []

    n_episodes = len(episodes) if max_episodes is None else min(len(episodes), max_episodes)
    for ep_idx in range(n_episodes):
        ep = episodes[ep_idx]
        start = int(ep["dataset_from_index"])
        end = int(ep["dataset_to_index"])
        ep_len = end - start
        if ep_len <= 0:
            continue

        tail = min(success_tail_frames, ep_len)
        for offset in range(ep_len - tail, ep_len):
            positives.append((start + offset, 1.0))

        neg_end = max(1, int(ep_len * negative_prefix_ratio))
        neg_end = min(neg_end, max(1, ep_len - tail))
        for _ in range(negatives_per_episode):
            frame_in_ep = int(rng.integers(0, neg_end))
            negatives.append((start + frame_in_ep, 0.0))

    if not positives or not negatives:
        raise ValueError("Could not derive any positive/negative frames from the source dataset.")

    if class_ratio is None:
        samples = positives + negatives
        rng.shuffle(samples)
        return samples

    return _subsample_to_class_ratio(positives, negatives, int(class_ratio[0]), int(class_ratio[1]), rng)


def _resolve_export_camera_keys(
    source_meta: LeRobotDatasetMetadata,
    camera_keys: list[str] | None,
) -> list[str]:
    available = list(source_meta.camera_keys)
    if not available:
        raise ValueError("Source dataset has no camera keys.")
    if camera_keys is None:
        return available
    missing = [key for key in camera_keys if key not in available]
    if missing:
        raise ValueError(f"camera_keys not found in source dataset: {missing}. Available: {available}")
    return list(camera_keys)


def _export_visual_feature_shape(
    source_shape: tuple | list,
    cam_key: str,
    image_preprocessing: RewardClassifierImagePreprocessingConfig | None,
) -> tuple[int, int, int]:
    if len(source_shape) == 3 and source_shape[0] in (1, 3):
        channels = int(source_shape[0])
        height, width = int(source_shape[1]), int(source_shape[2])
    elif len(source_shape) == 3:
        height, width, channels = (int(x) for x in source_shape)
    else:
        raise ValueError(f"Unexpected camera feature shape {source_shape} for {cam_key}")

    if image_preprocessing is not None and image_preprocessing.crop_params_dict:
        crop = image_preprocessing.crop_params_dict.get(cam_key)
        if crop is not None:
            height, width = int(crop[2]), int(crop[3])
    if image_preprocessing is not None and image_preprocessing.resize_size is not None:
        height, width = (int(x) for x in image_preprocessing.resize_size)
    return height, width, channels


def _build_export_visual_features(
    source_meta: LeRobotDatasetMetadata,
    camera_keys: list[str],
    *,
    image_preprocessing: RewardClassifierImagePreprocessingConfig | None = None,
    use_videos: bool = False,
) -> dict[str, dict]:
    features: dict[str, dict] = {}
    for key in camera_keys:
        src_feat = dict(source_meta.features[key])
        h, w, c = _export_visual_feature_shape(tuple(src_feat["shape"]), key, image_preprocessing)
        src_feat["shape"] = (h, w, c)
        if use_videos:
            src_feat["dtype"] = "video"
        else:
            src_feat["dtype"] = "image"
            src_feat.pop("info", None)
        features[key] = src_feat
    return features


def preprocess_export_image(
    value: np.ndarray,
    cam_key: str,
    image_preprocessing: RewardClassifierImagePreprocessingConfig | None,
) -> np.ndarray:
    """Apply optional crop/resize while exporting labeled frames (HWC uint8)."""
    if image_preprocessing is None or (
        image_preprocessing.resize_size is None and not image_preprocessing.crop_params_dict
    ):
        return value

    if value.ndim == 3 and value.shape[-1] in (1, 3):
        tensor = torch.from_numpy(value).permute(2, 0, 1).float()
    elif value.ndim == 3:
        tensor = torch.from_numpy(value).float()
        if tensor.shape[0] in (1, 3):
            pass
        else:
            raise ValueError(f"Unexpected image array shape {value.shape} for {cam_key}")
    else:
        raise ValueError(f"Unexpected image array shape {value.shape} for {cam_key}")

    if tensor.max() > 1.0:
        tensor = tensor / 255.0

    if image_preprocessing.crop_params_dict and cam_key in image_preprocessing.crop_params_dict:
        tensor = F.crop(tensor, *image_preprocessing.crop_params_dict[cam_key])
    if image_preprocessing.resize_size is not None:
        tensor = F.resize(tensor, list(image_preprocessing.resize_size))

    return tensor.clamp(0.0, 1.0).mul(255.0).byte().permute(1, 2, 0).contiguous().numpy()


def _frame_to_dataset_dict(
    item: dict[str, Any],
    camera_keys: list[str],
    reward: float,
    task: str,
    image_preprocessing: RewardClassifierImagePreprocessingConfig | None = None,
) -> dict[str, Any]:
    frame: dict[str, Any] = {"task": task, REWARD: np.array([reward], dtype=np.float32)}
    for cam_key in camera_keys:
        value = item[cam_key]
        if hasattr(value, "cpu"):
            value = value.cpu().numpy()
        if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[0] in (1, 3):
            value = np.transpose(value, (1, 2, 0))
        value = preprocess_export_image(value, cam_key, image_preprocessing)
        frame[cam_key] = value
    return frame


def split_train_eval_samples(
    samples: list[tuple[int, float]],
    *,
    eval_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Hold out a random subset of labeled frames for evaluation (HIL-SERL style)."""
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError(f"eval_ratio must be in (0, 1), got {eval_ratio}")
    if len(samples) < 2:
        raise ValueError("Need at least 2 labeled samples to create train/eval split.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(samples))
    n_eval = max(1, int(round(len(samples) * eval_ratio)))
    n_eval = min(n_eval, len(samples) - 1)
    eval_mask = set(perm[:n_eval].tolist())
    train_samples = [sample for idx, sample in enumerate(samples) if idx not in eval_mask]
    eval_samples = [sample for idx, sample in enumerate(samples) if idx in eval_mask]
    return train_samples, eval_samples


def _ensure_output_dataset_root_available(output_root: Path, *, overwrite: bool) -> None:
    """Validate export target before ``LeRobotDataset.create`` (which requires a missing directory)."""
    if not output_root.exists():
        return
    if overwrite:
        logger.warning("overwrite=true: removing existing reward classifier export at %s", output_root)
        shutil.rmtree(output_root)
        return
    raise FileExistsError(
        f"Output dataset directory already exists: {output_root}. "
        "LeRobotDataset.create requires a fresh directory. "
        "Remove it manually or set output.overwrite=true in your export config."
    )


def export_reward_classifier_dataset(
    source_repo_id: str,
    source_root: str | Path,
    output_repo_id: str,
    output_root: str | Path,
    samples: list[tuple[int, float]],
    *,
    camera_keys: list[str] | None = None,
    image_preprocessing: RewardClassifierImagePreprocessingConfig | None = None,
    use_videos: bool = False,
    overwrite: bool = False,
) -> LeRobotDataset:
    """Write extracted labeled frames into a **new** dataset (source is read-only)."""
    source_root = Path(source_root)
    output_root = Path(output_root)
    assert_source_dataset_readonly(source_root, output_dataset_root=output_root)

    source_meta = LeRobotDatasetMetadata(source_repo_id, root=source_root)
    task = (
        str(source_meta.tasks.index[0])
        if source_meta.tasks is not None and len(source_meta.tasks) > 0
        else "reward_classifier"
    )
    export_camera_keys = _resolve_export_camera_keys(source_meta, camera_keys)
    features = _build_export_visual_features(
        source_meta,
        export_camera_keys,
        image_preprocessing=image_preprocessing,
        use_videos=use_videos,
    )
    features[REWARD] = {"dtype": "float32", "shape": (1,), "names": None}

    if image_preprocessing is not None:
        logger.info(
            "Export image preprocessing enabled for cameras %s (use_videos=%s)",
            export_camera_keys,
            use_videos,
        )
    else:
        logger.info("Exporting full-resolution frames for cameras %s", export_camera_keys)

    _ensure_output_dataset_root_available(output_root, overwrite=overwrite)

    out_ds = LeRobotDataset.create(
        output_repo_id,
        source_meta.fps,
        root=output_root,
        robot_type=source_meta.robot_type,
        features=features,
        use_videos=use_videos,
        image_writer_threads=4,
        image_writer_processes=0,
    )
    out_ds.writer.start_image_writer(num_processes=0, num_threads=4)

    source = LeRobotDataset(source_repo_id, root=source_root, return_uint8=True)
    n_pos = sum(1 for _, reward in samples if reward >= 0.5)
    n_neg = len(samples) - n_pos
    logger.info(
        "Exporting reward dataset (read-only source=%s): %d samples (%d pos / %d neg) -> %s",
        source_root,
        len(samples),
        n_pos,
        n_neg,
        output_root,
    )

    for global_idx, reward in samples:
        item = source[global_idx]
        frame = _frame_to_dataset_dict(
            item,
            export_camera_keys,
            reward,
            task,
            image_preprocessing=image_preprocessing,
        )
        out_ds.add_frame(frame)
        out_ds.save_episode()

    out_ds.finalize()
    logger.info("Saved reward classifier dataset to %s (%d frames)", output_root, len(samples))
    return out_ds


def export_train_eval_reward_classifier_datasets(
    source_repo_id: str,
    source_root: str | Path,
    train_output_repo_id: str,
    train_output_root: str | Path,
    eval_output_repo_id: str,
    eval_output_root: str | Path,
    samples: list[tuple[int, float]],
    *,
    eval_ratio: float = 0.2,
    camera_keys: list[str] | None = None,
    image_preprocessing: RewardClassifierImagePreprocessingConfig | None = None,
    use_videos: bool = False,
    seed: int = 42,
    overwrite: bool = False,
) -> tuple[LeRobotDataset, LeRobotDataset]:
    """Export separate train and evaluation datasets from labeled frame samples."""
    train_samples, eval_samples = split_train_eval_samples(samples, eval_ratio=eval_ratio, seed=seed)
    logger.info(
        "Train/eval split: %d train / %d eval frames (eval_ratio=%.2f)",
        len(train_samples),
        len(eval_samples),
        eval_ratio,
    )
    train_ds = export_reward_classifier_dataset(
        source_repo_id,
        source_root,
        train_output_repo_id,
        train_output_root,
        train_samples,
        camera_keys=camera_keys,
        image_preprocessing=image_preprocessing,
        use_videos=use_videos,
        overwrite=overwrite,
    )
    eval_ds = export_reward_classifier_dataset(
        source_repo_id,
        source_root,
        eval_output_repo_id,
        eval_output_root,
        eval_samples,
        camera_keys=camera_keys,
        image_preprocessing=image_preprocessing,
        use_videos=use_videos,
        overwrite=overwrite,
    )
    return train_ds, eval_ds


def build_reward_classifier_dataset_from_annotations(
    source_repo_id: str,
    source_root: str | Path,
    annotation_path: str | Path,
    output_repo_id: str,
    output_root: str | Path,
    *,
    eval_output_repo_id: str | None = None,
    eval_output_root: str | Path | None = None,
    eval_split_ratio: float = 0.2,
    max_frames_per_segment: int | None = None,
    balance: bool = True,
    class_ratio: tuple[int, int] | list[int] | None = None,
    camera_keys: list[str] | None = None,
    image_preprocessing: RewardClassifierImagePreprocessingConfig | None = None,
    use_videos: bool = False,
    seed: int = 42,
    overwrite: bool = False,
) -> LeRobotDataset | tuple[LeRobotDataset, LeRobotDataset]:
    """Build train (and optional eval) datasets from manually annotated segments."""
    assert_source_dataset_readonly(
        source_root,
        output_dataset_root=output_root,
        annotation_path=annotation_path,
    )
    if eval_output_root is not None:
        assert_source_dataset_readonly(source_root, output_dataset_root=eval_output_root)

    source_meta = LeRobotDatasetMetadata(source_repo_id, root=source_root)
    source_meta.ensure_readable()
    if source_meta.episodes is None:
        raise RuntimeError("Episode metadata is unavailable for the source dataset.")

    annotations = RewardClassifierAnnotations.load(annotation_path)
    samples = select_manual_annotation_frame_labels(
        annotations,
        source_meta.episodes,
        max_frames_per_segment=max_frames_per_segment,
        balance=balance,
        class_ratio=class_ratio,
        seed=seed,
    )
    if eval_output_repo_id and eval_output_root:
        return export_train_eval_reward_classifier_datasets(
            source_repo_id,
            source_root,
            output_repo_id,
            output_root,
            eval_output_repo_id,
            eval_output_root,
            samples,
            eval_ratio=eval_split_ratio,
            camera_keys=camera_keys,
            image_preprocessing=image_preprocessing,
            use_videos=use_videos,
            seed=seed,
            overwrite=overwrite,
        )
    return export_reward_classifier_dataset(
        source_repo_id,
        source_root,
        output_repo_id,
        output_root,
        samples,
        camera_keys=camera_keys,
        image_preprocessing=image_preprocessing,
        use_videos=use_videos,
        overwrite=overwrite,
    )


def build_balanced_reward_classifier_dataset(
    source_repo_id: str,
    source_root: str | Path,
    output_repo_id: str,
    output_root: str | Path,
    *,
    eval_output_repo_id: str | None = None,
    eval_output_root: str | Path | None = None,
    eval_split_ratio: float = 0.2,
    success_tail_frames: int = 15,
    negative_prefix_ratio: float = 0.5,
    negatives_per_episode: int = 2,
    max_episodes: int | None = None,
    class_ratio: tuple[int, int] | list[int] | None = (1, 1),
    camera_keys: list[str] | None = None,
    image_preprocessing: RewardClassifierImagePreprocessingConfig | None = None,
    use_videos: bool = False,
    seed: int = 42,
    overwrite: bool = False,
) -> LeRobotDataset | tuple[LeRobotDataset, LeRobotDataset]:
    """Create train (and optional eval) datasets using demo-tail heuristics."""
    assert_source_dataset_readonly(source_root, output_dataset_root=output_root)
    if eval_output_root is not None:
        assert_source_dataset_readonly(source_root, output_dataset_root=eval_output_root)

    source_meta = LeRobotDatasetMetadata(source_repo_id, root=source_root)
    source_meta.ensure_readable()
    if source_meta.episodes is None:
        raise RuntimeError("Episode metadata is unavailable for the source dataset.")

    samples = select_demo_heuristic_frame_labels(
        source_meta.episodes,
        success_tail_frames=success_tail_frames,
        negative_prefix_ratio=negative_prefix_ratio,
        negatives_per_episode=negatives_per_episode,
        max_episodes=max_episodes,
        class_ratio=class_ratio,
        seed=seed,
    )
    if eval_output_repo_id and eval_output_root:
        return export_train_eval_reward_classifier_datasets(
            source_repo_id,
            source_root,
            output_repo_id,
            output_root,
            eval_output_repo_id,
            eval_output_root,
            samples,
            eval_ratio=eval_split_ratio,
            camera_keys=camera_keys,
            image_preprocessing=image_preprocessing,
            use_videos=use_videos,
            seed=seed,
            overwrite=overwrite,
        )
    return export_reward_classifier_dataset(
        source_repo_id,
        source_root,
        output_repo_id,
        output_root,
        samples,
        camera_keys=camera_keys,
        image_preprocessing=image_preprocessing,
        use_videos=use_videos,
        overwrite=overwrite,
    )
