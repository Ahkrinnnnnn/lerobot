# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Orchestration for inspect / annotate / build reward-classifier datasets."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .annotator import AnnotatorConfig, run_reward_classifier_annotator
from .annotations import RewardClassifierAnnotations
from .pipeline_config import RewardClassifierPipelineConfig
from .dataset import (
    build_balanced_reward_classifier_dataset,
    build_reward_classifier_dataset_from_annotations,
    inspect_reward_classifier_dataset,
)
from .paths import assert_source_dataset_readonly

logger = logging.getLogger(__name__)


def run_reward_classifier_pipeline(cfg: RewardClassifierPipelineConfig) -> None:
    source_root = cfg.source.root
    if source_root is None and cfg.mode in ("annotate", "build"):
        raise ValueError("source.root is required for annotate/build modes.")

    if source_root is not None:
        assert_source_dataset_readonly(
            source_root,
            output_dataset_root=cfg.output.dataset_root if cfg.mode == "build" else None,
            annotation_path=cfg.output.annotation_path if cfg.mode in ("annotate", "build") else None,
            report_path=cfg.output.report_path,
        )
        if cfg.mode == "build":
            assert_source_dataset_readonly(source_root, output_dataset_root=cfg.output.eval_dataset_root)

    if cfg.mode == "annotate":
        if source_root is None:
            raise ValueError("source.root is required for annotate mode.")
        run_reward_classifier_annotator(
            AnnotatorConfig(
                source_repo_id=cfg.source.repo_id,
                source_root=source_root,
                annotation_path=cfg.output.annotation_path,
                episode_start=cfg.episode_start,
                episode_indices=cfg.episode_indices,
                playback_fps=cfg.playback_fps,
            )
        )
        ann = RewardClassifierAnnotations.load(cfg.output.annotation_path)
        logger.info("Annotation summary: %s", json.dumps(ann.to_dict(), indent=2, ensure_ascii=False))
        return

    inspection = inspect_reward_classifier_dataset(cfg.source.repo_id, root=source_root)
    report = inspection.to_dict()
    logger.info("Dataset inspection:\n%s", json.dumps(report, indent=2, ensure_ascii=False))

    if cfg.output.report_path:
        Path(cfg.output.report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.output.report_path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Wrote inspection report to %s", cfg.output.report_path)

    if cfg.mode == "inspect":
        if not inspection.ready_for_training:
            logger.warning("Dataset is NOT ready for reward-classifier training.")
            for rec in inspection.recommendations:
                logger.warning("  - %s", rec)
        else:
            logger.info("Dataset is ready for reward-classifier training.")
        return

    if cfg.mode != "build":
        raise ValueError(f"Unknown mode '{cfg.mode}'. Use 'inspect', 'annotate', or 'build'.")

    if source_root is None:
        raise ValueError("source.root is required for build mode.")

    if cfg.label_source == "annotations":
        if not Path(cfg.output.annotation_path).is_file():
            raise FileNotFoundError(
                f"Annotation file not found: {cfg.output.annotation_path}. "
                "Run annotate mode first (see examples/reward_classifier/*_annotate.json)."
            )
        build_reward_classifier_dataset_from_annotations(
            source_repo_id=cfg.source.repo_id,
            source_root=source_root,
            annotation_path=cfg.output.annotation_path,
            output_repo_id=cfg.output.dataset_repo_id,
            output_root=cfg.output.dataset_root,
            eval_output_repo_id=cfg.output.eval_dataset_repo_id,
            eval_output_root=cfg.output.eval_dataset_root,
            eval_split_ratio=cfg.output.eval_split_ratio,
            max_frames_per_segment=cfg.max_frames_per_segment,
            balance=cfg.balance,
            class_ratio=cfg.class_ratio,
            camera_keys=cfg.camera_keys,
            image_preprocessing=cfg.image_preprocessing,
            use_videos=cfg.use_videos,
            seed=cfg.seed,
            overwrite=cfg.output.overwrite,
        )
    elif cfg.label_source == "heuristic":
        build_balanced_reward_classifier_dataset(
            source_repo_id=cfg.source.repo_id,
            source_root=source_root,
            output_repo_id=cfg.output.dataset_repo_id,
            output_root=cfg.output.dataset_root,
            eval_output_repo_id=cfg.output.eval_dataset_repo_id,
            eval_output_root=cfg.output.eval_dataset_root,
            eval_split_ratio=cfg.output.eval_split_ratio,
            success_tail_frames=cfg.success_tail_frames,
            negative_prefix_ratio=cfg.negative_prefix_ratio,
            negatives_per_episode=cfg.negatives_per_episode,
            max_episodes=cfg.max_episodes,
            class_ratio=cfg.class_ratio if cfg.class_ratio is not None else (1, 1),
            camera_keys=cfg.camera_keys,
            image_preprocessing=cfg.image_preprocessing,
            use_videos=cfg.use_videos,
            seed=cfg.seed,
            overwrite=cfg.output.overwrite,
        )
    else:
        raise ValueError(f"Unknown label_source '{cfg.label_source}'. Use 'annotations' or 'heuristic'.")

    out_inspection = inspect_reward_classifier_dataset(
        cfg.output.dataset_repo_id,
        root=cfg.output.dataset_root,
    )
    eval_inspection = inspect_reward_classifier_dataset(
        cfg.output.eval_dataset_repo_id,
        root=cfg.output.eval_dataset_root,
    )
    logger.info(
        "Built train dataset ready_for_training=%s at %s",
        out_inspection.ready_for_training,
        cfg.output.dataset_root,
    )
    logger.info(
        "Built eval dataset ready_for_training=%s at %s (eval_split_ratio=%.2f)",
        eval_inspection.ready_for_training,
        cfg.output.eval_dataset_root,
        cfg.output.eval_split_ratio,
    )
