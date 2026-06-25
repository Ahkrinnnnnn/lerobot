# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

import numpy as np
import pytest

from lerobot.rewards.classifier.annotations import EpisodeRewardAnnotation, RewardClassifierAnnotations
from lerobot.rewards.classifier.dataset import (
    _subsample_to_class_ratio,
    preprocess_export_image,
    select_demo_heuristic_frame_labels,
    select_manual_annotation_frame_labels,
    split_train_eval_samples,
)
from lerobot.rewards.classifier.dataset import _subsample_segment_frames
from lerobot.rewards.classifier.paths import assert_source_dataset_readonly


class FakeEpisode:
    def __init__(self, start: int, length: int):
        self._data = {
            "dataset_from_index": start,
            "dataset_to_index": start + length,
        }

    def __getitem__(self, key):
        return self._data[key]


def test_preprocess_export_image_crop_and_resize():
    from lerobot.rewards.classifier.configuration_classifier import RewardClassifierImagePreprocessingConfig

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[200:260, 300:360] = 255
    prep = RewardClassifierImagePreprocessingConfig(
        crop_params_dict={"observation.images.top": (0, 90, 240, 480)},
        resize_size=(128, 128),
    )
    out = preprocess_export_image(image, "observation.images.top", prep)
    assert out.shape == (128, 128, 3)


def test_select_demo_heuristic_frame_labels_balanced():
    episodes = [FakeEpisode(0, 100), FakeEpisode(100, 80)]
    samples = select_demo_heuristic_frame_labels(
        episodes,
        success_tail_frames=10,
        negative_prefix_ratio=0.5,
        negatives_per_episode=1,
        seed=0,
    )
    labels = [label for _, label in samples]
    assert len(labels) % 2 == 0
    assert sum(labels) == len(labels) / 2
    assert all(idx >= 0 for idx, _ in samples)


def test_manual_annotation_frame_labels_one_to_five_ratio():
    episodes = [FakeEpisode(0, 100)]
    ann = RewardClassifierAnnotations(
        source_repo_id="local/test",
        source_root="/tmp/test",
        fps=30,
        camera_keys=["observation.images.top"],
        episodes={
            "0": EpisodeRewardAnnotation(
                success_segments=[[0, 49]],
                failure_segments=[[50, 99]],
            ),
        },
    )
    positives = [(i, 1.0) for i in range(50)]
    negatives = [(i + 50, 0.0) for i in range(50)]
    rng = np.random.default_rng(0)
    samples = _subsample_to_class_ratio(positives, negatives, 1, 5, rng)
    labels = [label for _, label in samples]
    n_pos = sum(1 for label in labels if label == 1.0)
    n_neg = sum(1 for label in labels if label == 0.0)
    assert n_pos == 10
    assert n_neg == 50


def test_manual_annotation_frame_labels():
    episodes = [FakeEpisode(0, 100), FakeEpisode(100, 50)]
    ann = RewardClassifierAnnotations(
        source_repo_id="local/test",
        source_root="/tmp/test",
        fps=30,
        camera_keys=["observation.images.top"],
        episodes={
            "0": EpisodeRewardAnnotation(success_segments=[[90, 99]], failure_segments=[[0, 10]]),
            "1": EpisodeRewardAnnotation(success_segments=[[40, 45]], failure_segments=[[5, 8]]),
        },
    )
    samples = select_manual_annotation_frame_labels(ann, episodes, class_ratio=(1, 1), seed=0)
    pos = [idx for idx, label in samples if label == 1.0]
    neg = [idx for idx, label in samples if label == 0.0]
    assert len(pos) == len(neg)
    assert 90 in pos and 0 in neg


def test_annotation_save_load(tmp_path):
    path = tmp_path / "ann.json"
    ann = RewardClassifierAnnotations(
        source_repo_id="local/test",
        source_root="/tmp/test",
        episodes={"3": EpisodeRewardAnnotation(success_segments=[[1, 5]], failure_segments=[[10, 12]])},
    )
    ann.save(path)
    loaded = RewardClassifierAnnotations.load(path)
    assert loaded.episodes["3"].success_segments == [[1, 5]]


def test_split_train_eval_samples():
    samples = [(i, float(i % 2)) for i in range(100)]
    train, eval_ = split_train_eval_samples(samples, eval_ratio=0.2, seed=0)
    assert len(train) == 80
    assert len(eval_) == 20
    assert len(set(train) & set(eval_)) == 0


def test_assert_source_readonly_blocks_output_inside_source(tmp_path):
    source = tmp_path / "source_ds"
    source.mkdir()
    bad_output = source / "labeled"
    with pytest.raises(ValueError, match="must not be inside"):
        assert_source_dataset_readonly(source, output_dataset_root=bad_output)


def test_assert_source_readonly_allows_external_output(tmp_path):
    source = tmp_path / "source_ds"
    source.mkdir()
    good_output = tmp_path / "outputs" / "labeled"
    assert_source_dataset_readonly(source, output_dataset_root=good_output)
