# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.rewards.classifier.configuration_classifier import (
    RewardClassifierConfig,
    RewardClassifierImagePreprocessingConfig,
)
from lerobot.rewards.classifier.processor_classifier import make_classifier_processor


def test_classifier_processor_includes_crop_resize_step():
    config = RewardClassifierConfig(
        image_preprocessing=RewardClassifierImagePreprocessingConfig(
            crop_params_dict={"observation.images.top": (10, 20, 100, 120)},
            resize_size=(128, 128),
        ),
    )
    preprocessor, _ = make_classifier_processor(config)
    step_types = [step.__class__.__name__ for step in preprocessor.steps]
    assert step_types[0] == "ImageCropResizeProcessorStep"
    assert "NormalizerProcessorStep" in step_types


def test_reward_classifier_config_syncs_input_shapes_on_resize():
    config = RewardClassifierConfig(
        image_preprocessing=RewardClassifierImagePreprocessingConfig(resize_size=(128, 128)),
        input_features={
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        },
    )
    assert config.input_features["observation.images.top"].shape == (3, 128, 128)
