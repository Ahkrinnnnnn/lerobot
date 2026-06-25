#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Standalone real-robot reward classifier pipeline: inspect, annotate, export."""

import logging

from lerobot.configs import parser
from lerobot.rewards.classifier.pipeline_config import RewardClassifierPipelineConfig
from lerobot.rewards.classifier.pipeline import run_reward_classifier_pipeline
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


@parser.wrap()
def reward_classifier(cfg: RewardClassifierPipelineConfig) -> None:
    init_logging()
    logger.info("Reward classifier pipeline | mode=%s", cfg.mode)
    run_reward_classifier_pipeline(cfg)


def main() -> None:
    reward_classifier()


if __name__ == "__main__":
    main()
