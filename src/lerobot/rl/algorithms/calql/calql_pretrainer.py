# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

import logging
from collections.abc import Iterator

from lerobot.rl.algorithms.residual_sac.residual_sac_algorithm import ResidualSACAlgorithm
from lerobot.types import BatchType

logger = logging.getLogger(__name__)


class CalQLPretrainer:
    """Cal-QL critic pre-training on offline buffer (PLD Algorithm 1 initialization)."""

    def __init__(self, algorithm: ResidualSACAlgorithm, calql_alpha: float = 1.0):
        self.algorithm = algorithm
        self.calql_alpha = calql_alpha

    def pretrain(
        self,
        batch_iterator: Iterator[BatchType],
        num_steps: int,
        log_interval: int = 1000,
    ) -> list[float]:
        """Run Cal-QL critic updates for ``num_steps`` gradient steps."""
        losses: list[float] = []
        self.algorithm.config.use_calql = True
        self.algorithm.config.calql_alpha = self.calql_alpha

        for step in range(num_steps):
            batch = next(batch_iterator)
            stats = self.algorithm.update_critic_only(batch)
            loss = stats.losses["loss_critic"]
            losses.append(loss)
            if (step + 1) % log_interval == 0:
                logger.info("Cal-QL pretrain step %d/%d | loss_critic=%.4f", step + 1, num_steps, loss)

        self.algorithm.config.use_calql = False
        return losses
