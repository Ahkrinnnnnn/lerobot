# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Evaluate a binary reward classifier on a held-out dataset."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from .modeling_classifier import Classifier
from lerobot.rewards.pretrained import PreTrainedRewardModel


@dataclass
class RewardClassifierEvalMetrics:
    """Classification metrics on a held-out evaluation dataset."""

    accuracy: float
    loss: float
    num_samples: int
    num_correct: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "eval_accuracy": self.accuracy,
            "eval_loss": self.loss,
            "eval_num_samples": self.num_samples,
            "eval_num_correct": self.num_correct,
        }


@torch.inference_mode()
def evaluate_reward_classifier(
    model: PreTrainedRewardModel,
    dataloader: DataLoader,
    preprocessor,
    *,
    device: torch.device | str | None = None,
) -> RewardClassifierEvalMetrics:
    """Compute average loss and classification accuracy on an evaluation dataset."""
    if not isinstance(model, Classifier):
        raise TypeError(f"Expected Classifier, got {type(model)}")

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_batches = 0

    for batch in dataloader:
        for key, value in batch.items():
            if "image" in key and isinstance(value, torch.Tensor) and value.dtype == torch.uint8:
                batch[key] = value.to(dtype=torch.float32) / 255.0
        batch = preprocessor(batch)
        loss, output_dict = model.forward(batch)
        batch_size = int(output_dict["total"])
        total_loss += loss.item()
        total_correct += int(output_dict["correct"])
        total_samples += batch_size
        num_batches += 1

    if total_samples == 0:
        raise ValueError("Evaluation dataloader is empty.")

    return RewardClassifierEvalMetrics(
        accuracy=100.0 * total_correct / total_samples,
        loss=total_loss / max(num_batches, 1),
        num_samples=total_samples,
        num_correct=total_correct,
    )
