# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Shared reward-classifier loading and inference (training deploy + HIL/PLD rollout)."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import OBS_IMAGE
from lerobot.utils.hub import resolve_local_model_path

logger = logging.getLogger(__name__)


def build_reward_classifier_batch(
    images: dict[str, Any],
    input_features: dict[str, Any],
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    """Map observation image dict keys to classifier ``input_features`` batch tensors (BCHW float)."""
    batch: dict[str, torch.Tensor] = {}
    for feat_key in input_features:
        if not feat_key.startswith(OBS_IMAGE):
            continue
        short = feat_key.replace("observation.images.", "")
        value = None
        for cand in (feat_key, short, f"observation.images.{short}"):
            if cand in images:
                value = images[cand]
                break
        if value is None:
            continue

        if isinstance(value, torch.Tensor):
            tensor = value
        else:
            tensor = torch.from_numpy(np.asarray(value))
        if tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255.0
        if tensor.ndim == 3:
            if tensor.shape[-1] in (1, 3):
                tensor = tensor.permute(2, 0, 1)
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 4 and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(0, 3, 1, 2)
        batch[feat_key] = tensor.to(device)
    return batch


def _move_batch_tensors_to_device(batch: dict[str, Any], device: str | torch.device) -> dict[str, Any]:
    """Move tensor values in a batch dict to ``device`` (preprocessor may leave tensors on CPU)."""
    target = torch.device(device)
    return {k: v.to(target) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


@dataclass
class RewardClassifierRuntime:
    """Loaded classifier + optional saved preprocessor for rollout inference."""

    classifier: Any
    preprocessor: PolicyProcessorPipeline | None = None
    device: str = "cuda"
    success_threshold: float = 0.5
    success_reward: float = 1.0
    terminate_on_success: bool = True
    last_success_prob: float | None = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str | Path,
        *,
        device: str = "cuda",
        success_threshold: float = 0.5,
        success_reward: float = 1.0,
        terminate_on_success: bool = True,
    ) -> RewardClassifierRuntime:
        from lerobot.rewards.classifier.modeling_classifier import Classifier

        model_path = resolve_local_model_path(pretrained_path)
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"Reward classifier directory not found: {model_path}\n"
                "Train it first (lerobot-train + examples/reward_classifier/stack_the_base_train.json), "
                "or remove reward_classifier.path and use manual_fallback (s=success, f=failure)."
            )

        classifier = Classifier.from_pretrained(str(model_path))
        classifier.to(device)
        classifier.eval()

        preprocessor = None
        for config_name in ("processor.json", "classifier_preprocessor.json"):
            processor_file = model_path / config_name
            if processor_file.is_file():
                preprocessor = PolicyProcessorPipeline.from_pretrained(
                    model_path,
                    config_filename=config_name,
                )
                logger.info("Loaded reward classifier preprocessor from %s", processor_file)
                break
        if preprocessor is None:
            logger.warning(
                "No reward classifier preprocessor found under %s — "
                "inference will skip crop/resize/normalize (success detection may fail).",
                model_path,
            )

        logger.info("Loaded reward classifier from %s", pretrained_path)
        return cls(
            classifier=classifier,
            preprocessor=preprocessor,
            device=device,
            success_threshold=success_threshold,
            success_reward=success_reward,
            terminate_on_success=terminate_on_success,
        )

    @torch.inference_mode()
    def predict_success_prob(self, images: dict[str, Any]) -> float | None:
        """Return P(success) from the classifier, or None if no images were matched."""
        batch = build_reward_classifier_batch(images, self.classifier.config.input_features, self.device)
        if not batch:
            return None
        if self.preprocessor is not None:
            batch = self.preprocessor(batch)
        batch = _move_batch_tensors_to_device(batch, self.device)
        images_list = [batch[key] for key in self.classifier.config.input_features if key.startswith(OBS_IMAGE)]
        output = self.classifier.predict(images_list)
        if self.classifier.config.num_classes == 2:
            return float(output.probabilities.squeeze().item())
        return float(output.probabilities.max(dim=-1).values.squeeze().item())

    @torch.inference_mode()
    def predict_success(self, images: dict[str, Any]) -> bool:
        prob = self.predict_success_prob(images)
        self.last_success_prob = prob
        if prob is None:
            logger.debug("Reward classifier: no camera images matched in observation keys %s", list(images))
            return False
        success = prob > self.success_threshold
        if success:
            logger.info(
                "Reward classifier: SUCCESS P(success)=%.3f > threshold=%.2f",
                prob,
                self.success_threshold,
            )
        else:
            logger.debug(
                "Reward classifier: P(success)=%.3f threshold=%.2f → fail",
                prob,
                self.success_threshold,
            )
        return success

    @torch.inference_mode()
    def predict_reward_and_done(self, images: dict[str, Any]) -> tuple[float, bool]:
        if self.predict_success(images):
            return self.success_reward, self.terminate_on_success
        return 0.0, False
