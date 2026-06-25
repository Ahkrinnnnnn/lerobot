# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode
from lerobot.configs.rewards import RewardModelConfig
from lerobot.optim import AdamWConfig, LRSchedulerConfig, OptimizerConfig
from lerobot.utils.constants import OBS_IMAGE


@dataclass
class RewardClassifierImagePreprocessingConfig:
    """Optional crop/resize for reward-classifier images (export and/or train-time)."""

    crop_params_dict: dict[str, tuple[int, int, int, int]] | None = None
    resize_size: tuple[int, int] | None = None


@RewardModelConfig.register_subclass(name="reward_classifier")
@dataclass
class RewardClassifierConfig(RewardModelConfig):
    """Configuration for the Reward Classifier model."""

    name: str = "reward_classifier"
    num_classes: int = 2
    hidden_dim: int = 256
    latent_dim: int = 256
    image_embedding_pooling_dim: int = 8
    dropout_rate: float = 0.1
    model_name: str = "lerobot/resnet10"
    device: str = "cpu"
    model_type: str = "cnn"  # "transformer" or "cnn"
    num_cameras: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    image_preprocessing: RewardClassifierImagePreprocessingConfig | None = None
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.image_preprocessing is not None and self.image_preprocessing.resize_size is not None:
            resize_size = self.image_preprocessing.resize_size
            for key, feat in self.input_features.items():
                if key.startswith(OBS_IMAGE) and len(feat.shape) == 3:
                    feat.shape = (feat.shape[0], *resize_size)

    @property
    def observation_delta_indices(self) -> list | None:
        return None

    @property
    def action_delta_indices(self) -> list | None:
        return None

    @property
    def reward_delta_indices(self) -> list | None:
        return None

    def get_optimizer_preset(self) -> OptimizerConfig:
        return AdamWConfig(
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            grad_clip_norm=self.grad_clip_norm,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None

    def validate_features(self) -> None:
        """Validate feature configurations."""
        image_keys = [key for key in self.input_features if key.startswith(OBS_IMAGE)]
        if not image_keys:
            raise ValueError(
                "You must provide an image observation (key starting with 'observation.image') in the input features"
            )
        if self.num_cameras != len(image_keys):
            raise ValueError(
                f"num_cameras={self.num_cameras} but input_features defines {len(image_keys)} image keys: "
                f"{image_keys}"
            )
