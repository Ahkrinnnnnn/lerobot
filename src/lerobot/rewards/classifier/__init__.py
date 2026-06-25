# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""Binary reward classifier: model, data prep, training helpers, and deployment."""

from .annotations import EpisodeRewardAnnotation, RewardClassifierAnnotations
from .configuration_classifier import RewardClassifierConfig
from .detector import RewardClassifierDetector
from .dataset import inspect_reward_classifier_dataset
from .eval import RewardClassifierEvalMetrics, evaluate_reward_classifier
from .modeling_classifier import Classifier
from .pipeline_config import RewardClassifierPipelineConfig, RewardClassifierRuntimeConfig
from .processor_classifier import make_classifier_processor
from .runtime import RewardClassifierRuntime, build_reward_classifier_batch

__all__ = [
    # Model + train
    "RewardClassifierConfig",
    "Classifier",
    "make_classifier_processor",
    "RewardClassifierRuntime",
    "build_reward_classifier_batch",
    "RewardClassifierEvalMetrics",
    "evaluate_reward_classifier",
    # Data prep CLI
    "EpisodeRewardAnnotation",
    "RewardClassifierAnnotations",
    "RewardClassifierPipelineConfig",
    "inspect_reward_classifier_dataset",
    # Deploy
    "RewardClassifierRuntimeConfig",
    "RewardClassifierDetector",
]
