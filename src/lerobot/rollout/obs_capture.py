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

"""Capture policy observations at the inference boundary (fresh camera + proprio)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from lerobot.processor import RobotObservation, RobotProcessorPipeline

from .robot_wrapper import ThreadSafeRobot

PolicyObsCaptureFn = Callable[[], dict[str, Any]]


def make_policy_obs_capture_fn(
    robot_wrapper: ThreadSafeRobot,
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
) -> PolicyObsCaptureFn:
    """Return a callable that reads cameras + proprio immediately before policy inference."""

    def capture_policy_observation() -> dict[str, Any]:
        fn = robot_wrapper.get_observation
        try:
            obs_raw = fn(include_images=True)  # type: ignore[misc]
        except TypeError:
            obs_raw = fn()
        return robot_observation_processor(obs_raw)

    return capture_policy_observation
