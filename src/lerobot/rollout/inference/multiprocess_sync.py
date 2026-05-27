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

"""Multiprocess synchronous inference: policy runs in a spawn child process."""

from __future__ import annotations

import logging
import multiprocessing
import queue
from contextlib import nullcontext
from copy import copy
from typing import Any

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline

from .base import InferenceEngine

logger = logging.getLogger(__name__)


def _multiprocessing_context() -> multiprocessing.context.BaseContext:
    """Use spawn so the inference child is CUDA-safe after parent init."""
    return multiprocessing.get_context("spawn")


def _sync_inference_worker(
    obs_queue: Any,
    latest_action_ref: dict,
    latest_action_lock: Any,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    dataset_features: dict,
    ordered_action_keys: list[str],
    task: str,
    robot_type: str,
    device,
) -> None:
    """Runs in the inference subprocess; mirrors :class:`SyncInferenceEngine.get_action`."""
    while True:
        observation_frame = obs_queue.get()
        if observation_frame is None:
            break
        try:
            observation = copy(observation_frame)
            autocast_ctx = (
                torch.autocast(device_type=device.type)
                if device.type == "cuda" and policy.config.use_amp
                else nullcontext()
            )
            with torch.inference_mode(), autocast_ctx:
                observation = prepare_observation_for_inference(
                    observation, device, task, robot_type
                )
                observation = preprocessor(observation)
                action = policy.select_action(observation)
                action = postprocessor(action)
            action_tensor = action.squeeze(0).cpu()
            action_dict = make_robot_action(action_tensor, dataset_features)
            reordered = torch.tensor([action_dict[k] for k in ordered_action_keys])
            with latest_action_lock:
                latest_action_ref["action"] = reordered
        except Exception as e:
            logger.warning("MultiprocessSync[inference_err] exc=%r", e)


class MultiprocessSyncInferenceEngine(InferenceEngine):
    """Synchronous inference in a dedicated spawn subprocess.

    The main control loop enqueues observation frames and reads the latest
    action from a shared reference so ``send_action`` is never blocked by
    policy forward passes.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        dataset_features: dict,
        ordered_action_keys: list[str],
        task: str,
        device: str | None,
        robot_type: str,
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._dataset_features = dataset_features
        self._ordered_action_keys = ordered_action_keys
        self._task = task
        self._device = torch.device(device or "cpu")
        self._robot_type = robot_type

        self._mp_ctx: multiprocessing.context.BaseContext | None = None
        self._manager: Any = None
        self._latest_action_ref: Any = None
        self._latest_action_lock: Any = None
        self._obs_queue: Any = None
        self._inference_process: multiprocessing.Process | None = None

        logger.info(
            "MultiprocessSyncInferenceEngine initialized (device=%s, action_keys=%d)",
            self._device,
            len(ordered_action_keys),
        )

    def start(self) -> None:
        """Spawn the inference subprocess."""
        self._mp_ctx = _multiprocessing_context()
        self._manager = self._mp_ctx.Manager()
        self._latest_action_ref = self._manager.dict()
        self._latest_action_ref["action"] = None
        self._latest_action_lock = self._mp_ctx.Lock()
        self._obs_queue = self._mp_ctx.Queue(1)

        self._inference_process = self._mp_ctx.Process(
            target=_sync_inference_worker,
            kwargs=dict(
                obs_queue=self._obs_queue,
                latest_action_ref=self._latest_action_ref,
                latest_action_lock=self._latest_action_lock,
                policy=self._policy,
                preprocessor=self._preprocessor,
                postprocessor=self._postprocessor,
                dataset_features=self._dataset_features,
                ordered_action_keys=self._ordered_action_keys,
                task=self._task,
                robot_type=self._robot_type,
                device=self._device,
            ),
            daemon=True,
        )
        self._inference_process.start()
        logger.info("MultiprocessSyncInferenceEngine subprocess started")

    def stop(self) -> None:
        """Signal the inference subprocess to exit and join."""
        if self._obs_queue is not None:
            try:
                self._obs_queue.put(None)
            except Exception:
                pass
        if self._inference_process is not None and self._inference_process.is_alive():
            self._inference_process.join(timeout=5.0)
            if self._inference_process.is_alive():
                logger.warning("MultiprocessSync inference subprocess did not exit within 5s")
        self._inference_process = None
        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None
        logger.info("MultiprocessSyncInferenceEngine stopped")

    def reset(self) -> None:
        """Reset policy and processor state in the parent (child reloads on next obs)."""
        logger.info("Resetting multiprocess sync inference state")
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        if self._latest_action_ref is not None:
            with self._latest_action_lock:
                self._latest_action_ref["action"] = None

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Enqueue ``obs_frame`` for inference and return the latest action tensor."""
        if obs_frame is not None and self._obs_queue is not None:
            try:
                self._obs_queue.put_nowait(obs_frame)
            except queue.Full:
                pass
        if self._latest_action_ref is None:
            return None
        with self._latest_action_lock:
            action = self._latest_action_ref.get("action")
        return action.clone() if action is not None else None

    def read_latest_action(self) -> torch.Tensor | None:
        """Return the latest action without enqueueing a new observation."""
        if self._latest_action_ref is None:
            return None
        with self._latest_action_lock:
            action = self._latest_action_ref.get("action")
        return action.clone() if action is not None else None
