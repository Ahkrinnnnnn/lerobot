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

"""Real-Time Chunking inference engine.

A background thread produces action chunks asynchronously via
:meth:`policy.predict_action_chunk`.  The main control loop polls
``get_action`` for the next ready action; observations flow the other
way via ``notify_observation``.
"""

from __future__ import annotations

import logging
import math
import time
import traceback
from collections.abc import Callable
from threading import Event, Lock, Thread
from typing import Any

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc import ActionQueue, LatencyTracker, reanchor_relative_rtc_prefix
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import (
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
)
from lerobot.utils.feature_utils import build_dataset_frame

from ..robot_wrapper import ThreadSafeRobot
from .base import InferenceEngine

logger = logging.getLogger(__name__)

# How long the RTC loop sleeps when paused, idle, or backpressured by a full queue.
_RTC_IDLE_SLEEP_S: float = 0.01
# Backoff between transient inference errors (per consecutive failure).
_RTC_ERROR_RETRY_DELAY_S: float = 0.5
# Consecutive transient errors tolerated before giving up and propagating shutdown.
_RTC_MAX_CONSECUTIVE_ERRORS: int = 10
# Hard timeout for joining the RTC thread on stop().
_RTC_JOIN_TIMEOUT_S: float = 3.0


def compute_rtc_delay_steps(
    inference_latency_s: float,
    obs_capture_latency_s: float,
    time_per_chunk: float,
) -> int:
    """Convert combined inference + obs-capture latency to control-step delay."""
    total_s = inference_latency_s + obs_capture_latency_s
    if total_s <= 0 or time_per_chunk <= 0:
        return 0
    return math.ceil(total_s / time_per_chunk)


# ---------------------------------------------------------------------------
# RTC helpers
# ---------------------------------------------------------------------------


def _setup_relative_action_steps(
    preprocessor: PolicyProcessorPipeline,
    policy: PreTrainedPolicy,
    action_feature_names: list[str] | None = None,
) -> tuple[RelativeActionsProcessorStep | None, NormalizerProcessorStep | None]:
    """Locate relative/normalizer processor steps and configure action names."""
    relative_step = next(
        (s for s in preprocessor.steps if isinstance(s, RelativeActionsProcessorStep) and s.enabled),
        None,
    )
    normalizer_step = next(
        (s for s in preprocessor.steps if isinstance(s, NormalizerProcessorStep)),
        None,
    )
    if relative_step is not None and relative_step.action_names is None:
        cfg_names = getattr(policy.config, "action_feature_names", None)
        if cfg_names:
            relative_step.action_names = list(cfg_names)
        elif action_feature_names:
            relative_step.action_names = list(action_feature_names)
    return relative_step, normalizer_step


def run_rtc_inference_step(
    *,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    obs: dict,
    dataset_features: dict,
    task: str,
    robot_type: str,
    device: torch.device,
    rtc_config: RTCConfig,
    inference_delay: int,
    prev_actions: torch.Tensor | None,
    prev_abs: torch.Tensor | None,
    relative_step: RelativeActionsProcessorStep | None,
    normalizer_step: NormalizerProcessorStep | None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Run one RTC forward pass. Returns ``(original, processed, latency_s)``."""
    start = time.perf_counter()

    obs_batch = build_dataset_frame(dataset_features, obs, prefix="observation")
    obs_batch = prepare_observation_for_inference(obs_batch, device, task, robot_type)
    obs_batch["task"] = [task]

    preprocessed = preprocessor(obs_batch)

    rtc_prev = prev_actions
    if rtc_prev is not None and relative_step is not None:
        raw_state = relative_step.get_cached_state()
        if raw_state is not None and prev_abs is not None and prev_abs.numel() > 0:
            rtc_prev = reanchor_relative_rtc_prefix(
                prev_actions_absolute=prev_abs,
                current_state=raw_state,
                relative_step=relative_step,
                normalizer_step=normalizer_step,
                policy_device=device,
            )

    if rtc_prev is not None:
        rtc_prev = _normalize_prev_actions_length(rtc_prev, target_steps=rtc_config.execution_horizon)

    actions = policy.predict_action_chunk(
        preprocessed, inference_delay=inference_delay, prev_chunk_left_over=rtc_prev
    )

    original = actions.squeeze(0).clone()
    processed = postprocessor(actions).squeeze(0)
    return original, processed, time.perf_counter() - start


def _normalize_prev_actions_length(prev_actions: torch.Tensor, target_steps: int) -> torch.Tensor:
    """Pad or truncate RTC prefix actions to a fixed length for stable compiled inference."""
    if prev_actions.ndim != 2:
        raise ValueError(f"Expected 2D [T, A] tensor, got shape={tuple(prev_actions.shape)}")
    steps, action_dim = prev_actions.shape
    if steps == target_steps:
        return prev_actions
    if steps > target_steps:
        return prev_actions[:target_steps]
    padded = torch.zeros((target_steps, action_dim), dtype=prev_actions.dtype, device=prev_actions.device)
    padded[:steps] = prev_actions
    return padded


# ---------------------------------------------------------------------------
# RTCInferenceEngine
# ---------------------------------------------------------------------------


class RTCInferenceEngine(InferenceEngine):
    """Async RTC inference: a background thread produces action chunks.

    ``get_action`` pops the next action from the shared queue (or
    returns ``None`` if the queue is empty).  When ``policy_obs_capture_fn``
    is set, observations are captured at the inference boundary instead of
    via ``notify_observation``.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        robot_wrapper: ThreadSafeRobot,
        rtc_config: RTCConfig,
        dataset_features: dict,
        task: str,
        fps: float,
        device: str | None,
        use_torch_compile: bool = False,
        compile_warmup_inferences: int = 2,
        rtc_queue_threshold: int = 30,
        shutdown_event: Event | None = None,
        policy_obs_capture_fn: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._robot = robot_wrapper
        self._rtc_config = rtc_config
        self._dataset_features = dataset_features
        self._task = task
        self._fps = fps
        self._device = device or "cpu"
        self._use_torch_compile = use_torch_compile
        self._compile_warmup_inferences = compile_warmup_inferences
        self._rtc_queue_threshold = rtc_queue_threshold

        self._action_queue: ActionQueue | None = None
        self._obs_holder: dict[str, Any] = {}
        self._obs_lock = Lock()
        self._policy_active = Event()
        self._compile_warmup_done = Event()
        self._shutdown_event = Event()
        self._rtc_error = Event()
        self._global_shutdown_event = shutdown_event
        self._policy_obs_capture_fn = policy_obs_capture_fn
        self._rtc_thread: Thread | None = None

        if not self._use_torch_compile:
            self._compile_warmup_done.set()
            logger.info("RTCInferenceEngine initialized (torch.compile disabled, no warmup needed)")
        else:
            logger.info(
                "RTCInferenceEngine initialized (torch.compile enabled, %d warmup inferences)",
                compile_warmup_inferences,
            )

        action_names = [k for k in robot_wrapper.action_features if k.endswith(".pos")]
        self._relative_step, self._normalizer_step = _setup_relative_action_steps(
            preprocessor, policy, action_feature_names=action_names
        )
        if self._relative_step is not None:
            logger.info("Relative actions enabled: RTC prefix will be re-anchored")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        """True once torch.compile warmup is complete (or immediately if compile is disabled)."""
        return self._compile_warmup_done.is_set()

    @property
    def failed(self) -> bool:
        """True if the RTC background thread exited due to an unrecoverable error."""
        return self._rtc_error.is_set()

    @property
    def action_queue(self) -> ActionQueue | None:
        """The shared action queue between the RTC thread and the main loop."""
        return self._action_queue

    def start(self) -> None:
        """Launch the RTC background thread."""
        self._action_queue = ActionQueue(self._rtc_config)
        self._obs_holder = {
            "obs": None,
            "robot_type": self._robot.robot_type,
        }
        self._shutdown_event.clear()
        self._rtc_thread = Thread(
            target=self._rtc_loop,
            daemon=True,
            name="RTCInference",
        )
        self._rtc_thread.start()
        logger.info("RTC inference thread started")

    def stop(self) -> None:
        """Signal the RTC thread to stop and wait for it."""
        logger.info("Stopping RTC inference thread...")
        self._shutdown_event.set()
        self._policy_active.clear()
        if self._rtc_thread is not None and self._rtc_thread.is_alive():
            self._rtc_thread.join(timeout=_RTC_JOIN_TIMEOUT_S)
            if self._rtc_thread.is_alive():
                logger.warning("RTC thread did not join within %.1fs", _RTC_JOIN_TIMEOUT_S)
            else:
                logger.info("RTC inference thread stopped")
            self._rtc_thread = None

    def pause(self) -> None:
        """Pause the RTC background thread."""
        logger.info("Pausing RTC inference thread")
        self._policy_active.clear()

    def resume(self) -> None:
        """Resume the RTC background thread."""
        logger.info("Resuming RTC inference thread")
        self._policy_active.set()

    def reset(self) -> None:
        """Reset the policy, processors, and action queue."""
        logger.info("Resetting RTC inference state (policy + processors + queue)")
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        if self._action_queue is not None:
            self._action_queue.clear()

    # ------------------------------------------------------------------
    # Action production (called from main thread)
    # ------------------------------------------------------------------

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Pop the next action from the RTC queue (ignores ``obs_frame``)."""
        if self._action_queue is None:
            return None
        return self._action_queue.get()

    def queue_size(self) -> int:
        """Number of unconsumed actions in the RTC queue (0 if not started)."""
        if self._action_queue is None:
            return 0
        return self._action_queue.qsize()

    def notify_observation(self, obs: dict) -> None:
        """Publish the latest observation for the RTC thread to consume."""
        with self._obs_lock:
            self._obs_holder["obs"] = obs

    # ------------------------------------------------------------------
    # RTC: background inference thread
    # ------------------------------------------------------------------

    def _rtc_loop(self) -> None:
        """Background thread that generates action chunks via RTC."""
        try:
            latency_tracker = LatencyTracker()
            obs_capture_tracker = LatencyTracker()
            time_per_chunk = 1.0 / self._fps
            policy_device = torch.device(self._device)

            warmup_required = max(1, self._compile_warmup_inferences) if self._use_torch_compile else 0
            inference_count = 0
            consecutive_errors = 0

            while not self._shutdown_event.is_set():
                if not self._policy_active.is_set():
                    time.sleep(_RTC_IDLE_SLEEP_S)
                    continue

                queue = self._action_queue
                if queue is None:
                    time.sleep(_RTC_IDLE_SLEEP_S)
                    continue

                if queue.qsize() <= self._rtc_queue_threshold:
                    try:
                        obs_capture_s = 0.0
                        if self._policy_obs_capture_fn is not None:
                            obs_capture_start = time.perf_counter()
                            obs = self._policy_obs_capture_fn()
                            obs_capture_s = time.perf_counter() - obs_capture_start
                        else:
                            with self._obs_lock:
                                obs = self._obs_holder.get("obs")
                            if obs is None:
                                time.sleep(_RTC_IDLE_SLEEP_S)
                                continue

                        idx_before = queue.get_action_index()
                        prev_actions = queue.get_left_over()
                        prev_abs = queue.get_processed_left_over()

                        prev_inference_s = latency_tracker.max() or 0.0
                        prev_obs_capture_s = obs_capture_tracker.max() or 0.0
                        delay = compute_rtc_delay_steps(
                            prev_inference_s, prev_obs_capture_s, time_per_chunk
                        )

                        original, processed, new_latency = run_rtc_inference_step(
                            policy=self._policy,
                            preprocessor=self._preprocessor,
                            postprocessor=self._postprocessor,
                            obs=obs,
                            dataset_features=self._dataset_features,
                            task=self._task,
                            robot_type=self._robot.robot_type,
                            device=policy_device,
                            rtc_config=self._rtc_config,
                            inference_delay=delay,
                            prev_actions=prev_actions,
                            prev_abs=prev_abs,
                            relative_step=self._relative_step,
                            normalizer_step=self._normalizer_step,
                        )
                        new_delay = compute_rtc_delay_steps(
                            new_latency, obs_capture_s, time_per_chunk
                        )

                        inference_count += 1
                        consecutive_errors = 0
                        is_warmup = self._use_torch_compile and inference_count <= warmup_required
                        if is_warmup:
                            latency_tracker.reset()
                            obs_capture_tracker.reset()
                        else:
                            latency_tracker.add(new_latency)
                            obs_capture_tracker.add(obs_capture_s)

                        queue.merge(original, processed, new_delay, idx_before)

                        if (
                            is_warmup
                            and inference_count >= warmup_required
                            and not self._compile_warmup_done.is_set()
                        ):
                            self._compile_warmup_done.set()
                            logger.info("Compile warmup complete (%d inferences)", inference_count)

                        logger.debug("RTC inference latency=%.2fs, queue=%d", new_latency, queue.qsize())

                    except Exception as e:
                        consecutive_errors += 1
                        if self._policy_obs_capture_fn is not None and consecutive_errors == 1:
                            logger.warning(
                                "Failed to capture policy observation at inference boundary: %s", e
                            )
                        logger.error(
                            "RTC inference error (%d/%d): %s",
                            consecutive_errors,
                            _RTC_MAX_CONSECUTIVE_ERRORS,
                            e,
                        )
                        logger.debug(traceback.format_exc())
                        if consecutive_errors >= _RTC_MAX_CONSECUTIVE_ERRORS:
                            # Persistent failure: stop retrying and propagate shutdown.
                            raise
                        time.sleep(_RTC_ERROR_RETRY_DELAY_S)
                else:
                    time.sleep(_RTC_IDLE_SLEEP_S)

        except Exception as e:
            logger.error("Fatal error in RTC thread: %s", e)
            logger.error(traceback.format_exc())
            self._rtc_error.set()
            # Unblock any warmup waiters so the main loop doesn't spin forever
            self._compile_warmup_done.set()
            # Signal the top-level shutdown so strategies exit their control loops
            if self._global_shutdown_event is not None:
                self._global_shutdown_event.set()
