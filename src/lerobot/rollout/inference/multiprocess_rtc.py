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

"""Multiprocess RTC inference: policy runs in a spawn child (isolated GIL from main I/O)."""

from __future__ import annotations

import logging
import math
import multiprocessing
import queue
import time
import traceback
from threading import Event, Lock, Thread
from typing import Any

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc import ActionQueue, LatencyTracker
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.processor import PolicyProcessorPipeline

from .base import InferenceEngine
from .multiprocess_sync import _multiprocessing_context
from .rtc import _RTC_ERROR_RETRY_DELAY_S, _RTC_IDLE_SLEEP_S, _RTC_JOIN_TIMEOUT_S, run_rtc_inference_step

logger = logging.getLogger(__name__)

_RTC_MAX_CONSECUTIVE_ERRORS = 10


def _obs_to_picklable(obs: dict) -> dict:
    """Ensure observation values are picklable for spawn IPC (CPU numpy)."""
    out: dict = {}
    for k, v in obs.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().numpy()
        else:
            out[k] = v
    return out


def _rtc_subprocess_worker(
    job_queue: Any,
    result_queue: Any,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    hw_features: dict,
    task: str,
    robot_type: str,
    device,
    rtc_config: RTCConfig,
    action_feature_names: list[str] | None,
) -> None:
    """Spawn child: run RTC forwards so the parent GIL is free for cam.read() / robot I/O."""
    from .rtc import _setup_relative_action_steps

    policy_device = torch.device(device)
    relative_step, normalizer_step = _setup_relative_action_steps(
        preprocessor, policy, action_feature_names=action_feature_names
    )
    inference_count = 0

    while True:
        job = job_queue.get()
        if job is None:
            break

        if job["type"] == "reset":
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            inference_count = 0
            continue

        if job["type"] != "infer":
            continue

        try:
            prev_actions = job.get("prev_actions")
            prev_abs = job.get("prev_abs")
            if prev_actions is not None:
                prev_actions = prev_actions.to(policy_device)
            if prev_abs is not None:
                prev_abs = prev_abs.to(policy_device)

            original, processed, latency_s = run_rtc_inference_step(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                obs=job["obs"],
                hw_features=hw_features,
                task=task,
                robot_type=robot_type,
                device=policy_device,
                rtc_config=rtc_config,
                inference_delay=job["delay"],
                prev_actions=prev_actions,
                prev_abs=prev_abs,
                relative_step=relative_step,
                normalizer_step=normalizer_step,
            )
            inference_count += 1
            result_queue.put(
                {
                    "ok": True,
                    "original": original.cpu(),
                    "processed": processed.cpu(),
                    "idx_before": job["idx_before"],
                    "latency_s": latency_s,
                    "inference_count": inference_count,
                }
            )
        except Exception as e:
            logger.error("MultiprocessRTC[inference_err] exc=%r", e)
            logger.debug(traceback.format_exc())
            result_queue.put({"ok": False, "error": repr(e)})


class MultiprocessRTCInferenceEngine(InferenceEngine):
    """RTC with inference in a dedicated spawn subprocess (like deploy v1).

    The parent keeps :class:`ActionQueue` for the control loop to consume.
    A lightweight coordinator thread dispatches obs to the child and merges
    returned chunks — no policy forward passes on the main thread GIL.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        rtc_config: RTCConfig,
        hw_features: dict,
        task: str,
        robot_type: str,
        fps: float,
        device: str | None,
        action_feature_names: list[str] | None = None,
        use_torch_compile: bool = False,
        compile_warmup_inferences: int = 2,
        rtc_queue_threshold: int = 30,
        shutdown_event: Event | None = None,
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._rtc_config = rtc_config
        self._hw_features = hw_features
        self._task = task
        self._robot_type = robot_type
        self._fps = fps
        self._device = device or "cpu"
        self._action_feature_names = action_feature_names
        self._use_torch_compile = use_torch_compile
        self._compile_warmup_inferences = compile_warmup_inferences
        self._rtc_queue_threshold = rtc_queue_threshold
        self._global_shutdown_event = shutdown_event

        self._action_queue: ActionQueue | None = None
        self._obs_holder: dict[str, Any] = {"obs": None}
        self._obs_lock = Lock()
        self._policy_active = Event()
        self._compile_warmup_done = Event()
        self._shutdown_event = Event()
        self._rtc_error = Event()
        self._coordinator_thread: Thread | None = None

        self._mp_ctx: multiprocessing.context.BaseContext | None = None
        self._job_queue: Any = None
        self._result_queue: Any = None
        self._inference_process: multiprocessing.Process | None = None
        self._job_in_flight = False
        self._consecutive_errors = 0

        if not self._use_torch_compile:
            self._compile_warmup_done.set()
            logger.info(
                "MultiprocessRTCInferenceEngine initialized (spawn child, torch.compile disabled)"
            )
        else:
            logger.info(
                "MultiprocessRTCInferenceEngine initialized (spawn child, %d compile warmup inferences)",
                compile_warmup_inferences,
            )

    @property
    def ready(self) -> bool:
        return self._compile_warmup_done.is_set()

    @property
    def failed(self) -> bool:
        return self._rtc_error.is_set()

    def start(self) -> None:
        """Spawn the inference subprocess and start the merge coordinator thread."""
        self._action_queue = ActionQueue(self._rtc_config)
        self._shutdown_event.clear()
        self._job_in_flight = False
        self._consecutive_errors = 0

        self._mp_ctx = _multiprocessing_context()
        self._job_queue = self._mp_ctx.Queue(1)
        self._result_queue = self._mp_ctx.Queue()

        self._inference_process = self._mp_ctx.Process(
            target=_rtc_subprocess_worker,
            kwargs=dict(
                job_queue=self._job_queue,
                result_queue=self._result_queue,
                policy=self._policy,
                preprocessor=self._preprocessor,
                postprocessor=self._postprocessor,
                hw_features=self._hw_features,
                task=self._task,
                robot_type=self._robot_type,
                device=self._device,
                rtc_config=self._rtc_config,
                action_feature_names=self._action_feature_names,
            ),
            daemon=True,
        )
        self._inference_process.start()

        self._coordinator_thread = Thread(
            target=self._coordinator_loop,
            daemon=True,
            name="MultiprocessRTCCoordinator",
        )
        self._coordinator_thread.start()
        logger.info("MultiprocessRTCInferenceEngine subprocess and coordinator started")

    def stop(self) -> None:
        """Stop coordinator and inference subprocess."""
        logger.info("Stopping MultiprocessRTCInferenceEngine...")
        self._shutdown_event.set()
        self._policy_active.clear()

        if self._job_queue is not None:
            try:
                self._job_queue.put(None)
            except Exception:
                pass

        if self._coordinator_thread is not None and self._coordinator_thread.is_alive():
            self._coordinator_thread.join(timeout=_RTC_JOIN_TIMEOUT_S)

        if self._inference_process is not None and self._inference_process.is_alive():
            self._inference_process.join(timeout=_RTC_JOIN_TIMEOUT_S)
            if self._inference_process.is_alive():
                logger.warning("MultiprocessRTC subprocess did not join within %.1fs", _RTC_JOIN_TIMEOUT_S)
        self._inference_process = None
        self._coordinator_thread = None
        logger.info("MultiprocessRTCInferenceEngine stopped")

    def pause(self) -> None:
        logger.info("Pausing MultiprocessRTC inference")
        self._policy_active.clear()

    def resume(self) -> None:
        logger.info("Resuming MultiprocessRTC inference")
        self._policy_active.set()

    def reset(self) -> None:
        logger.info("Resetting MultiprocessRTC inference state")
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        if self._action_queue is not None:
            self._action_queue.clear()
        self._job_in_flight = False
        self._consecutive_errors = 0
        if self._job_queue is not None:
            try:
                self._job_queue.put_nowait({"type": "reset"})
            except queue.Full:
                try:
                    self._job_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._job_queue.put_nowait({"type": "reset"})
                except queue.Full:
                    logger.warning("Could not send reset to MultiprocessRTC child (queue busy)")

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        if self._action_queue is None:
            return None
        return self._action_queue.get()

    def queue_size(self) -> int:
        if self._action_queue is None:
            return 0
        return self._action_queue.qsize()

    def notify_observation(self, obs: dict) -> None:
        with self._obs_lock:
            self._obs_holder["obs"] = obs

    def _coordinator_loop(self) -> None:
        """Dispatch inference jobs to the child and merge returned chunks."""
        try:
            latency_tracker = LatencyTracker()
            time_per_chunk = 1.0 / self._fps
            warmup_required = max(1, self._compile_warmup_inferences) if self._use_torch_compile else 0

            while not self._shutdown_event.is_set():
                self._drain_results(latency_tracker, time_per_chunk, warmup_required)

                if not self._policy_active.is_set():
                    time.sleep(_RTC_IDLE_SLEEP_S)
                    continue

                action_queue = self._action_queue
                with self._obs_lock:
                    obs = self._obs_holder.get("obs")
                if action_queue is None or obs is None:
                    time.sleep(_RTC_IDLE_SLEEP_S)
                    continue

                if action_queue.qsize() > self._rtc_queue_threshold:
                    time.sleep(_RTC_IDLE_SLEEP_S)
                    continue

                if self._job_in_flight:
                    time.sleep(_RTC_IDLE_SLEEP_S)
                    continue

                latency = latency_tracker.max()
                delay = math.ceil(latency / time_per_chunk) if latency else 0
                idx_before = action_queue.get_action_index()
                prev_actions = action_queue.get_left_over()
                prev_abs = action_queue.get_processed_left_over()

                job = {
                    "type": "infer",
                    "obs": _obs_to_picklable(obs),
                    "idx_before": idx_before,
                    "delay": delay,
                    "prev_actions": prev_actions.cpu() if prev_actions is not None else None,
                    "prev_abs": prev_abs.cpu() if prev_abs is not None else None,
                }
                try:
                    self._job_queue.put_nowait(job)
                    self._job_in_flight = True
                except queue.Full:
                    pass

                time.sleep(_RTC_IDLE_SLEEP_S)

        except Exception as e:
            logger.error("Fatal error in MultiprocessRTC coordinator: %s", e)
            logger.error(traceback.format_exc())
            self._rtc_error.set()
            self._compile_warmup_done.set()
            if self._global_shutdown_event is not None:
                self._global_shutdown_event.set()

    def _drain_results(
        self,
        latency_tracker: LatencyTracker,
        time_per_chunk: float,
        warmup_required: int,
    ) -> None:
        action_queue = self._action_queue
        if action_queue is None or self._result_queue is None:
            return

        while True:
            try:
                result = self._result_queue.get_nowait()
            except queue.Empty:
                break

            self._job_in_flight = False

            if not result.get("ok", False):
                self._consecutive_errors += 1
                logger.error(
                    "MultiprocessRTC inference error (%d/%d): %s",
                    self._consecutive_errors,
                    _RTC_MAX_CONSECUTIVE_ERRORS,
                    result.get("error"),
                )
                if self._consecutive_errors >= _RTC_MAX_CONSECUTIVE_ERRORS:
                    raise RuntimeError(result.get("error", "unknown RTC subprocess error"))
                time.sleep(_RTC_ERROR_RETRY_DELAY_S)
                continue

            self._consecutive_errors = 0
            latency_s = result["latency_s"]
            new_delay = math.ceil(latency_s / time_per_chunk)
            inference_count = result["inference_count"]
            is_warmup = self._use_torch_compile and inference_count <= warmup_required

            if is_warmup:
                latency_tracker.reset()
            else:
                latency_tracker.add(latency_s)

            original = result["original"]
            processed = result["processed"]
            action_queue.merge(original, processed, new_delay, result["idx_before"])
            q_after = action_queue.qsize()

            if inference_count == 1:
                logger.info(
                    "MultiprocessRTC first chunk: chunk_len=%d action_dim=%d inference=%.2fs "
                    "delay_steps=%d queue_after=%d",
                    processed.shape[0],
                    processed.shape[-1],
                    latency_s,
                    new_delay,
                    q_after,
                )
            if q_after == 0:
                logger.warning(
                    "MultiprocessRTC merge emptied queue: chunk_len=%d delay_steps=%d "
                    "(inference %.2fs at fps=%.1f)",
                    processed.shape[0],
                    new_delay,
                    latency_s,
                    self._fps,
                )

            if (
                is_warmup
                and inference_count >= warmup_required
                and not self._compile_warmup_done.is_set()
            ):
                self._compile_warmup_done.set()
                logger.info("MultiprocessRTC compile warmup complete (%d inferences)", inference_count)
