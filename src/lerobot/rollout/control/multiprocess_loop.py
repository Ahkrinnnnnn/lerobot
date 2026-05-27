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

"""Fixed-rate control loop with decoupled inference (deploy v2)."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep

from ..inference.multiprocess_sync import MultiprocessSyncInferenceEngine
from ..inference.rtc import RTCInferenceEngine

if TYPE_CHECKING:
    from ..context import RolloutContext
    from ..strategies.core import RolloutStrategy

logger = logging.getLogger(__name__)


def _get_robot_observation(robot, *, include_images: bool = True) -> dict[str, Any]:
    """Call ``get_observation``; use ``include_images=False`` when the robot supports it."""
    fn = robot.get_observation
    if not include_images:
        try:
            return fn(include_images=False)  # type: ignore[misc]
        except TypeError:
            pass
    return fn()


def _current_pose_action(obs_processed: dict, ordered_keys: list[str]) -> dict[str, float]:
    """Hold-action: map current joint positions to action keys."""
    return {k: obs_processed[k] for k in ordered_keys if k in obs_processed}


def multiprocess_control_loop(
    ctx: RolloutContext,
    strategy: RolloutStrategy,
    control_loop_stats: dict | None = None,
) -> None:
    """Run a fixed-rate control loop; inference runs async (RTC thread or sync subprocess).

    The main process always ticks at ``cfg.fps`` (times ``interpolation_multiplier``) and
    sends actions every tick. Policy inference must not block ``send_action``.
    """
    cfg = ctx.runtime.cfg
    engine = strategy._engine
    interpolator = strategy._interpolator
    if engine is None or interpolator is None:
        raise RuntimeError("Strategy engine/interpolator not initialized; call setup() first")

    robot = ctx.hardware.robot_wrapper
    processors = ctx.processors
    features = ctx.data.dataset_features
    ordered_keys = ctx.data.ordered_action_keys
    shutdown = ctx.runtime.shutdown_event

    control_interval = interpolator.get_control_interval(cfg.fps)
    is_rtc = isinstance(engine, RTCInferenceEngine)
    is_mp_sync = isinstance(engine, MultiprocessSyncInferenceEngine)

    engine.resume()

    start_time = time.perf_counter()
    next_tick_t = time.perf_counter()
    loop_i = 0
    last_robot_action_sent: dict | None = None
    cached_obs_processed: dict | None = None

    logger.info(
        "Multiprocess control loop started (fps=%.2f, interval=%.4fs, rtc=%s, mp_sync=%s)",
        cfg.fps,
        control_interval,
        is_rtc,
        is_mp_sync,
    )

    try:
        while not shutdown.is_set():
            if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                logger.info("Duration limit reached (%.0fs)", cfg.duration)
                break

            next_tick_t += control_interval
            precise_sleep(max(0.0, next_tick_t - time.perf_counter()))
            loop_start = time.perf_counter()
            loop_i += 1

            if strategy._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                continue

            if is_rtc:
                obs_raw = robot.get_observation()
                obs_processed = processors.robot_observation_processor(obs_raw)
                engine.notify_observation(obs_processed)
                cached_obs_processed = obs_processed
            else:
                obs_raw = _get_robot_observation(robot, include_images=False)
                obs_processed = processors.robot_observation_processor(obs_raw)

                if is_mp_sync:
                    try:
                        obs_for_policy = _get_robot_observation(robot, include_images=True)
                        obs_policy_processed = processors.robot_observation_processor(obs_for_policy)
                        cached_obs_processed = obs_policy_processed
                        obs_frame = build_dataset_frame(features, obs_policy_processed, prefix=OBS_STR)
                        engine.get_action(obs_frame)
                    except Exception as e:
                        logger.warning("Failed to enqueue observation for inference: %s", e)

            if interpolator.needs_new_action():
                obs_for_frame = cached_obs_processed if cached_obs_processed is not None else obs_processed
                obs_frame = build_dataset_frame(features, obs_for_frame, prefix=OBS_STR)
                if is_rtc:
                    action_tensor = engine.get_action(obs_frame)
                elif is_mp_sync:
                    action_tensor = engine.read_latest_action()
                else:
                    action_tensor = engine.get_action(obs_frame)
                if action_tensor is not None:
                    interpolator.add(action_tensor)

            interp = interpolator.get()
            if interp is not None and len(interp) == len(ordered_keys):
                action_dict = {k: interp[i].item() for i, k in enumerate(ordered_keys)}
            else:
                action_dict = _current_pose_action(obs_processed, ordered_keys)

            try:
                processed = processors.robot_action_processor((action_dict, obs_raw))
                robot.send_action(processed)
                last_robot_action_sent = dict(processed)
                if control_loop_stats is not None:
                    control_loop_stats["sends"] = control_loop_stats.get("sends", 0) + 1
            except Exception as e:
                logger.warning("send_action failed: %s", e)
                if last_robot_action_sent is not None:
                    try:
                        robot.send_action(last_robot_action_sent)
                    except Exception as retry_e:
                        logger.warning("send_action retry failed: %s", retry_e)

            strategy._log_telemetry(obs_processed, action_dict, ctx.runtime)

            dt_s = time.perf_counter() - loop_start
            if control_loop_stats is not None:
                control_loop_stats["frames"] = control_loop_stats.get("frames", 0) + 1
                if dt_s > control_interval:
                    overruns = control_loop_stats.get("overruns", 0) + 1
                    control_loop_stats["overruns"] = overruns
                    if overruns <= 5 or overruns % 30 == 0:
                        logger.warning(
                            "Control loop overrun: step took %.1f ms (period %.1f ms); overruns=%d",
                            dt_s * 1e3,
                            control_interval * 1e3,
                            overruns,
                        )
