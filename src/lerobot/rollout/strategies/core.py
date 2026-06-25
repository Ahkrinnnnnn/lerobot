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

"""Rollout strategy ABC and shared action-dispatch helper."""

from __future__ import annotations

import abc
import logging
import math
import time
from typing import TYPE_CHECKING

from lerobot.datasets.utils import DEFAULT_VIDEO_FILE_SIZE_IN_MB
from lerobot.utils.action_interpolator import ActionInterpolator
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import log_rerun_data

from ..collect_intervention import CollectInterventionController
from ..inference import InferenceEngine

if TYPE_CHECKING:
    from ..configs import RolloutStrategyConfig
    from ..context import HardwareContext, ProcessorContext, RolloutContext, RuntimeContext

logger = logging.getLogger(__name__)

_MANUAL_RESET_BANNER = "!" * 72


def _log_manual_scene_reset_open(
    episode_index: int,
    wait_s: float,
    pause_key: str,
    *,
    intervention: CollectInterventionController | None,
) -> None:
    """Highly visible banner: inference paused — operator should reset the scene."""
    wait_i = max(1, int(math.ceil(wait_s)))
    logger.warning("")
    logger.warning(_MANUAL_RESET_BANNER)
    logger.warning(
        "!!! [MANUAL SCENE RESET] Episode %d — INFERENCE PAUSED FOR %ds !!!",
        episode_index,
        wait_i,
    )
    logger.warning(
        "!!! Reposition objects / fix the scene NOW (arm should be at home) !!!"
    )
    if pause_key and intervention is not None:
        logger.warning(
            "!!! Press '%s' to PAUSE / RESUME the countdown (same key as collect pause) !!!",
            pause_key,
        )
    logger.warning(_MANUAL_RESET_BANNER)
    logger.warning("")


def _log_manual_scene_reset_countdown(episode_index: int, remaining_s: int) -> None:
    logger.warning(
        ">>> [MANUAL SCENE RESET] Episode %d — %ds remaining — finish scene setup <<<",
        episode_index,
        remaining_s,
    )


def _log_manual_scene_reset_close(episode_index: int) -> None:
    logger.warning("")
    logger.warning(_MANUAL_RESET_BANNER)
    logger.warning(
        "!!! [MANUAL SCENE RESET] Episode %d — TIME UP — resuming robot control !!!",
        episode_index,
    )
    logger.warning(_MANUAL_RESET_BANNER)
    logger.warning("")


def _wait_for_manual_scene_reset_window(
    episode_index: int,
    wait_s: float,
    *,
    pause_key: str = "space",
    collect_intervention: CollectInterventionController | None = None,
) -> None:
    """Count down the manual scene-reset window; reuse session pause key if provided."""
    if collect_intervention is not None:
        collect_intervention.clear_paused()
    _log_manual_scene_reset_open(
        episode_index,
        wait_s,
        pause_key,
        intervention=collect_intervention,
    )
    deadline = time.perf_counter() + wait_s
    last_announced = -1
    last_pause_reminder = 0.0
    while True:
        if collect_intervention is not None and collect_intervention.paused:
            now = time.perf_counter()
            if now - last_pause_reminder >= 5.0:
                logger.warning(
                    ">>> [MANUAL SCENE RESET] countdown PAUSED — press '%s' to resume <<<",
                    pause_key,
                )
                last_pause_reminder = now
            precise_sleep(0.2)
            continue

        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        sec_left = int(math.ceil(remaining))
        if sec_left != last_announced:
            _log_manual_scene_reset_countdown(episode_index, sec_left)
            last_announced = sec_left
        precise_sleep(min(1.0, remaining))
    if collect_intervention is not None:
        collect_intervention.clear_paused()
    _log_manual_scene_reset_close(episode_index)


class RolloutStrategy(abc.ABC):
    """Abstract base for rollout execution strategies.

    Each concrete strategy implements a self-contained control loop with
    its own recording/interaction semantics.  Strategies are mutually
    exclusive — only one runs per session.
    """

    def __init__(self, config: RolloutStrategyConfig) -> None:
        self.config = config
        self._engine: InferenceEngine | None = None
        self._interpolator: ActionInterpolator | None = None
        self._warmup_flushed: bool = False
        self._cached_obs_processed: dict | None = None

    def _init_engine(self, ctx: RolloutContext) -> None:
        """Attach the inference engine and action interpolator, then start the backend.

        Creates an :class:`ActionInterpolator` from the config's
        ``interpolation_multiplier`` and starts the inference engine.
        Call this from ``setup()`` so strategies share identical
        initialisation without duplicating code.
        """
        self._interpolator = ActionInterpolator(multiplier=ctx.runtime.cfg.interpolation_multiplier)
        self._engine = ctx.policy.inference
        logger.info("Debug: starting inference engine (%s)...", type(self._engine).__name__)
        self._engine.reset()
        self._engine.start()
        self._warmup_flushed = False
        self._cached_obs_processed = None
        logger.info("Debug: inference engine started (%s)", type(self._engine).__name__)

    def _process_observation_and_notify(self, processors: ProcessorContext, obs_raw: dict) -> dict:
        """Run the observation processor and notify the engine — throttled to policy ticks.

        Callers are responsible for calling ``robot.get_observation()`` every loop
        iteration so ``obs_raw`` stays fresh for the action post-processor.  This
        helper gates only the comparatively expensive bits — the processor pipeline
        and ``engine.notify_observation`` — to fire when the interpolator signals
        it needs a new action (once per ``interpolation_multiplier`` ticks).  On
        interpolated ticks the cached ``obs_processed`` is reused.

        With ``interpolation_multiplier == 1`` this is equivalent to the unthrottled
        path: ``needs_new_action()`` is True every tick.

        The cache is implicitly invalidated whenever ``interpolator.reset()`` is
        called (warmup completion, DAgger phase transitions back to AUTONOMOUS),
        because reset makes ``needs_new_action()`` return True on the next call.
        """
        if self._cached_obs_processed is None or self._interpolator.needs_new_action():
            obs_processed = processors.robot_observation_processor(obs_raw)
            self._engine.notify_observation(obs_processed)
            self._cached_obs_processed = obs_processed
        return self._cached_obs_processed

    def _handle_warmup(self, use_torch_compile: bool, loop_start: float, control_interval: float) -> bool:
        """Handle torch.compile warmup phase.

        Returns ``True`` if the caller should ``continue`` (still warming
        up).  On the first post-warmup iteration the engine and
        interpolator are reset so stale warmup state is discarded.
        """
        engine = self._engine
        interpolator = self._interpolator
        if not use_torch_compile:
            return False
        if not engine.ready:
            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)
            return True
        if not self._warmup_flushed:
            logger.info("Warmup complete — flushing stale state and resuming engine")
            engine.reset()
            interpolator.reset()
            self._warmup_flushed = True
            engine.resume()
        return False

    def _teardown_hardware(self, hw: HardwareContext, return_to_initial_position: bool = True) -> None:
        """Stop the inference engine, optionally return robot to initial position, and disconnect hardware."""
        if self._engine is not None:
            logger.info("Stopping inference engine...")
            self._engine.stop()
        robot = hw.robot_wrapper.inner
        if robot.is_connected:
            if return_to_initial_position and hw.initial_position:
                logger.info("Returning robot to initial position before shutdown...")
                self._return_to_initial_position(hw)
            elif not return_to_initial_position:
                logger.info(
                    "Skipping return-to-initial-position (disabled by config); leaving robot in final pose."
                )
            logger.info("Disconnecting robot...")
            robot.disconnect()
        teleop = hw.teleop
        if teleop is not None and teleop.is_connected:
            logger.info("Disconnecting teleoperator...")
            teleop.disconnect()

    @staticmethod
    def _return_to_initial_position(hw: HardwareContext, duration_s: float = 3.0, fps: int = 50) -> None:
        """Smoothly interpolate the robot back to its initial position."""
        robot = hw.robot_wrapper
        target = hw.initial_position
        gripper_key = "gripper.pos"
        try:
            current_obs = robot.get_observation()
            current_pos = {k: v for k, v in current_obs.items() if k in target}
            if gripper_key in target:
                open_val = float(target[gripper_key])
                logger.info(
                    "=== Homing === opening gripper to %.0f before moving arm (avoid carrying object home)",
                    open_val,
                )
                try:
                    robot.send_action({gripper_key: open_val})
                    precise_sleep(0.4)
                except Exception as e:
                    logger.warning("Could not open gripper before homing: %s", e)
            steps = max(int(duration_s * fps), 1)
            for step in range(1, steps + 1):
                t = step / steps
                interp = {}
                for k in current_pos:
                    if k == gripper_key:
                        continue
                    if k not in target:
                        continue
                    interp[k] = current_pos[k] * (1 - t) + target[k] * t
                if interp:
                    robot.send_action(interp)
                precise_sleep(1 / fps)
        except Exception as e:
            logger.warning("Could not return to initial position: %s", e)

    def _wait_between_episodes(
        self,
        ctx: RolloutContext,
        *,
        episode_index: int,
        episode_reset_time_s: float,
        return_to_initial_duration_s: float = 3.0,
        manual_scene_reset_pause_key: str = "space",
        collect_intervention: CollectInterventionController | None = None,
    ) -> None:
        """Pause inference, home the arm, then wait for manual scene reset.

        ``episode_reset_time_s`` is the full manual-reset window **after homing**
        (homing time is extra, not subtracted from the countdown).
        """
        return_home = (
            ctx.runtime.cfg.return_to_initial_position and bool(ctx.hardware.initial_position)
        )

        logger.info(
            "=== Episode %d RESET START === pausing inference%s",
            episode_index,
            " — homing arm to initial position" if return_home else "",
        )

        if self._engine is not None:
            self._engine.pause()

        reset_start = time.perf_counter()
        if return_home:
            logger.info("=== Episode %d RESET === moving arm to initial position...", episode_index)
            self._return_to_initial_position(
                ctx.hardware, duration_s=return_to_initial_duration_s
            )
        else:
            logger.info(
                "=== Episode %d RESET === skipping homing (return_to_initial_position=false or no initial pose)",
                episode_index,
            )

        homing_s = time.perf_counter() - reset_start
        if episode_reset_time_s > 0:
            logger.info(
                "=== Episode %d RESET === homing took %.1fs — starting %.0fs manual scene window",
                episode_index,
                homing_s,
                episode_reset_time_s,
            )
            _wait_for_manual_scene_reset_window(
                episode_index,
                episode_reset_time_s,
                pause_key=manual_scene_reset_pause_key,
                collect_intervention=collect_intervention,
            )

        if self._engine is not None:
            self._engine.resume()
        logger.info("=== Episode %d RESET COMPLETE === resuming collection", episode_index)

    @staticmethod
    def _log_telemetry(
        obs_processed: dict | None,
        action_dict: dict | None,
        runtime_ctx: RuntimeContext,
    ) -> None:
        """Log observation/action telemetry to Rerun if display_data is enabled."""
        cfg = runtime_ctx.cfg
        if not cfg.display_data:
            return
        log_rerun_data(
            observation=obs_processed,
            action=action_dict,
            compress_images=cfg.display_compressed_images,
        )

    @abc.abstractmethod
    def setup(self, ctx: RolloutContext) -> None:
        """Strategy-specific initialisation (keyboard listeners, buffers, etc.)."""

    @abc.abstractmethod
    def run(self, ctx: RolloutContext) -> None:
        """Main rollout loop.  Returns when shutdown is requested or duration expires."""

    @abc.abstractmethod
    def teardown(self, ctx: RolloutContext) -> None:
        """Cleanup: save dataset, stop threads, disconnect hardware."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def safe_push_to_hub(dataset, tags=None, private=False) -> bool:
    """Push dataset to hub, skipping if no episodes have been saved.

    Returns ``True`` if the push was attempted, ``False`` if skipped.
    """
    if dataset.num_episodes == 0:
        logger.warning("No episodes saved — skipping push to hub")
        return False
    dataset.push_to_hub(tags=tags, private=private)
    return True


def estimate_max_episode_seconds(
    dataset_features: dict,
    fps: float,
    target_size_mb: float = DEFAULT_VIDEO_FILE_SIZE_IN_MB,
) -> float:
    """Conservatively estimate how many seconds of video will exceed *target_size_mb*.

    Each camera produces its own video file, so the episode duration is
    driven by the **slowest** camera to fill ``target_size_mb`` — i.e.
    the one with the fewest pixels per frame (lowest bitrate).

    Uses a deliberately **low** bits-per-pixel estimate so the computed
    duration is *longer* than reality.  By the time the timer fires the
    actual video file is guaranteed to have crossed the target size,
    which aligns episode boundaries with the dataset's video-file
    chunking — each ``push_to_hub`` uploads complete files rather than
    re-uploading a still-growing one.

    The estimate ignores codec-specific settings (CRF, preset) on purpose:
    we only need a rough lower bound on bitrate, not a precise prediction.

    Falls back to 300 s (5 min) when no video features are present.
    """
    # 0.1 bits-per-pixel is a *low* estimate for CRF-30 streaming video of
    # robot footage (real-world is typically 0.1 – 0.3 bpp).  Under-
    # estimating the bitrate over-estimates the time → the episode will be
    # *larger* than target_size_mb when we save, which is what we want.
    conservative_bpp = 0.1

    # Collect per-camera pixel counts — each camera has its own video file.
    camera_pixels = []
    for feat in dataset_features.values():
        if feat.get("dtype") == "video":
            shape = feat.get("shape", ())

            # (H, W, C) — bits-per-pixel is a per-spatial-pixel metric,
            # so we exclude the channel dimension from the count.
            if len(shape) == 3:
                pixels = shape[0] * shape[1]
                camera_pixels.append(pixels)
            else:
                raise ValueError(f"Unexpected video feature shape: {shape}")

    if not camera_pixels:
        return 300.0

    # Use the smallest camera: it produces the lowest bitrate and therefore
    # takes the longest to reach the target — the conservative choice.
    min_pixels = min(camera_pixels)
    bits_per_frame = min_pixels * conservative_bpp
    bytes_per_second = (bits_per_frame * fps) / 8

    # Guard against division by zero just in case
    if bytes_per_second <= 0:
        return 300.0

    return (target_size_mb * 1024 * 1024) / bytes_per_second


# ---------------------------------------------------------------------------
# Shared action-dispatch helper
# ---------------------------------------------------------------------------


def send_next_action(
    obs_processed: dict,
    obs_raw: dict,
    ctx: RolloutContext,
    interpolator: ActionInterpolator,
) -> dict | None:
    """Dispatch the next action to the robot.

    Pulls the next action tensor from the inference engine, feeds the
    interpolator, and sends the interpolated action through the
    ``robot_action_processor`` to the robot.  Works identically for
    sync and async backends — the rollout strategy never needs to branch.

    Returns the action dict that was sent, or ``None`` if no action was
    ready (e.g. empty async queue, interpolator not yet primed).
    """
    engine = ctx.policy.inference
    features = ctx.data.dataset_features
    ordered_keys = ctx.data.ordered_action_keys

    if interpolator.needs_new_action():
        obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
        action_tensor = engine.get_action(obs_frame)
        if action_tensor is not None:
            interpolator.add(action_tensor.cpu())

    interp = interpolator.get()
    if interp is None:
        return None

    if len(interp) != len(ordered_keys):
        raise ValueError(f"Interpolated tensor length ({len(interp)}) != action keys ({len(ordered_keys)})")
    action_dict = {k: interp[i].item() for i, k in enumerate(ordered_keys)}
    processed = ctx.processors.robot_action_processor((action_dict, obs_raw))
    ctx.hardware.robot_wrapper.send_action(processed)
    return action_dict
