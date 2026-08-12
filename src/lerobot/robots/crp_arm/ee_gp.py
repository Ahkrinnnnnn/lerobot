#!/usr/bin/env python

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

"""Shared CRP end-effector GP helpers (recording + HIL-SERL)."""

from __future__ import annotations

import logging
import multiprocessing
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lerobot.tools import TrajectoryProcessor
from lerobot.utils.robot_utils import precise_sleep

if TYPE_CHECKING:
    from .crp_arm import CRPArm

logger = logging.getLogger(__name__)

# Scale applied to OMY EE position delta (after ``ros_ee_pose_position_scale``), before adding to CRP pose.
EE_OMY_DELTA_SCALE = 1.5

# OMY ``rh_r1_joint`` gripper → CRP GOT0 (0=closed, 1000=open).
OMY_GRIPPER_RH_R1_CLIP_LO = -15.0
OMY_GRIPPER_RH_R1_CLIP_HI = 15.0
CRP_GRIPPER_GOT0_MAX = 1000

# ``init_matrix(..., group_size=5)`` layout expected by CRP ``set_GPs``.
GP_GROUP_SIZE = 5
DEFAULT_GP_START_INDEX = 10
DEFAULT_GP_STREAM_HZ = 100.0

# Default axis signs for ``gp = p0 + scale * step * (omy - ref)``. Config overlays these.
DEFAULT_EE_STEP_SIZES: dict[str, float] = {"x": 1.0, "y": 1.0, "z": 1.0}

# Consecutive-sample EE stability window before latching ref / arming GP deltas.
EE_STABLE_SAMPLES = 8
EE_STABLE_MAX_SPAN = 1.0  # same units as scaled EE (mm when scale=1000)
EE_STABLE_POLL_S = 0.02

# Throttle ``send_GPs`` pose logs. ``None`` or ``<=0`` disables logging.
GP_SEND_LOG_INTERVAL_S: float | None = None
_gp_send_log_last_mono: list[float] = [0.0]


def resolve_ee_step_sizes(cfg_steps: dict[str, float] | None) -> dict[str, float]:
    """Merge config step sizes onto ``DEFAULT_EE_STEP_SIZES`` (config keys win)."""
    out = {k: float(v) for k, v in DEFAULT_EE_STEP_SIZES.items()}
    if cfg_steps:
        for key, val in cfg_steps.items():
            out[str(key)] = float(val)
    return out


def resolve_ee_delta_scale(cfg_scale: float | None, default: float = EE_OMY_DELTA_SCALE) -> float:
    """Use config delta scale when set; otherwise ``default``."""
    if cfg_scale is None:
        return float(default)
    return float(cfg_scale)


def wait_stable_omy_ee_xyz(
    get_ee_fn,
    *,
    n_samples: int = EE_STABLE_SAMPLES,
    max_span: float = EE_STABLE_MAX_SPAN,
    poll_s: float = EE_STABLE_POLL_S,
    stop_fn=None,
    log_prefix: str = "EE stable",
) -> list[float]:
    """Block until ``n_samples`` consecutive EE xyz stay within ``max_span`` per axis.

    ``get_ee_fn`` should return the same shape as ``OMYL100.get_ros_end_effector_xyz_rpy_deg``
    (``(xyz, rpy)`` or ``None``). Returns the mean xyz of the stable window.
    """
    buf: list[list[float]] = []
    t0 = time.perf_counter()
    last_log = t0
    while True:
        if stop_fn is not None and stop_fn():
            raise RuntimeError(f"{log_prefix}: stopped before EE stabilized")
        ee = get_ee_fn()
        if ee is not None:
            xyz = [float(ee[0][i]) for i in range(3)]
            buf.append(xyz)
            if len(buf) > n_samples:
                buf.pop(0)
            if len(buf) == n_samples:
                span = [max(s[i] for s in buf) - min(s[i] for s in buf) for i in range(3)]
                if all(s <= max_span for s in span):
                    mean = [sum(s[i] for s in buf) / n_samples for i in range(3)]
                    logger.info(
                        "%s: ok after %.2fs mean=%.4f %.4f %.4f span=%.4f %.4f %.4f",
                        log_prefix,
                        time.perf_counter() - t0,
                        mean[0],
                        mean[1],
                        mean[2],
                        span[0],
                        span[1],
                        span[2],
                    )
                    return mean
        now = time.perf_counter()
        if now - last_log >= 2.0:
            last_log = now
            logger.info("%s: waiting (%d/%d consecutive samples)...", log_prefix, len(buf), n_samples)
        time.sleep(poll_s)


def log_gp_vec6(msg: str, vec6: Sequence[float]) -> None:
    """Log one 6-D endpose: xyz and rx/ry/rz in degrees (roll, pitch, yaw)."""
    if len(vec6) < 6:
        return
    x, y, z, rx, ry, rz = (float(vec6[i]) for i in range(6))
    logger.info("%s xyz=(%.6f %.6f %.6f) rx_ry_rz_deg=(%.6f %.6f %.6f)", msg, x, y, z, rx, ry, rz)


def log_gp_vec6_throttled(msg: str, vec6: Sequence[float]) -> None:
    if GP_SEND_LOG_INTERVAL_S is None or float(GP_SEND_LOG_INTERVAL_S) <= 0:
        return
    now = time.monotonic()
    if now - _gp_send_log_last_mono[0] < float(GP_SEND_LOG_INTERVAL_S):
        return
    _gp_send_log_last_mono[0] = now
    log_gp_vec6(msg, vec6)


def log_gp_points_matrix(msg: str, rows: list[list[float]]) -> None:
    """Log GP matrix: row count and first row as xyz + rx/ry/rz (deg)."""
    if not rows or len(rows[0]) < 6:
        return
    n = len(rows)
    same = n > 1 and all(r == rows[0] for r in rows[1:])
    suffix = f" ({n} duplicate rows)" if same and n > 1 else f" ({n} rows, logging first row)"
    log_gp_vec6(msg + suffix, rows[0])


def apply_xyz_delta_to_endpose(
    p0: Sequence[float],
    dxyz: Sequence[float],
    *,
    hold_rpy: Sequence[float] | None = None,
) -> list[float]:
    """Return absolute 6-D GP pose: ``p0[:3] + dxyz``; orientation = ``hold_rpy`` or ``p0[3:6]``.

    No handwritten fixed orientation — callers latch current CRP rpy (recording / HIL).
    """
    if hold_rpy is None:
        if len(p0) < 6:
            raise ValueError("p0 must include rpy (len>=6) when hold_rpy is omitted")
        hold_rpy = (p0[3], p0[4], p0[5])
    return [
        float(p0[0]) + float(dxyz[0]),
        float(p0[1]) + float(dxyz[1]),
        float(p0[2]) + float(dxyz[2]),
        float(hold_rpy[0]),
        float(hold_rpy[1]),
        float(hold_rpy[2]),
    ]


def omy_relative_xyz_to_gp6(
    p0_xyz: Sequence[float],
    omy_now_xyz: Sequence[float],
    omy_ref_xyz: Sequence[float],
    *,
    hold_rpy: Sequence[float],
    step_sizes: dict[str, float] | None = None,
    scale: float = EE_OMY_DELTA_SCALE,
) -> list[float]:
    """Relative OMY xyz → absolute GP6; orientation held at ``hold_rpy`` (never from OMY).

    ``crp_xyz = p0 + scale * step_sizes[axis] * (omy_now - omy_ref)``.
    Negative ``step_sizes`` invert an axis (e.g. ROS Z-up vs CRP user frame).
    """
    sizes = step_sizes or {}
    dxyz = [
        float(scale) * float(sizes.get(axis, 1.0)) * (float(omy_now_xyz[i]) - float(omy_ref_xyz[i]))
        for i, axis in enumerate(("x", "y", "z"))
    ]
    return apply_xyz_delta_to_endpose(p0_xyz, dxyz, hold_rpy=hold_rpy)


def wrap_angle_delta_deg(now: float, ref: float) -> float:
    """Signed shortest angle difference ``now - ref`` in degrees, in ``(-180, 180]``."""
    d = float(now) - float(ref)
    d = (d + 180.0) % 360.0 - 180.0
    return d


def ee_action_to_crp_endpose_list(action: dict[str, float]) -> list[float]:
    """Build a 6-DOF GP vector from ``ee.*`` keys; no coordinate transform."""
    return [
        float(action.get("ee.x", 0.0)),
        float(action.get("ee.y", 0.0)),
        float(action.get("ee.z", 0.0)),
        float(action.get("ee.roll", 0.0)),
        float(action.get("ee.pitch", 0.0)),
        float(action.get("ee.yaw", 0.0)),
    ]


def omy_rh_r1_to_got0(raw: float) -> int:
    """Clamp OMY gripper joint value to [-15, 15], map linearly to GOT0 in [0, 1000]."""
    raw = raw * 180 / 3.14
    c = max(OMY_GRIPPER_RH_R1_CLIP_LO, min(OMY_GRIPPER_RH_R1_CLIP_HI, float(raw)))
    span = OMY_GRIPPER_RH_R1_CLIP_HI - OMY_GRIPPER_RH_R1_CLIP_LO
    v = float(CRP_GRIPPER_GOT0_MAX) * (c - OMY_GRIPPER_RH_R1_CLIP_LO) / span
    return int(max(0, min(CRP_GRIPPER_GOT0_MAX, round(v))))


def read_omy_ee_xyz(teleop: Any) -> tuple[float, float, float] | None:
    """Scaled OMY EE xyz from the same helper used by recording / HIL."""
    if teleop is None or not hasattr(teleop, "get_ros_end_effector_xyz_rpy_deg"):
        return None
    ee = teleop.get_ros_end_effector_xyz_rpy_deg()
    if ee is None:
        return None
    return (float(ee[0][0]), float(ee[0][1]), float(ee[0][2]))


def send_gp_endpose6(
    robot: CRPArm,
    trajectory_processor: TrajectoryProcessor,
    vec6: list[float],
    *,
    start_index: int = DEFAULT_GP_START_INDEX,
    group_size: int = GP_GROUP_SIZE,
    switch_to_gp_mode: bool = True,
) -> None:
    """Send latest 6-D endpose as a ``group_size``×6 GP matrix.

    Pass ``switch_to_gp_mode=False`` to preload registers without changing GI;
    call ``robot.ensure_gp_mode()`` when teleop is ready to follow GP.
    """
    mat = trajectory_processor.init_matrix([float(x) for x in vec6], group_size=group_size)
    log_gp_vec6_throttled(f"send_GPs({start_index}) before", vec6)
    robot.send_GPs(start_index, mat, switch_to_gp_mode=switch_to_gp_mode)


def run_omy_relative_gp_stream(
    *,
    robot: CRPArm,
    teleop: Any,
    trajectory_processor: TrajectoryProcessor,
    stop_event: threading.Event,
    p0_xyz: Sequence[float],
    hold_rpy: Sequence[float],
    omy_ref_xyz: Sequence[float],
    init_pose6: Sequence[float],
    step_sizes: dict[str, float] | None = None,
    delta_scale: float = EE_OMY_DELTA_SCALE,
    start_index: int = DEFAULT_GP_START_INDEX,
    group_size: int = GP_GROUP_SIZE,
    stream_hz: float = DEFAULT_GP_STREAM_HZ,
    use_gripper: bool = True,
    latest_got0_holder: list[int] | None = None,
    got0_lock: threading.Lock | None = None,
    log_prefix: str = "OMY EE→CRP",
    log_interval_s: float = 0.5,
    relatch_omy_ref_on_start: bool = True,
) -> None:
    """Blocking GP stream used by **both** recording and HIL intervention.

    Process-safe contract:
    - Must run in the **same process** that owns ``robot`` (CRP SDK + ``_motion_cmd_lock``).
    - Must sample OMY from the **same** ``teleop`` that latched ``omy_ref_xyz`` (no second ROS node).
    - ``p0_xyz`` / ``hold_rpy`` are frozen; ``omy_ref`` may be re-latched on the first in-stream
      sample so the opening frame has zero delta (avoids latch→start noise jumps).
    """
    p0 = (float(p0_xyz[0]), float(p0_xyz[1]), float(p0_xyz[2]))
    hold = (float(hold_rpy[0]), float(hold_rpy[1]), float(hold_rpy[2]))
    ref = (float(omy_ref_xyz[0]), float(omy_ref_xyz[1]), float(omy_ref_xyz[2]))
    latest_gp6 = [float(x) for x in init_pose6]
    latest_got0 = int(latest_got0_holder[0]) if latest_got0_holder else 0
    last_got_sent = latest_got0
    cmd_lock = threading.Lock()
    hz = max(1.0, float(stream_hz))
    period = 1.0 / hz
    log_last = 0.0
    next_t = time.perf_counter()
    need_relatch = bool(relatch_omy_ref_on_start)
    proprio_refresh_every = max(1, int(round(hz / 10.0)))  # ~10 Hz joint cache refresh
    tick_i = 0

    logger.info(
        "%s stream start @%.0fHz p0=(%.4f %.4f %.4f) ref=(%.4f %.4f %.4f) hold_rpy=(%.2f %.2f %.2f) "
        "relatch_on_start=%s",
        log_prefix,
        hz,
        *p0,
        *ref,
        *hold,
        need_relatch,
    )

    while not stop_event.is_set():
        if not getattr(robot, "is_connected", True):
            logger.info("%s: robot disconnected — stopping stream", log_prefix)
            stop_event.set()
            return

        # Space released → cut OMY control immediately (do not wait for next actor step).
        kb = getattr(teleop, "_hil_keyboard", None)
        if kb is not None and not bool(getattr(kb, "intervening", True)):
            logger.info("%s: Space released — stopping OMY→GP stream (no further cmds)", log_prefix)
            stop_event.set()
            return

        omy_now = read_omy_ee_xyz(teleop)
        if omy_now is not None:
            if need_relatch:
                # Opening frame: OMY now becomes ref → delta=0; keep publishing hold pose.
                ref = (float(omy_now[0]), float(omy_now[1]), float(omy_now[2]))
                need_relatch = False
                hold_pose = apply_xyz_delta_to_endpose(p0, (0.0, 0.0, 0.0), hold_rpy=hold)
                with cmd_lock:
                    latest_gp6 = list(hold_pose)
                logger.info(
                    "%s re-latched omy_ref on first stream sample xyz=(%.4f %.4f %.4f) — "
                    "holding CRP p0 (opening delta=0)",
                    log_prefix,
                    *ref,
                )
            else:
                gp6 = omy_relative_xyz_to_gp6(
                    p0,
                    omy_now,
                    ref,
                    hold_rpy=hold,
                    step_sizes=step_sizes,
                    scale=delta_scale,
                )
                dxyz = [gp6[i] - p0[i] for i in range(3)]
                with cmd_lock:
                    latest_gp6 = list(gp6)
                now_m = time.monotonic()
                if now_m - log_last >= float(log_interval_s):
                    log_last = now_m
                    logger.info(
                        "%s @%.0fHz: omy_now=%.4f %.4f %.4f ref=%.4f %.4f %.4f "
                        "delta=%.4f %.4f %.4f gp6=[%.4f %.4f %.4f %.2f %.2f %.2f]",
                        log_prefix,
                        hz,
                        omy_now[0],
                        omy_now[1],
                        omy_now[2],
                        ref[0],
                        ref[1],
                        ref[2],
                        dxyz[0],
                        dxyz[1],
                        dxyz[2],
                        gp6[0],
                        gp6[1],
                        gp6[2],
                        gp6[3],
                        gp6[4],
                        gp6[5],
                    )

        if use_gripper and teleop is not None and hasattr(teleop, "get_gripper_raw"):
            try:
                got0 = int(omy_rh_r1_to_got0(float(teleop.get_gripper_raw())))
            except Exception:
                got0 = latest_got0
            with cmd_lock:
                latest_got0 = got0
            if latest_got0_holder is not None:
                if got0_lock is not None:
                    with got0_lock:
                        latest_got0_holder[0] = got0
                else:
                    latest_got0_holder[0] = got0

        with cmd_lock:
            vec = list(latest_gp6)
            got_send = int(latest_got0)

        # Re-check after OMY sample: Space-up may have cut mid-iteration.
        if stop_event.is_set():
            return
        kb = getattr(teleop, "_hil_keyboard", None)
        if kb is not None and not bool(getattr(kb, "intervening", True)):
            logger.info("%s: Space released before send — drop this OMY GP", log_prefix)
            stop_event.set()
            return

        try:
            send_gp_endpose6(
                robot,
                trajectory_processor,
                vec,
                start_index=start_index,
                group_size=group_size,
            )
        except Exception:
            logger.exception("%s send_GPs failed — stopping stream", log_prefix)
            stop_event.set()
            return

        # HIL in-process stream: mirror commanded GP into EE cache (no extra SDK read).
        # Recording uses measured EE via fork shared memory instead — see crp_record_omy_ee_inc.
        if hasattr(robot, "update_ee_cache_from_pose6"):
            robot.update_ee_cache_from_pose6(vec)
        tick_i += 1
        if tick_i % proprio_refresh_every == 0 and hasattr(robot, "refresh_proprio_cache"):
            robot.refresh_proprio_cache()

        # Avoid 100Hz GOT spam when unchanged (extra SDK load next to joint refresh).
        if use_gripper and hasattr(robot, "set_GOT") and got_send != last_got_sent:
            try:
                robot.set_GOT(0, got_send)
                last_got_sent = got_send
            except Exception:
                logger.warning(
                    "%s set_GOT failed (got0=%s); continuing",
                    log_prefix,
                    got_send,
                    exc_info=True,
                )

        next_t += period
        sleep_s = next_t - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            next_t = time.perf_counter()


# ---------------------------------------------------------------------------
# Multiprocess stream: fork GP sender (inherits CRP SDK) + spawn OMY reader (fresh rclpy)
# ---------------------------------------------------------------------------


def _fork_mp_ctx() -> "multiprocessing.context.BaseContext":
    """``fork`` so the GP sender inherits the live CRP SDK / TrajectoryProcessor."""
    return multiprocessing.get_context("fork")


def _spawn_mp_ctx() -> "multiprocessing.context.BaseContext":
    """``spawn`` so the OMY worker gets a fresh ``rclpy`` interpreter."""
    return multiprocessing.get_context("spawn")


def _spawn_omy_gp6_worker(
    *,
    teleop_cfg: dict[str, Any],
    omy_action_stop: Any,
    gp_cmd_lock: Any,
    latest_gp6: Any,
    shared_p0: Any,
    shared_omy_ref: Any,
    shared_hold_rpy: Any,
    step_sizes: dict[str, float],
    delta_scale: float,
    stream_hz: float,
    ros_gripper_joint_name: str,
    gripper_got_lock: Any,
    latest_got0_holder: Any,
    latest_obs_for_action_lock: Any = None,
    latest_obs_for_action: Any = None,
    omy_display_action_lock: Any = None,
    omy_display_action: Any = None,
) -> None:
    """``spawn`` process: poll OMY EE / gripper into Manager buffers (no CRP SDK).

    Holds ``latest_gp6`` at the parent init pose until spawn EE is near the parent-latched
    ``shared_omy_ref`` (or a short timeout), then writes relative GP deltas. Does not rewrite ref.
    """
    import os

    import rclpy as _rclpy

    from lerobot.processor import make_default_processors
    from lerobot.teleoperators.OMY_L100 import OMYL100
    from lerobot.teleoperators.OMY_L100.config_OMY_L100 import OMYL100Config

    _orig_create_node = _rclpy.create_node

    def _create_node_spawn(name: str, *args: Any, **kwargs: Any):
        return _orig_create_node(f"{name}_spawn_{os.getpid()}", *args, **kwargs)

    _rclpy.create_node = _create_node_spawn
    teleop_action_processor, robot_action_processor, _unused_obs = make_default_processors()
    teleop: Any = None
    try:
        cfg = OMYL100Config(**teleop_cfg)
        cfg.hil_ee_delta = False
        teleop = OMYL100(cfg)
        teleop.connect()
        hz = max(1.0, float(stream_hz))
        period = 1.0 / hz
        t_next = time.perf_counter()
        sizes = dict(step_sizes or {})
        # Hold ``latest_gp6`` (= parent init_pose) until spawn EE is near parent-latched ref.
        # Avoids opening jump from a warm-up / mismatched first Pose on the new ROS node
        # without rewriting ``shared_omy_ref`` (fork never re-latches ref).
        open_eps = 15.0
        opened = False
        open_wait_t0 = time.perf_counter()
        while not omy_action_stop.is_set():
            act = teleop.get_action()

            obs_for_action: dict[str, Any] = {}
            if latest_obs_for_action_lock is not None and latest_obs_for_action is not None:
                with latest_obs_for_action_lock:
                    obs_for_action = dict(latest_obs_for_action.copy())

            act_processed = teleop_action_processor((act, obs_for_action))
            robot_action_to_send_tamp = robot_action_processor((act_processed, obs_for_action))

            ee_pair = teleop.get_ros_end_effector_xyz_rpy_deg()
            if ee_pair is not None:
                omy_now = [float(ee_pair[0][i]) for i in range(3)]
                p0 = [float(shared_p0[i]) for i in range(3)]
                hold = [float(shared_hold_rpy[i]) for i in range(3)]
                ref = [float(shared_omy_ref[i]) for i in range(3)]
                if not opened:
                    drift = max(abs(omy_now[i] - ref[i]) for i in range(3))
                    waited = time.perf_counter() - open_wait_t0
                    if drift <= open_eps or waited >= 2.0:
                        opened = True
                        logging.getLogger(__name__).info(
                            "OMY spawn EE armed (drift=%.3f eps=%.3f waited=%.2fs) — releasing GP deltas",
                            drift,
                            open_eps,
                            waited,
                        )
                if opened:
                    gp6 = omy_relative_xyz_to_gp6(
                        p0,
                        omy_now,
                        ref,
                        hold_rpy=hold,
                        step_sizes=sizes,
                        scale=float(delta_scale),
                    )
                    with gp_cmd_lock:
                        for _i in range(6):
                            latest_gp6[_i] = gp6[_i]

            if ros_gripper_joint_name:
                got0 = omy_rh_r1_to_got0(float(robot_action_to_send_tamp.get("gripper.pos", 0.0)))
                if gripper_got_lock is not None and latest_got0_holder is not None:
                    with gripper_got_lock:
                        latest_got0_holder[0] = got0

            if omy_display_action_lock is not None and omy_display_action is not None:
                with omy_display_action_lock:
                    omy_display_action.clear()
                    omy_display_action.update(act_processed)

            t_next += period
            dt = t_next - time.perf_counter()
            if dt > 0:
                precise_sleep(dt)
            else:
                t_next = time.perf_counter()
    finally:
        _rclpy.create_node = _orig_create_node
        if teleop is not None:
            try:
                teleop.disconnect()
            except Exception:
                logging.getLogger(__name__).exception("OMY spawn worker disconnect failed")


@dataclass
class MpGpStreamHandle:
    """Handle for fork GP sender + spawn OMY worker + Manager shared buffers."""

    omy_action_stop: Any
    gp_sender_stop: Any
    gp_sender: Any
    omy_worker: Any
    manager: Any
    latest_got0_holder: Any | None = None
    gripper_got_lock: Any | None = None
    latest_gp6: Any | None = None
    omy_display_action_lock: Any | None = None
    omy_display_action: Any | None = None
    latest_obs_for_action_lock: Any | None = None
    latest_obs_for_action: Any | None = None


def start_omy_relative_gp_mp_stream(
    *,
    robot: CRPArm,
    trajectory_processor: TrajectoryProcessor,
    teleop_cfg: dict[str, Any],
    p0_xyz: Sequence[float],
    hold_rpy: Sequence[float],
    omy_ref_xyz: Sequence[float],
    init_pose6: Sequence[float],
    step_sizes: dict[str, float] | None = None,
    delta_scale: float = EE_OMY_DELTA_SCALE,
    start_index: int = DEFAULT_GP_START_INDEX,
    group_size: int = GP_GROUP_SIZE,
    stream_hz: float = DEFAULT_GP_STREAM_HZ,
    use_gripper: bool = True,
    ros_gripper_joint_name: str = "",
    init_got0: int = 0,
    log_prefix: str = "MP OMY EE→CRP",
) -> MpGpStreamHandle:
    """Start fork ``send_GPs`` sender then spawn OMY EE updater (HIL intervention path).

    Sync objects shared with spawn come from ``Manager`` (never a plain ``fork_ctx.Event()`` —
    that SemLock cannot cross into a spawn child). GP sender stop uses ``fork_ctx.Event()``
    and is not passed to spawn.

    Parent must enable proprio/EE cache and avoid CRP SDK calls while the fork
    sender is alive (otherwise Thrift dies under concurrent use). Callers should
    preload GP registers and ``ensure_gp_mode()`` before starting this stream.
    """
    fork_ctx = _fork_mp_ctx()
    spawn_ctx = _spawn_mp_ctx()
    manager = fork_ctx.Manager()

    # Shared with spawn (Manager proxies).
    omy_action_stop = manager.Event()
    gp_cmd_lock = manager.Lock()
    shared_p0 = manager.list([float(p0_xyz[0]), float(p0_xyz[1]), float(p0_xyz[2])])
    shared_omy_ref = manager.list(
        [float(omy_ref_xyz[0]), float(omy_ref_xyz[1]), float(omy_ref_xyz[2])]
    )
    shared_hold_rpy = manager.list(
        [float(hold_rpy[0]), float(hold_rpy[1]), float(hold_rpy[2])]
    )
    latest_gp6 = manager.list([float(x) for x in init_pose6])
    omy_display_action_lock = manager.Lock()
    omy_display_action = manager.dict()
    latest_obs_for_action_lock = manager.Lock()
    latest_obs_for_action = manager.dict()

    sizes = resolve_ee_step_sizes(step_sizes)
    grip_name = str(ros_gripper_joint_name or "")
    use_grip = bool(use_gripper and grip_name)
    gripper_got_lock = manager.Lock() if use_grip else None
    latest_got0_holder = manager.list([int(init_got0)]) if use_grip else None

    # Fork-only stop (not shared with spawn).
    gp_sender_stop = fork_ctx.Event()

    hz = max(1.0, float(stream_hz))
    period = 1.0 / hz

    def _fixed_rate_gp_sender() -> None:
        # Hot path = set_GPs (+ rare set_GOT) only. No read_joints during stream.
        t_next = time.perf_counter()
        last_got0_sent = -10**9
        while not gp_sender_stop.is_set():
            with gp_cmd_lock:
                vec = [float(x) for x in latest_gp6]
            try:
                send_gp_endpose6(
                    robot,
                    trajectory_processor,
                    vec,
                    start_index=start_index,
                    group_size=group_size,
                    switch_to_gp_mode=False,
                )
            except Exception:
                logging.getLogger(__name__).exception(
                    "%s send_GPs failed — stopping stream", log_prefix
                )
                gp_sender_stop.set()
                omy_action_stop.set()
                return
            # HIL MP stream: commanded pose into EE cache only (no measured read on hot path).
            if hasattr(robot, "update_ee_cache_from_pose6"):
                robot.update_ee_cache_from_pose6(vec)
            if use_grip and latest_got0_holder is not None and gripper_got_lock is not None:
                with gripper_got_lock:
                    got_sent = int(latest_got0_holder[0])
                if got_sent != last_got0_sent:
                    try:
                        robot.set_GOT(0, got_sent)
                        last_got0_sent = got_sent
                    except Exception:
                        logging.getLogger(__name__).warning(
                            "%s set_GOT failed (got0=%s)", log_prefix, got_sent, exc_info=True
                        )
            t_next += period
            dt = t_next - time.perf_counter()
            if dt > 0:
                precise_sleep(dt)
            else:
                t_next = time.perf_counter()

    gp_sender = fork_ctx.Process(target=_fixed_rate_gp_sender, name="crp_gp_sender", daemon=True)
    gp_sender.start()

    omy_worker = spawn_ctx.Process(
        target=_spawn_omy_gp6_worker,
        name="omy_ee_gp_update",
        daemon=True,
        kwargs={
            "teleop_cfg": dict(teleop_cfg),
            "omy_action_stop": omy_action_stop,
            "gp_cmd_lock": gp_cmd_lock,
            "latest_gp6": latest_gp6,
            "shared_p0": shared_p0,
            "shared_omy_ref": shared_omy_ref,
            "shared_hold_rpy": shared_hold_rpy,
            "step_sizes": sizes,
            "delta_scale": float(delta_scale),
            "stream_hz": hz,
            "ros_gripper_joint_name": grip_name if use_grip else "",
            "gripper_got_lock": gripper_got_lock,
            "latest_got0_holder": latest_got0_holder,
            "latest_obs_for_action_lock": latest_obs_for_action_lock,
            "latest_obs_for_action": latest_obs_for_action,
            "omy_display_action_lock": omy_display_action_lock,
            "omy_display_action": omy_display_action,
        },
    )
    omy_worker.start()
    logger.info(
        "%s MP stream start @%.0fHz (fork GP → spawn OMY) "
        "p0=(%.4f %.4f %.4f) ref=(%.4f %.4f %.4f) hold_rpy=(%.2f %.2f %.2f)",
        log_prefix,
        hz,
        float(p0_xyz[0]),
        float(p0_xyz[1]),
        float(p0_xyz[2]),
        float(omy_ref_xyz[0]),
        float(omy_ref_xyz[1]),
        float(omy_ref_xyz[2]),
        float(hold_rpy[0]),
        float(hold_rpy[1]),
        float(hold_rpy[2]),
    )
    return MpGpStreamHandle(
        omy_action_stop=omy_action_stop,
        gp_sender_stop=gp_sender_stop,
        gp_sender=gp_sender,
        omy_worker=omy_worker,
        manager=manager,
        latest_got0_holder=latest_got0_holder,
        gripper_got_lock=gripper_got_lock,
        latest_gp6=latest_gp6,
        omy_display_action_lock=omy_display_action_lock,
        omy_display_action=omy_display_action,
        latest_obs_for_action_lock=latest_obs_for_action_lock,
        latest_obs_for_action=latest_obs_for_action,
    )


def stop_omy_relative_gp_mp_stream(handle: MpGpStreamHandle | None, *, join_timeout_s: float = 2.0) -> None:
    """Signal both stops and join fork GP sender + spawn OMY worker."""
    if handle is None:
        return
    for ev_name in ("omy_action_stop", "gp_sender_stop"):
        ev = getattr(handle, ev_name, None)
        if ev is None:
            continue
        try:
            ev.set()
        except Exception:
            pass
    for proc in (handle.gp_sender, handle.omy_worker):
        if proc is None:
            continue
        try:
            if proc.is_alive():
                proc.join(timeout=join_timeout_s)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)
        except Exception:
            logger.warning("Error stopping GP MP worker %s", getattr(proc, "name", proc), exc_info=True)
    try:
        handle.manager.shutdown()
    except Exception:
        pass
