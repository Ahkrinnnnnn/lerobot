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

"""
Record a LeRobot dataset: OMY_L100 teleop → CRP arm via **incremental GP** from ROS EE xyz
(no joint-angle mapping to CRP).

Control (episode-relative; OMY EE xyz and CRP share the same frame):
  ``crp_xyz = p0 + scale * step_sizes * (omy_now - omy_ref)``; rpy held at latched CRP pose.

Process layout (CRP Thrift is not safe to share — fork owns SDK during the stream):
  - **spawn**: ROS EE→``latest_gp6`` @stream Hz; gripper via ``get_gripper_raw``→GOT
    (processors/display ~30Hz only).
  - **fork**: ``set_GPs`` @stream Hz; ``set_GOT`` **only when value changes** (dedup).
  - **parent**: cameras + ``add_frame`` at dataset fps; no CRP SDK while fork lives.

Gripper timing (smooth checkpoint):
  - Until armed: no ``set_GOT``.
  - At arm: parent writes held/seed GOT once, then starts fork (dedup seeded to that value).
  - After ``omy_gp_armed``: spawn updates GOT buffer from ``get_gripper_raw``; ignore
    OMY until |ΔGOT| ≥ ``GOT_FOLLOW_DELTA`` vs seed (avoid re-arm snap-shut).
  - Fork never spams unchanged GOT next to ``set_GPs``.


Dataset labeling (OMY commands EE+gripper; joints are measured on the CRP):
  - ``action.ee`` / gripper: current command (``latest_gp6`` / GOT).
  - ``obs.joint``: measured ``read_joints`` at frame t (fork samples ~dataset fps).
  - ``action.joint``: measured joints at frame t+1 (next-state target). The final
    camera/obs tick is dropped (no next joint) — last transition still uses that
    last measurement as ``action.joint`` of the previous row.
  - ``obs.ee`` / ``obs.gripper``: previous frame's action (1-frame delayed command).
    Do **not** sample ``read_end_pose`` while streaming — that starved GP / froze EE before.

Arming:
  1. Spawn waits for stable EE, latches ``omy_ref``, signals ready (no deltas yet).
  2. Parent preloads GP, ``ensure_gp_mode``, GI56 ON, **starts fork** (streams init pose).
  3. Then ``omy_gp_armed`` — spawn re-latches ``omy_ref`` (opening delta=0) and starts deltas.
  Until armed: no GP/GJ/GOT/GI→GP; GI56 stays off.

Teardown: stop spawn/fork first, then GI56 OFF; keep last GP. Never write GI while fork owns SDK.

Between episodes: ``reset_teleop=false`` (default) hold still for manual reset;
  ``true`` re-arms OMY teleop during the reset phase.

Helpers: ``lerobot.robots.crp_arm.ee_gp``. Gripper: JointState → GOT0 / ``gripper.pos``.

Example: ``python -m lerobot.scripts.crp_record_omy_ee_inc --config_path=examples/hilserl/record/crp_omy_record.json``
(``--teleop.port`` is required by schema but unused; OMY is ROS-based).
"""

# TrajectoryProcessor lives in lerobot.tools.TrajProcessor (file TrajProcessor.py);
# lerobot.tools.__init__ re-exports it — use the package import, not lerobot.tools.TrajectoryProcessor.
from lerobot.tools import TrajectoryProcessor

import logging
import multiprocessing
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import Robot, RobotConfig, make_robot_from_config

# Side-effect imports: register RobotConfig subclasses with draccus (not re-exported from lerobot.robots).
import lerobot.robots.bi_so_follower  # noqa: F401
import lerobot.robots.hope_jr  # noqa: F401
import lerobot.robots.koch_follower  # noqa: F401
import lerobot.robots.so_follower  # noqa: F401 — so100_follower / so101_follower

from lerobot.teleoperators import Teleoperator, TeleoperatorConfig, make_teleoperator_from_config

import lerobot.teleoperators.bi_so_leader  # noqa: F401
import lerobot.teleoperators.homunculus  # noqa: F401
import lerobot.teleoperators.koch_leader  # noqa: F401
import lerobot.teleoperators.so_leader  # noqa: F401
from lerobot.teleoperators.koch_leader import KochLeader
from lerobot.teleoperators.OMY_L100 import OMYL100  # noqa: F401 — registers OMY_L100 teleop config
from lerobot.teleoperators.OMY_L100.OMY_L100 import EE_STATES_TOPIC
from lerobot.teleoperators.so_leader import SO100Leader, SO101Leader
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop

# Import CRPArm after ROS teleop imports; native SDK loads lazily on first robot connect.
from lerobot.robots.crp_arm import CRPArm
from lerobot.robots.crp_arm.ee_gp import (
    DEFAULT_EE_STEP_SIZES,
    DEFAULT_GP_STREAM_HZ,
    EE_OMY_DELTA_SCALE,
    GP_GROUP_SIZE,
    ee_action_to_crp_endpose_list,
    log_gp_points_matrix,
    omy_relative_xyz_to_gp6,
    omy_rh_r1_to_got0,
    resolve_ee_delta_scale,
    resolve_ee_step_sizes,
    send_gp_endpose6,
    wait_stable_omy_ee_xyz,
)
from lerobot.common.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.constants import ACTION
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


def _record_multiprocessing_context() -> "multiprocessing.context.BaseContext":
    """Use ``fork`` so the GP sender child can use the already-created ``robot`` / ``trajectory_processor``."""
    return multiprocessing.get_context("fork")


def _omy_spawn_context() -> "multiprocessing.context.BaseContext":
    """Fresh interpreter for OMY / EE workers so ``rclpy`` in spawn children is valid."""
    return multiprocessing.get_context("spawn")


# Recording-only log throttle for spawn / fork debug lines.
CRP_EE_SPAWN_LOG_INTERVAL_S = 0.5

# After arm / re-arm: hold commanded GOT until OMY gripper moves by this much (GOT units).
# Prevents snap-shut when the master spring returns near closed at episode start.
GOT_FOLLOW_DELTA = 40

# ANSI colors for terminal readiness cues (no CRP LED API in CrpRobotPy bindings).
_ANSI_RESET = "\033[0m"
_ANSI_BOLD_RED = "\033[1;91m"
_ANSI_BOLD_YELLOW = "\033[1;93m"


_ANSI_BOLD_CYAN = "\033[1;96m"


def _print_ee_wait_stage(msg: str, *, color: str = _ANSI_BOLD_YELLOW) -> None:
    """Stage cue during EE wait (not yet safe to teleop)."""
    print(f"{color}{msg}{_ANSI_RESET}", flush=True)


def _print_teleop_ready_banner(*, phase: str = "record", reset_teleop: bool = False) -> None:
    """Loud cue for record (GP armed) or reset (manual hold / optional teleop)."""
    line = "=" * 72
    if phase == "reset":
        if reset_teleop:
            print(
                f"{_ANSI_BOLD_CYAN}{line}\n"
                f"  ◆ 复位阶段遥操就绪（用 OMY 摆场景 / 回位，本段不写入 episode）\n"
                f"  reset_teleop=true：臂+夹爪跟随 OMY。右键结束复位→save_episode。\n"
                f"{line}{_ANSI_RESET}",
                flush=True,
            )
        else:
            print(
                f"{_ANSI_BOLD_CYAN}{line}\n"
                f"  ◆ 复位阶段：机械臂保持不动（reset_teleop=false，手动复原）\n"
                f"  不遥操臂/夹爪；本段不写入 episode。右键结束复位→save_episode。\n"
                f"{line}{_ANSI_RESET}",
                flush=True,
            )
        return
    print(
        f"{_ANSI_BOLD_RED}{line}\n"
        f"  ★★★  录制阶段 EE→GP 已就绪：可以开始遥操（动 OMY）  ★★★\n"
        f"  右键=结束本段→复位→写盘；左键=结束本段→复位→清空缓冲重录。\n"
        f"{line}{_ANSI_RESET}",
        flush=True,
    )


def _spawn_omy_ee_gp_action_worker(
    *,
    teleop_cfg: dict[str, Any],
    omy_action_stop: Any,
    omy_stream_ready: Any,
    omy_gp_armed: Any,
    gp_cmd_lock: Any,
    latest_gp6: Any,
    shared_p0: Any,
    shared_omy_ref: Any,
    shared_hold_rpy: Any,
    step_sizes: dict[str, float],
    delta_scale: float,
    latest_obs_for_action_lock: Any,
    latest_obs_for_action: Any,
    omy_display_action_lock: Any,
    omy_display_action: Any,
    gripper_got_lock: Any,
    latest_got0_holder: Any,
    ros_gripper_joint_name: str,
) -> None:
    """``spawn`` process: ROS + EE→GP into shared-memory buffers (Array/Lock)."""
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
    teleop: OMYL100 | None = None
    _ee_dbg_last = [0.0]
    _log = logging.getLogger(__name__)
    resolved_steps = resolve_ee_step_sizes(step_sizes)
    scale = resolve_ee_delta_scale(delta_scale)
    try:
        cfg = OMYL100Config(**teleop_cfg)
        teleop = OMYL100(cfg)
        teleop.connect()

        # Stabilize EE in this process before any delta GP; then signal parent to arm CRP GP mode.
        # Do not overwrite parent-seeded GOT here (held across episodes until arm).
        ref_xyz = wait_stable_omy_ee_xyz(
            teleop.get_ros_end_effector_xyz_rpy_deg,
            stop_fn=omy_action_stop.is_set,
            log_prefix="OMY EE spawn stable",
        )
        for _i in range(3):
            shared_omy_ref[_i] = float(ref_xyz[_i])
        omy_stream_ready.set()
        _log.info(
            "OMY EE spawn: ref ready omy_ref=%s step_sizes=%s scale=%.2f — "
            "waiting for parent omy_gp_armed (no delta until then)",
            ref_xyz,
            resolved_steps,
            scale,
        )
        while not omy_action_stop.is_set() and not omy_gp_armed.wait(timeout=0.5):
            pass
        if omy_action_stop.is_set():
            return
        # Re-latch omy_ref at arm time (parent may have spent time preloading / starting
        # fork). Opening delta must be 0 — otherwise the first GP jump feels steep.
        _arm_ee = teleop.get_ros_end_effector_xyz_rpy_deg()
        if _arm_ee is not None:
            for _i in range(3):
                shared_omy_ref[_i] = float(_arm_ee[0][_i])
            p0_hold = [float(shared_p0[i]) for i in range(3)] + [
                float(shared_hold_rpy[i]) for i in range(3)
            ]
            with gp_cmd_lock:
                for _i in range(6):
                    latest_gp6[_i] = p0_hold[_i]
            _log.info(
                "OMY EE spawn: omy_gp_armed — re-latched omy_ref=%s (delta0); starting EE→GP",
                [float(_arm_ee[0][i]) for i in range(3)],
            )
        else:
            _log.info("OMY EE spawn: omy_gp_armed — starting EE→GP deltas (no EE for re-latch)")

        # Hot path @stream Hz (smooth checkpoint): EE→GP + raw gripper→GOT only.
        # Processors / Manager display ~30Hz — 100Hz Manager+processor stuttered GP,
        # especially while GOT was changing during grasp.
        period = 1.0 / DEFAULT_GP_STREAM_HZ
        display_every = max(1, int(round(DEFAULT_GP_STREAM_HZ / 30.0)))
        t_next = time.perf_counter()
        _none_streak = 0
        tick = 0
        # Hold seed GOT until OMY moves enough (same as re-arm snap-shut guard).
        _grip_seed: int | None = None
        if gripper_got_lock is not None and latest_got0_holder is not None:
            with gripper_got_lock:
                _grip_seed = int(latest_got0_holder[0])
        while not omy_action_stop.is_set():
            ee_pair = teleop.get_ros_end_effector_xyz_rpy_deg()
            if ee_pair is None:
                _none_streak += 1
                if _none_streak in (1, 50, 200) or _none_streak % 500 == 0:
                    _log.warning(
                        "OMY EE spawn: get_ros_end_effector_xyz_rpy_deg() is None "
                        "(streak=%d) — CRP GP held at last cmd (arm looks frozen)",
                        _none_streak,
                    )
            else:
                _none_streak = 0
                omy_now = [float(ee_pair[0][i]) for i in range(3)]
                p0 = [float(shared_p0[i]) for i in range(3)]
                ref = [float(shared_omy_ref[i]) for i in range(3)]
                hold = [float(shared_hold_rpy[i]) for i in range(3)]
                gp6 = omy_relative_xyz_to_gp6(
                    p0,
                    omy_now,
                    ref,
                    hold_rpy=hold,
                    step_sizes=resolved_steps,
                    scale=scale,
                )
                with gp_cmd_lock:
                    for _i in range(6):
                        latest_gp6[_i] = float(gp6[_i])
                _now_m = time.monotonic()
                if _now_m - _ee_dbg_last[0] >= CRP_EE_SPAWN_LOG_INTERVAL_S:
                    _ee_dbg_last[0] = _now_m
                    d = [gp6[i] - p0[i] for i in range(3)]
                    _log.info(
                        "OMY EE→CRP (spawn): omy=%.3f %.3f %.3f delta=%.3f %.3f %.3f gp=%.3f %.3f %.3f",
                        omy_now[0],
                        omy_now[1],
                        omy_now[2],
                        d[0],
                        d[1],
                        d[2],
                        gp6[0],
                        gp6[1],
                        gp6[2],
                    )

            # Gripper: get_gripper_raw → GOT (bypass processors on the 100Hz path).
            if ros_gripper_joint_name:
                if hasattr(teleop, "get_gripper_raw"):
                    raw_g = float(teleop.get_gripper_raw())
                else:
                    raw_g = float(teleop.get_action().get("gripper.pos", 0.0))
                got0 = int(omy_rh_r1_to_got0(raw_g))
                if _grip_seed is not None and abs(got0 - _grip_seed) < GOT_FOLLOW_DELTA:
                    got0 = _grip_seed
                else:
                    _grip_seed = None
                if gripper_got_lock is not None and latest_got0_holder is not None:
                    with gripper_got_lock:
                        latest_got0_holder[0] = got0

            tick += 1
            if tick % display_every == 0:
                act = teleop.get_action()
                obs_for_action: RobotObservation = {}
                if latest_obs_for_action_lock is not None and latest_obs_for_action is not None:
                    with latest_obs_for_action_lock:
                        obs_for_action = dict(latest_obs_for_action.copy())
                act_processed = teleop_action_processor((act, obs_for_action))
                _ = robot_action_processor((act_processed, obs_for_action))
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
                logging.getLogger(__name__).exception("OMY EE spawn worker disconnect failed")


def _resolve_crp_read_joints_value(read_joints: dict[str, Any], motor: str) -> float:
    """Look up one joint angle from CrpRobotPy ``read_joints()`` (keys vary by SDK)."""
    candidates: list[str] = [motor, motor.upper(), motor.lower()]
    if motor.startswith("j") and len(motor) >= 2 and motor[1:].isdigit():
        n = motor[1:]
        candidates.extend((f"joint{n}", f"Joint{n}"))
    for k in candidates:
        if k in read_joints:
            return float(read_joints[k])
    return 0.0


def _action_values_for_dataset_from_crp_joints(
    dataset: LeRobotDataset, read_joints: dict[str, Any]
) -> dict[str, float]:
    """Build flat action dict matching ``dataset.features['action']['names']`` for ``build_dataset_frame``."""
    spec = dataset.features.get("action") or {}
    names: list[str] = list(spec.get("names") or [])
    if not names:
        return {f"j{i}.pos": _resolve_crp_read_joints_value(read_joints, f"j{i}") for i in range(1, 7)}

    out: dict[str, float] = {}
    for name in names:
        if name in read_joints:
            out[name] = float(read_joints[name])
            continue
        if name.endswith(".pos"):
            motor = name.removesuffix(".pos")
            out[name] = _resolve_crp_read_joints_value(read_joints, motor)
        else:
            out[name] = float(read_joints.get(name, 0.0))
    return out


_EE_ACTION_NAMES = ("ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw")


def _action_values_for_dataset_from_gp6(gp6: list[float] | tuple[float, ...]) -> dict[str, float]:
    """Map commanded GP pose6 → action keys (teleop command, not measured EE)."""
    if len(gp6) < 6:
        raise ValueError(f"gp6 needs 6 values, got {len(gp6)}")
    return {name: float(gp6[i]) for i, name in enumerate(_EE_ACTION_NAMES)}


def _joints_dict_from_shared6(joints6: Any) -> dict[str, float]:
    """``Array('d', 6)`` / sequence → ``j1.pos``…``j6.pos``."""
    return {f"j{i}.pos": float(joints6[i - 1]) for i in range(1, 7)}


def _flush_pending_dataset_frame(
    dataset: LeRobotDataset,
    pending: dict[str, Any],
    *,
    joint_action: dict[str, float] | None,
) -> None:
    """Write a buffered frame; optional ``joint_action`` fills ``j*.pos`` (next-state target)."""
    action_values = dict(pending["action_values"])
    if joint_action:
        action_values.update(joint_action)
    action_frame = build_dataset_frame(dataset.features, action_values, prefix="action")
    dataset.add_frame(
        {
            **pending["observation_frame"],
            **action_frame,
            "task": pending["task"],
        }
    )


def _publish_stream_obs_cache(
    robot: "CRPArm",
    *,
    prev: dict[str, float] | None,
    measured_joints: dict[str, float] | None = None,
) -> None:
    """Parent-side cache publish (no SDK): measured joints + delayed EE/gripper command."""
    snap: dict[str, float] = {}
    if robot.config.enable_joint:
        src = measured_joints if measured_joints is not None else prev
        if src:
            for i in range(1, 7):
                key = f"j{i}.pos"
                if key in src:
                    snap[key] = float(src[key])
    if prev is not None and robot.config.use_gripper_feature and "gripper.pos" in prev:
        snap["gripper.pos"] = float(prev["gripper.pos"])
    if snap:
        robot.set_proprio_cache_snapshot(snap)
    if prev is not None and robot.config.enable_ee and all(name in prev for name in _EE_ACTION_NAMES):
        robot.update_ee_cache_from_pose6([float(prev[name]) for name in _EE_ACTION_NAMES])


def _build_dataset_action_values(
    dataset: LeRobotDataset,
    *,
    joint_snapshot: dict[str, Any] | None,
    gp6: list[float] | tuple[float, ...] | None,
    gripper_got0: float | None,
) -> dict[str, float]:
    """Fill action dict from joints and/or commanded GP according to ``dataset.features['action'].names``."""
    names: list[str] = list((dataset.features.get("action") or {}).get("names") or [])
    out: dict[str, float] = {}
    if joint_snapshot is not None:
        out.update(_action_values_for_dataset_from_crp_joints(dataset, joint_snapshot))
    if gp6 is not None:
        ee_vals = _action_values_for_dataset_from_gp6(gp6)
        for name in names:
            if name in ee_vals:
                out[name] = ee_vals[name]
        if not names:
            out.update(ee_vals)
    if gripper_got0 is not None and (not names or "gripper.pos" in names):
        out["gripper.pos"] = float(gripper_got0)
    # Drop keys not in the dataset action schema when names are known.
    if names:
        out = {n: float(out.get(n, 0.0)) for n in names}
    return out


@dataclass
class DatasetRecordConfig:
    repo_id: str
    single_task: str
    root: str | Path | None = None
    fps: int = 30
    episode_time_s: int | float = 60
    reset_time_s: int | float = 60
    num_episodes: int = 50
    video: bool = True
    push_to_hub: bool = False
    private: bool = False
    tags: list[str] | None = None
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1
    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    teleop: TeleoperatorConfig | None = None
    policy: PreTrainedConfig | None = None
    display_data: bool = False
    play_sounds: bool = True
    resume: bool = False
    # Between episodes: False = hold still (manual scene reset); True = re-arm OMY teleop.
    reset_teleop: bool = False
    speed_ratio: int = 80
    ee_delta_scale: float = EE_OMY_DELTA_SCALE
    ee_step_sizes: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_EE_STEP_SIZES)
    )

    def __post_init__(self):
        # Config / CLI values overlay code defaults (partial dicts keep missing axes at default).
        self.ee_step_sizes = resolve_ee_step_sizes(self.ee_step_sizes)
        self.ee_delta_scale = resolve_ee_delta_scale(self.ee_delta_scale)
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.teleop is None and self.policy is None:
            raise ValueError("Choose a policy, a teleoperator or both to control the robot")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    trajectory_processor: TrajectoryProcessor | None = None,
    ee_step_sizes: dict[str, float] | None = None,
    ee_delta_scale: float = 1.5,
    phase: str = "record",
    held_got0: list[int | None] | None = None,
    reset_teleop: bool = False,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")
    # Record always; reset only when reset_teleop=True.
    omy_stream = phase != "reset" or reset_teleop

    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (
                        SO100Leader,
                        SO101Leader,
                        KochLeader,
                        OMYL100,
                    ),
                )
            ),
            None,
        )

        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. Currently only supported for LeKiwi robot."
            )

    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    mp_ctx = None
    mp_manager = None
    omy_action_stop: Any = None
    omy_action_update_process: Any = None
    omy_display_action_lock: Any = None
    omy_display_action: RobotAction | None = None
    latest_obs_for_action_lock: Any = None
    latest_obs_for_action: RobotObservation | None = None
    if isinstance(teleop, OMYL100) and policy is None and omy_stream:
        mp_ctx = _record_multiprocessing_context()
        mp_manager = mp_ctx.Manager()
        omy_action_stop = mp_manager.Event()
        omy_display_action_lock = mp_manager.Lock()
        omy_display_action = mp_manager.dict()
        latest_obs_for_action_lock = mp_manager.Lock()
        latest_obs_for_action = mp_manager.dict()

    timestamp = 0
    start_episode_t = time.perf_counter()

    gp_sender_stop: Any = None
    gp_sender_process: Any = None
    gp_cmd_lock: Any = None
    latest_gp6: Any = None
    latest_got0_holder: list[int] | Any | None = None
    gripper_got_lock: Any = None
    # EE/gripper obs = previous action; joints = fork-measured shared buffer when enabled.
    prev_action_for_obs: dict[str, float] | None = None
    latest_joints6: Any = None
    # When enable_joint: delay add_frame 1 tick so action.joint = next measured joints.
    pending_dataset_frame: dict[str, Any] | None = None
    p0_crp: tuple[float, float, float] | None = None
    omy_ref_xyz: list[float] | None = None

    # GP/OMY init + main loop share one try/finally so Ctrl+C still stops workers.
    try:
        if phase == "reset" and not reset_teleop:
            _print_teleop_ready_banner(phase="reset", reset_teleop=False)
        elif trajectory_processor is not None:
            if isinstance(teleop, OMYL100) and isinstance(robot, CRPArm) and policy is None and omy_stream:
                if mp_ctx is None or mp_manager is None:
                    raise RuntimeError("OMY multiprocess workers require a multiprocessing context.")
                # Until GP armed: no send_GPs / send_GJs / set_GOT / GI→GP (GI56 stays off).
                if hasattr(robot, "wait_controller_ready"):
                    robot.wait_controller_ready(timeout_s=1.5)
                p0 = robot.get_current_endpose()
                p0_crp = (float(p0[0]), float(p0[1]), float(p0[2]))
                _log = logging.getLogger(__name__)
                _log.info(
                    "CRP initial endpose p0: xyz=(%.6f %.6f %.6f) rx_ry_rz_deg=(%.6f %.6f %.6f) "
                    "(p0_crp = this xyz, used as incremental GP position origin)",
                    float(p0[0]),
                    float(p0[1]),
                    float(p0[2]),
                    float(p0[3]),
                    float(p0[4]),
                    float(p0[5]),
                )
                _cfg_ee = (getattr(teleop.config, "ros_end_effector_pose_topic", "") or "").strip()
                _resolved_ee_topic = _cfg_ee or EE_STATES_TOPIC
                _log.info(
                    "EE wait [1/4] Entering OMY EE init: p0_crp=%s teleop.is_connected=%s "
                    "ros_end_effector_pose_topic config=%r → effective PoseStamped topic=%r",
                    p0_crp,
                    getattr(teleop, "is_connected", None),
                    getattr(teleop.config, "ros_end_effector_pose_topic", ""),
                    _resolved_ee_topic,
                )
                _log.info(
                    "EE wait [2/4] Blocking until get_ros_end_effector_xyz_rpy_deg() is not None "
                    "(ROS must deliver first PoseStamped on %r; spin thread must be running).",
                    _resolved_ee_topic,
                )
                _wait_t0 = time.perf_counter()
                _last_stall_log = _wait_t0
                _stall_iv = 2.0
                _iter = 0
                while teleop.get_ros_end_effector_xyz_rpy_deg() is None:
                    _iter += 1
                    _now = time.perf_counter()
                    if _now - _last_stall_log >= _stall_iv:
                        _last_stall_log = _now
                        _ga = teleop.get_action()
                        _log.warning(
                            "EE wait [2/4] still blocked after %.1fs (iter≈%d): get_ros_end_effector_xyz_rpy_deg() "
                            "is still None. Check: (1) publisher on %r + QoS vs subscriber depth=10 "
                            "(2) ROS_DOMAIN_ID (3) joint_states OK? j1.pos=%.4f teleop.is_connected=%s",
                            _now - _wait_t0,
                            _iter,
                            _resolved_ee_topic,
                            float(_ga.get("j1.pos", 0.0)),
                            getattr(teleop, "is_connected", None),
                        )
                    time.sleep(0.01)
                _dt_wait = time.perf_counter() - _wait_t0
                _log.info(
                    "EE wait [3/4] First non-None EE after %.3fs (%d wait-loop iterations); "
                    "now waiting for stable EE window.",
                    _dt_wait,
                    _iter,
                )
                omy_ref_xyz = wait_stable_omy_ee_xyz(
                    teleop.get_ros_end_effector_xyz_rpy_deg,
                    log_prefix="OMY EE parent stable",
                )
                # Re-read CRP pose after OMY is stable so p0/hold match the true start.
                p0 = robot.get_current_endpose()
                p0_crp = (float(p0[0]), float(p0[1]), float(p0[2]))
                hold_rpy = (float(p0[3]), float(p0[4]), float(p0[5]))
                _log.info(
                    "EE wait [4/4] Stable parent omy_ref=%s; refreshed p0 xyz=(%.4f %.4f %.4f) "
                    "hold_rpy=(%.4f %.4f %.4f) — starting spawn (no send_GPs / no GI=GP until armed)",
                    omy_ref_xyz,
                    p0_crp[0],
                    p0_crp[1],
                    p0_crp[2],
                    *hold_rpy,
                )
                _print_ee_wait_stage(
                    "[EE GP] 4/4 父进程 EE 已稳 — 仍在等 spawn 稳定 + GP arm，先别动手"
                )
                init_pose = [
                    p0_crp[0],
                    p0_crp[1],
                    p0_crp[2],
                    hold_rpy[0],
                    hold_rpy[1],
                    hold_rpy[2],
                ]
                # Hot-path GP/GOT: shared-memory Array/Lock — NOT Manager.list (100Hz proxy stutter).
                # Must be created with the **spawn** context: these objects are pickled into the
                # OMY spawn worker. Default/fork SemLock cannot be shared into spawn.
                # The later fork GP sender inherits the same mappings from the parent.
                omy_spawn_ctx = _omy_spawn_context()
                shared_p0 = omy_spawn_ctx.Array("d", [p0_crp[0], p0_crp[1], p0_crp[2]], lock=False)
                shared_omy_ref = omy_spawn_ctx.Array(
                    "d", [omy_ref_xyz[0], omy_ref_xyz[1], omy_ref_xyz[2]], lock=False
                )
                shared_hold_rpy = omy_spawn_ctx.Array(
                    "d", [hold_rpy[0], hold_rpy[1], hold_rpy[2]], lock=False
                )
                omy_stream_ready = mp_manager.Event()
                omy_gp_armed = mp_manager.Event()
                gp_cmd_lock = omy_spawn_ctx.Lock()
                latest_gp6 = omy_spawn_ctx.Array("d", list(init_pose), lock=False)
                _grip_name = str(getattr(teleop.config, "ros_gripper_joint_name", "") or "")
                if _grip_name:
                    latest_got0_holder = omy_spawn_ctx.Array("i", [0], lock=False)
                    gripper_got_lock = omy_spawn_ctx.Lock()
                    if held_got0 is not None and held_got0[0] is not None:
                        latest_got0_holder[0] = int(held_got0[0])
                    else:
                        init_grip = float(teleop.get_action().get("gripper.pos", 0.0))
                        latest_got0_holder[0] = int(omy_rh_r1_to_got0(init_grip))

                assert omy_action_stop is not None
                _steps = resolve_ee_step_sizes(ee_step_sizes)
                _scale = resolve_ee_delta_scale(ee_delta_scale)
                omy_action_update_process = omy_spawn_ctx.Process(
                    target=_spawn_omy_ee_gp_action_worker,
                    name="omy_ee_gp_update",
                    daemon=True,
                    kwargs={
                        "teleop_cfg": asdict(teleop.config),
                        "omy_action_stop": omy_action_stop,
                        "omy_stream_ready": omy_stream_ready,
                        "omy_gp_armed": omy_gp_armed,
                        "gp_cmd_lock": gp_cmd_lock,
                        "latest_gp6": latest_gp6,
                        "shared_p0": shared_p0,
                        "shared_omy_ref": shared_omy_ref,
                        "shared_hold_rpy": shared_hold_rpy,
                        "step_sizes": _steps,
                        "delta_scale": _scale,
                        "latest_obs_for_action_lock": latest_obs_for_action_lock,
                        "latest_obs_for_action": latest_obs_for_action,
                        "omy_display_action_lock": omy_display_action_lock,
                        "omy_display_action": omy_display_action,
                        "gripper_got_lock": gripper_got_lock,
                        "latest_got0_holder": latest_got0_holder,
                        "ros_gripper_joint_name": _grip_name,
                    },
                )
                omy_action_update_process.start()
                _log.info(
                    "Waiting for OMY spawn EE stable + omy_stream_ready "
                    "(CRP stays without GP/GI until then; pendant may use moveabsj)..."
                )
                while not omy_stream_ready.wait(timeout=2.0):
                    if not omy_action_update_process.is_alive():
                        raise RuntimeError("OMY EE spawn process exited before omy_stream_ready")
                    _log.info("Still waiting for omy_stream_ready (spawn stabilizing EE)...")

                # Arm CRP only after OMY spawn ref is ready.
                # Keep the 4/4-latched p0/hold_rpy — do NOT re-read endpose here.
                # A second get_current_endpose() during spawn wait has returned a
                # distant/wrong pose (e.g. 658→467 + rpy flip) and preload+GI→GP
                # then jerked the arm to that target.
                init_pose = [
                    float(p0_crp[0]),
                    float(p0_crp[1]),
                    float(p0_crp[2]),
                    float(hold_rpy[0]),
                    float(hold_rpy[1]),
                    float(hold_rpy[2]),
                ]
                for _i in range(3):
                    shared_p0[_i] = init_pose[_i]
                    shared_hold_rpy[_i] = init_pose[_i + 3]
                with gp_cmd_lock:
                    for _i in range(6):
                        latest_gp6[_i] = init_pose[_i]
                init_matrix = trajectory_processor.init_matrix(init_pose, group_size=GP_GROUP_SIZE)
                log_gp_points_matrix("send_GPs arm preload (no GI switch) before", init_matrix)
                # Preload registers while pendant can still be in joint / moveabsj.
                robot.send_GPs(10, init_matrix, switch_to_gp_mode=False)
                robot.send_GPs(20, init_matrix, switch_to_gp_mode=False)
                # Arm GOT once before fork owns SDK. Seed fork dedup to this value so the
                # first GP tick does not immediately re-send the same GOT.
                _arm_got0: int | None = None
                if _grip_name and latest_got0_holder is not None:
                    if held_got0 is not None and held_got0[0] is not None:
                        latest_got0_holder[0] = int(held_got0[0])
                    _arm_got0 = int(latest_got0_holder[0])
                    robot.set_GOT(0, _arm_got0)
                robot.ensure_gp_mode()
                robot.set_motion_enabled(True)
                _log.info(
                    "CRP GP mode+GI56 ON: preload xyz=(%.4f %.4f %.4f) hold_rpy=(%.4f %.4f %.4f) "
                    "got0=%s — start fork before releasing OMY deltas",
                    *init_pose[:3],
                    *init_pose[3:],
                    _arm_got0,
                )

                # Parent obs from cache only while fork owns SDK — seed from init_pose,
                # do NOT refresh_proprio / get_observation here (delays fork + steals SDK).
                robot.enable_proprio_cache()
                robot.update_ee_cache_from_pose6(init_pose)
                prev_action_for_obs = {}
                _seed_j = [0.0] * 6
                latest_joints6 = mp_ctx.Array("d", _seed_j, lock=False)
                if robot.config.enable_ee:
                    for _i, _name in enumerate(_EE_ACTION_NAMES):
                        prev_action_for_obs[_name] = float(init_pose[_i])
                if robot.config.use_gripper_feature:
                    if _arm_got0 is not None:
                        prev_action_for_obs["gripper.pos"] = float(_arm_got0)
                    else:
                        prev_action_for_obs["gripper.pos"] = 0.0

                gp_sender_stop = mp_ctx.Event()
                _sample_joints = bool(robot.config.enable_joint)
                # GP first every tick; joint read ~dataset fps (not every 100Hz, not EE).
                _joint_every = max(1, int(round(DEFAULT_GP_STREAM_HZ / max(1.0, float(fps)))))
                _fork_got_seed = int(_arm_got0) if _arm_got0 is not None else -10**9

                def _fixed_rate_gp_sender() -> None:
                    """Fork: GP @stream Hz; set_GOT only when value changes; optional joint samples."""
                    assert latest_gp6 is not None
                    assert gp_cmd_lock is not None
                    assert gp_sender_stop is not None
                    sdk = robot.crp_arm_robot
                    period = 1.0 / DEFAULT_GP_STREAM_HZ
                    t_next = time.perf_counter()
                    tick = 0
                    last_got0_sent = _fork_got_seed
                    while not gp_sender_stop.is_set():
                        with gp_cmd_lock:
                            row = [float(latest_gp6[i]) for i in range(6)]
                        mat = [row[:] for _ in range(GP_GROUP_SIZE)]
                        sdk.set_GPs(10, mat)
                        if gripper_got_lock is not None and latest_got0_holder is not None:
                            with gripper_got_lock:
                                got_sent = int(latest_got0_holder[0])
                            if got_sent != last_got0_sent:
                                sdk.set_GOT(0, got_sent)
                                last_got0_sent = got_sent
                        tick += 1
                        if (
                            _sample_joints
                            and latest_joints6 is not None
                            and tick % _joint_every == 0
                        ):
                            try:
                                jraw = sdk.read_joints() or {}
                                for _i in range(1, 7):
                                    latest_joints6[_i - 1] = _resolve_crp_read_joints_value(
                                        jraw, f"j{_i}"
                                    )
                            except Exception:
                                pass
                        t_next += period
                        dt = t_next - time.perf_counter()
                        if dt > 0:
                            precise_sleep(dt)
                        else:
                            t_next = time.perf_counter()

                # Fork must stream init_pose BEFORE spawn writes deltas — otherwise the first
                # send jumps from preload to accumulated OMY motion (stall then steep catch-up).
                gp_sender_process = mp_ctx.Process(target=_fixed_rate_gp_sender, name="crp_gp_sender", daemon=True)
                gp_sender_process.start()
                _log.info(
                    "Fork started: GP@%.0fHz; joint_sample=%s every %d ticks (~%.0fHz); "
                    "obs.ee=prev action (enable_ee=%s).",
                    DEFAULT_GP_STREAM_HZ,
                    _sample_joints,
                    _joint_every,
                    DEFAULT_GP_STREAM_HZ / _joint_every,
                    robot.config.enable_ee,
                )
                print("[EE GP] fork streaming init pose; releasing OMY→GP deltas")
                omy_gp_armed.set()
                _print_teleop_ready_banner(phase=phase, reset_teleop=reset_teleop)
            else:
                init_matrix = trajectory_processor.init_matrix(robot.get_current_endpose(), group_size=GP_GROUP_SIZE)
                log_gp_points_matrix("send_GPs init non-OMY reg10/20 before", init_matrix)
                robot.send_GPs(10, init_matrix)
                robot.send_GPs(20, init_matrix)

        while timestamp < control_time_s:
            start_loop_t = time.perf_counter()

            if events["exit_early"]:
                events["exit_early"] = False
                break

            # Manual reset: poll cameras / keyboard only — never command the arm.
            if phase == "reset" and not reset_teleop:
                if isinstance(robot, CRPArm):
                    _ = robot.get_observation()
                dt_s = time.perf_counter() - start_loop_t
                precise_sleep(max(1 / fps - dt_s, 0.0))
                timestamp = time.perf_counter() - start_episode_t
                continue

            # Obs: measured joints (shared) + delayed EE/gripper; no parent SDK while fork lives.
            if isinstance(robot, CRPArm) and (
                prev_action_for_obs is not None or latest_joints6 is not None
            ):
                _meas = (
                    _joints_dict_from_shared6(latest_joints6)
                    if robot.config.enable_joint and latest_joints6 is not None
                    else None
                )
                _publish_stream_obs_cache(
                    robot, prev=prev_action_for_obs, measured_joints=_meas
                )

            obs = robot.get_observation()
            if latest_obs_for_action_lock is not None and latest_obs_for_action is not None:
                with latest_obs_for_action_lock:
                    latest_obs_for_action.clear()
                    latest_obs_for_action.update(obs)
            obs_processed = robot_observation_processor(obs)

            if policy is not None or dataset is not None:
                observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix="observation")

            if policy is not None and preprocessor is not None and postprocessor is not None:
                action_values = predict_action(
                    observation=observation_frame,
                    policy=policy,
                    device=get_safe_torch_device(policy.config.device),
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=policy.config.use_amp,
                    task=single_task,
                    robot_type=robot.robot_type,
                )

                action_names = dataset.features["action"]["names"]
                act_processed_policy: RobotAction = {
                    f"{name}": float(action_values[i]) for i, name in enumerate(action_names)
                }

            elif policy is None and isinstance(teleop, Teleoperator):
                if isinstance(teleop, OMYL100) and gp_sender_process is not None:
                    if omy_display_action_lock is not None and omy_display_action is not None:
                        with omy_display_action_lock:
                            act_processed_teleop = omy_display_action.copy()
                    else:
                        act_processed_teleop = {}
                else:
                    act = teleop.get_action()
                    act_processed_teleop = teleop_action_processor((act, obs))

            elif policy is None and isinstance(teleop, list):
                arm_action = teleop_arm.get_action()
                arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
                keyboard_action = teleop_keyboard.get_action()
                base_action = robot._from_keyboard_to_base_action(keyboard_action)
                act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
                act_processed_teleop = teleop_action_processor((act, obs))
            else:
                logging.info(
                    "No policy or teleoperator provided, skipping action generation."
                    "This is likely to happen when resetting the environment without a teleop device."
                    "The robot won't be at its rest position at the start of the next episode."
                )
                continue

            if policy is not None and act_processed_policy is not None:
                action_values = act_processed_policy
                robot_action_to_send = robot_action_processor((act_processed_policy, obs))
                if trajectory_processor is not None:
                    if isinstance(robot, CRPArm):
                        send_gp_endpose6(robot, trajectory_processor, list(robot_action_to_send))
                    else:
                        trajectory_processor.write_point(robot_action_to_send)
                        _pts = trajectory_processor.read_points()
                        log_gp_points_matrix("send_GPs(10) policy before", _pts)
                        _ = robot.send_GPs(10, _pts)
                else:
                    _ = robot.send_endpose(robot_action_to_send)
            else:
                action_values = act_processed_teleop
                robot_action_to_send_tamp = robot_action_processor((act_processed_teleop, obs))

                if isinstance(teleop, OMYL100):
                    if not isinstance(robot, CRPArm):
                        raise TypeError("OMY EE incremental teleop requires robot.type=crp_arm (CRPArm.send_GPs).")
                    if trajectory_processor is None:
                        raise RuntimeError("trajectory_processor is required for OMY EE incremental streaming (send_GPs).")
                    if gp_sender_process is None:
                        # Not armed yet (before EE wait 4/4 + arm) or stream already torn down.
                        # Never send_GPs/set_GOT from the main loop in this gap.
                        pass
                    # When armed, fork sender owns all send_GPs / set_GOT.
                else:
                    robot_action_to_send = ee_action_to_crp_endpose_list(robot_action_to_send_tamp)
                    if trajectory_processor is not None:
                        if isinstance(robot, CRPArm):
                            send_gp_endpose6(robot, trajectory_processor, robot_action_to_send)
                        else:
                            trajectory_processor.write_point(robot_action_to_send)
                            _pts = trajectory_processor.read_points()
                            log_gp_points_matrix("send_GPs(10) teleop before", _pts)
                            _ = robot.send_GPs(10, _pts)
                    else:
                        _ = robot.send_endpose(robot_action_to_send)

            if dataset is not None:
                # action.ee/GOT = command at t; action.joint = measured at t+1 (buffered write).
                # prev_action_for_obs only tracks EE/gripper (delayed obs), not joints.
                gp6_cmd: list[float] | None = None
                joints_now: dict[str, float] | None = None
                use_next_joint_action = False
                if isinstance(robot, CRPArm):
                    use_next_joint_action = bool(
                        robot.config.enable_joint and latest_joints6 is not None
                    )
                    if use_next_joint_action:
                        joints_now = _joints_dict_from_shared6(latest_joints6)
                    if robot.config.enable_ee and latest_gp6 is not None:
                        if gp_cmd_lock is not None:
                            with gp_cmd_lock:
                                gp6_cmd = [float(latest_gp6[i]) for i in range(6)]
                        else:
                            gp6_cmd = [float(latest_gp6[i]) for i in range(6)]
                    joint_snapshot = None if use_next_joint_action else (
                        {f"j{i}": float(joints_now[f"j{i}.pos"]) for i in range(1, 7)}
                        if joints_now is not None
                        else None
                    )
                else:
                    joint_snapshot = {
                        k: float(v)
                        for k, v in (robot.crp_arm_robot.read_joints() or {}).items()
                    }

                gripper_got0: float | None = None
                action_names_ds = list((dataset.features.get("action") or {}).get("names") or [])
                if (
                    "gripper.pos" in action_names_ds
                    and isinstance(teleop, OMYL100)
                    and getattr(teleop.config, "ros_gripper_joint_name", "")
                    and policy is None
                ):
                    if gripper_got_lock is not None and latest_got0_holder is not None:
                        with gripper_got_lock:
                            gripper_got0 = float(latest_got0_holder[0])
                    else:
                        gripper_got0 = float(
                            omy_rh_r1_to_got0(float(robot_action_to_send_tamp.get("gripper.pos", 0.0)))
                        )

                crp_arm_joint_values = _build_dataset_action_values(
                    dataset,
                    joint_snapshot=joint_snapshot,
                    gp6=gp6_cmd,
                    gripper_got0=gripper_got0,
                )

                if use_next_joint_action:
                    if pending_dataset_frame is not None and joints_now is not None:
                        _flush_pending_dataset_frame(
                            dataset,
                            pending_dataset_frame,
                            joint_action=joints_now,
                        )
                    pending_dataset_frame = {
                        "observation_frame": observation_frame,
                        "action_values": crp_arm_joint_values,
                        "task": single_task,
                    }
                else:
                    action_frame = build_dataset_frame(
                        dataset.features, crp_arm_joint_values, prefix="action"
                    )
                    dataset.add_frame(
                        {**observation_frame, **action_frame, "task": single_task}
                    )

                if prev_action_for_obs is not None:
                    for _name in _EE_ACTION_NAMES:
                        if _name in crp_arm_joint_values:
                            prev_action_for_obs[_name] = float(crp_arm_joint_values[_name])
                    if "gripper.pos" in crp_arm_joint_values:
                        prev_action_for_obs["gripper.pos"] = float(crp_arm_joint_values["gripper.pos"])

            if display_data:
                log_rerun_data(observation=obs_processed, action=action_values)

            dt_s = time.perf_counter() - start_loop_t
            precise_sleep(max(1 / fps - dt_s, 0.0))

            timestamp = time.perf_counter() - start_episode_t

    finally:
        # Drop trailing pending row: it has obs but no next joint for action.
        # Its joints already became action.joint of the previous written transition.
        pending_dataset_frame = None
        # Stop EE/GP workers FIRST — never write GI/SDK while the fork still owns the client
        # (concurrent set_GI + send_GPs caused left-key ``read user pose failed`` / abort).
        if (
            held_got0 is not None
            and gripper_got_lock is not None
            and latest_got0_holder is not None
        ):
            try:
                with gripper_got_lock:
                    held_got0[0] = int(latest_got0_holder[0])
            except Exception:
                logging.getLogger(__name__).debug("teardown: could not latch held_got0", exc_info=True)
        if gp_sender_stop is not None:
            gp_sender_stop.set()
        if omy_action_stop is not None:
            omy_action_stop.set()
        if gp_sender_process is not None and gp_sender_process.pid is not None:
            gp_sender_process.join(timeout=5.0)
        if omy_action_update_process is not None and omy_action_update_process.pid is not None:
            omy_action_update_process.join(timeout=8.0)
        if isinstance(robot, CRPArm):
            robot.disable_proprio_cache()
            robot.clear_ee_cache()  # never latch next arm from episode-start EE cache
            try:
                robot.set_motion_enabled(False)
            except Exception:
                logging.getLogger(__name__).warning(
                    "teardown: failed to clear motion enable GI", exc_info=True
                )
            # Brief settle so the next record_loop's get_current_endpose is accepted.
            robot.wait_controller_ready(timeout_s=1.0)
        logging.getLogger(__name__).info(
            "EE/GP stream stopped; GI56 OFF; hold last GP; held_got0=%s",
            None if held_got0 is None else held_got0[0],
        )
        print("[EE GP] stream stopped; GI56 move OFF; hold last GP")
        if mp_manager is not None:
            mp_manager.shutdown()


@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="recording")

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None
    if isinstance(teleop, OMYL100):
        logging.info(
            "crp_record_omy_ee_inc: EE xyz from OMYL100.get_ros_end_effector_xyz_rpy_deg(); "
            "ros_end_effector_pose_topic=%r (empty → OMY_L100.EE_STATES_TOPIC).",
            getattr(teleop.config, "ros_end_effector_pose_topic", ""),
        )

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(
                action=robot.action_features,
            ),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )
    if (
        cfg.teleop is not None
        and cfg.teleop.type == "OMY_L100"
        and cfg.robot.type == "crp_arm"
        and getattr(cfg.teleop, "ros_gripper_joint_name", "")
    ):
        dataset_features = combine_feature_dicts(
            dataset_features,
            hw_to_dataset_features({"gripper.pos": float}, ACTION, use_video=cfg.dataset.video),
        )

    if cfg.resume:
        num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
        dataset = LeRobotDataset.resume(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
            image_writer_threads=(
                cfg.dataset.num_image_writer_threads_per_camera * num_cameras if num_cameras > 0 else 0
            ),
        )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
    else:
        sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

    policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)
    preprocessor = None
    postprocessor = None
    if cfg.policy is not None:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.device},
                "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
            },
        )

    robot.connect()
    if teleop is not None:
        teleop.connect()

    listener, events = init_keyboard_listener()

    print("当前速度比：", robot.get_speed_ratio())
    robot.set_speed_ratio(int(cfg.speed_ratio))
    print("当前速度比：", robot.get_speed_ratio())
    # Always print so axis signs are visible even if logging is filtered.
    print(
        f"[EE GP] applied ee_step_sizes={cfg.ee_step_sizes} "
        f"ee_delta_scale={cfg.ee_delta_scale} "
        f"(code defaults step={DEFAULT_EE_STEP_SIZES} scale={EE_OMY_DELTA_SCALE})"
    )
    logging.getLogger(__name__).info(
        "RecordConfig EE overrides applied: ee_step_sizes=%s ee_delta_scale=%.4f speed_ratio=%s",
        cfg.ee_step_sizes,
        cfg.ee_delta_scale,
        cfg.speed_ratio,
    )

    trajectory_processor = TrajectoryProcessor()

    with VideoEncodingManager(dataset):
        recorded_episodes = 0
        # Persist last commanded GOT across record↔reset so re-arm does not snap gripper.
        held_got0: list[int | None] = [None]
        while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
            log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
            record_loop(
                robot=robot,
                events=events,
                fps=cfg.dataset.fps,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset=dataset,
                control_time_s=cfg.dataset.episode_time_s,
                single_task=cfg.dataset.single_task,
                display_data=cfg.display_data,
                trajectory_processor=trajectory_processor,
                ee_step_sizes=cfg.ee_step_sizes,
                ee_delta_scale=cfg.ee_delta_scale,
                phase="record",
                held_got0=held_got0,
            )

            # Between episodes: reset wait (cyan). reset_teleop toggles OMY re-arm vs hold-still.
            if not events["stop_recording"] and (
                (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
            ):
                log_say("Reset the environment", cfg.play_sounds)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop if cfg.reset_teleop else None,
                    control_time_s=cfg.dataset.reset_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    trajectory_processor=trajectory_processor if cfg.reset_teleop else None,
                    ee_step_sizes=cfg.ee_step_sizes,
                    ee_delta_scale=cfg.ee_delta_scale,
                    phase="reset",
                    held_got0=held_got0,
                    reset_teleop=cfg.reset_teleop,
                )

            if events["rerecord_episode"]:
                log_say("Re-record episode", cfg.play_sounds)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            recorded_episodes += 1

    log_say("Stop recording", cfg.play_sounds, blocking=True)

    robot.disconnect()
    if teleop is not None:
        teleop.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()

    if cfg.dataset.push_to_hub:
        dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    log_say("Exiting", cfg.play_sounds)
    return dataset


def main():
    record()


if __name__ == "__main__":
    main()
