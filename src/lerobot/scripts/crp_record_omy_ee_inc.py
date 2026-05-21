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
Record a LeRobot dataset while teleoperating a CRP arm from an OMY_L100 master using **incremental GP**
commands from ROS **end-effector xyz** (no joint-angle mapping to CRP).

Control (episode-relative incremental; **OMY EE xyz uses the same frame as CRP**, no kinematics):
  ``crp_xyz = p0_crp + scaling_factor * ( xyz_now - xyz_ref )`` with fixed
  roll/pitch/yaw ``-180°, 0°, 0°``.  OMY EE xyz (after ``ros_ee_pose_position_scale``) comes from
  ``OMYL100.get_ros_end_effector_xyz_rpy_deg()`` — same ``PoseStamped`` subscription as the teleop node
  (``OMYL100Config.ros_end_effector_pose_topic``; empty uses ``OMY_L100.EE_STATES_TOPIC``).
  No separate EE ROS process in this script.  Main process waits without timeout for the first non-``None`` EE
  reading before starting the GP stream.  Arm commands use ``send_GPs`` only.

Gripper: unchanged — ``sensor_msgs/JointState`` + ``ros_gripper_joint_name`` → ``set_GOT`` /
dataset ``gripper.pos`` via ``_omy_rh_r1_to_got0``.

Multiprocessing: a ``spawn`` worker owns ROS + polls EE / gripper at ``CRP_GP_STREAM_HZ``; a ``fork`` worker runs ``send_GPs`` + ``set_GOT`` at the same cadence so **camera / dataset fps does not throttle** arm commands. Main loop only copies ``omy_display_action`` for processors / logging.

Other teleops (non-OMY): GP command from ``ee.x`` … ``ee.yaw`` in the processed action dict, assumed already in the CRP frame (pass-through, no transform).

Prerequisites:
  - OMY publishes ``JointState`` and (for arm GP) ``PoseStamped`` (see ``OMYL100Config.ros_end_effector_pose_topic``).
  - ``third_party/CrpRobotPy`` on disk; ``load_CrpRobotPy()`` configures ``sys.path`` / ``LD_LIBRARY_PATH``.

Example:

```shell
python -m lerobot.scripts.crp_record_omy_ee_inc \\
    --robot.type=crp_arm \\
    --robot.port=/dev/ttyUSB0 \\
    --robot.cameras='{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \\
    --teleop.type=OMY_L100 \\
    --teleop.port=dummy \\
    --dataset.repo_id=<user>/crp_omy_ee_inc \\
    --dataset.num_episodes=10 \\
    --dataset.single_task="Pick and place" \\
    --display_data=false
```

Note: ``--teleop.port`` is required by the config schema but unused by OMY (connection is ROS-based).
"""

# TrajectoryProcessor lives in lerobot.tools.TrajProcessor (file TrajProcessor.py);
# lerobot.tools.__init__ re-exports it — use the package import, not lerobot.tools.TrajectoryProcessor.
from lerobot.tools import TrajectoryProcessor
from lerobot.tools.lib_loader import load_CrpRobotPy

import logging
import math
import multiprocessing
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any, Sequence

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
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

# Load CRP SDK path/libraries only after ROS teleop imports. This avoids
# CRP shared-library preloading interfering with ROS2 Python imports.
load_CrpRobotPy()
import lerobot.robots.crp_arm  # noqa: F401
from lerobot.robots.crp_arm import CRPArm
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


def _record_multiprocessing_context() -> "multiprocessing.context.BaseContext":
    """Use ``fork`` so the GP sender child can use the already-created ``robot`` / ``trajectory_processor``."""
    return multiprocessing.get_context("fork")


def _omy_spawn_context() -> "multiprocessing.context.BaseContext":
    """Fresh interpreter for OMY / EE workers so ``rclpy`` in spawn children is valid."""
    return multiprocessing.get_context("spawn")


def _spawn_omy_ee_gp_action_worker(
    *,
    teleop_cfg: dict[str, Any],
    omy_action_stop: Any,
    gp_cmd_lock: Any,
    latest_gp6: Any,
    shared_p0: Any,
    shared_omy_ref: Any,
    latest_obs_for_action_lock: Any,
    latest_obs_for_action: Any,
    omy_display_action_lock: Any,
    omy_display_action: Any,
    gripper_got_lock: Any,
    latest_got0_holder: Any,
    ros_gripper_joint_name: str,
) -> None:
    """``spawn`` process: ROS node + EE incremental GP targets + gripper GOT0 into Manager buffers."""
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
    try:
        cfg = OMYL100Config(**teleop_cfg)
        teleop = OMYL100(cfg)
        teleop.connect()
        period = 1.0 / CRP_GP_STREAM_HZ
        t_next = time.perf_counter()
        while not omy_action_stop.is_set():
            act = teleop.get_action()

            obs_for_action: RobotObservation = {}
            if latest_obs_for_action_lock is not None and latest_obs_for_action is not None:
                with latest_obs_for_action_lock:
                    obs_for_action = dict(latest_obs_for_action.copy())

            act_processed = teleop_action_processor((act, obs_for_action))
            robot_action_to_send_tamp = robot_action_processor((act_processed, obs_for_action))

            ee_pair = teleop.get_ros_end_effector_xyz_rpy_deg()
            if ee_pair is not None:
                omy_now = [float(ee_pair[0][i]) for i in range(3)]
                p0 = [float(shared_p0[i]) for i in range(3)]
                ref = [float(shared_omy_ref[i]) for i in range(3)]
                d = [CRP_EE_OMY_DELTA_SCALE * (float(omy_now[i]) - ref[i]) for i in range(3)]
                gp6 = [
                    p0[0] + d[0],
                    p0[1] + d[1],
                    p0[2] + d[2],
                    CRP_EE_FIXED_ROLL_DEG,
                    CRP_EE_FIXED_PITCH_DEG,
                    CRP_EE_FIXED_YAW_DEG,
                ]
                with gp_cmd_lock:
                    # Avoid ``latest_gp6[:] = gp6`` on ``Manager().list`` — slice assign can fail to sync
                    # reliably across processes; fork GP sender would then repeat the initial pose forever.
                    for _i in range(6):
                        latest_gp6[_i] = gp6[_i]
                _now_m = time.monotonic()
                if _now_m - _ee_dbg_last[0] >= CRP_EE_SPAWN_LOG_INTERVAL_S:
                    _ee_dbg_last[0] = _now_m
                    _gt = None
                    if gripper_got_lock is not None and latest_got0_holder is not None:
                        with gripper_got_lock:
                            _gt = int(latest_got0_holder[0])
                    logging.getLogger(__name__).info(
                        "OMY EE→CRP (spawn): omy_xyz_scaled=%.4f %.4f %.4f ref=%.4f %.4f %.4f "
                        "delta=%.4f %.4f %.4f gp6=[%.4f %.4f %.4f %.2f %.2f %.2f] got0_target=%s",
                        omy_now[0],
                        omy_now[1],
                        omy_now[2],
                        ref[0],
                        ref[1],
                        ref[2],
                        d[0],
                        d[1],
                        d[2],
                        gp6[0],
                        gp6[1],
                        gp6[2],
                        gp6[3],
                        gp6[4],
                        gp6[5],
                        _gt,
                    )

            if ros_gripper_joint_name:
                got0 = _omy_rh_r1_to_got0(float(robot_action_to_send_tamp.get("gripper.pos", 0.0)))
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
                logging.getLogger(__name__).exception("OMY EE spawn worker disconnect failed")


# EE/GP streaming cadence (spawn worker poll + fork sender); decoupled from dataset/camera fps.
CRP_GP_STREAM_HZ = 100.0

# Scale applied to OMY EE position **delta** (after ``ros_ee_pose_position_scale``), before adding to ``p0_crp``.
CRP_EE_OMY_DELTA_SCALE = 1.0

# Fixed CRP command orientation (deg), end-effector frame (same convention as ``read_end_pose_user``).
CRP_EE_FIXED_ROLL_DEG = -180.0
CRP_EE_FIXED_PITCH_DEG = 0.0
CRP_EE_FIXED_YAW_DEG = 0.0

# OMY ``rh_r1_joint`` gripper position → CRP GOT0 (0=closed, 1000=open): clip to [-15, 15], then linear map.
OMY_GRIPPER_RH_R1_CLIP_LO = -15.0
OMY_GRIPPER_RH_R1_CLIP_HI = 15.0
CRP_GRIPPER_GOT0_MAX = 1000

# ``init_matrix(..., group_size=5)`` is used at startup; the fork sender must match that shape.
# Default ``TrajectoryProcessor`` has ``max_points=1``, so ``read_points()`` returns only **one** 6-vector row,
# which many CRP ``set_GPs`` paths ignore — arm appears frozen. Always send 5 duplicate rows of the latest pose.
CRP_GP_GROUP_SIZE = 5

# Throttle INFO logs from the OMY EE spawn worker (computed delta / gp6 / gripper target).
CRP_EE_SPAWN_LOG_INTERVAL_S = 0.5

# Throttle ``send_GPs`` pose logs from ``_crp_send_gp_endpose6`` (fork sender ~ ``CRP_GP_STREAM_HZ``). Set to
# ``None`` to log every tick (very verbose).
CRP_GP_SEND_LOG_INTERVAL_S: float | None = 0.2
_gp_send_log_last_mono: list[float] = [0.0]


def _log_gp_vec6(msg: str, vec6: Sequence[float]) -> None:
    """Log one 6-D endpose: xyz and rx/ry/rz in degrees (roll, pitch, yaw)."""
    if len(vec6) < 6:
        return
    x, y, z, rx, ry, rz = (float(vec6[i]) for i in range(6))
    logging.getLogger(__name__).info("%s xyz=(%.6f %.6f %.6f) rx_ry_rz_deg=(%.6f %.6f %.6f)", msg, x, y, z, rx, ry, rz)


def _log_gp_vec6_throttled(msg: str, vec6: Sequence[float]) -> None:
    if CRP_GP_SEND_LOG_INTERVAL_S is None:
        _log_gp_vec6(msg, vec6)
        return
    now = time.monotonic()
    if now - _gp_send_log_last_mono[0] < CRP_GP_SEND_LOG_INTERVAL_S:
        return
    _gp_send_log_last_mono[0] = now
    _log_gp_vec6(msg, vec6)


def _log_gp_points_matrix(msg: str, rows: list[list[float]]) -> None:
    """Log GP matrix: row count and first row as xyz + rx/ry/rz (deg)."""
    if not rows or len(rows[0]) < 6:
        return
    n = len(rows)
    same = n > 1 and all(r == rows[0] for r in rows[1:])
    suffix = f" ({n} duplicate rows)" if same and n > 1 else f" ({n} rows, logging first row)"
    _log_gp_vec6(msg + suffix, rows[0])


def _crp_send_gp_endpose6(robot: CRPArm, trajectory_processor: TrajectoryProcessor, vec6: list[float]) -> None:
    """Send latest 6-D endpose as a ``CRP_GP_GROUP_SIZE``×6 GP matrix (same layout as startup ``init_matrix``)."""
    mat = trajectory_processor.init_matrix([float(x) for x in vec6], group_size=CRP_GP_GROUP_SIZE)
    _log_gp_vec6_throttled("send_GPs(10) before", vec6)
    robot.send_GPs(10, mat)


def _omy_rh_r1_to_got0(raw: float) -> int:
    """Clamp OMY gripper joint value to [-15, 15], map linearly to GOT0 in [0, 1000]."""
    raw = raw * 180 / 3.14
    c = max(OMY_GRIPPER_RH_R1_CLIP_LO, min(OMY_GRIPPER_RH_R1_CLIP_HI, float(raw)))
    span = OMY_GRIPPER_RH_R1_CLIP_HI - OMY_GRIPPER_RH_R1_CLIP_LO
    v = float(CRP_GRIPPER_GOT0_MAX) * (c - OMY_GRIPPER_RH_R1_CLIP_LO) / span
    return int(max(0, min(CRP_GRIPPER_GOT0_MAX, round(v))))


def _ee_action_to_crp_endpose_list(action: dict[str, float]) -> list[float]:
    """Build a 6-DOF GP vector from ``ee.*`` keys; no coordinate transform (same frame as CRP)."""
    return [
        float(action.get("ee.x", 0.0)),
        float(action.get("ee.y", 0.0)),
        float(action.get("ee.z", 0.0)),
        float(action.get("ee.roll", 0.0)),
        float(action.get("ee.pitch", 0.0)),
        float(action.get("ee.yaw", 0.0)),
    ]


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

    def __post_init__(self):
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
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

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
    if isinstance(teleop, OMYL100) and policy is None:
        mp_ctx = _record_multiprocessing_context()
        mp_manager = mp_ctx.Manager()
        omy_action_stop = mp_manager.Event()
        omy_display_action_lock = mp_manager.Lock()
        omy_display_action = mp_manager.dict()
        latest_obs_for_action_lock = mp_manager.Lock()
        latest_obs_for_action = mp_manager.dict()

    timestamp = 0
    start_episode_t = time.perf_counter()

    # Dedup ``set_GOT`` in the fork GP sender (and fallback main-thread path).
    last_got0_sent = [-10**9]

    gp_sender_stop: Any = None
    gp_sender_process: Any = None
    gp_cmd_lock: Any = None
    latest_gp6: Any = None
    latest_got0_holder: list[int] | Any | None = None
    gripper_got_lock: Any = None

    # Episode-local incremental EE state (OMY ROS xyz → CRP GP); copied into Manager lists for spawn worker.
    p0_crp: tuple[float, float, float] | None = None
    omy_ref_xyz: list[float] | None = None

    # Initialize GP registers on the CRP controller.
    if trajectory_processor is not None:
        if isinstance(teleop, OMYL100) and isinstance(robot, CRPArm) and policy is None:
            if mp_ctx is None or mp_manager is None:
                raise RuntimeError("OMY multiprocess workers require a multiprocessing context.")
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
                "EE wait [3/4] First non-None EE after %.3fs (%d wait-loop iterations).",
                _dt_wait,
                _iter,
            )
            _ee0 = teleop.get_ros_end_effector_xyz_rpy_deg()
            assert _ee0 is not None
            omy_ref_xyz = [float(_ee0[0][0]), float(_ee0[0][1]), float(_ee0[0][2])]
            _log.info(
                "EE wait [4/4] First OMY EE scaled xyz=%s — starting GP stream (shared_omy_ref / send_GPs).",
                omy_ref_xyz,
            )
            init_pose = [
                p0_crp[0],
                p0_crp[1],
                p0_crp[2],
                CRP_EE_FIXED_ROLL_DEG,
                CRP_EE_FIXED_PITCH_DEG,
                CRP_EE_FIXED_YAW_DEG,
            ]
            shared_p0 = mp_manager.list([p0_crp[0], p0_crp[1], p0_crp[2]])
            shared_omy_ref = mp_manager.list([omy_ref_xyz[0], omy_ref_xyz[1], omy_ref_xyz[2]])
            gp_cmd_lock = mp_manager.Lock()
            latest_gp6 = mp_manager.list(list(init_pose))
            init_matrix = trajectory_processor.init_matrix(init_pose, group_size=CRP_GP_GROUP_SIZE)
            _log_gp_points_matrix("send_GPs init OMY reg10/20 before", init_matrix)
            robot.send_GPs(10, init_matrix)
            robot.send_GPs(20, init_matrix)
            _grip_name = str(getattr(teleop.config, "ros_gripper_joint_name", "") or "")
            if _grip_name:
                latest_got0_holder = mp_manager.list([0])
                gripper_got_lock = mp_manager.Lock()
                init_grip = float(teleop.get_action().get("gripper.pos", 0.0))
                _init_got = _omy_rh_r1_to_got0(init_grip)
                robot.set_GOT(0, _init_got)
                latest_got0_holder[0] = _init_got
                last_got0_sent[0] = _init_got
            gp_sender_stop = mp_ctx.Event()
            _gp_fork_log_last = [0.0]

            def _fixed_rate_gp_sender() -> None:
                period = 1.0 / CRP_GP_STREAM_HZ
                t_next = time.perf_counter()
                assert latest_gp6 is not None
                assert gp_cmd_lock is not None
                assert gp_sender_stop is not None
                _log = logging.getLogger(__name__)
                while not gp_sender_stop.is_set():
                    with gp_cmd_lock:
                        vec = list(latest_gp6)
                    _crp_send_gp_endpose6(robot, trajectory_processor, vec)
                    got_sent = None
                    if gripper_got_lock is not None and latest_got0_holder is not None:
                        with gripper_got_lock:
                            got_sent = int(latest_got0_holder[0])
                        robot.set_GOT(0, got_sent)
                    _tn = time.monotonic()
                    if _tn - _gp_fork_log_last[0] >= CRP_EE_SPAWN_LOG_INTERVAL_S:
                        _gp_fork_log_last[0] = _tn
                        _log.info(
                            "CRP GP fork sent: vec6=%s rows=%d got0=%s",
                            vec,
                            CRP_GP_GROUP_SIZE,
                            got_sent,
                        )
                    t_next += period
                    dt = t_next - time.perf_counter()
                    if dt > 0:
                        precise_sleep(dt)
                    else:
                        t_next = time.perf_counter()

            gp_sender_process = mp_ctx.Process(target=_fixed_rate_gp_sender, name="crp_gp_sender", daemon=True)
            gp_sender_process.start()

            assert omy_action_stop is not None
            omy_spawn_ctx = _omy_spawn_context()
            omy_action_update_process = omy_spawn_ctx.Process(
                target=_spawn_omy_ee_gp_action_worker,
                name="omy_ee_gp_update",
                daemon=True,
                kwargs={
                    "teleop_cfg": asdict(teleop.config),
                    "omy_action_stop": omy_action_stop,
                    "gp_cmd_lock": gp_cmd_lock,
                    "latest_gp6": latest_gp6,
                    "shared_p0": shared_p0,
                    "shared_omy_ref": shared_omy_ref,
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
        else:
            init_matrix = trajectory_processor.init_matrix(robot.get_current_endpose(), group_size=CRP_GP_GROUP_SIZE)
            _log_gp_points_matrix("send_GPs init non-OMY reg10/20 before", init_matrix)
            robot.send_GPs(10, init_matrix)
            robot.send_GPs(20, init_matrix)

    try:
        while timestamp < control_time_s:
            start_loop_t = time.perf_counter()

            if events["exit_early"]:
                events["exit_early"] = False
                break

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
                        _crp_send_gp_endpose6(robot, trajectory_processor, list(robot_action_to_send))
                    else:
                        trajectory_processor.write_point(robot_action_to_send)
                        _pts = trajectory_processor.read_points()
                        _log_gp_points_matrix("send_GPs(10) policy before", _pts)
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
                        if p0_crp is None or omy_ref_xyz is None:
                            raise RuntimeError("OMY EE incremental state was not initialized (p0_crp / omy_ref_xyz).")
                        omy_now = None
                        _ee_fb = teleop.get_ros_end_effector_xyz_rpy_deg()
                        if _ee_fb is not None:
                            omy_now = [float(_ee_fb[0][i]) for i in range(3)]
                        if omy_now is not None:
                            d = [
                                CRP_EE_OMY_DELTA_SCALE * (float(omy_now[i]) - float(omy_ref_xyz[i]))
                                for i in range(3)
                            ]
                            px = p0_crp[0] + d[0]
                            py = p0_crp[1] + d[1]
                            pz = p0_crp[2] + d[2]
                            endpose_cmd = [
                                px,
                                py,
                                pz,
                                CRP_EE_FIXED_ROLL_DEG,
                                CRP_EE_FIXED_PITCH_DEG,
                                CRP_EE_FIXED_YAW_DEG,
                            ]
                            _crp_send_gp_endpose6(robot, trajectory_processor, endpose_cmd)
                        if getattr(teleop.config, "ros_gripper_joint_name", ""):
                            _got_fb = _omy_rh_r1_to_got0(float(robot_action_to_send_tamp.get("gripper.pos", 0.0)))
                            if _got_fb != last_got0_sent[0]:
                                robot.set_GOT(0, _got_fb)
                                last_got0_sent[0] = _got_fb
                else:
                    robot_action_to_send = _ee_action_to_crp_endpose_list(robot_action_to_send_tamp)
                    if trajectory_processor is not None:
                        if isinstance(robot, CRPArm):
                            _crp_send_gp_endpose6(robot, trajectory_processor, robot_action_to_send)
                        else:
                            trajectory_processor.write_point(robot_action_to_send)
                            _pts = trajectory_processor.read_points()
                            _log_gp_points_matrix("send_GPs(10) teleop before", _pts)
                            _ = robot.send_GPs(10, _pts)
                    else:
                        _ = robot.send_endpose(robot_action_to_send)

            if dataset is not None:
                # Same joint snapshot as in this ``get_observation()`` call (avoids a second
                # ``read_joints()`` round-trip per frame on CRP).
                if isinstance(robot, CRPArm):
                    joint_snapshot = {
                        k.removesuffix(".pos"): float(v)
                        for k, v in obs.items()
                        if k.endswith(".pos") and isinstance(v, (int, float))
                    }
                    crp_arm_joint_values = _action_values_for_dataset_from_crp_joints(
                        dataset, joint_snapshot
                    )
                else:
                    crp_arm_joint = robot.crp_arm_robot.read_joints()
                    crp_arm_joint_values = _action_values_for_dataset_from_crp_joints(
                        dataset, crp_arm_joint
                    )
                action_names_ds = list((dataset.features.get("action") or {}).get("names") or [])
                if (
                    "gripper.pos" in action_names_ds
                    and isinstance(teleop, OMYL100)
                    and getattr(teleop.config, "ros_gripper_joint_name", "")
                    and policy is None
                ):
                    if gripper_got_lock is not None and latest_got0_holder is not None:
                        with gripper_got_lock:
                            _g_ds = float(latest_got0_holder[0])
                    else:
                        _g_ds = float(
                            _omy_rh_r1_to_got0(float(robot_action_to_send_tamp.get("gripper.pos", 0.0)))
                        )
                    crp_arm_joint_values = {**crp_arm_joint_values, "gripper.pos": _g_ds}

                action_frame = build_dataset_frame(dataset.features, crp_arm_joint_values, prefix="action")
                frame = {**observation_frame, **action_frame, "task": single_task}
                dataset.add_frame(frame)

            if display_data:
                log_rerun_data(observation=obs_processed, action=action_values)

            dt_s = time.perf_counter() - start_loop_t
            precise_sleep(max(1 / fps - dt_s, 0.0))

            timestamp = time.perf_counter() - start_episode_t

    finally:
        if gp_sender_stop is not None:
            gp_sender_stop.set()
        if omy_action_stop is not None:
            omy_action_stop.set()
        if gp_sender_process is not None:
            gp_sender_process.join(timeout=5.0)
        if omy_action_update_process is not None:
            omy_action_update_process.join(timeout=8.0)
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
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
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
    robot.set_speed_ratio(20)
    print("当前速度比：", robot.get_speed_ratio())

    trajectory_processor = TrajectoryProcessor()

    with VideoEncodingManager(dataset):
        recorded_episodes = 0
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
            )

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
                    teleop=teleop,
                    control_time_s=cfg.dataset.reset_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    trajectory_processor=trajectory_processor,
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
