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
Record a LeRobot dataset while teleoperating a CRP arm from an OMY_L100 master (ROS joint states).

Same data layout as ``crp_record.py`` (observation = CRP cameras + state; action = CRP joint positions
from ``read_joints()``), plus ``gripper.pos`` (0..1000 GOT0 command) when using ``OMY_L100`` with
``ros_gripper_joint_name`` set. For ``OMY_L100`` teleop, a ``spawn`` child process reconnects ROS and polls
``get_action()`` (``fork`` after ``rclpy`` would leave stale joint data); a ``fork`` child runs ``write_joint`` +
``send_GJs`` at fixed rate. ``multiprocessing.Manager`` proxies synchronize shared joint buffers. Other teleops
still use
``get_omy_endpose2Crp`` and GP / ``send_endpose`` when applicable.

Prerequisites:
  - OMY publishes ``sensor_msgs/JointState`` (see ``OMY_L100`` teleoperator, e.g. ``/leader/joint_states``).
  - ``third_party/CrpRobotPy`` on disk; ``load_CrpRobotPy()`` configures ``sys.path`` / ``LD_LIBRARY_PATH``.

Example:

```shell
lerobot-record-crp-omy \\
    --robot.type=crp_arm \\
    --robot.port=/dev/ttyUSB0 \\
    --robot.cameras='{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \\
    --teleop.type=OMY_L100 \\
    --teleop.port=dummy \\
    --dataset.repo_id=<user>/crp_omy_dataset \\
    --dataset.num_episodes=10 \\
    --dataset.single_task="Pick and place" \\
    --display_data=false
```

Note: ``--teleop.port`` is required by the config schema but unused by OMY (connection is ROS-based).
"""

# TrajectoryProcessor lives in lerobot.tools.TrajProcessor (file TrajProcessor.py);
# lerobot.tools.__init__ re-exports it — use the package import, not lerobot.tools.TrajectoryProcessor.
from lerobot.tools import TrajectoryProcessor
from lerobot.tools.kinematics import get_omy_endpose2Crp
from lerobot.tools.lib_loader import load_CrpRobotPy

import logging
import math
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
    """Use ``fork`` so child processes can access already-created robot/teleop objects."""
    return multiprocessing.get_context("fork")


def _omy_spawn_context() -> "multiprocessing.context.BaseContext":
    """Fresh interpreter for OMY worker so ``rclpy`` / subscriptions are valid (not ``fork`` after ROS connect)."""
    return multiprocessing.get_context("spawn")


def _spawn_omy_joint_action_worker(
    *,
    teleop_cfg: dict[str, Any],
    omy_action_stop: Any,
    joint_cmd_lock: Any,
    latest_joint_vec: Any,
    smoothed_init: list[float],
    latest_obs_for_action_lock: Any,
    latest_obs_for_action: Any,
    omy_display_action_lock: Any,
    omy_display_action: Any,
    gripper_got_lock: Any,
    latest_got0_holder: Any,
    ros_gripper_joint_name: str,
) -> None:
    """OMY joint polling in a ``spawn`` process: own ROS node + ``get_action()`` loop (see module docstring)."""
    import os

    import rclpy as _rclpy

    from lerobot.processor import make_default_processors
    from lerobot.teleoperators.OMY_L100 import OMYL100
    from lerobot.teleoperators.OMY_L100.config_OMY_L100 import OMYL100Config

    _orig_create_node = _rclpy.create_node

    def _create_node_spawn(name: str, *args: Any, **kwargs: Any):
        return _orig_create_node(f"{name}_spawn_{os.getpid()}", *args, **kwargs)

    _rclpy.create_node = _create_node_spawn
    teleop_action_processor, robot_action_processor, _unused_obs_proc = make_default_processors()
    teleop: OMYL100 | None = None
    try:
        cfg = OMYL100Config(**teleop_cfg)
        teleop = OMYL100(cfg)
        teleop.connect()
        omy_use_degrees = teleop.config.use_degrees
        smoothed_joint_vec = list(smoothed_init)
        period = 1.0 / CRP_JOINT_STREAM_HZ
        t_next = time.perf_counter()
        while not omy_action_stop.is_set():
            act = teleop.get_action()

            obs_for_action: RobotObservation = {}
            if latest_obs_for_action_lock is not None and latest_obs_for_action is not None:
                with latest_obs_for_action_lock:
                    obs_for_action = dict(latest_obs_for_action.copy())

            act_processed = teleop_action_processor((act, obs_for_action))
            robot_action_to_send_tamp = robot_action_processor((act_processed, obs_for_action))
            joint_action = _omy_joint_teleop_to_crp_action(
                robot_action_to_send_tamp,
                omy_use_degrees=omy_use_degrees,
            )
            joint_vec = [float(joint_action.get(f"j{i}.pos", 0.0)) for i in range(1, 7)]
            _smooth_joint_vec_inplace(smoothed_joint_vec, joint_vec, CRP_JOINT_INERTIA_BLEND)

            with joint_cmd_lock:
                latest_joint_vec[:] = smoothed_joint_vec

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
                logging.getLogger(__name__).exception("OMY spawn worker disconnect failed")


# OMY→CRP joint streaming: keep a fixed ``send_GJs`` cadence and align target updates to the same rate
# so ``TrajectoryProcessor`` is not filled with duplicate rows between slow (e.g. dataset) fps updates.
CRP_JOINT_STREAM_HZ = 100.0

# First-order low-pass on joint targets (OMY→CRP): each tick at ``CRP_JOINT_STREAM_HZ``, blend this
# fraction of the *new* master target into the running command. Acts as a low-pass filter: dampens
# high-frequency jitter when the OMY_L100 shakes or the JointState stream is noisy, and softens
# abrupt stop/start on the follower. Trade-off: lower value ⇒ smoother / less transmitted shake but
# more lag vs the master; higher value ⇒ snappier tracking. Does not fix mechanical play on OMY or
# slow ``send_GJs`` (see latency logs). 1.0 (or ≤0) = off (raw teleop). Try ~0.10–0.18 if shake persists.
CRP_JOINT_INERTIA_BLEND = 0.22

# OMY ``rh_r1_joint`` gripper position → CRP GOT0 (0=closed, 1000=open): clip to [-15, 15], then linear map.
OMY_GRIPPER_RH_R1_CLIP_LO = -15.0
OMY_GRIPPER_RH_R1_CLIP_HI = 15.0
CRP_GRIPPER_GOT0_MAX = 1000


def _omy_rh_r1_to_got0(raw: float) -> int:
    """Clamp OMY gripper joint value to [-15, 15], map linearly to GOT0 in [0, 1000]."""
    raw = raw
    c = max(OMY_GRIPPER_RH_R1_CLIP_LO, min(OMY_GRIPPER_RH_R1_CLIP_HI, float(raw)))
    span = OMY_GRIPPER_RH_R1_CLIP_HI - OMY_GRIPPER_RH_R1_CLIP_LO
    v = float(CRP_GRIPPER_GOT0_MAX) * (c - OMY_GRIPPER_RH_R1_CLIP_LO) / span 
    return int(max(0, min(CRP_GRIPPER_GOT0_MAX, round(v))))



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


def _wrap_deg_pm180(deg: float) -> float:
    """Wrap angle in degrees to the interval [-180, 180]."""
    return (deg + 180.0) % 360.0 - 180.0


def _smooth_joint_vec_inplace(smoothed: list[float], raw_target: list[float], blend: float) -> None:
    """In-place: ``smoothed[i] <- (1-blend)*smoothed[i] + blend*raw_target[i]`` for six joints."""
    b = float(blend)
    if b >= 1.0 or b <= 0.0:
        smoothed[:] = raw_target
        return
    bb = min(1.0, b)
    o = 1.0 - bb
    for i in range(6):
        smoothed[i] = o * smoothed[i] + bb * float(raw_target[i])


def _omy_deg_to_crp_deg(joint_index: int, omy_deg: float) -> float:
    """Map one OMY joint angle (degrees) to CRP joint angle (degrees). Result wrapped to [-180, 180]."""
    if joint_index == 1:
        # Same zero, same direction
        crp = omy_deg
    elif joint_index == 2:
        # OMY 0° → CRP 90°, direction inverted
        crp = 90.0 - omy_deg
    elif joint_index == 3:
        # OMY 90° → CRP 0°, direction inverted  =>  crp = 90 - omy
        crp = 90.0 - omy_deg
    elif joint_index == 4:
        # Same zero, direction inverted
        crp = - ( omy_deg % 360 )
    elif joint_index == 5:
        # OMY 90° → CRP 0°, same direction  =>  crp = omy - 90
        crp = ( omy_deg % 360 ) - 90.0
    elif joint_index == 6:
        # OMY -180° → CRP 0°, same direction  =>  crp = omy + 180
        crp = - ( omy_deg % 360 ) + 180.0
        crp = max(-70.0, min(70.0, crp))  # CRP j6 workspace limit
    else:
        raise ValueError(f"joint_index must be 1..6, got {joint_index}")
    return _wrap_deg_pm180(crp)


def _omy_joint_teleop_to_crp_action(
    teleop_action: dict[str, float],
    *,
    omy_use_degrees: bool,
) -> dict[str, float]:
    """Build ``j*.pos`` targets (degrees) for CRP; used with ``send_GJs`` after OMY→CRP joint mapping."""
    out: dict[str, float] = {}
    for j in range(1, 7):
        k = f"j{j}.pos"
        if k not in teleop_action:
            continue
        v = float(teleop_action[k])
        if not omy_use_degrees:
            v = math.degrees(v)
        out[k] = _omy_deg_to_crp_deg(j, v)
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

    gj_sender_stop: Any = None
    gj_sender_process: Any = None
    joint_cmd_lock: Any = None
    latest_joint_vec: list[float] | None = None
    smoothed_joint_vec: list[float] | None = None
    latest_got0_holder: list[int] | None = None
    gripper_got_lock: Any = None
    # Dedup ``set_GOT`` in the OMY update loop (and rare fallback); always defined for closure/fallback.
    last_got0_sent = [-10**9]

    # Initialize GP (end-pose) or GJ (joint) registers on the CRP controller.
    if trajectory_processor is not None:
        if isinstance(teleop, OMYL100) and isinstance(robot, CRPArm):
            if mp_ctx is None:
                raise RuntimeError("OMY multiprocess workers require a multiprocessing context.")
            crp_raw = robot.crp_arm_robot.read_joints()
            init_j = [_resolve_crp_read_joints_value(crp_raw, f"j{i}") for i in range(1, 7)]
            init_matrix = trajectory_processor.init_matrix(init_j, group_size=5)
            robot.send_GJs(10, init_matrix)
            robot.send_GJs(20, init_matrix)
            if getattr(teleop.config, "ros_gripper_joint_name", ""):
                latest_got0_holder = mp_manager.list([0]) if mp_manager is not None else [0]
                gripper_got_lock = mp_manager.Lock() if mp_manager is not None else None
                init_grip = float(teleop.get_action().get("gripper.pos", 0.0))
                _init_got = _omy_rh_r1_to_got0(init_grip)
                robot.set_GOT(0, _init_got)
                latest_got0_holder[0] = _init_got
                last_got0_sent[0] = _init_got
            latest_joint_vec = mp_manager.list(init_j) if mp_manager is not None else list(init_j)
            smoothed_joint_vec = list(init_j)
            joint_cmd_lock = mp_manager.Lock() if mp_manager is not None else None
            gj_sender_stop = mp_ctx.Event() if mp_ctx is not None else None

            def _fixed_rate_gj_sender() -> None:
                period = 1.0 / CRP_JOINT_STREAM_HZ
                t_next = time.perf_counter()
                assert latest_joint_vec is not None
                assert joint_cmd_lock is not None
                assert gj_sender_stop is not None
                while not gj_sender_stop.is_set():
                    with joint_cmd_lock:
                        vec = list(latest_joint_vec)
                    # Always refresh trajectory + send at ``CRP_JOINT_STREAM_HZ`` (independent of inertia /
                    # min-delta), so the controller sees a stable command cadence.
                    trajectory_processor.write_joint(vec)
                    robot.send_GJs(10, trajectory_processor.read_joints())
                    if gripper_got_lock is not None and latest_got0_holder is not None:
                        with gripper_got_lock:
                            got_target = int(latest_got0_holder[0])
                        if got_target != last_got0_sent[0]:
                            robot.set_GOT(0, got_target)
                            last_got0_sent[0] = got_target
                    t_next += period
                    dt = t_next - time.perf_counter()
                    if dt > 0:
                        precise_sleep(dt)
                    else:
                        t_next = time.perf_counter()

            gj_sender_process = mp_ctx.Process(
                target=_fixed_rate_gj_sender, name="crp_gj_sender", daemon=True
            )
            gj_sender_process.start()

            omy_spawn_ctx = _omy_spawn_context()
            assert joint_cmd_lock is not None
            assert latest_joint_vec is not None
            assert smoothed_joint_vec is not None
            assert omy_action_stop is not None
            _grip_name = str(getattr(teleop.config, "ros_gripper_joint_name", "") or "")
            omy_action_update_process = omy_spawn_ctx.Process(
                target=_spawn_omy_joint_action_worker,
                name="omy_action_update",
                daemon=True,
                kwargs={
                    "teleop_cfg": asdict(teleop.config),
                    "omy_action_stop": omy_action_stop,
                    "joint_cmd_lock": joint_cmd_lock,
                    "latest_joint_vec": latest_joint_vec,
                    "smoothed_init": list(smoothed_joint_vec),
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
            init_matrix = trajectory_processor.init_matrix(robot.get_current_endpose(), group_size=5)
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
                if isinstance(teleop, OMYL100):
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
                    trajectory_processor.write_point(robot_action_to_send)
                    _ = robot.send_GPs(10, trajectory_processor.read_points())
                else:
                    _ = robot.send_endpose(robot_action_to_send)
            else:
                action_values = act_processed_teleop
                robot_action_to_send_tamp = robot_action_processor((act_processed_teleop, obs))

                if isinstance(teleop, OMYL100):
                    if not isinstance(robot, CRPArm):
                        raise TypeError("OMY joint teleop requires robot.type=crp_arm (CRPArm.send_GJs).")
                    if trajectory_processor is None:
                        raise RuntimeError(
                            "trajectory_processor is required for OMY joint streaming (send_GJs)."
                        )
                    # Joint targets are produced by a dedicated worker process to decouple command freshness
                    # from observation/dataset latency in the main loop.
                    if joint_cmd_lock is None or latest_joint_vec is None:
                        joint_action = _omy_joint_teleop_to_crp_action(
                            robot_action_to_send_tamp,
                            omy_use_degrees=teleop.config.use_degrees,
                        )
                        joint_vec = [float(joint_action.get(f"j{i}.pos", 0.0)) for i in range(1, 7)]
                        if smoothed_joint_vec is not None:
                            _smooth_joint_vec_inplace(smoothed_joint_vec, joint_vec, CRP_JOINT_INERTIA_BLEND)
                            trajectory_processor.write_joint(smoothed_joint_vec)
                        else:
                            trajectory_processor.write_joint(joint_vec)
                        robot.send_GJs(10, trajectory_processor.read_joints())
                        if getattr(teleop.config, "ros_gripper_joint_name", ""):
                            _got_fb = _omy_rh_r1_to_got0(
                                float(robot_action_to_send_tamp.get("gripper.pos", 0.0))
                            )
                            if _got_fb != last_got0_sent[0]:
                                robot.set_GOT(0, _got_fb)
                                last_got0_sent[0] = _got_fb
                            if gripper_got_lock is not None and latest_got0_holder is not None:
                                with gripper_got_lock:
                                    latest_got0_holder[0] = _got_fb
                else:
                    robot_action_to_send = get_omy_endpose2Crp(robot_action_to_send_tamp)
                    if trajectory_processor is not None:
                        trajectory_processor.write_point(robot_action_to_send)
                        _ = robot.send_GPs(10, trajectory_processor.read_points())
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
                    and gripper_got_lock is not None
                    and latest_got0_holder is not None
                ):
                    with gripper_got_lock:
                        crp_arm_joint_values = {
                            **crp_arm_joint_values,
                            "gripper.pos": float(latest_got0_holder[0]),
                        }

                action_frame = build_dataset_frame(dataset.features, crp_arm_joint_values, prefix="action")
                frame = {**observation_frame, **action_frame, "task": single_task}
                dataset.add_frame(frame)

            if display_data:
                log_rerun_data(observation=obs_processed, action=action_values)

            dt_s = time.perf_counter() - start_loop_t
            precise_sleep(max(1 / fps - dt_s, 0.0))

            timestamp = time.perf_counter() - start_episode_t

    finally:
        if gj_sender_stop is not None:
            gj_sender_stop.set()
        if omy_action_stop is not None:
            omy_action_stop.set()
        if gj_sender_process is not None:
            gj_sender_process.join(timeout=5.0)
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
