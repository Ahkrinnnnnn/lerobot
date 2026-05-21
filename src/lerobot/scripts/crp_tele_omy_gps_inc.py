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
Record a LeRobot dataset while teleoperating CRP with OMY_L100 using **incremental GP**
end-effector commands (same dataset layout as ``crp_record_omy.py``).

Control (from-start incremental):
  ``p_cmd = clip( p0_crp + ( get_omy_endpose2Crp(ee_now)[:3] - get_omy_endpose2Crp(ee_ref)[:3] ) )``
  with fixed roll/pitch/yaw from config (or current pose at start).   ``ee_*`` for ``get_omy_endpose2Crp`` comes from either (a) optional ROS
  ``geometry_msgs/PoseStamped`` on ``--teleop.ros_end_effector_pose_topic`` (FK from the
  manipulator node), scaled by ``ros_ee_pose_position_scale`` (default SI m→mm), or (b) if
  that topic is unset or has no message yet, from OMY ``j*.pos`` as before.

Recording follows ``crp_record_omy.py``: ``build_dataset_frame`` for observation/action,
``dataset.add_frame``, episodes / reset / hub like ``record()``.

Multiprocessing (aligned with ``lerobot_deploy.py``): ``spawn`` context, an ``RLock`` around all
``CrpRobotPy`` / camera I/O, fixed-rate ticks (``next_tick_t`` + ``precise_sleep`` at the start of
each step). With ``--policy.path=...``, a daemon **inference** process consumes a size-1
``obs_queue`` of picklable observation frames and writes ``latest_action_ref`` (Manager dict);
the main process never runs ``predict_action`` on-policy when MP is active. Without a policy,
the main loop uses a **fast** ``get_observation(include_images=False)`` before ``send_GPs`` and
a second **full** read for dataset / cameras.

Example:

```shell
python -m lerobot.scripts.crp_tele_omy_gps_inc \\
  --robot.type=crp_arm \\
  --robot.port=/dev/ttyUSB0 \\
  --robot.cameras='{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \\
  --teleop.type=OMY_L100 \\
  --teleop.port=dummy \\
  --dataset.repo_id=<user>/crp_omy_gps_inc \\
  --dataset.num_episodes=10 \\
  --dataset.single_task="Pick and place" \\
  --display_data=false
```
"""

import logging
import math
import queue
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import rerun as rr

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
from lerobot.scripts.lerobot_deploy import (
    _current_pose_as_action,
    _deploy_get_robot_observation,
    _deploy_multiprocessing_context,
    _inference_worker,
)
import lerobot.robots.bi_so_follower  # noqa: F401
import lerobot.robots.hope_jr  # noqa: F401
import lerobot.robots.koch_follower  # noqa: F401
import lerobot.robots.so_follower  # noqa: F401
import lerobot.robots.crp_arm  # noqa: F401
from lerobot.robots.crp_arm import CRPArm
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig, make_teleoperator_from_config
import lerobot.teleoperators.bi_so_leader  # noqa: F401
import lerobot.teleoperators.homunculus  # noqa: F401
import lerobot.teleoperators.koch_leader  # noqa: F401
import lerobot.teleoperators.so_leader  # noqa: F401
import lerobot.teleoperators.OMY_L100  # noqa: F401
from lerobot.teleoperators.OMY_L100.OMY_L100 import OMYL100
from lerobot.tools.TrajProcessor import TrajectoryProcessor
from lerobot.tools.kinematics import get_omy_endpose2Crp
from lerobot.tools.lib_loader import load_CrpRobotPy
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

load_CrpRobotPy()

OMY_GRIPPER_RH_R1_CLIP_LO = -15.0
OMY_GRIPPER_RH_R1_CLIP_HI = 15.0
CRP_GRIPPER_GOT0_MAX = 1000


def _gps_inc_get_obs_locked(
    robot: Robot, robot_io_lock: Any, *, include_images: bool
) -> dict[str, Any]:
    """``lerobot_deploy``-style: all ``get_observation`` / SDK access under the same reentrant lock."""
    with robot_io_lock:
        return _deploy_get_robot_observation(robot, include_images=include_images)


def _omy_rh_r1_to_got0(raw: float) -> int:
    """Map OMY ``rh_r1_joint``-style value to GOT0 [0, 1000] (same as ``crp_record_omy``)."""
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
    """Build flat action dict matching ``dataset.features['action']['names']`` (``crp_record_omy``)."""
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


def _clip_crp_xyz_mm(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Clip commanded CRP translation (mm) to the same safe box as ``map_omy2crp`` output."""
    return (
        float(np.clip(x, 320.0, 800.0)),
        float(np.clip(y, -400.0, 400.0)),
        float(np.clip(z, -300.0, 300.0)),
    )


def _joint_action_to_omy_ee_dict(
    robot_action: dict[str, float], *, omy_use_degrees: bool
) -> dict[str, float]:
    """
    Build ``ee.*`` for ``get_omy_endpose2Crp`` from OMY ``j*.pos`` (incremental path).

    Joints are passed through as task-space scalars; ``map_omy2crp`` / ``get_omy_endpose2Crp``
    clip and transform. Deltas ``crp(now) - crp(ref)`` only need a consistent parametrization.
    """
    j = [float(robot_action.get(f"j{n}.pos", 0.0)) for n in range(1, 7)]
    if not omy_use_degrees:
        j = [math.degrees(x) for x in j]
    return {
        "ee.x": j[0],
        "ee.y": j[1],
        "ee.z": j[2],
        "ee.roll": j[3],
        "ee.pitch": j[4],
        "ee.yaw": j[5],
    }


def _crp_xyz_from_omy_action(robot_action: dict[str, float], *, omy_use_degrees: bool) -> list[float]:
    ee = _joint_action_to_omy_ee_dict(robot_action, omy_use_degrees=omy_use_degrees)
    return get_omy_endpose2Crp(ee)[:3]


def _crp_xyz_from_teleop(
    teleop: OMYL100, robot_action: dict[str, float], *, omy_use_degrees: bool
) -> list[float]:
    """Prefer ROS ``PoseStamped`` EE if configured; else joint-based ``_crp_xyz_from_omy_action``."""
    ros_ee = teleop.get_ros_end_effector_xyz_rpy_deg()
    if ros_ee is not None:
        (x, y, z), (roll, pitch, yaw) = ros_ee
        ee = {
            "ee.x": float(x),
            "ee.y": float(y),
            "ee.z": float(z),
            "ee.roll": float(roll),
            "ee.pitch": float(pitch),
            "ee.yaw": float(yaw),
        }
        return get_omy_endpose2Crp(ee)[:3]
    return _crp_xyz_from_omy_action(robot_action, omy_use_degrees=omy_use_degrees)


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
class GpsIncRecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    teleop: TeleoperatorConfig
    policy: PreTrainedConfig | None = None
    display_data: bool = False
    play_sounds: bool = True
    resume: bool = False
    # Fixed CRP command orientation (deg, xyz), same as ``read_end_pose_user``.
    rpydeg_roll: float = -180.0
    rpydeg_pitch: float = 0.0
    rpydeg_yaw: float = 0.0
    rpy_from_current_on_start: bool = False

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


@safe_stop_image_writer
def record_loop_gps_inc(
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
    dataset: LeRobotDataset,
    teleop: Teleoperator,
    *,
    single_task: str,
    display_data: bool,
    trajectory_processor: TrajectoryProcessor,
    control_time_s: int | float,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
    rpydeg_roll: float = -180.0,
    rpydeg_pitch: float = 0.0,
    rpydeg_yaw: float = 0.0,
    rpy_from_current_on_start: bool = False,
) -> None:
    if dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")
    if not isinstance(teleop, OMYL100):
        raise TypeError("record_loop_gps_inc expects teleop.type=OMY_L100")
    if not isinstance(robot, CRPArm):
        raise TypeError("record_loop_gps_inc expects robot.type=crp_arm (CRPArm)")

    omy_use_deg = bool(getattr(teleop.config, "use_degrees", True))
    for _ in range(10):
        a = teleop.get_action()
        if a.get("j1.pos", 0.0) or a.get("j2.pos", 0.0):
            break
        time.sleep(0.01)

    mp_ctx = _deploy_multiprocessing_context()
    robot_io_lock = mp_ctx.RLock()

    with robot_io_lock:
        p0 = robot.get_current_endpose()
    rpy: tuple[float, float, float]
    if rpy_from_current_on_start:
        rpy = (float(p0[3]), float(p0[4]), float(p0[5]))
    else:
        rpy = (rpydeg_roll, rpydeg_pitch, rpydeg_yaw)
    p0_crp = (float(p0[0]), float(p0[1]), float(p0[2]))

    for _ in range(3):
        time.sleep(0.002)
    ref_obs = _gps_inc_get_obs_locked(robot, robot_io_lock, include_images=True)
    ra = teleop.get_action()
    ta = teleop_action_processor((ra, ref_obs))
    robot_action_ref: dict[str, Any] = dict(robot_action_processor((ta, ref_obs)))
    crp_ref_xyz: list[float] = _crp_xyz_from_teleop(teleop, robot_action_ref, omy_use_degrees=omy_use_deg)

    with robot_io_lock:
        init_matrix = trajectory_processor.init_matrix(robot.get_current_endpose(), group_size=5)
        robot.send_GPs(10, init_matrix)
        robot.send_GPs(20, init_matrix)

    use_policy_mp = policy is not None and preprocessor is not None and postprocessor is not None
    period_s = 1.0 / float(fps)
    control_loop_stats: dict[str, int] = {}

    action_manager: Any = None
    inference_process: Any = None
    obs_queue: Any = None
    latest_action_ref: Any = None
    latest_action_lock: Any = None
    device = get_safe_torch_device(policy.config.device) if use_policy_mp else None

    if use_policy_mp:
        action_manager = mp_ctx.Manager()
        try:
            latest_action_ref = action_manager.dict()
            latest_action_ref["action"] = None
            latest_action_lock = mp_ctx.Lock()
            obs_queue = mp_ctx.Queue(1)
            inference_process = mp_ctx.Process(
                target=_inference_worker,
                kwargs=dict(
                    obs_queue=obs_queue,
                    latest_action_ref=latest_action_ref,
                    latest_action_lock=latest_action_lock,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    dataset_features=dataset.features,
                    single_task=single_task,
                    robot_type=robot.robot_type,
                    use_amp=policy.config.use_amp,
                    device=device,
                ),
                daemon=True,
            )
            inference_process.start()

            with robot_io_lock:
                seed_obs = _deploy_get_robot_observation(robot, include_images=True)
            seed_proc = robot_observation_processor(seed_obs)
            seed_frame = build_dataset_frame(dataset.features, dict(seed_proc), prefix=OBS_STR)
            try:
                obs_queue.put_nowait(seed_frame)
            except queue.Full:
                pass
        except Exception:
            if inference_process is not None and getattr(inference_process, "is_alive", lambda: False)():
                try:
                    if obs_queue is not None:
                        obs_queue.put(None)
                except Exception:
                    pass
                inference_process.join(timeout=2.0)
            if action_manager is not None:
                action_manager.shutdown()
            raise

    start_episode_t = time.perf_counter()
    main_loop_i = 0
    next_tick_t = time.perf_counter()
    timestamp = 0.0

    try:
        while timestamp < float(control_time_s):
            if main_loop_i > 0:
                next_tick_t += period_s
                precise_sleep(max(0.0, next_tick_t - time.perf_counter()))

            start_loop_t = time.perf_counter()
            main_loop_i += 1

            if events["exit_early"]:
                events["exit_early"] = False
                break

            teleop_action: RobotAction
            action_values: Any
            obs_processed: RobotObservation
            obs_full: dict[str, Any]

            if use_policy_mp:
                obs_fast = _gps_inc_get_obs_locked(robot, robot_io_lock, include_images=False)
                obs_fast_processed = robot_observation_processor(obs_fast)
                with latest_action_lock:
                    policy_action = latest_action_ref["action"]
                if policy_action is not None and len(policy_action) == len(robot.action_features):
                    act_processed_policy = policy_action
                else:
                    act_processed_policy = _current_pose_as_action(
                        obs_fast_processed, robot.action_features
                    )
                robot_action_to_send = robot_action_processor((act_processed_policy, obs_fast))
                trajectory_processor.write_point(robot_action_to_send)
                with robot_io_lock:
                    robot.send_GPs(10, trajectory_processor.read_points())
                teleop_action = act_processed_policy
                action_values = act_processed_policy

                obs_full = _gps_inc_get_obs_locked(robot, robot_io_lock, include_images=True)
                obs_processed = robot_observation_processor(obs_full)
                observation_frame = build_dataset_frame(
                    dataset.features, obs_processed, prefix=OBS_STR
                )
                try:
                    obs_queue.put_nowait(observation_frame)
                except queue.Full:
                    pass
            else:
                obs_fast = _gps_inc_get_obs_locked(robot, robot_io_lock, include_images=False)
                raw_action = teleop.get_action()
                teleop_action = teleop_action_processor((raw_action, obs_fast))
                robot_action_to_send = dict(robot_action_processor((teleop_action, obs_fast)))
                crp_now_xyz = _crp_xyz_from_teleop(
                    teleop, robot_action_to_send, omy_use_degrees=omy_use_deg
                )
                d = [float(crp_now_xyz[i] - crp_ref_xyz[i]) for i in range(3)]
                px, py, pz = _clip_crp_xyz_mm(
                    p0_crp[0] + d[0], p0_crp[1] + d[1], p0_crp[2] + d[2]
                )
                endpose: list[float] = [px, py, pz, rpy[0], rpy[1], rpy[2]]
                trajectory_processor.write_point(endpose)
                with robot_io_lock:
                    robot.send_GPs(10, trajectory_processor.read_points())
                action_values = teleop_action

                obs_full = _gps_inc_get_obs_locked(robot, robot_io_lock, include_images=True)
                obs_processed = robot_observation_processor(obs_full)
                observation_frame = build_dataset_frame(
                    dataset.features, obs_processed, prefix=OBS_STR
                )

            if isinstance(robot, CRPArm):
                joint_snapshot = {
                    k.removesuffix(".pos"): float(v)
                    for k, v in obs_full.items()
                    if k.endswith(".pos") and isinstance(v, (int, float))
                }
                crp_arm_joint_values = _action_values_for_dataset_from_crp_joints(dataset, joint_snapshot)
            else:
                with robot_io_lock:
                    crp_arm_joint = robot.crp_arm_robot.read_joints()
                crp_arm_joint_values = _action_values_for_dataset_from_crp_joints(dataset, crp_arm_joint)

            action_names_ds = list((dataset.features.get("action") or {}).get("names") or [])
            if (
                "gripper.pos" in action_names_ds
                and getattr(teleop.config, "ros_gripper_joint_name", "")
                and policy is None
            ):
                g_raw = float(teleop_action.get("gripper.pos", 0.0))
                crp_arm_joint_values = {
                    **crp_arm_joint_values,
                    "gripper.pos": float(_omy_rh_r1_to_got0(g_raw)),
                }

            action_frame = build_dataset_frame(dataset.features, crp_arm_joint_values, prefix="action")
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

            if display_data:
                log_rerun_data(observation=obs_processed, action=action_values)

            dt_s = time.perf_counter() - start_loop_t
            if dt_s > period_s:
                control_loop_stats["overruns"] = control_loop_stats.get("overruns", 0) + 1
                ov = control_loop_stats["overruns"]
                if ov <= 5 or ov % 30 == 0:
                    logging.warning(
                        "GpsInc[pipeline overrun]: step took %.1f ms (period %.1f ms); total overruns=%d.",
                        dt_s * 1e3,
                        period_s * 1e3,
                        ov,
                    )
            timestamp = time.perf_counter() - start_episode_t
    finally:
        if inference_process is not None and obs_queue is not None:
            try:
                obs_queue.put(None)
            except Exception:
                pass
            inference_process.join(timeout=5.0)
        if action_manager is not None:
            try:
                action_manager.shutdown()
            except Exception:
                pass

    if control_loop_stats.get("overruns", 0) > 0:
        logging.info(
            "GpsInc episode: %d control-loop overruns (fps=%d, period=%.1f ms).",
            control_loop_stats["overruns"],
            fps,
            period_s * 1e3,
        )


@parser.wrap()
def record_gps_inc(cfg: GpsIncRecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="recording")

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)
    if not isinstance(teleop, OMYL100):
        raise TypeError("This script expects --teleop.type=OMY_L100")

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )
    if getattr(cfg.teleop, "ros_gripper_joint_name", "") and cfg.robot.type == "crp_arm":
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
    teleop.connect()

    listener, events = init_keyboard_listener()

    print("current speed ratio:", robot.get_speed_ratio())
    robot.set_speed_ratio(20)
    print("current speed ratio:", robot.get_speed_ratio())

    trajectory_processor = TrajectoryProcessor()

    with VideoEncodingManager(dataset):
        recorded_episodes = 0
        while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
            log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
            record_loop_gps_inc(
                robot=robot,
                events=events,
                fps=cfg.dataset.fps,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                dataset=dataset,
                teleop=teleop,
                single_task=cfg.dataset.single_task,
                display_data=cfg.display_data,
                trajectory_processor=trajectory_processor,
                control_time_s=cfg.dataset.episode_time_s,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                rpydeg_roll=cfg.rpydeg_roll,
                rpydeg_pitch=cfg.rpydeg_pitch,
                rpydeg_yaw=cfg.rpydeg_yaw,
                rpy_from_current_on_start=cfg.rpy_from_current_on_start,
            )

            if not events["stop_recording"] and (
                (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
            ):
                log_say("Reset the environment", cfg.play_sounds)
                record_loop_gps_inc(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    dataset=dataset,
                    teleop=teleop,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    trajectory_processor=trajectory_processor,
                    control_time_s=cfg.dataset.reset_time_s,
                    policy=None,
                    preprocessor=None,
                    postprocessor=None,
                    rpydeg_roll=cfg.rpydeg_roll,
                    rpydeg_pitch=cfg.rpydeg_pitch,
                    rpydeg_yaw=cfg.rpydeg_yaw,
                    rpy_from_current_on_start=cfg.rpy_from_current_on_start,
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
    teleop.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()

    if cfg.dataset.push_to_hub:
        dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    log_say("Exiting", cfg.play_sounds)
    return dataset


def main() -> None:
    record_gps_inc()


if __name__ == "__main__":
    main()
