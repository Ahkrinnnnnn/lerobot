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

import logging
import math
import threading
import time

from scipy.spatial.transform import Rotation as R_scipy

from lerobot.robots.crp_arm.ee_gp import EE_OMY_DELTA_SCALE, omy_rh_r1_to_got0, wrap_angle_delta_deg
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_OMY_L100 import OMYL100Config

# rclpy is an optional runtime dependency used when this teleoperator is connected
try:
    import rclpy
    from geometry_msgs.msg import Pose as RosPose
    from geometry_msgs.msg import PoseStamped
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
        qos_profile_sensor_data,
    )
    from sensor_msgs.msg import JointState

    _EE_QOS_RELIABLE = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        durability=DurabilityPolicy.VOLATILE,
    )
except Exception:  # pragma: no cover - if ROS2 not available, node will not connect
    rclpy = None
    JointState = None
    PoseStamped = None
    RosPose = None
    qos_profile_sensor_data = None
    _EE_QOS_RELIABLE = None

logger = logging.getLogger(__name__)

# ``ros2 launch open_manipulator_bringup omy_l100_leader_ai.launch.py`` uses ``PushRosNamespace('leader')``.
JOINT_STATES_TOPIC = "/leader/joint_states"
EE_STATES_TOPIC = "/end_effector_pose"
ROS_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")


class OMYL100(Teleoperator):

    config_class = OMYL100Config
    name = "OMY_L100"

    def __init__(self, config: OMYL100Config):
        super().__init__(config)
        self.config = config
        self.OMY_joints = {    
        "j1": float,
        "j2": float, 
        "j3": float,
        "j4": float,
        "j5": float,
        "j6": float,
        }
        # ROS related members
        self._ros_node = None
        self._ros_thread = None
        self._ros_running = False
        self._last_joint_state = None
        self._joint_lock = threading.Lock()
        # Set together with ``_last_joint_state`` in the JointState callback (same clock as ``time.perf_counter()``).
        self._last_joint_recv_mono: float | None = None
        self._last_joint_recv_seq: int = 0
        # After ``get_action()``: source of the returned dict (for E2E latency to CRP).
        self._action_source_recv_mono: float | None = None
        self._action_source_recv_seq: int = 0
        # Gap between the last two JointState callbacks (ms); updated in callback (GIL / spin health).
        self._prev_joint_cb_mono: float | None = None
        self._last_joint_cb_interval_ms: float = 0.0
        # Optional ``geometry_msgs/PoseStamped`` or ``Pose`` (FK EE from collision / manipulator node).
        self._ee_pose_lock = threading.Lock()
        self._last_ee_pose = None
        self._logged_first_ee_pose = False
        # HIL EE-delta state (only used when ``config.hil_ee_delta`` is True).
        self._hil_prev_xyz: tuple[float, float, float] | None = None
        self._hil_prev_rpy: tuple[float, float, float] | None = None
        self._hil_keyboard = None

    @property
    def action_features(self) -> dict[str, type]:
        if getattr(self.config, "hil_ee_delta", False):
            ft: dict[str, type] = {"delta_x": float, "delta_y": float, "delta_z": float}
            if getattr(self.config, "hil_use_gripper", True):
                ft["gripper"] = float
            return ft
        ft = {f"{joint}.pos": joint_type for joint, joint_type in self.OMY_joints.items()}
        if self.config.ros_gripper_joint_name:
            ft["gripper.pos"] = float
        return ft

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}
 
    @property
    def is_connected(self) -> bool:
        return getattr(self, "_ros_node", None) is not None and getattr(self, "_ros_thread", None) is not None and getattr(self, "_ros_thread").is_alive()


    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # initialize ROS if available
        if rclpy is None or JointState is None:
            raise RuntimeError("ROS2 (rclpy) or sensor_msgs is not available in this environment")

        try:
            # rclpy.init() may raise if already initialized; ignore in that case
            rclpy.init()
        except Exception:
            # ignore initialization errors (likely already initialized)
            pass

        # create node and subscription
        self._ros_node = rclpy.create_node(f"{self.name.lower()}_teleoperator_node")

        def _joint_cb(msg: JointState):
            now = time.perf_counter()
            with self._joint_lock:
                self._last_joint_state = msg
                if self._prev_joint_cb_mono is not None:
                    self._last_joint_cb_interval_ms = (now - self._prev_joint_cb_mono) * 1000
                else:
                    self._last_joint_cb_interval_ms = 0.0
                self._prev_joint_cb_mono = now
                self._last_joint_recv_mono = now
                self._last_joint_recv_seq = self._last_joint_recv_seq + 1

        topic = getattr(self.config, "joint_states_topic", JOINT_STATES_TOPIC)
        self._ros_node.create_subscription(JointState, topic, _joint_cb, 10)
        logger.info(f"{self} subscribing to JointState on {topic}")

        # Config default is often ``""``; fall back to module ``EE_STATES_TOPIC``.
        ee_topic = (getattr(self.config, "ros_end_effector_pose_topic", "") or EE_STATES_TOPIC).strip()
        if ee_topic and not ee_topic.startswith("/"):
            ee_topic = f"/{ee_topic.lstrip('/')}"
        if ee_topic:
            msg_kind = (getattr(self.config, "ros_end_effector_pose_msg_type", "pose") or "pose").lower()
            msg_kind = msg_kind.replace("-", "_")
            if msg_kind in ("pose",):
                ee_msg_cls = RosPose
                ee_label = "Pose"
            else:
                ee_msg_cls = PoseStamped
                ee_label = "PoseStamped"

            qos_mode = (getattr(self.config, "ros_ee_pose_qos", "reliable") or "reliable").lower()
            if qos_mode in ("reliable", "default", "system_default"):
                ee_qos = _EE_QOS_RELIABLE
                qos_label = "RELIABLE depth=10"
            else:
                ee_qos = qos_profile_sensor_data
                qos_label = "sensor (BEST_EFFORT)"

            if ee_msg_cls is None:
                logger.warning(
                    "%s: ros_end_effector_pose_topic=%r set but geometry_msgs/%s is unavailable",
                    self,
                    ee_topic,
                    ee_label,
                )
            elif ee_qos is None:
                logger.warning("%s: rclpy QoS unavailable; skipping EE pose subscription", self)
            else:

                def _ee_pose_cb(msg):
                    with self._ee_pose_lock:
                        self._last_ee_pose = msg
                    if not self._logged_first_ee_pose:
                        self._logged_first_ee_pose = True
                        logger.debug(
                            "%s first %s on %r QoS=%s (get_ros_end_effector_xyz_rpy_deg ready)",
                            self,
                            ee_label,
                            ee_topic,
                            qos_label,
                        )

                self._ros_node.create_subscription(ee_msg_cls, ee_topic, _ee_pose_cb, ee_qos)
                logger.info("%s subscribing to %s (EE, %s) on %s", self, ee_label, qos_label, ee_topic)

        # run a spin loop in a background thread so callbacks are processed
        self._ros_running = True

        def _spin_loop():
            try:
                while self._ros_running:
                    rclpy.spin_once(self._ros_node, timeout_sec=0.1)
            except Exception:
                logger.exception("Exception in ROS spin loop")

        self._ros_thread = threading.Thread(target=_spin_loop, name=f"{self.name}_ros_spin", daemon=True)
        self._ros_thread.start()

        # Run any configuration steps
        self.configure()
        if getattr(self.config, "hil_ee_delta", False):
            from lerobot.teleoperators.keyboard_hil_events import KeyboardHilEvents

            self._hil_keyboard = KeyboardHilEvents(
                intervene_key=getattr(self.config, "hil_intervene_key", "space"),
                success_key=getattr(self.config, "hil_success_key", "s"),
                failure_key=getattr(self.config, "hil_failure_key", "f"),
                rerecord_key=getattr(self.config, "hil_rerecord_key", "r"),
                reset_pause_key=getattr(self.config, "hil_reset_pause_key", None),
            )
            self._hil_keyboard.start()
            self.reset_reference()
        logger.info(f"{self} connected.")


    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass


    def configure(self) -> None:
        pass

    def get_gripper_raw(self) -> float:
        """Return OMY gripper joint raw value (same source as recording → ``omy_rh_r1_to_got0``)."""
        joints = self._get_joint_action()
        return float(joints.get("gripper.pos", 0.0))

    def reset_reference(self) -> None:
        """Latch HIL EE reference so the next ``get_action`` incremental delta starts from zero."""
        ee = self.get_ros_end_effector_xyz_rpy_deg()
        if ee is None:
            self._hil_prev_xyz = None
            self._hil_prev_rpy = None
            return
        self._hil_prev_xyz = tuple(float(v) for v in ee[0])
        self._hil_prev_rpy = tuple(float(v) for v in ee[1])

    def get_teleop_events(self) -> dict:
        """HIL control events from the shared keyboard helper (requires ``hil_ee_delta``)."""
        from lerobot.teleoperators.utils import TeleopEvents

        if self._hil_keyboard is None:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }
        return self._hil_keyboard.consume_events()

    def get_action(self) -> dict[str, float]:
        """Return joint positions, or HIL EE deltas when ``hil_ee_delta`` is enabled."""
        if getattr(self.config, "hil_ee_delta", False):
            return self._get_hil_ee_delta_action()
        return self._get_joint_action()

    def _get_hil_ee_delta_action(self) -> dict[str, float]:
        """Incremental EE xyz+rpy delta (+ optional gripper) for HIL-SERL EE action space."""
        ee = self.get_ros_end_effector_xyz_rpy_deg()
        norm = float(getattr(self.config, "hil_ee_delta_norm", 1.0) or 1.0)
        if ee is None:
            action = {
                "delta_x": 0.0,
                "delta_y": 0.0,
                "delta_z": 0.0,
                "delta_roll": 0.0,
                "delta_pitch": 0.0,
                "delta_yaw": 0.0,
            }
        else:
            xyz = tuple(float(v) for v in ee[0])
            rpy = tuple(float(v) for v in ee[1])
            if self._hil_prev_xyz is None or self._hil_prev_rpy is None:
                self._hil_prev_xyz = xyz
                self._hil_prev_rpy = rpy
                dxyz = (0.0, 0.0, 0.0)
                drpy = (0.0, 0.0, 0.0)
            else:
                dxyz = tuple(
                    EE_OMY_DELTA_SCALE * (xyz[i] - self._hil_prev_xyz[i]) / norm for i in range(3)
                )
                drpy = tuple(wrap_angle_delta_deg(rpy[i], self._hil_prev_rpy[i]) / norm for i in range(3))
                self._hil_prev_xyz = xyz
                self._hil_prev_rpy = rpy
            action = {
                "delta_x": dxyz[0],
                "delta_y": dxyz[1],
                "delta_z": dxyz[2],
                "delta_roll": drpy[0],
                "delta_pitch": drpy[1],
                "delta_yaw": drpy[2],
            }

        if getattr(self.config, "hil_use_gripper", True):
            action["gripper"] = float(omy_rh_r1_to_got0(self.get_gripper_raw()))
        return action

    def _get_joint_action(self) -> dict[str, float]:
        """Return latest joint positions from `sensor_msgs/JointState` (degrees when `use_degrees` is True).

        Only scans ``JointState`` for the arm (+ optional gripper) names. Avoids ``dict(zip(all names))``:
        ``crp_record_omy`` polls this at ~500 Hz; a large ``JointState`` would otherwise allocate huge dicts
        every call and starve other threads (cameras, control loops).
        """
        with self._joint_lock:
            js = self._last_joint_state
            recv_mono = self._last_joint_recv_mono
            recv_seq = self._last_joint_recv_seq

        zeros = {f"{j}.pos": 0.0 for j in self.OMY_joints}
        if self.config.ros_gripper_joint_name:
            zeros["gripper.pos"] = 0.0
        if js is None:
            logger.debug(f"{self} no joint state received yet, returning zeros")
            self._action_source_recv_mono = None
            self._action_source_recv_seq = 0
            return zeros

        ros_joint_names = getattr(self.config, "ros_joint_names", ROS_JOINT_NAMES)
        gname = self.config.ros_gripper_joint_name
        needed: set[str] = set(ros_joint_names)
        if gname:
            needed.add(gname)
        need_count = len(needed)

        name_to_pos: dict[str, float] = {}
        for n, p in zip(js.name, js.position):
            if n in needed:
                name_to_pos[n] = float(p)
                if len(name_to_pos) >= need_count:
                    break

        action: dict[str, float] = {}
        for joint_key, ros_name in zip(self.OMY_joints.keys(), ros_joint_names):
            key = f"{joint_key}.pos"
            if ros_name in name_to_pos:
                val = name_to_pos[ros_name]
                if self.config.use_degrees:
                    val = math.degrees(val)
                action[key] = val
            else:
                action[key] = 0.0

        if gname:
            if gname in name_to_pos:
                gval = name_to_pos[gname]
                if self.config.use_degrees and self.config.gripper_apply_use_degrees:
                    gval = math.degrees(gval)
                action["gripper.pos"] = gval
            else:
                action["gripper.pos"] = 0.0

        self._action_source_recv_mono = recv_mono
        self._action_source_recv_seq = recv_seq
        return action

    def get_ros_end_effector_xyz_rpy_deg(self) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
        """Latest FK pose from optional ``Pose`` / ``PoseStamped`` topic, or None if no message yet."""
        with self._ee_pose_lock:
            m = self._last_ee_pose
        if m is None:
            return None
        scale = float(getattr(self.config, "ros_ee_pose_position_scale", 1000.0))
        if RosPose is not None and isinstance(m, RosPose):
            p, q = m.position, m.orientation
        elif PoseStamped is not None and isinstance(m, PoseStamped):
            p, q = m.pose.position, m.pose.orientation
        else:
            # Duck-type (e.g. mocks): prefer ``.pose`` like ``PoseStamped``.
            pose = getattr(m, "pose", m)
            p = pose.position
            q = pose.orientation
        x = float(p.x) * scale
        y = float(p.y) * scale
        z = float(p.z) * scale
        roll, pitch, yaw = R_scipy.from_quat([float(q.x), float(q.y), float(q.z), float(q.w)]).as_euler(
            "xyz", degrees=True
        )
        return ((x, y, z), (float(roll), float(pitch), float(yaw)))

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError


    def disconnect(self) -> None:
        if not self.is_connected:
            DeviceNotConnectedError(f"{self} is not connected.")

        if self._hil_keyboard is not None:
            try:
                self._hil_keyboard.stop()
            except Exception:
                logger.exception("%s failed to stop HIL keyboard", self)
            self._hil_keyboard = None

        # stop ROS spin thread and destroy node
        try:
            self._ros_running = False
            if self._ros_thread is not None:
                self._ros_thread.join(timeout=1.0)
            if self._ros_node is not None:
                try:
                    self._ros_node.destroy_node()
                except Exception:
                    # ignore destroy errors
                    pass
        finally:
            self._ros_thread = None
            self._ros_node = None
            self._logged_first_ee_pose = False
            self._hil_prev_xyz = None
            self._hil_prev_rpy = None

        logger.info(f"{self} disconnected.")
