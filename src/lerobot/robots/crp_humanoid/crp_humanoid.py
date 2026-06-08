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

"""CRP humanoid: ROS observation feedback and command publishing (ti5_interfaces).

Feedback (observation):
  - ``/multi_motor_state`` — body + arms (20 motors, position in rad)
  - ``/joint_skillful_hand_state`` — dexterous hand finger positions

Commands (``send_action`` / deploy):
  - ``/motor_command`` — ``MotorCommand`` (position mode by default)
  - ``/skillfulHand_command`` — ``SkillfulHandCommand`` (L/R per message)

Cameras (OrbbecSDK v2 via ``pyorbbecsdk2``, default 640x480 @ 30 FPS):
  - ``head``, ``left_wrist``, ``right_wrist`` — USB Orbbec (``OrbbecCameraConfig``)
  - Set a unique ``serial_number`` per camera when multiple devices are connected
  - Override with ``--robot.cameras`` or use ``OrbbecCamera.find_cameras()`` to list devices
"""

from __future__ import annotations

import logging
import threading
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.ros.camera_ros import RosImageCamera
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ._ti5_ros import (
    SKILLFUL_HAND_L,
    SKILLFUL_HAND_R,
    RosSpinSession,
    default_hand_obs_keys,
    default_motor_stems,
    default_ros_node_name,
    hand_side_keys,
    hand_side_present_in_action,
    hand_targets_from_action,
    import_ti5_messages,
    motor_stem_for_id,
    motor_targets_from_action,
    resolve_ros_topic,
    rclpy_available,
)
from .config_crp_humanoid import CRPHumanoidConfig

logger = logging.getLogger(__name__)


class CRPHumanoid(Robot):
    config_class = CRPHumanoidConfig
    name = "crp_humanoid"

    def __init__(self, config: CRPHumanoidConfig):
        super().__init__(config)
        self.config = config
        self._motor_stems = config.ros_joint_names or default_motor_stems(config.num_body_motors)
        self._hand_obs_keys = default_hand_obs_keys(config.hand_fingers_per_side)
        self._state_keys = tuple(f"{stem}.pos" for stem in self._motor_stems) + self._hand_obs_keys
        self._joints_ft = {key: float for key in self._state_keys}
        self.cameras = make_cameras_from_configs(config.cameras)

        self._ros = RosSpinSession()
        self._state_lock = threading.Lock()
        self._last_motor_positions: dict[str, float] = {
            f"{stem}.pos": 0.0 for stem in self._motor_stems
        }
        self._last_hand_positions: dict[str, float] = dict.fromkeys(self._hand_obs_keys, 0.0)

        self._ros_feedback_enabled = bool(
            config.multi_motor_state_topic.strip() or config.joint_skillful_hand_state_topic.strip()
        )
        self._ros_command_enabled = bool(
            config.motor_command_topic.strip() or config.skillful_hand_command_topic.strip()
        )
        self._ros_enabled = self._ros_feedback_enabled or self._ros_command_enabled

        self._motor_cmd_pub = None
        self._hand_cmd_pub = None
        self._MotorCommand = None
        self._SkillfulHandCommand = None

        self._warned_stub_state = False
        self._connected = False

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @property
    def _ros_image_cameras(self) -> list[RosImageCamera]:
        return [cam for cam in self.cameras.values() if isinstance(cam, RosImageCamera)]

    @property
    def _needs_ros_node(self) -> bool:
        return self._ros_enabled or bool(self._ros_image_cameras)

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._joints_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict(self._joints_ft)

    @property
    def is_connected(self) -> bool:
        if not self._connected:
            return False
        if self.cameras and not all(cam.is_connected for cam in self.cameras.values()):
            return False
        if self._needs_ros_node and not self._ros.is_alive:
            return False
        return True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _state_fallback(self) -> dict[str, float]:
        with self._state_lock:
            return {**self._last_motor_positions, **self._last_hand_positions}

    def _update_hand_state_from_fingerpos(self, fingerpos: list[int] | tuple[int, ...]) -> None:
        fps = self.config.hand_fingers_per_side
        left_keys = hand_side_keys("l", fps)
        right_keys = hand_side_keys("r", fps)
        n = len(fingerpos)

        if n >= 2 * fps:
            for i, key in enumerate(left_keys):
                self._last_hand_positions[key] = float(fingerpos[i])
            for i, key in enumerate(right_keys):
                self._last_hand_positions[key] = float(fingerpos[fps + i])
        elif n >= fps:
            for i, key in enumerate(left_keys):
                self._last_hand_positions[key] = float(fingerpos[i])
        else:
            for i, key in enumerate(self._hand_obs_keys):
                if i < n:
                    self._last_hand_positions[key] = float(fingerpos[i])

    def _init_ros_node(self) -> None:
        if not self._needs_ros_node:
            logger.warning("%s: no ROS topics configured — state/commands/cameras disabled.", self)
            return
        if not rclpy_available():
            raise ImportError("rclpy is required for crp_humanoid ROS I/O but is not installed.")

        node_name = self.config.ros_node_name.strip() or default_ros_node_name("crp_humanoid")
        if not self._ros.is_alive:
            self._ros.start(node_name)
        node = self._ros.node
        ns = self.config.ros_namespace

        if not (self._ros_feedback_enabled or self._ros_command_enabled):
            logger.info("%s ROS node ready (camera subscriptions only).", self)
            return

        MultiMotorState, JointSkillfulHandState, MotorCommand, SkillfulHandCommand = import_ti5_messages()
        self._MotorCommand = MotorCommand
        self._SkillfulHandCommand = SkillfulHandCommand

        if self._ros_feedback_enabled and self.config.multi_motor_state_topic.strip():
            topic = resolve_ros_topic(self.config.multi_motor_state_topic, ns)

            def _multi_motor_cb(msg: MultiMotorState) -> None:
                with self._state_lock:
                    for state in msg.states:
                        stem = motor_stem_for_id(int(state.motor_id), self.config.ros_joint_names)
                        key = f"{stem}.pos"
                        if key in self._last_motor_positions:
                            self._last_motor_positions[key] = float(state.position)

            node.create_subscription(MultiMotorState, topic, _multi_motor_cb, 10)
            logger.info("%s subscribing to MultiMotorState on %s", self, topic)

        if self._ros_feedback_enabled and self.config.joint_skillful_hand_state_topic.strip():
            topic = resolve_ros_topic(self.config.joint_skillful_hand_state_topic, ns)

            def _hand_state_cb(msg: JointSkillfulHandState) -> None:
                with self._state_lock:
                    self._update_hand_state_from_fingerpos(list(msg.fingerpos))

            node.create_subscription(JointSkillfulHandState, topic, _hand_state_cb, 10)
            logger.info("%s subscribing to JointSkillfulHandState on %s", self, topic)

        if self._ros_command_enabled and self.config.motor_command_topic.strip():
            topic = resolve_ros_topic(self.config.motor_command_topic, ns)
            self._motor_cmd_pub = node.create_publisher(MotorCommand, topic, 10)
            logger.info("%s publishing MotorCommand on %s", self, topic)

        if self._ros_command_enabled and self.config.skillful_hand_command_topic.strip():
            topic = resolve_ros_topic(self.config.skillful_hand_command_topic, ns)
            self._hand_cmd_pub = node.create_publisher(SkillfulHandCommand, topic, 10)
            logger.info("%s publishing SkillfulHandCommand on %s", self, topic)

    def _shutdown_ros_node(self) -> None:
        self._motor_cmd_pub = None
        self._hand_cmd_pub = None
        self._MotorCommand = None
        self._SkillfulHandCommand = None
        self._ros.shutdown()

    def _publish_motor_command(self, action: dict[str, float], fallback: dict[str, float]) -> dict[str, float]:
        if self._motor_cmd_pub is None or self._MotorCommand is None:
            return {}

        pairs = motor_targets_from_action(
            action, self._motor_stems, self.config.ros_joint_names, fallback
        )
        if not pairs:
            return {}

        msg = self._MotorCommand()
        msg.control_mode = int(self.config.motor_control_mode)
        msg.motor_ids = [motor_id for motor_id, _ in pairs]
        msg.targets = [target for _, target in pairs]
        self._motor_cmd_pub.publish(msg)

        sent: dict[str, float] = {}
        for motor_id, target in pairs:
            stem = motor_stem_for_id(motor_id, self.config.ros_joint_names)
            sent[f"{stem}.pos"] = float(target)
        return sent

    def _publish_hand_command(
        self,
        action: dict[str, float],
        fallback: dict[str, float],
        side: str,
        hand_id: int,
    ) -> dict[str, float]:
        if self._hand_cmd_pub is None or self._SkillfulHandCommand is None:
            return {}
        if not hand_side_present_in_action(action, side, self.config.hand_fingers_per_side):
            return {}

        targets = hand_targets_from_action(
            action, side, self.config.hand_fingers_per_side, fallback
        )
        msg = self._SkillfulHandCommand()
        msg.control_skillful_hand_id = int(hand_id)
        msg.multfingercw = int(self.config.skillful_hand_multfingercw)
        msg.fingerpara1 = [float(v) for v in targets]
        msg.fingerpara2 = []
        self._hand_cmd_pub.publish(msg)

        keys = hand_side_keys(side, self.config.hand_fingers_per_side)
        return {key: float(targets[i]) for i, key in enumerate(keys)}

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        if self._needs_ros_node:
            self._init_ros_node()
            if self._ros.is_alive:
                for cam in self._ros_image_cameras:
                    cam.attach_ros_node(self._ros.node)
        for cam in self.cameras.values():
            cam.connect()
        self._connected = True
        logger.info("%s connected.", self)

    def disconnect(self) -> None:
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()
        self._shutdown_ros_node()
        self._connected = False
        logger.info("%s disconnected.", self)

    def get_observation(self, include_images: bool = True) -> RobotObservation:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        with self._state_lock:
            obs_dict: dict[str, Any] = {
                **self._last_motor_positions,
                **self._last_hand_positions,
            }

        if not self._ros_feedback_enabled and not self._warned_stub_state:
            self._warned_stub_state = True
            logger.warning(
                "%s: no ROS feedback topics — joint/hand observations stay at zero.", self
            )

        if include_images:
            n_retries = int(self.config.camera_async_read_retries)
            for cam_key, cam in self.cameras.items():
                for attempt in range(n_retries + 1):
                    try:
                        obs_dict[cam_key] = cam.read()
                        break
                    except RuntimeError:
                        if attempt >= n_retries:
                            raise
                        time.sleep(0.005)

        return obs_dict

    def send_action(self, action: RobotAction) -> RobotAction:
        """Publish ``MotorCommand`` / ``SkillfulHandCommand`` from a flat action dict."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if not self._ros_command_enabled:
            logger.debug("%s send_action ignored (no command topics configured)", self)
            return action

        fallback = self._state_fallback()
        sent: dict[str, float] = {}

        sent.update(self._publish_motor_command(action, fallback))
        sent.update(
            self._publish_hand_command(action, fallback, "l", SKILLFUL_HAND_L)
        )
        sent.update(
            self._publish_hand_command(action, fallback, "r", SKILLFUL_HAND_R)
        )

        for key, value in action.items():
            if key not in sent and isinstance(value, (int, float)):
                sent[key] = float(value)

        return sent

    def joint_positions_snapshot(self) -> dict[str, float]:
        """Flat state snapshot for dataset action when ``action_source=robot``."""
        return self._state_fallback()
