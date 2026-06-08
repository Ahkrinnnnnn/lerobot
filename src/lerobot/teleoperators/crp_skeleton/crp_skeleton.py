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

"""CRP skeleton teleoperator: pass-through labels from robot command ROS topics.

Subscribes to the **commands sent to the robot** (post exoskeleton mapping / smoothing):
  - ``/motor_command`` — body + arms (``ti5_interfaces/msg/MotorCommand``)
  - ``/skillfulHand_command`` — dexterous hands L/R (``ti5_interfaces/msg/SkillfulHandCommand``)

No LeRobot processor steps — values are recorded verbatim for ``lerobot-record-humanoid``.
"""

from __future__ import annotations

import logging
import threading
from functools import cached_property
from typing import Any

from lerobot.robots.crp_humanoid._ti5_ros import (
    SKILLFUL_HAND_L,
    SKILLFUL_HAND_R,
    RosSpinSession,
    default_dataset_action_keys,
    default_ros_node_name,
    hand_side_keys,
    import_ti5_messages,
    motor_stem_for_id,
    resolve_ros_topic,
    rclpy_available,
)
from lerobot.types import RobotAction
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_crp_skeleton import CRPSkeletonConfig

logger = logging.getLogger(__name__)


class CRPSkeleton(Teleoperator):
    config_class = CRPSkeletonConfig
    name = "crp_skeleton"

    def __init__(self, config: CRPSkeletonConfig):
        super().__init__(config)
        self.config = config
        self._all_action_keys = config.ros_action_keys or default_dataset_action_keys(
            config.num_body_motors, config.hand_fingers_per_side
        )
        self._action_keys = config.ros_action_keys or config.stub_action_keys
        self._action_lock = threading.Lock()
        self._last_action: dict[str, float] = dict.fromkeys(self._all_action_keys, 0.0)

        self._ros = RosSpinSession()
        self._ros_motor_enabled = bool(config.motor_command_topic.strip())
        self._ros_hand_enabled = bool(config.skillful_hand_command_topic.strip())
        self._ros_configured = self._ros_motor_enabled or self._ros_hand_enabled
        self._warned_stub_action = False
        self._connected = False

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {key: float for key in self._action_keys}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        if not self._connected:
            return False
        if self._ros_configured and not self._ros.is_alive:
            return False
        return True

    def _hand_side_for_command(self, hand_id: int) -> str | None:
        if hand_id == SKILLFUL_HAND_R:
            return "r"
        if hand_id == SKILLFUL_HAND_L:
            return "l"
        return None

    def _update_hand_command(self, hand_id: int, fingerpara1: list[float] | tuple[float, ...]) -> None:
        side = self._hand_side_for_command(int(hand_id))
        if side is None:
            return
        keys = hand_side_keys(side, self.config.hand_fingers_per_side)
        for i, key in enumerate(keys):
            if i < len(fingerpara1) and key in self._last_action:
                self._last_action[key] = float(fingerpara1[i])

    def _init_ros_node(self) -> None:
        if not self._ros_configured:
            logger.warning("%s: no ROS command topics configured — actions will stay at zero.", self)
            return
        if not rclpy_available():
            raise ImportError("rclpy is required for crp_skeleton ROS commands but is not installed.")

        _, _, MotorCommand, SkillfulHandCommand = import_ti5_messages()
        node_name = self.config.ros_node_name.strip() or default_ros_node_name("crp_skeleton_record")
        node = self._ros.start(node_name)
        ns = self.config.ros_namespace

        if self._ros_motor_enabled:
            topic = resolve_ros_topic(self.config.motor_command_topic, ns)

            def _motor_command_cb(msg: MotorCommand) -> None:
                with self._action_lock:
                    for motor_id, target in zip(msg.motor_ids, msg.targets):
                        stem = motor_stem_for_id(int(motor_id), self.config.ros_joint_names)
                        key = f"{stem}.pos"
                        if key in self._last_action:
                            self._last_action[key] = float(target)

            node.create_subscription(MotorCommand, topic, _motor_command_cb, 10)
            logger.info("%s subscribing to MotorCommand on %s", self, topic)

        if self._ros_hand_enabled:
            topic = resolve_ros_topic(self.config.skillful_hand_command_topic, ns)

            def _hand_command_cb(msg: SkillfulHandCommand) -> None:
                with self._action_lock:
                    self._update_hand_command(msg.control_skillful_hand_id, list(msg.fingerpara1))

            node.create_subscription(SkillfulHandCommand, topic, _hand_command_cb, 10)
            logger.info("%s subscribing to SkillfulHandCommand on %s", self, topic)

    def _shutdown_ros_node(self) -> None:
        self._ros.shutdown()

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        if self._ros_configured:
            self._init_ros_node()
        self._connected = True
        logger.info("%s connected (ROS pass-through command labels).", self)

    def disconnect(self) -> None:
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self._shutdown_ros_node()
        self._connected = False
        logger.info("%s disconnected.", self)

    def get_action(self) -> RobotAction:
        """Latest robot command targets from ROS (no unit / frame conversion here)."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        with self._action_lock:
            out = {k: float(self._last_action.get(k, 0.0)) for k in self._action_keys}

        if not self._ros_configured and not self._warned_stub_action:
            self._warned_stub_action = True
            logger.warning(
                "%s: returning stub actions (clear motor_command_topic / "
                "skillful_hand_command_topic for intentional stub mode).",
                self,
            )

        return out

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def calibrate(self) -> None:
        pass

    @property
    def is_calibrated(self) -> bool:
        return True
