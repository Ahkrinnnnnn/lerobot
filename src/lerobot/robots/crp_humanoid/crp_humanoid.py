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

"""CRP humanoid follower: observation + cameras for passive dataset recording.

Robot motion is assumed to be commanded by external ROS2 nodes. This class does not
send motion commands during ``lerobot-record-humanoid`` (``send_action`` is a no-op).
"""

from __future__ import annotations

import logging
import threading
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_crp_humanoid import CRPHumanoidConfig

logger = logging.getLogger(__name__)

try:
    import rclpy
    from sensor_msgs.msg import JointState

    _RCLPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    rclpy = None  # type: ignore[assignment]
    JointState = None  # type: ignore[assignment,misc]
    _RCLPY_AVAILABLE = False


def _default_joint_names() -> tuple[str, ...]:
    return tuple(f"j{i}" for i in range(1, 7))


class CRPHumanoid(Robot):
    config_class = CRPHumanoidConfig
    name = "crp_humanoid"

    def __init__(self, config: CRPHumanoidConfig):
        super().__init__(config)
        self.config = config
        self._joint_names = config.ros_joint_names or _default_joint_names()
        self._joints_ft = {f"{n}.pos": float for n in self._joint_names}
        self.cameras = make_cameras_from_configs(config.cameras)

        self._ros_node = None
        self._ros_thread: threading.Thread | None = None
        self._ros_running = False
        self._joint_lock = threading.Lock()
        self._last_joint_positions: dict[str, float] = dict.fromkeys(self._joint_names, 0.0)
        self._ros_configured = bool(config.joint_states_topic.strip())
        self._warned_stub_joints = False
        self._connected = False

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

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
        if self.cameras:
            return all(cam.is_connected for cam in self.cameras.values())
        # TODO(ros): require ROS spin thread alive when ``joint_states_topic`` is set.
        return True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _init_ros_node(self) -> None:
        """TODO(ros): Initialize ``rclpy``, create node, subscribe to ``joint_states_topic``."""
        if not self._ros_configured:
            logger.warning(
                "%s: joint_states_topic is empty — joint observations will stay at 0 until configured.",
                self,
            )
            return
        if not _RCLPY_AVAILABLE:
            raise ImportError(
                "rclpy is required for crp_humanoid joint_states_topic but is not installed."
            )
        # TODO(ros): rclpy.init(); create node (self.config.ros_node_name or auto);
        # TODO(ros): apply self.config.ros_namespace to topic remapping;
        # TODO(ros): subscribe JointState on joint_states_topic, update _last_joint_positions in callback;
        # TODO(ros): start self._ros_spin_loop in self._ros_thread.
        raise NotImplementedError(
            "CRPHumanoid ROS subscription is not implemented yet. "
            "Fill in _init_ros_node() or run without joint_states_topic for camera-only tests."
        )

    def _shutdown_ros_node(self) -> None:
        """TODO(ros): Stop spin thread, destroy node, shutdown rclpy if this process owns it."""
        self._ros_running = False
        if self._ros_thread is not None:
            self._ros_thread.join(timeout=2.0)
            self._ros_thread = None
        self._ros_node = None

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        if self._ros_configured:
            self._init_ros_node()
        for cam in self.cameras.values():
            cam.connect()
        self._connected = True
        logger.info("%s connected (passive recording — no motion commands from LeRobot).", self)

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self._shutdown_ros_node()
        for cam in self.cameras.values():
            cam.disconnect()
        self._connected = False
        logger.info("%s disconnected.", self)

    def get_observation(self, include_images: bool = True) -> RobotObservation:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        with self._joint_lock:
            obs_dict: dict[str, Any] = {
                f"{name}.pos": float(self._last_joint_positions.get(name, 0.0))
                for name in self._joint_names
            }

        if not self._ros_configured and not self._warned_stub_joints:
            self._warned_stub_joints = True
            logger.warning("%s: returning stub joint positions (configure joint_states_topic).", self)

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
        """No-op: external ROS2 stack controls the humanoid during recording."""
        logger.debug("%s send_action ignored (external control): keys=%s", self, list(action.keys()))
        return action

    def joint_positions_snapshot(self) -> dict[str, float]:
        """Flat ``{name}.pos``: value`` for dataset action when ``action_source=robot``."""
        with self._joint_lock:
            return {f"{n}.pos": float(self._last_joint_positions.get(n, 0.0)) for n in self._joint_names}
