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

"""CRP skeleton teleoperator: pass-through labels from an external ROS2 processing node.

The skeleton stack publishes **final** action vectors (units, frame, key names). This class
only buffers the latest message for ``lerobot-record-humanoid`` — no LeRobot processor steps.
"""

from __future__ import annotations

import logging
import threading
from functools import cached_property
from typing import Any

from lerobot.types import RobotAction
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_crp_skeleton import CRPSkeletonConfig

logger = logging.getLogger(__name__)

try:
    import rclpy

    _RCLPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    rclpy = None  # type: ignore[assignment]
    _RCLPY_AVAILABLE = False


class CRPSkeleton(Teleoperator):
    config_class = CRPSkeletonConfig
    name = "crp_skeleton"

    def __init__(self, config: CRPSkeletonConfig):
        super().__init__(config)
        self.config = config
        self._action_keys = config.ros_action_keys or config.stub_action_keys
        self._action_lock = threading.Lock()
        self._last_action: dict[str, float] = dict.fromkeys(self._action_keys, 0.0)

        self._ros_node = None
        self._ros_thread: threading.Thread | None = None
        self._ros_running = False
        self._ros_configured = bool(config.ros_action_topic.strip())
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
        # TODO(ros): also require ROS spin thread when ros_action_topic is set.
        return self._connected

    def _init_ros_node(self) -> None:
        """TODO(ros): Subscribe on ``ros_action_topic`` and copy fields into ``_last_action``."""
        if not self._ros_configured:
            logger.warning(
                "%s: ros_action_topic is empty — actions will stay at stub zeros until configured.",
                self,
            )
            return
        if not _RCLPY_AVAILABLE:
            raise ImportError("rclpy is required for crp_skeleton ros_action_topic but is not installed.")
        # TODO(ros): rclpy.init(); create node; subscribe; in callback assign self._last_action[key]=...
        raise NotImplementedError(
            "CRPSkeleton ROS subscription is not implemented yet. "
            "Implement _init_ros_node() to mirror the skeleton node's processed action message."
        )

    def _shutdown_ros_node(self) -> None:
        """TODO(ros): Stop spin thread and destroy node."""
        self._ros_running = False
        if self._ros_thread is not None:
            self._ros_thread.join(timeout=2.0)
            self._ros_thread = None
        self._ros_node = None

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        if self._ros_configured:
            self._init_ros_node()
        self._connected = True
        logger.info("%s connected (ROS pass-through action labels).", self)

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self._shutdown_ros_node()
        self._connected = False
        logger.info("%s disconnected.", self)

    def get_action(self) -> RobotAction:
        """Latest processed action from the skeleton ROS node (no unit / frame conversion here)."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        with self._action_lock:
            out = {k: float(self._last_action.get(k, 0.0)) for k in self._action_keys}

        if not self._ros_configured and not self._warned_stub_action:
            self._warned_stub_action = True
            logger.warning("%s: returning stub actions (configure ros_action_topic).", self)

        return out

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def calibrate(self) -> None:
        pass

    @property
    def is_calibrated(self) -> bool:
        return True
