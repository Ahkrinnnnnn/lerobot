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

"""Shared rclpy spin helpers for LeRobot ROS integrations."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

try:
    import rclpy as _rclpy

    _RCLPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _rclpy = None  # type: ignore[assignment]
    _RCLPY_AVAILABLE = False


def rclpy_available() -> bool:
    return _RCLPY_AVAILABLE


def resolve_ros_topic(topic: str, namespace: str = "") -> str:
    topic = topic.strip()
    if not topic:
        return ""
    if not topic.startswith("/"):
        topic = f"/{topic.lstrip('/')}"
    ns = namespace.strip().rstrip("/")
    if ns and not ns.startswith("/"):
        ns = f"/{ns}"
    if ns and not topic.startswith(f"{ns}/"):
        return f"{ns}{topic}"
    return topic


def init_rclpy() -> None:
    if not _RCLPY_AVAILABLE:
        raise ImportError("rclpy is required but is not installed.")
    try:
        _rclpy.init()
    except Exception:
        pass


def default_ros_node_name(prefix: str) -> str:
    return f"{prefix}_{os.getpid()}"


class RosSpinSession:
    """Background ``rclpy.spin_once`` loop for a single node."""

    def __init__(self) -> None:
        self._node: Any | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    @property
    def node(self) -> Any:
        if self._node is None:
            raise RuntimeError("ROS node is not initialized.")
        return self._node

    @property
    def is_alive(self) -> bool:
        return self._node is not None and self._thread is not None and self._thread.is_alive()

    def start(self, node_name: str) -> Any:
        init_rclpy()
        self._node = _rclpy.create_node(node_name)
        self._running = True

        def _spin_loop() -> None:
            try:
                while self._running and self._node is not None:
                    _rclpy.spin_once(self._node, timeout_sec=0.1)
            except Exception:
                logger.exception("Exception in ROS spin loop for %s", node_name)

        self._thread = threading.Thread(target=_spin_loop, name=f"{node_name}_spin", daemon=True)
        self._thread.start()
        return self._node

    def shutdown(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None
