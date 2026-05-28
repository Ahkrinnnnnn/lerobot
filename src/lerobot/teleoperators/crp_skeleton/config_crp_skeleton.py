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

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("crp_skeleton")
@dataclass
class CRPSkeletonConfig(TeleoperatorConfig):
    """Configuration for CRP skeleton labels during passive humanoid recording.

    The external ROS2 skeleton node publishes **already processed** actions (scaling,
    frame, naming). LeRobot records ``get_action()`` verbatim — no processor pipeline.
    """

    # Unused for ROS-only hardware; kept for CLI compatibility.
    port: str = "ros"

    # TODO(ros): Topic from the CRP skeleton processing node (message type TBD).
    ros_action_topic: str = ""
    # Dataset / ``get_action()`` keys exactly as published (e.g. ``ee.x``, ``j1.pos``).
    ros_action_keys: tuple[str, ...] = ()

    # TODO(ros): Optional ROS node name; empty → auto ``crp_skeleton_record_<pid>``.
    ros_node_name: str = ""
    # TODO(ros): Namespace prefix for topics (e.g. ``/skeleton``).
    ros_namespace: str = ""

    # Fallback stub keys when ``ros_action_keys`` is empty (development only).
    stub_action_keys: tuple[str, ...] = field(
        default_factory=lambda: tuple(f"j{i}.pos" for i in range(1, 7))
    )
