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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("OMY_L100")
@dataclass
class OMYL100Config(TeleoperatorConfig):
    # Port to connect to the arm
    port: str

    use_degrees: bool = True
    # With `ros2 launch open_manipulator_bringup omy_l100_leader_ai.launch.py`, joints are under `/leader`.
    joint_states_topic: str = "/leader/joint_states"
    # URDF joint names in order j1..j6 (must match `sensor_msgs/JointState` names).
    ros_joint_names: tuple[str, ...] = (
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    )
    # Gripper (or tool) joint name on the same `sensor_msgs/JointState` message (e.g. OMY leader).
    ros_gripper_joint_name: str = "rh_r1_joint"
    # If True, apply `math.degrees` to the gripper joint position like the arm joints when `use_degrees` is True.
    # If False, pass the raw `JointState.position` value (typical for prismatic / arbitrary units).
    gripper_apply_use_degrees: bool = False
    # Optional FK end-effector from ROS (e.g. namespaced ``leader/end_effector_pose``).
    # Empty string: ``OMYL100.connect()`` uses module default ``EE_STATES_TOPIC`` (see ``OMY_L100.py``).
    ros_end_effector_pose_topic: str = ""
    # Message type on that topic: ``pose`` (``geometry_msgs/Pose``, common on ``/end_effector_pose``) or
    # ``pose_stamped`` for ``PoseStamped`` FK streams.
    ros_end_effector_pose_msg_type: str = "pose"
    # ``reliable``: KEEP_LAST 10 RELIABLE (typical ``self_collision_node`` / leader EE). ``sensor``: BEST_EFFORT.
    ros_ee_pose_qos: str = "reliable"
    # Multiply EE position (``Pose`` / ``PoseStamped.pose``) — typical SI: meters → mm for CRP-style inputs.
    ros_ee_pose_position_scale: float = 1.0


