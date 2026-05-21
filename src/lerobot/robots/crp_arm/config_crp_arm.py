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

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("crp_arm")
@dataclass
class CRPArmConfig(RobotConfig):
    # Port to connect to the arm
    port: str

    # ip: str = "192.168.0.100"
    
    # disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    # cameras

    #############################等待配置###############################
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # Reserved for compatibility; ``CRPArm.get_observation`` uses synchronous ``cam.read()`` (not
    # ``async_read``) so this timeout is unused.
    camera_async_read_timeout_ms: float = 1500.0
    # Retries after ``RuntimeError`` from synchronous ``read()`` (transient USB / driver glitches).
    camera_async_read_retries: int = 2

    # Set to `True` for backward compatibility with previous policies/datasets
    use_degrees: bool = False

    # Adds ``gripper.pos`` to **action** (and ``get_observation`` raw dict), not to ``observation.state``
    # (stays 6 joints) so SmolVLA / datasets match ``crp_record_omy`` (gripper in action only).
    use_gripper_feature: bool = False

    # If set, ``connect()`` applies ``set_speed_ratio`` after the arm is ready (e.g. 20 like ``crp_record_omy``).
    speed_ratio_on_connect: int | None = None

    # ``send_GJs`` register indices (must match controller / ``crp_record_omy``).
    gj_register_primary: int = 10
    gj_register_secondary_init: int = 20
    # ``init_matrix`` row count and ``TrajectoryProcessor.max_joints`` (``crp_record_omy`` uses 5).
    gj_trajectory_group_size: int = 5
