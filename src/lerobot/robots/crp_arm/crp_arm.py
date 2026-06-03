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
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_crp_arm import CRPArmConfig


# import the CrpRobotPy
from CrpRobotPy import CrpRobotPy, RobotMode


logger = logging.getLogger(__name__)


def _six_joint_values_from_sdk_dict(crp_raw: dict[str, Any]) -> list[float]:
    """Map ``read_joints()`` dict to ``[j1..j6]`` (same key fallbacks as ``crp_record_omy``)."""
    out: list[float] = []
    for i in range(1, 7):
        motor = f"j{i}"
        candidates = (motor, motor.upper(), motor.lower(), f"joint{i}", f"Joint{i}")
        v = 0.0
        for k in candidates:
            if k in crp_raw:
                v = float(crp_raw[k])
                break
        out.append(v)
    return out


class CRPArm(Robot):

    config_class = CRPArmConfig
    name = "crp_arm"

    def __init__(self, config: CRPArmConfig):
        super().__init__(config)
        self.config = config
        
        self.crp_arm_robot = CrpRobotPy()

        self.crp_joints = {    
        "j1": float,
        "j2": float, 
        "j3": float,
        "j4": float,
        "j5": float,
        "j6": float,
        }

        self.cameras = make_cameras_from_configs(config.cameras)
        self._gj_traj = None

    # @property
    # def _motors_ft(self) -> dict[str, type]:
    #     return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{joint}.pos": joint_type for joint, joint_type in self.crp_joints.items()}
    
    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    
    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        # Joints + cameras only. ``gripper.pos`` is intentionally omitted here so
        # ``observation.state`` stays 6-D like ``crp_record_omy`` (gripper lives in ``action`` only).
        # ``get_observation`` still adds ``gripper.pos`` to the raw dict when ``use_gripper_feature``
        # for deploy hold-pose and logging.
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        out = dict(self._motors_ft)
        if self.config.use_gripper_feature:
            out["gripper.pos"] = float
        return out


    @property
    def is_connected(self) -> bool:
        return self.crp_arm_robot.is_connected() and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        # self.bus.connect() #连接电机
        # if not self.is_calibrated and calibrate:
        #     logger.info(
        #         "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
        #     )
        #     self.calibrate()

        self.crp_arm_robot.connect(self.config.port)
        self.crp_arm_robot.servo_power_on()
        # self.crp_arm_robot.switch_work_mode(RobotMode.Manual)
        self.crp_arm_robot.switch_work_mode(RobotMode.Auto)

        from lerobot.tools import TrajectoryProcessor

        self._gj_traj = TrajectoryProcessor(
            max_points=1,
            max_joints=self.config.gj_trajectory_group_size,
        )
        init_j = _six_joint_values_from_sdk_dict(self.crp_arm_robot.read_joints())
        init_matrix = self._gj_traj.init_matrix(init_j, self.config.gj_trajectory_group_size)
        self.send_GJs(self.config.gj_register_primary, init_matrix)
        self.send_GJs(self.config.gj_register_secondary_init, init_matrix)

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        sr = getattr(self.config, "speed_ratio_on_connect", None)
        if sr is not None:
            self.set_speed_ratio(int(sr))
        logger.info(f"{self} connected.")


    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        # self.bus.disconnect(self.config.disable_torque_on_disconnect)  #断连电机

        self.crp_arm_robot.servo_power_off()
        self.crp_arm_robot.disconnect()

        for cam in self.cameras.values():
            cam.disconnect()

        self._gj_traj = None
        logger.info(f"{self} disconnected.")




    # 校准
    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass


    def configure(self) -> None:
        pass



    def get_observation(self, include_images: bool = True) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()

        # obs_dict = self.bus.sync_read("Present_Position")
        # obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}

        crp_joints_dict = self.crp_arm_robot.read_joints()

        obs_dict = {f"{motor}.pos": val for motor, val in crp_joints_dict.items()}

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        if include_images:
            # Synchronous read on the caller thread (same as lerobot_camera_stream). Background
            # async_read + long timeouts starved under GIL when recording threads compete with the
            # camera read thread, yielding multi-second gaps between frames.
            n_retries = int(getattr(self.config, "camera_async_read_retries", 2))
            for cam_key, cam in self.cameras.items():
                start = time.perf_counter()
                for attempt in range(n_retries + 1):
                    try:
                        obs_dict[cam_key] = cam.read()
                        break
                    except RuntimeError:
                        if attempt >= n_retries:
                            raise
                        logger.warning(
                            "%s camera %r read failed (%s/%s attempts); retrying after short delay",
                            self,
                            cam_key,
                            attempt + 1,
                            n_retries + 1,
                        )
                        time.sleep(0.005)
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        if self.config.use_gripper_feature:
            obs_dict["gripper.pos"] = float(self.get_GOT(0))

        return obs_dict


    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command arm joints via ``TrajectoryProcessor`` + ``set_GJs`` (same pattern as ``crp_record_omy``).

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            the action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        joint_keys = [f"j{i}.pos" for i in range(1, 7)]
        if all(k in action for k in joint_keys):
            joint_vec = [float(action[k]) for k in joint_keys]
        else:
            crp_raw = self.crp_arm_robot.read_joints()
            joint_vec = _six_joint_values_from_sdk_dict(crp_raw)
            for i, k in enumerate(joint_keys):
                if k in action:
                    joint_vec[i] = float(action[k])

        sent: dict[str, Any] = {f"j{i}.pos": joint_vec[i - 1] for i in range(1, 7)}

        if self._gj_traj is None:
            raise DeviceNotConnectedError(
                f"{self}: GJ trajectory buffer missing; connect() must finish successfully."
            )
        self._gj_traj.write_joint(joint_vec)
        self.send_GJs(self.config.gj_register_primary, self._gj_traj.read_joints())
        if "gripper.pos" in action and self.config.use_gripper_feature:
            got0 = int(max(0, min(1000, round(float(action["gripper.pos"])))))
            self.set_GOT(0, got0)
            sent["gripper.pos"] = float(got0)

        return sent


    def send_endpose(self, endpose: list[float]):
        if len(endpose) != 6:
            print("crp_arm: [ERROR] endpose必须包含6个值")
            return

        try:
            _ = self.crp_arm_robot.movel_user(endpose)
        except Exception as e:
            print("crp_arm: [ERROR] movel_user机械臂运动失败", e)




    def send_GPs(self, start_index: int, GPs: list[float]):
        self.crp_arm_robot.set_GPs(start_index, GPs)
        return

    def send_GJs(self, start_index: int, GJs: list[float]):
        self.crp_arm_robot.set_GJs(start_index, GJs)
        return

    def set_GI(self, index: int, value: int) -> bool:
        self.crp_arm_robot.set_GI(index, value)
        return

    def get_GI(self, index: int) -> int:
        return self.crp_arm_robot.get_GI(index)

    def set_GOT(self, index: int, value: int) -> bool:
        """Set gripper output target register (e.g. GOT0: 0=closed .. 1000=open, per controller)."""
        return bool(self.crp_arm_robot.set_GOT(index, value))

    def get_GOT(self, index: int) -> int:
        """Read gripper output target register (int64 in SDK; typically 0..1000 for GOT0)."""
        return int(self.crp_arm_robot.get_GOT(index))


    def set_speed_ratio(self, ratio: int):
        self.crp_arm_robot.set_speed_ratio(ratio)
        return
    
    def get_speed_ratio(self) -> int:
        return self.crp_arm_robot.get_speed_ratio()
    
    def get_current_endpose(self) -> list[float]:
        """
        获取当前末端位置姿态
        返回: [x, y, z, roll, pitch, yaw]
        """
        x, y, z, roll, pitch, yaw = self.crp_arm_robot.read_end_pose_user()
        return [x, y, z, roll, pitch, yaw]
    