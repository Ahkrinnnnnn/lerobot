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
import threading
import time
from collections.abc import Sequence
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_crp_arm import CRPArmConfig
from ._sdk import import_crp_robot_py


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

        CrpRobotPy, _ = import_crp_robot_py()
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
        # Cache last GI mode written for GP/GJ switching (S55 / motion_mode_gi_index).
        self._motion_mode_gi_cached: int | None = None
        # Cache last MOVE-enable GI (S56 / motion_enable_gi_index).
        self._motion_enable_gi_cached: int | None = None
        self._gj_send_log_remaining: int = 5

        # In-process SDK serialization (threads in the same process).
        # A fork child that inherits this client must not race the parent on Thrift.
        self._motion_cmd_lock = threading.RLock()
        # When set (dict), get_observation serves joints/EE from cache (no SDK).
        # Recording: parent fills cache from fork shared-memory snapshots.
        self._proprio_cache: dict[str, float] | None = None
        self._proprio_cache_lock = threading.Lock()
        self._last_ee_cache: dict[str, float] | None = None

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

    
    @property
    def _ee_ft(self) -> dict[str, type]:
        return {
            "ee.x": float,
            "ee.y": float,
            "ee.z": float,
            "ee.roll": float,
            "ee.pitch": float,
            "ee.yaw": float,
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        # Cameras always. Joints / EE / gripper gated by config flags.
        features: dict[str, type | tuple] = {**self._cameras_ft}
        if self.config.enable_joint:
            features.update(self._motors_ft)
        if self.config.enable_ee:
            features.update(self._ee_ft)
        if self.config.use_gripper_feature:
            features["gripper.pos"] = float
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        # Feature keys only; recording fills obs from prev action, action EE from command.
        out: dict[str, type] = {}
        if self.config.enable_joint:
            out.update(self._motors_ft)
        if self.config.enable_ee:
            out.update(self._ee_ft)
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
        _, RobotMode = import_crp_robot_py()
        # self.crp_arm_robot.switch_work_mode(RobotMode.Manual)
        self.crp_arm_robot.switch_work_mode(RobotMode.Auto)
        # MOVE gate off immediately after Auto — pendant must not ``mv`` on leftover targets.
        self.set_motion_enabled(False)

        from lerobot.tools import TrajectoryProcessor

        self._gj_traj = TrajectoryProcessor(
            max_points=1,
            max_joints=self.config.gj_trajectory_group_size,
        )
        init_j = _six_joint_values_from_sdk_dict(self.crp_arm_robot.read_joints())
        # Cache init joints so callers after send_GJs can avoid read_joints
        # (the controller rejects ``getCurrentJoint`` while a GJ trajectory is
        # executing). See ``get_last_cached_joints``.
        self._last_joints_cache: dict[str, float] = {
            f"j{i}.pos": float(init_j[i - 1]) for i in range(1, 7)
        }
        # Keep TrajectoryProcessor buffer in sync (init_matrix alone does not write joints).
        self._gj_traj.write_joint(init_j)
        logger.info(
            "%s connect: MOVE gate off (GI56); j cache=[%.3f %.3f %.3f %.3f %.3f %.3f]",
            self,
            *init_j,
        )

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        sr = getattr(self.config, "speed_ratio_on_connect", None)
        if sr is not None:
            self.set_speed_ratio(int(sr))
        self._gj_send_log_remaining = 5
        logger.info(f"{self} connected.")


    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        # self.bus.disconnect(self.config.disable_torque_on_disconnect)  #断连电机

        try:
            self.set_motion_enabled(False)
        except Exception:
            logger.warning("%s disconnect: failed to clear motion enable GI", self, exc_info=True)

        self.crp_arm_robot.servo_power_off()
        self.crp_arm_robot.disconnect()

        for cam in self.cameras.values():
            cam.disconnect()

        self._gj_traj = None
        self._motion_mode_gi_cached = None
        self._motion_enable_gi_cached = None
        logger.info(f"{self} disconnected.")




    # 校准
    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass


    def configure(self) -> None:
        pass



    def read_observation_cameras(self) -> dict[str, Any]:
        """Read camera frames without touching arm proprio (safe to call outside robot I/O lock).

        Synchronous ``cam.read()`` on the caller thread (same as ``lerobot_camera_stream``).
        Background ``async_read`` + long timeouts starved under GIL when recording threads
        compete with the camera read thread, yielding multi-second gaps between frames.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        images: dict[str, Any] = {}
        n_retries = int(getattr(self.config, "camera_async_read_retries", 2))
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            for attempt in range(n_retries + 1):
                try:
                    images[cam_key] = cam.read()
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
        return images

    def _read_proprio_observation(self) -> dict[str, Any]:
        """Joint (+ optional GOT) observation.

        Cache mode (``_proprio_cache is not None``): return the latest snapshot
        published by ``set_proprio_cache_snapshot`` / ``refresh_proprio_cache``.
        Direct SDK mode: ``read_joints`` with retries under ``_motion_cmd_lock``.
        """
        with self._proprio_cache_lock:
            cached_live = self._proprio_cache
        if cached_live is not None:
            if cached_live:
                return dict(cached_live)
            stale = getattr(self, "_last_joints_cache", {})
            if stale:
                return dict(stale)

        start = time.perf_counter()
        last_err: Exception | None = None
        obs_dict: dict[str, Any] = {}
        for attempt in range(3):
            try:
                with self._motion_cmd_lock:
                    crp_joints_dict = self.crp_arm_robot.read_joints()
                    obs_dict = {f"{motor}.pos": val for motor, val in crp_joints_dict.items()}
                    if self.config.use_gripper_feature:
                        obs_dict["gripper.pos"] = float(self.crp_arm_robot.get_GOT(0))
                self._last_joints_cache = dict(obs_dict)
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                logger.warning(
                    "%s read_joints failed (%s/%s): %s",
                    self,
                    attempt + 1,
                    3,
                    exc,
                )
                time.sleep(0.05 * (attempt + 1))
        if last_err is not None:
            cached = getattr(self, "_last_joints_cache", {})
            if cached:
                logger.warning("%s using cached joints (stale) — read_joints unavailable", self)
                return dict(cached)
            raise last_err
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")
        return obs_dict

    def _read_ee_observation(self) -> dict[str, float]:
        """EE xyz+rpy when ``enable_ee``. Cache mode uses measured snapshots only."""
        with self._proprio_cache_lock:
            cache_on = self._proprio_cache is not None
            ee_cached = self._last_ee_cache
        if cache_on:
            if ee_cached:
                return dict(ee_cached)
            return {}

        try:
            with self._motion_cmd_lock:
                x, y, z, roll, pitch, yaw = self.crp_arm_robot.read_end_pose_user()
            ee = {
                "ee.x": float(x),
                "ee.y": float(y),
                "ee.z": float(z),
                "ee.roll": float(roll),
                "ee.pitch": float(pitch),
                "ee.yaw": float(yaw),
            }
            self._last_ee_cache = dict(ee)
            return ee
        except Exception as exc:
            logger.warning("%s read_end_pose_user failed: %s", self, exc)
            if ee_cached:
                return dict(ee_cached)
            return {}

    def get_observation(self, include_images: bool = True) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict: dict[str, Any] = {}
        if include_images:
            # Cameras first, then proprio so ``observation.state`` matches image capture time.
            obs_dict.update(self.read_observation_cameras())

        obs_dict.update(self._read_proprio_observation())

        if self.config.enable_ee:
            obs_dict.update(self._read_ee_observation())

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
        with self._motion_cmd_lock:
            # read_joints after a prior send_GJs/send_GPs is rejected by the
            # controller with ``getCurrentJoint failed``; fall back to cache.
            try:
                present = _six_joint_values_from_sdk_dict(self.crp_arm_robot.read_joints())
                self._last_joints_cache = {
                    f"j{i}.pos": float(present[i - 1]) for i in range(1, 7)
                }
            except Exception:
                present = [
                    float(self._last_joints_cache[f"j{i}.pos"]) for i in range(1, 7)
                ]
        if all(k in action for k in joint_keys):
            joint_vec = [float(action[k]) for k in joint_keys]
        else:
            joint_vec = list(present)
            for i, k in enumerate(joint_keys):
                if k in action:
                    joint_vec[i] = float(action[k])

        # Cap per-step joint jump (absolute IL/RL targets can slam toward dataset mid-range).
        if self.config.max_relative_target is not None:
            goal_present = {f"j{i}": (joint_vec[i - 1], present[i - 1]) for i in range(1, 7)}
            safe = ensure_safe_goal_position(goal_present, self.config.max_relative_target)
            joint_vec = [float(safe[f"j{i}"]) for i in range(1, 7)]

        sent: dict[str, Any] = {f"j{i}.pos": joint_vec[i - 1] for i in range(1, 7)}

        if self._gj_traj is None:
            raise DeviceNotConnectedError(
                f"{self}: GJ trajectory buffer missing; connect() must finish successfully."
            )
        if self._gj_send_log_remaining > 0:
            delta = [joint_vec[i] - present[i] for i in range(6)]
            logger.info(
                "%s send_GJs target j=[%.3f %.3f %.3f %.3f %.3f %.3f] "
                "present=[%.3f %.3f %.3f %.3f %.3f %.3f] delta=[%.3f %.3f %.3f %.3f %.3f %.3f]",
                self,
                *joint_vec,
                *present,
                *delta,
            )
            self._gj_send_log_remaining -= 1

        self._gj_traj.write_joint(joint_vec)
        # Policy / env intentionally commands motion — open pendant MOVE gate (GI56).
        self.set_motion_enabled(True)
        # Preload GJ while avoiding a GI flip onto stale registers when coming from GP.
        self.send_GJs(
            self.config.gj_register_primary,
            self._gj_traj.read_joints(),
            switch_to_gj_mode=True,
        )
        if "gripper.pos" in action and self.config.use_gripper_feature:
            desired = int(max(0, min(1000, round(float(action["gripper.pos"])))))
            try:
                present_got = int(self.get_GOT(0))
            except Exception:
                present_got = desired
            # Untrained / wrong-stats policies often denorm gripper to ~0 (closed) or ~1.
            # Rate-limit GOT so a single step cannot slam the jaw.
            max_grip_step = 30
            if abs(desired - present_got) > max_grip_step:
                desired = present_got + (max_grip_step if desired > present_got else -max_grip_step)
                logger.warning(
                    "%s gripper GOT clipped present=%s → cmd limited to %s (policy asked large jump)",
                    self,
                    present_got,
                    desired,
                )
            self.set_GOT(0, desired)
            sent["gripper.pos"] = float(desired)

        return sent


    def send_endpose(self, endpose: list[float]):
        if len(endpose) != 6:
            print("crp_arm: [ERROR] endpose必须包含6个值")
            return

        try:
            _ = self.crp_arm_robot.movel_user(endpose)
        except Exception as e:
            print("crp_arm: [ERROR] movel_user机械臂运动失败", e)








    def send_GPs(self, start_index: int, GPs: list[float], *, switch_to_gp_mode: bool = True):
        """Write GP registers. By default also sets GI to GP mode (``gp_mode_gi_value``).

        Pass ``switch_to_gp_mode=False`` to preload registers while the teach-pendant may still
        be in joint / ``moveabsj`` (GI=GJ). Call ``ensure_gp_mode()`` only when teleop is armed.
        """
        with self._motion_cmd_lock:
            if switch_to_gp_mode:
                self._ensure_motion_mode_gi(self.config.gp_mode_gi_value)
            self.crp_arm_robot.set_GPs(start_index, GPs)
        return

    def send_GJs(self, start_index: int, GJs: list[float], *, switch_to_gj_mode: bool = True):
        """Write GJ registers. By default also sets GI to GJ mode.

        Pass ``switch_to_gj_mode=False`` to preload registers while still in GP mode,
        then call ``ensure_gj_mode()`` — otherwise GI flips before registers update and
        the pendant snaps to stale joint targets.
        """
        with self._motion_cmd_lock:
            if switch_to_gj_mode:
                self._ensure_motion_mode_gi(self.config.gj_mode_gi_value)
            self.crp_arm_robot.set_GJs(start_index, GJs)
        return

    def ensure_gp_mode(self) -> None:
        """Switch teach-pendant mode flag to GP (``set_GI`` → ``gp_mode_gi_value``)."""
        with self._motion_cmd_lock:
            self._ensure_motion_mode_gi(self.config.gp_mode_gi_value)

    def ensure_gj_mode(self) -> None:
        """Switch teach-pendant mode flag to GJ / joint (``set_GI`` → ``gj_mode_gi_value``)."""
        with self._motion_cmd_lock:
            self._ensure_motion_mode_gi(self.config.gj_mode_gi_value)

    def set_motion_enabled(self, enabled: bool) -> None:
        """Teach-pendant MOVE gate (``motion_enable_gi_index``, default GI56).

        Pendant program should only ``mv`` when this register is on. Writing GP/GJ alone
        does not open the gate — call with ``True`` at arm-ready (recording
        ``omy_gp_armed`` / HIL Space armed) or from intentional ``send_action``.
        """
        index = self.config.motion_enable_gi_index
        if index is None:
            return
        value = (
            int(self.config.motion_enable_on_value)
            if enabled
            else int(self.config.motion_enable_off_value)
        )
        with self._motion_cmd_lock:
            if self._motion_enable_gi_cached == value:
                return
            self.crp_arm_robot.set_GI(int(index), value)
            self._motion_enable_gi_cached = value
        logger.info("%s set_GI(%s)=%s (motion_enable=%s)", self, index, value, enabled)

    def is_motion_enabled(self) -> bool:
        """Return True if MOVE gate cache matches the configured ON value."""
        index = self.config.motion_enable_gi_index
        if index is None:
            return True
        return self._motion_enable_gi_cached == int(self.config.motion_enable_on_value)

    def hold_current_joints_gj(
        self, joints: Sequence[float] | None = None, *, settle_timeout_s: float = 0.8
    ) -> list[float]:
        """Preload joints into GJ registers, then GI→GJ.

        Order matters: write primary+secondary GJ **before** flipping GI. Switching GI
        first makes the pendant chase stale joint targets (violent snap on Space release).

        Prefer an explicit ``joints`` snapshot taken **before** stopping the GP stream
        (cache is still warm). After GP, ``read_joints`` often fails briefly — falling
        back to a pre-intervention cache would yank the arm back and feel like a huge
        post-intervention jump.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if self._gj_traj is None:
            raise DeviceNotConnectedError(
                f"{self}: GJ trajectory buffer missing; connect() must finish successfully."
            )

        joint_list: list[float] | None = None
        source = "arg"
        if joints is not None and len(joints) >= 6:
            joint_list = [float(joints[i]) for i in range(6)]
        else:
            source = "read"
            if settle_timeout_s > 0:
                self.wait_controller_ready(timeout_s=float(settle_timeout_s))
            with self._motion_cmd_lock:
                try:
                    joint_list = _six_joint_values_from_sdk_dict(self.crp_arm_robot.read_joints())
                    self._last_joints_cache = {
                        f"j{i}.pos": float(joint_list[i - 1]) for i in range(1, 7)
                    }
                except Exception:
                    joint_list = None
            if joint_list is None:
                source = "last_joints_cache"
                cache = getattr(self, "_last_joints_cache", {}) or {}
                try:
                    joint_list = [float(cache[f"j{i}.pos"]) for i in range(1, 7)]
                except Exception as exc:
                    raise DeviceNotConnectedError(
                        f"{self}: cannot resolve joints for GJ hold ({exc})"
                    ) from exc

        assert joint_list is not None
        self._gj_traj.write_joint(joint_list)
        mat = self._gj_traj.read_joints()
        # Preload while still in GP (or whatever mode), then flip GI.
        self.send_GJs(self.config.gj_register_primary, mat, switch_to_gj_mode=False)
        self.send_GJs(self.config.gj_register_secondary_init, mat, switch_to_gj_mode=False)
        self.ensure_gj_mode()
        # HIL handoff / hold: allow pendant to track the refreshed GJ target.
        self.set_motion_enabled(True)
        # Hold gripper at current GOT so untrained policy cannot slam close on the same frame.
        if self.config.use_gripper_feature:
            try:
                got0 = int(self.get_GOT(0))
                self.set_GOT(0, got0)
            except Exception:
                logger.warning("%s hold_current_joints_gj: get/set GOT failed", self, exc_info=True)
        logger.info(
            "%s GJ hold j=[%.3f %.3f %.3f %.3f %.3f %.3f] (source=%s, preload then GI→GJ)",
            self,
            *joint_list,
            source,
        )
        return [float(x) for x in joint_list]

    def _ensure_motion_mode_gi(self, value: int) -> None:
        """Set teach-pendant mode flag (GI / ``S55``) when switching between GP and GJ.

        Must be called while holding ``_motion_cmd_lock`` (``send_GPs`` / ``send_GJs``).
        """
        index = self.config.motion_mode_gi_index
        if index is None:
            return
        if self._motion_mode_gi_cached == int(value):
            return
        self.crp_arm_robot.set_GI(int(index), int(value))
        self._motion_mode_gi_cached = int(value)
        logger.info(
            "%s set_GI(%s)=%s (GP=%s GJ=%s)",
            self,
            index,
            value,
            self.config.gp_mode_gi_value,
            self.config.gj_mode_gi_value,
        )

    def set_GI(self, index: int, value: int) -> bool:
        with self._motion_cmd_lock:
            self.crp_arm_robot.set_GI(index, value)
            # Keep cache coherent when callers write GI directly.
            if self.config.motion_mode_gi_index is not None and int(index) == int(
                self.config.motion_mode_gi_index
            ):
                self._motion_mode_gi_cached = int(value)
            if self.config.motion_enable_gi_index is not None and int(index) == int(
                self.config.motion_enable_gi_index
            ):
                self._motion_enable_gi_cached = int(value)
        return True

    def get_GI(self, index: int) -> int:
        with self._motion_cmd_lock:
            return self.crp_arm_robot.get_GI(index)

    def set_GOT(self, index: int, value: int) -> bool:
        """Set gripper output target register (e.g. GOT0: 0=closed .. 1000=open, per controller)."""
        with self._motion_cmd_lock:
            return bool(self.crp_arm_robot.set_GOT(index, value))

    def get_GOT(self, index: int) -> int:
        """Read gripper output target register (int64 in SDK; typically 0..1000 for GOT0)."""
        with self._motion_cmd_lock:
            return int(self.crp_arm_robot.get_GOT(index))

    def wait_controller_ready(
        self, timeout_s: float = 2.0, poll_interval_s: float = 0.05
    ) -> bool:
        """Poll ``read_joints`` until it succeeds or ``timeout_s`` elapses.

        Use after a GP/GJ mode switch + high-frequency stream start: the controller
        briefly rejects ``getCurrentJoint`` while it settles into the new mode. Polling
        adapts to the real settle time instead of a fixed sleep — fast controllers start
        immediately, slow ones get the time they need.
        """
        deadline = time.perf_counter() + float(timeout_s)
        while time.perf_counter() < deadline:
            try:
                with self._motion_cmd_lock:
                    self.crp_arm_robot.read_joints()
                return True
            except Exception:
                time.sleep(poll_interval_s)
        return False

    def enable_proprio_cache(self) -> None:
        """Enable proprio/EE cache mode (called before starting the GP stream thread).

        While enabled, ``get_observation`` serves joints/EE from cache instead of
        calling the SDK, avoiding races with the 100 Hz GP stream thread.
        """
        with self._proprio_cache_lock:
            self._proprio_cache = {}

    def get_last_cached_joints(self) -> dict[str, float]:
        """Return the most recent cached joint dict (connect-time or stream-refreshed).

        Use when ``read_joints`` would race a just-sent GJ/GP trajectory and the
        controller would reject it with ``getCurrentJoint failed``.
        """
        return dict(getattr(self, "_last_joints_cache", {}))

    def disable_proprio_cache(self) -> None:
        """Disable proprio cache mode (called after stopping the GP stream thread)."""
        with self._proprio_cache_lock:
            self._proprio_cache = None

    def clear_ee_cache(self) -> None:
        """Drop latched EE cache so the next ``get_current_endpose`` must hit the SDK.

        Without this, a failed read after teardown can fall back to the **episode-start**
        pose and the next arm preload snaps the arm ``home''.
        """
        with self._proprio_cache_lock:
            self._last_ee_cache = None

    def update_ee_cache_from_pose6(self, pose6: Sequence[float]) -> None:
        """Publish EE into the observation cache (no SDK call)."""
        ee = {
            "ee.x": float(pose6[0]),
            "ee.y": float(pose6[1]),
            "ee.z": float(pose6[2]),
            "ee.roll": float(pose6[3]),
            "ee.pitch": float(pose6[4]),
            "ee.yaw": float(pose6[5]),
        }
        with self._proprio_cache_lock:
            self._last_ee_cache = ee

    def set_proprio_cache_snapshot(self, obs: dict[str, float]) -> None:
        """Publish joint/GOT into the proprio cache (parent-side; fork RAM is separate)."""
        snap = {str(k): float(v) for k, v in obs.items()}
        self._last_joints_cache = dict(snap)
        with self._proprio_cache_lock:
            if self._proprio_cache is not None:
                self._proprio_cache = snap

    def refresh_proprio_cache(self) -> bool:
        """SDK ``read_joints`` (+ GOT) into cache. Same-process only; after fork use snapshots."""
        try:
            with self._motion_cmd_lock:
                crp_joints_dict = self.crp_arm_robot.read_joints()
                got0 = (
                    int(self.crp_arm_robot.get_GOT(0))
                    if self.config.use_gripper_feature
                    else 0
                )
            obs = {f"{motor}.pos": float(val) for motor, val in crp_joints_dict.items()}
            if self.config.use_gripper_feature:
                obs["gripper.pos"] = float(got0)
            self.set_proprio_cache_snapshot(obs)
            return True
        except Exception as exc:
            logger.warning("%s refresh_proprio_cache failed: %s", self, exc)
            return False

    def set_speed_ratio(self, ratio: int):
        with self._motion_cmd_lock:
            self.crp_arm_robot.set_speed_ratio(ratio)
        return

    def get_speed_ratio(self) -> int:
        with self._motion_cmd_lock:
            return self.crp_arm_robot.get_speed_ratio()

    def stop_move(self) -> None:
        """Emergency-stop all ongoing controller motion (GJ/GP trajectories)."""
        with self._motion_cmd_lock:
            self.crp_arm_robot.stop_move()

    def get_current_endpose(self, *, allow_cache_fallback: bool = False) -> list[float]:
        """Return ``[x, y, z, roll, pitch, yaw]`` (user frame).

        Retries briefly after GP-stream stop / mode changes. Cache fallback is **off**
        by default — a stale cache (often the previous episode origin) would preload
        GP wrong and snap the arm when MOVE enable opens. Use
        ``allow_cache_fallback=True`` only for non-critical reads.
        """
        last_err: Exception | None = None
        for attempt in range(8):
            try:
                with self._motion_cmd_lock:
                    x, y, z, roll, pitch, yaw = self.crp_arm_robot.read_end_pose_user()
                pose = [float(x), float(y), float(z), float(roll), float(pitch), float(yaw)]
                with self._proprio_cache_lock:
                    self._last_ee_cache = {
                        "ee.x": pose[0],
                        "ee.y": pose[1],
                        "ee.z": pose[2],
                        "ee.roll": pose[3],
                        "ee.pitch": pose[4],
                        "ee.yaw": pose[5],
                    }
                return pose
            except Exception as exc:
                last_err = exc
                time.sleep(0.05 * (attempt + 1))
        if allow_cache_fallback:
            cached = getattr(self, "_last_ee_cache", None) or {}
            keys = ("ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw")
            if cached and all(k in cached for k in keys):
                logger.warning(
                    "%s get_current_endpose: SDK failed (%s); using last EE cache",
                    self,
                    last_err,
                )
                return [float(cached[k]) for k in keys]
        raise RuntimeError(f"{self}: read user pose failed after retries") from last_err
