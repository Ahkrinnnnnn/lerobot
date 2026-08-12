# !/usr/bin/env python

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
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets import LeRobotDataset
from lerobot.envs import HILSerlRobotEnvConfig
from lerobot.model import RobotKinematics
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    AddTeleopActionAsComplimentaryDataStep,
    AddTeleopEventsAsInfoStep,
    DataProcessorPipeline,
    DeviceProcessorStep,
    EnvTransition,
    GripperPenaltyProcessorStep,
    GymHILAdapterProcessorStep,
    ImageCropResizeProcessorStep,
    InterventionActionProcessorStep,
    MapDeltaActionToRobotActionStep,
    MapTensorToDeltaActionDictStep,
    Numpy2TorchActionProcessorStep,
    RewardClassifierProcessorStep,
    RobotActionToPolicyActionProcessorStep,
    RobotObservation,
    TimeLimitProcessorStep,
    Torch2NumpyActionProcessorStep,
    TransitionKey,
    VanillaObservationProcessorStep,
    create_transition,
    identity_transition,
)
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so_follower,
)
from lerobot.robots.crp_arm import CRPArm
from lerobot.robots.crp_arm.config_crp_arm import CRPArmConfig  # noqa: F401 — register crp_arm
from lerobot.robots.crp_arm.ee_gp import send_gp_endpose6
from lerobot.robots.crp_arm.hil_ee_processor import CRPDeltaEEToAbsoluteGPStep, CRPJointInterventionGPAssistStep
from lerobot.robots.robot import Robot
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    ForwardKinematicsJointsToEEObservation,
    GripperVelocityToJoint,
    InverseKinematicsRLStep,
)
from lerobot.teleoperators import (
    gamepad,  # noqa: F401
    keyboard,  # noqa: F401
    make_teleoperator_from_config,
    so_leader,  # noqa: F401
)
from lerobot.teleoperators.OMY_L100.config_OMY_L100 import OMYL100Config  # noqa: F401
from lerobot.teleoperators.OMY_L100.OMY_L100 import OMYL100  # noqa: F401
from lerobot.teleoperators.keyboard_hil_events import wait_for_manual_scene_reset
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.tools import TrajectoryProcessor
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGES, OBS_STATE, REWARD
from lerobot.utils.import_utils import require_package
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

logger = logging.getLogger(__name__)

DEFAULT_CRP_JOINT_NAMES = ("j1", "j2", "j3", "j4", "j5", "j6")


def _robot_has_bus(robot: Robot) -> bool:
    bus = getattr(robot, "bus", None)
    return bus is not None and getattr(bus, "motors", None) is not None


def _joint_names_from_robot(robot: Robot) -> list[str]:
    """Motor base names (without ``.pos``) for SO bus robots or CRP-style joints."""
    if _robot_has_bus(robot):
        return list(robot.bus.motors.keys())

    names: list[str] = []
    obs_features = getattr(robot, "observation_features", None)
    if isinstance(obs_features, dict):
        for key in obs_features:
            if isinstance(key, str) and key.endswith(".pos"):
                base = key.removesuffix(".pos")
                if base != "gripper":
                    names.append(base)
    if names:
        return names

    action_features = getattr(robot, "action_features", None)
    if isinstance(action_features, dict):
        for key in action_features:
            if isinstance(key, str) and key.endswith(".pos"):
                base = key.removesuffix(".pos")
                if base != "gripper":
                    names.append(base)
    if names:
        return names

    return list(DEFAULT_CRP_JOINT_NAMES)

from .joint_observations_processor import JointVelocityProcessorStep, MotorCurrentProcessorStep

logging.basicConfig(level=logging.INFO)


@dataclass
class DatasetConfig:
    """Configuration for dataset creation and management."""

    repo_id: str
    task: str
    root: str | None = None
    num_episodes_to_record: int = 5
    replay_episode: int | None = None
    push_to_hub: bool = False


@dataclass
class GymManipulatorConfig:
    """Main configuration for gym manipulator environment."""

    env: HILSerlRobotEnvConfig
    dataset: DatasetConfig
    mode: str | None = None  # Either "record", "replay", None
    device: str = "cpu"


def reset_follower_position(robot_arm: Robot, target_position: np.ndarray) -> None:
    """Reset robot arm to target position using smooth trajectory."""
    if _robot_has_bus(robot_arm):
        current_position_dict = robot_arm.bus.sync_read("Present_Position")
        current_position = np.array(
            [current_position_dict[name] for name in current_position_dict], dtype=np.float32
        )
        trajectory = torch.from_numpy(np.linspace(current_position, target_position, 50))
        for pose in trajectory:
            action_dict = dict(zip(current_position_dict, pose, strict=False))
            robot_arm.bus.sync_write("Goal_Position", action_dict)
            precise_sleep(0.015)
        return

    joint_names = _joint_names_from_robot(robot_arm)
    try:
        obs = robot_arm.get_observation(include_images=False)
    except TypeError:
        obs = robot_arm.get_observation()

    current_position = np.array(
        [float(obs.get(f"{name}.pos", 0.0)) for name in joint_names], dtype=np.float32
    )
    target = np.asarray(target_position, dtype=np.float32)
    if target.shape[0] != len(joint_names):
        raise ValueError(
            f"reset_pose length {target.shape[0]} does not match joint count {len(joint_names)}"
        )
    trajectory = np.linspace(current_position, target, 50)
    for pose in trajectory:
        action_dict = {f"{name}.pos": float(pose[i]) for i, name in enumerate(joint_names)}
        robot_arm.send_action(action_dict)
        precise_sleep(0.015)


class RobotEnv(gym.Env):
    """Gym environment for robotic control with human intervention support."""

    def __init__(
        self,
        robot,
        use_gripper: bool = False,
        display_cameras: bool = False,
        reset_pose: list[float] | None = None,
        reset_time_s: float = 5.0,
        action_mode: str = "joint",
        episode_reset_pause_key: str | None = None,
        crp_gp_start_index: int = 10,
        crp_gp_group_size: int = 5,
        teleop_device: Teleoperator | None = None,
        ee_include_rpy: bool = True,
    ) -> None:
        """Initialize robot environment with configuration options.

        Args:
            robot: Robot interface for hardware communication.
            use_gripper: Whether to include gripper in action space.
            display_cameras: Whether to show camera feeds during execution.
            reset_pose: Joint positions for environment reset.
            reset_time_s: Time to wait during reset (manual scene reset window).
            action_mode: ``joint`` (SO / GJ) or ``ee_gp`` (CRP absolute GP tensor).
            episode_reset_pause_key: Key that pauses the scene-reset countdown.
            crp_gp_start_index: GP register start index for CRP ``send_GPs``.
            crp_gp_group_size: TrajectoryProcessor group size for CRP GP matrix.
            teleop_device: Optional teleop (used for HIL keyboard reset pause / OMY latch).
            ee_include_rpy: When ``action_mode=ee_gp``, policy action includes δrpy (7D vs 4D).
        """
        super().__init__()

        self.robot = robot
        self.display_cameras = display_cameras
        self.action_mode = action_mode
        self.episode_reset_pause_key = episode_reset_pause_key
        self.crp_gp_start_index = crp_gp_start_index
        self.crp_gp_group_size = crp_gp_group_size
        self.teleop_device = teleop_device
        self.ee_include_rpy = ee_include_rpy
        self._trajectory_processor: TrajectoryProcessor | None = None
        self._episode_index = 0

        # Connect to the robot if not already connected.
        if not self.robot.is_connected:
            self.robot.connect()

        # Episode tracking.
        self.current_step = 0
        self.episode_data = None

        self._joint_names = _joint_names_from_robot(self.robot)
        self._image_keys = self.robot.cameras.keys()

        self.reset_pose = reset_pose
        self.reset_time_s = reset_time_s

        self.use_gripper = use_gripper

        self._raw_joint_positions = None
        # EE-GP delayed proprio: last commanded EE+gripper (matches recording obs=prev action).
        self._ee_command_cache: np.ndarray | None = None

        if self.action_mode == "ee_gp":
            self._trajectory_processor = TrajectoryProcessor()
            # Do not seed from a previous session's EE cache; GI56 stays off until arm-ready.
            if hasattr(self.robot, "clear_ee_cache"):
                self.robot.clear_ee_cache()
            if hasattr(self.robot, "set_motion_enabled"):
                self.robot.set_motion_enabled(False)
            self._seed_ee_command_cache()

        self._setup_spaces()

    def _seed_ee_command_cache(self) -> None:
        """Initialize EE command cache from a **fresh SDK** endpose (+ GOT).

        Never falls back to ``_last_ee_cache`` (recording: pre-arm reads must be live).
        """
        if hasattr(self.robot, "clear_ee_cache"):
            # Drop any latched pose so get_current_endpose cannot serve stale coordinates.
            self.robot.clear_ee_cache()
        pose = list(self.robot.get_current_endpose(allow_cache_fallback=False))
        grip = 0.0
        if self.use_gripper and hasattr(self.robot, "get_GOT"):
            try:
                grip = float(self.robot.get_GOT(0))
            except Exception:
                grip = 0.0
        self._ee_command_cache = np.asarray([*pose[:6], grip], dtype=np.float32)
        if hasattr(self.robot, "update_ee_cache_from_pose6"):
            self.robot.update_ee_cache_from_pose6(pose[:6])

    def _refresh_ee_command_cache_from_robot(self) -> None:
        """Pull latest GP command / EE cache published by the intervention stream."""
        cached = getattr(self.robot, "_last_ee_cache", None) or {}
        keys = ("ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw")
        if not all(k in cached for k in keys):
            return
        grip = 0.0
        if self.use_gripper:
            try:
                joints = (
                    self.robot.get_last_cached_joints()
                    if hasattr(self.robot, "get_last_cached_joints")
                    else {}
                )
                if "gripper.pos" in joints:
                    grip = float(joints["gripper.pos"])
                elif hasattr(self.robot, "get_GOT"):
                    grip = float(self.robot.get_GOT(0))
            except Exception:
                if self._ee_command_cache is not None and self._ee_command_cache.size >= 7:
                    grip = float(self._ee_command_cache[6])
        self._ee_command_cache = np.asarray(
            [float(cached[k]) for k in keys] + [grip],
            dtype=np.float32,
        )

    def _get_observation(self) -> RobotObservation:
        """Get current robot observation including joint positions and camera images."""
        obs_dict = self.robot.get_observation()
        raw_joint_joint_position = {f"{name}.pos": obs_dict[f"{name}.pos"] for name in self._joint_names}
        images = {key: obs_dict[key] for key in self._image_keys}

        if self.action_mode == "ee_gp":
            if self._ee_command_cache is None:
                self._seed_ee_command_cache()
            agent_pos = np.asarray(self._ee_command_cache, dtype=np.float32).copy()
            return {"agent_pos": agent_pos, "pixels": images, **raw_joint_joint_position}

        joint_positions = np.array([raw_joint_joint_position[f"{name}.pos"] for name in self._joint_names])
        return {"agent_pos": joint_positions, "pixels": images, **raw_joint_joint_position}

    def _setup_spaces(self) -> None:
        """Configure observation and action spaces based on robot capabilities."""
        current_observation = self._get_observation()

        observation_spaces = {}

        # Define observation spaces for images and other states.
        if current_observation is not None and "pixels" in current_observation:
            prefix = OBS_IMAGES
            observation_spaces = {
                f"{prefix}.{key}": gym.spaces.Box(
                    low=0, high=255, shape=current_observation["pixels"][key].shape, dtype=np.uint8
                )
                for key in current_observation["pixels"]
            }

        if current_observation is not None:
            agent_pos = current_observation["agent_pos"]
            observation_spaces[OBS_STATE] = gym.spaces.Box(
                low=0,
                high=10,
                shape=agent_pos.shape,
                dtype=np.float32,
            )

        self.observation_space = gym.spaces.Dict(observation_spaces)

        if self.action_mode == "joint":
            # CRP joint targets: j1..j6 (+ optional gripper), matching IL datasets like inserting_rod.
            n_joints = len(self._joint_names)
            action_dim = n_joints + (1 if self.use_gripper else 0)
            self.action_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(action_dim,),
                dtype=np.float32,
            )
        else:
            # Policy / teleop operate in EE delta space (xyz [+rpy] + optional gripper).
            action_dim = 6 if self.ee_include_rpy else 3
            bounds = {}
            bounds["min"] = -np.ones(action_dim)
            bounds["max"] = np.ones(action_dim)

            if self.use_gripper:
                action_dim += 1
                bounds["min"] = np.concatenate([bounds["min"], [0]])
                bounds["max"] = np.concatenate([bounds["max"], [2]])

            self.action_space = gym.spaces.Box(
                low=bounds["min"],
                high=bounds["max"],
                shape=(action_dim,),
                dtype=np.float32,
            )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[RobotObservation, dict[str, Any]]:
        """Reset environment to initial state."""
        # Close MOVE gate before scene rearrange so leftover GP cannot chase during reset.
        if self.action_mode == "ee_gp" and hasattr(self.robot, "set_motion_enabled"):
            self.robot.set_motion_enabled(False)
        if self.action_mode == "ee_gp" and hasattr(self.robot, "clear_ee_cache"):
            self.robot.clear_ee_cache()
            self._ee_command_cache = None

        start_time = time.perf_counter()
        if self.reset_pose is not None:
            log_say("Reset the environment.", play_sounds=True)
            reset_follower_position(self.robot, np.array(self.reset_pose))
            log_say("Reset the environment done.", play_sounds=True)

        remaining = max(self.reset_time_s - (time.perf_counter() - start_time), 0.0)
        keyboard_events = getattr(self.teleop_device, "_hil_keyboard", None) if self.teleop_device else None
        pause_key = self.episode_reset_pause_key
        if pause_key is None and keyboard_events is not None:
            pause_key = getattr(keyboard_events, "_reset_pause_key", None)
        if remaining > 0 and (
            keyboard_events is not None or (pause_key and self.action_mode == "ee_gp")
        ):
            wait_for_manual_scene_reset(
                remaining,
                episode_index=self._episode_index,
                pause_key=pause_key or "p",
                # Always pass keyboard when present so s/f/r latched during the wait
                # are cleared (even if pause_key is null and pause is unused).
                keyboard_events=keyboard_events,
            )
        else:
            precise_sleep(remaining)
        if keyboard_events is not None:
            keyboard_events.clear_episode_flags()

        if self.teleop_device is not None and hasattr(self.teleop_device, "reset_reference"):
            self.teleop_device.reset_reference()

        super().reset(seed=seed, options=options)

        # Reset episode tracking variables.
        self.current_step = 0
        self.episode_data = None
        self._episode_index += 1
        if self.action_mode == "ee_gp":
            # Re-seed command cache from pose after manual reset (GI56 still off until first send).
            self._seed_ee_command_cache()
        obs = self._get_observation()
        self._raw_joint_positions = {f"{key}.pos": obs[f"{key}.pos"] for key in self._joint_names}
        return obs, {TeleopEvents.IS_INTERVENTION: False}

    def _as_1d_float_action(self, action) -> np.ndarray:
        """Policy / processors often leave a batch dim ``(1, D)``; env.step needs ``(D,)``."""
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        return np.asarray(action, dtype=np.float32).reshape(-1)

    def _arm_ee_gp_motion_gate(self, seed_pose6: list[float]) -> None:
        """Recording-style arm-ready: preload GP → GI→GP → **then** GI56 ON.

        GI56 must stay off until GP registers hold the latched pose; opening the gate on
        stale GP (or after trusting an old EE cache) snaps the arm.
        """
        if self._trajectory_processor is None:
            self._trajectory_processor = TrajectoryProcessor()
        seed = [float(v) for v in seed_pose6[:6]]
        # Ensure we do not silently re-use a previous episode pose via cache.
        if hasattr(self.robot, "clear_ee_cache"):
            self.robot.clear_ee_cache()
        send_gp_endpose6(
            self.robot,
            self._trajectory_processor,
            seed,
            start_index=self.crp_gp_start_index,
            group_size=self.crp_gp_group_size,
            switch_to_gp_mode=False,
        )
        if hasattr(self.robot, "ensure_gp_mode"):
            self.robot.ensure_gp_mode()
        if hasattr(self.robot, "update_ee_cache_from_pose6"):
            self.robot.update_ee_cache_from_pose6(seed)
        if hasattr(self.robot, "set_motion_enabled"):
            self.robot.set_motion_enabled(True)
        logger.info(
            "HIL EE policy arm-ready: preloaded GP xyz=(%.3f %.3f %.3f) then GI56 ON "
            "(gate was closed; no MOVE before this)",
            seed[0],
            seed[1],
            seed[2],
        )

    def _send_ee_gp_action(self, action) -> None:
        """Send absolute GP pose (+ optional gripper) from the CRP EE processor."""
        values = self._as_1d_float_action(action).tolist()

        if len(values) < 6:
            raise ValueError(f"ee_gp action must have at least 6 values, got {len(values)}")

        vec6 = [float(v) for v in values[:6]]
        if self._trajectory_processor is None:
            self._trajectory_processor = TrajectoryProcessor()

        # Recording rule: until arm-ready, never open GI56. Fresh-latch pose, preload, then gate.
        gate_open = True
        if hasattr(self.robot, "is_motion_enabled"):
            gate_open = bool(self.robot.is_motion_enabled())
        if not gate_open:
            # Prefer a live SDK pose for the latch (not a leftover command cache from last ep).
            if hasattr(self.robot, "clear_ee_cache"):
                self.robot.clear_ee_cache()
            seed = list(self.robot.get_current_endpose(allow_cache_fallback=False))
            grip_seed = 0.0
            if self.use_gripper and hasattr(self.robot, "get_GOT"):
                try:
                    grip_seed = float(self.robot.get_GOT(0))
                except Exception:
                    grip_seed = 0.0
            self._ee_command_cache = np.asarray([*seed[:6], grip_seed], dtype=np.float32)
            self._arm_ee_gp_motion_gate(seed)
            # First armed command holds the latched pose (delta applied on subsequent steps).
            # Avoid chasing a policy target that was computed against a pre-latch reference.
            vec6 = list(seed[:6])
            values = list(seed[:6]) + ([grip_seed] if self.use_gripper else [])

        send_gp_endpose6(
            self.robot,
            self._trajectory_processor,
            vec6,
            start_index=self.crp_gp_start_index,
            group_size=self.crp_gp_group_size,
            switch_to_gp_mode=False,
        )

        grip = 0.0
        if self.use_gripper and len(values) > 6 and hasattr(self.robot, "set_GOT"):
            grip = float(values[6])
            # Discrete gamepad codes {0,1,2} → leave / close / open-ish; else treat as GOT0 value.
            if grip in (0.0, 1.0, 2.0) and grip == int(grip):
                if int(grip) == 0:
                    self.robot.set_GOT(0, 0)
                    grip = 0.0
                elif int(grip) == 2:
                    self.robot.set_GOT(0, 1000)
                    grip = 1000.0
                else:
                    # code 1 = leave current; keep latched GOT if we just armed.
                    if self._ee_command_cache is not None and self._ee_command_cache.size >= 7:
                        grip = float(self._ee_command_cache[6])
            else:
                grip = float(int(max(0, min(1000, round(grip)))))
                self.robot.set_GOT(0, int(grip))
        elif self._ee_command_cache is not None and self._ee_command_cache.size >= 7:
            grip = float(self._ee_command_cache[6])

        # Delayed proprio = last commanded EE (+ gripper), matching recording.
        self._ee_command_cache = np.asarray([*vec6, grip], dtype=np.float32)
        if hasattr(self.robot, "update_ee_cache_from_pose6"):
            self.robot.update_ee_cache_from_pose6(vec6)

    def step(self, action, *, apply_action: bool = True) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        """Execute one environment step with given action.

        Args:
            action: Policy / teleop action for this step.
            apply_action: If False, only refresh observation (used when joint-space HIL
                already sent a GP command during intervention — must not also ``send_GJs``).
        """
        if apply_action:
            if self.action_mode == "ee_gp":
                self._send_ee_gp_action(action)
            else:
                action = self._as_1d_float_action(action)
                joint_targets_dict = {
                    f"{key}.pos": float(action[i]) for i, key in enumerate(self._joint_names)
                }
                if self.use_gripper and action.shape[0] > len(self._joint_names):
                    joint_targets_dict["gripper.pos"] = float(action[len(self._joint_names)])
                self.robot.send_action(joint_targets_dict)
        elif self.action_mode == "ee_gp":
            # Intervention stream owns send_GPs; keep obs command-cache in sync.
            self._refresh_ee_command_cache_from_robot()

        obs = self._get_observation()

        self._raw_joint_positions = {f"{key}.pos": obs[f"{key}.pos"] for key in self._joint_names}

        if self.display_cameras:
            self.render()

        self.current_step += 1

        reward = 0.0
        terminated = False
        truncated = False

        return (
            obs,
            reward,
            terminated,
            truncated,
            {TeleopEvents.IS_INTERVENTION: False},
        )

    def render(self) -> None:
        """Display robot camera feeds."""
        import cv2

        current_observation = self._get_observation()
        if current_observation is not None:
            image_keys = [key for key in current_observation if "image" in key]

            for key in image_keys:
                cv2.imshow(key, cv2.cvtColor(current_observation[key].numpy(), cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

    def close(self) -> None:
        """Close environment and disconnect robot."""
        if self.robot.is_connected:
            self.robot.disconnect()

    def get_raw_joint_positions(self) -> dict[str, float]:
        """Get raw joint positions."""
        return self._raw_joint_positions


def make_robot_env(cfg: HILSerlRobotEnvConfig) -> tuple[gym.Env, Any]:
    """Create robot environment from configuration.

    Args:
        cfg: Environment configuration.

    Returns:
        Tuple of (gym environment, teleoperator device).
    """
    # Check if this is a GymHIL simulation environment
    if cfg.name == "gym_hil":
        assert cfg.robot is None and cfg.teleop is None, "GymHIL environment does not support robot or teleop"
        require_package("gym-hil", extra="hilserl", import_name="gym_hil")
        import gym_hil  # noqa: F401

        # Extract gripper settings with defaults
        use_gripper = cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else True
        gripper_penalty = cfg.processor.gripper.gripper_penalty if cfg.processor.gripper is not None else 0.0

        env = gym.make(
            f"gym_hil/{cfg.task}",
            image_obs=True,
            render_mode="human",
            use_gripper=use_gripper,
            gripper_penalty=gripper_penalty,
        )

        return env, None

    # Real robot environment
    assert cfg.robot is not None, "Robot config must be provided for real robot environment"
    assert cfg.teleop is not None, "Teleop config must be provided for real robot environment"

    robot = make_robot_from_config(cfg.robot)
    teleop_device = make_teleoperator_from_config(cfg.teleop)
    teleop_device.connect()

    # Create base environment with safe defaults
    use_gripper = cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else True
    display_cameras = (
        cfg.processor.observation.display_cameras if cfg.processor.observation is not None else False
    )
    reset_pose = cfg.processor.reset.fixed_reset_joint_positions if cfg.processor.reset is not None else None
    reset_time_s = cfg.processor.reset.reset_time_s if cfg.processor.reset is not None else 5.0
    episode_reset_pause_key = (
        cfg.processor.reset.episode_reset_pause_key if cfg.processor.reset is not None else None
    )
    # Fall back to teleop HIL pause key so a single JSON field (hil_reset_pause_key) is enough.
    if not episode_reset_pause_key and cfg.teleop is not None:
        episode_reset_pause_key = getattr(cfg.teleop, "hil_reset_pause_key", None)

    if cfg.processor.crp_ee is not None and cfg.processor.inverse_kinematics is not None:
        raise ValueError("processor.crp_ee and processor.inverse_kinematics are mutually exclusive")

    crp_action_space = getattr(cfg.processor, "crp_action_space", "ee") or "ee"
    if crp_action_space not in ("ee", "joint"):
        raise ValueError(f"processor.crp_action_space must be 'ee' or 'joint', got {crp_action_space!r}")
    if crp_action_space == "ee" and cfg.processor.crp_ee is None and cfg.processor.inverse_kinematics is None:
        # CRP EE path needs crp_ee; SO path uses inverse_kinematics instead.
        if cfg.robot is not None and getattr(cfg.robot, "type", None) == "crp_arm":
            raise ValueError("processor.crp_action_space='ee' requires processor.crp_ee for crp_arm")

    if crp_action_space == "joint":
        action_mode = "joint"
    elif cfg.processor.crp_ee is not None:
        action_mode = "ee_gp"
    else:
        action_mode = "joint"

    if isinstance(robot, CRPArm):
        logger.info(
            "HIL CRP action_space=%s → RobotEnv.action_mode=%s "
            "(joint = policy GJ via send_GJs; ee_gp = policy send_GPs)",
            crp_action_space,
            action_mode,
        )

    crp_gp_start_index = cfg.processor.crp_ee.gp_start_index if cfg.processor.crp_ee is not None else 10
    crp_gp_group_size = cfg.processor.crp_ee.gp_group_size if cfg.processor.crp_ee is not None else 5
    ee_include_rpy = (
        bool(cfg.processor.crp_ee.include_rpy) if cfg.processor.crp_ee is not None else True
    )

    env = RobotEnv(
        robot=robot,
        use_gripper=use_gripper,
        display_cameras=display_cameras,
        reset_pose=reset_pose,
        reset_time_s=reset_time_s,
        action_mode=action_mode,
        episode_reset_pause_key=episode_reset_pause_key,
        crp_gp_start_index=crp_gp_start_index,
        crp_gp_group_size=crp_gp_group_size,
        teleop_device=teleop_device,
        ee_include_rpy=ee_include_rpy,
    )

    return env, teleop_device


def make_processors(
    env: gym.Env, teleop_device: Teleoperator | None, cfg: HILSerlRobotEnvConfig, device: str = "cpu"
) -> tuple[
    DataProcessorPipeline[EnvTransition, EnvTransition], DataProcessorPipeline[EnvTransition, EnvTransition]
]:
    """Create environment and action processors.

    Args:
        env: Robot environment instance.
        teleop_device: Teleoperator device for intervention.
        cfg: Processor configuration.
        device: Target device for computations.

    Returns:
        Tuple of (environment processor, action processor).
    """
    terminate_on_success = (
        cfg.processor.reset.terminate_on_success if cfg.processor.reset is not None else True
    )

    if cfg.name == "gym_hil":
        action_pipeline_steps = [
            InterventionActionProcessorStep(terminate_on_success=terminate_on_success),
            Torch2NumpyActionProcessorStep(),
        ]

        env_pipeline_steps = [
            GymHILAdapterProcessorStep(),
            Numpy2TorchActionProcessorStep(),
            VanillaObservationProcessorStep(),
        ]

        # Add time limit processor if reset config exists
        if cfg.processor.reset is not None:
            env_pipeline_steps.append(
                TimeLimitProcessorStep(max_episode_steps=int(cfg.processor.reset.control_time_s * cfg.fps))
            )

        env_pipeline_steps.extend(
            [
                AddBatchDimensionProcessorStep(),
                DeviceProcessorStep(device=device),
            ]
        )

        return DataProcessorPipeline(
            steps=env_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
        ), DataProcessorPipeline(
            steps=action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
        )

    # Full processor pipeline for real robot environment
    motor_names = _joint_names_from_robot(env.robot)

    if cfg.processor.crp_ee is not None and cfg.processor.inverse_kinematics is not None:
        raise ValueError("processor.crp_ee and processor.inverse_kinematics are mutually exclusive")

    # Set up kinematics solver if inverse kinematics is configured
    kinematics_solver = None
    if cfg.processor.inverse_kinematics is not None:
        kinematics_solver = RobotKinematics(
            urdf_path=cfg.processor.inverse_kinematics.urdf_path,
            target_frame_name=cfg.processor.inverse_kinematics.target_frame_name,
            joint_names=motor_names,
        )

    env_pipeline_steps = [VanillaObservationProcessorStep()]

    if cfg.processor.observation is not None:
        if cfg.processor.observation.add_joint_velocity_to_observation:
            env_pipeline_steps.append(JointVelocityProcessorStep(dt=1.0 / cfg.fps))
        # Motor current requires Dynamixel/Feetech bus — skip for CRP.
        if cfg.processor.observation.add_current_to_observation and _robot_has_bus(env.robot):
            env_pipeline_steps.append(MotorCurrentProcessorStep(robot=env.robot))
        elif cfg.processor.observation.add_current_to_observation:
            logger.warning("add_current_to_observation ignored: robot has no motor bus")

    add_ee_pose = (
        cfg.processor.observation is not None and cfg.processor.observation.add_ee_pose_to_observation
    )
    if kinematics_solver is not None and add_ee_pose:
        env_pipeline_steps.append(
            ForwardKinematicsJointsToEEObservation(
                kinematics=kinematics_solver,
                motor_names=motor_names,
            )
        )

    if cfg.processor.image_preprocessing is not None:
        env_pipeline_steps.append(
            ImageCropResizeProcessorStep(
                crop_params_dict=cfg.processor.image_preprocessing.crop_params_dict,
                resize_size=cfg.processor.image_preprocessing.resize_size,
            )
        )

    # Add time limit processor if reset config exists
    if cfg.processor.reset is not None:
        env_pipeline_steps.append(
            TimeLimitProcessorStep(max_episode_steps=int(cfg.processor.reset.control_time_s * cfg.fps))
        )

    # Add gripper penalty processor if gripper config exists and enabled
    # Only add if max_gripper_pos is explicitly configured (required for normalization)
    if (
        cfg.processor.gripper is not None
        and cfg.processor.gripper.use_gripper
        and cfg.processor.max_gripper_pos is not None
    ):
        env_pipeline_steps.append(
            GripperPenaltyProcessorStep(
                penalty=cfg.processor.gripper.gripper_penalty,
                max_gripper_pos=cfg.processor.max_gripper_pos,
            )
        )

    if (
        cfg.processor.reward_classifier is not None
        and cfg.processor.reward_classifier.pretrained_path is not None
    ):
        env_pipeline_steps.append(
            RewardClassifierProcessorStep(
                pretrained_path=cfg.processor.reward_classifier.pretrained_path,
                device=device,
                success_threshold=cfg.processor.reward_classifier.success_threshold,
                success_reward=cfg.processor.reward_classifier.success_reward,
                terminate_on_success=terminate_on_success,
            )
        )

    env_pipeline_steps.append(AddBatchDimensionProcessorStep())
    env_pipeline_steps.append(DeviceProcessorStep(device=device))

    use_gripper = cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else False
    crp_action_space = getattr(cfg.processor, "crp_action_space", "ee") or "ee"

    action_pipeline_steps = [
        AddTeleopActionAsComplimentaryDataStep(teleop_device=teleop_device),
        AddTeleopEventsAsInfoStep(teleop_device=teleop_device),
    ]

    # OMY relative xyz → GP stream (recording arming). Labels: joint or ee_delta.
    include_rpy = bool(cfg.processor.crp_ee.include_rpy) if cfg.processor.crp_ee is not None else True
    if isinstance(env.robot, CRPArm) and cfg.processor.crp_ee is not None:
        action_pipeline_steps.append(
            CRPJointInterventionGPAssistStep(
                robot=env.robot,
                teleop=teleop_device,
                end_effector_step_sizes=cfg.processor.crp_ee.end_effector_step_sizes,
                ee_delta_scale=cfg.processor.crp_ee.ee_delta_scale,
                use_gripper=use_gripper,
                gp_start_index=cfg.processor.crp_ee.gp_start_index,
                gp_group_size=cfg.processor.crp_ee.gp_group_size,
                gp_secondary_index=cfg.processor.crp_ee.gp_secondary_index,
                omy_ee_ready_timeout_s=cfg.processor.crp_ee.hil_omy_ee_ready_timeout_s,
                gp_stream_hz=cfg.processor.crp_ee.hil_gp_stream_hz,
                rl_label_space="ee_delta" if crp_action_space == "ee" else "joint",
                include_rpy=include_rpy,
            )
        )

    action_pipeline_steps.append(
        InterventionActionProcessorStep(
            use_gripper=use_gripper,
            terminate_on_success=terminate_on_success,
            action_space="joint" if crp_action_space == "joint" else "ee",
            robot=env.robot if crp_action_space == "joint" else None,
            joint_names=motor_names,
            include_rpy=include_rpy if crp_action_space == "ee" else False,
        )
    )

    # CRP EE GP path (skips SO IK) — autonomous + intervention both in EE delta space
    if crp_action_space == "ee" and cfg.processor.crp_ee is not None:
        if not isinstance(env.robot, CRPArm):
            raise TypeError("processor.crp_ee requires robot.type=crp_arm")
        action_pipeline_steps.extend(
            [
                MapTensorToDeltaActionDictStep(use_gripper=use_gripper, include_rpy=include_rpy),
                CRPDeltaEEToAbsoluteGPStep(
                    robot=env.robot,
                    end_effector_step_sizes=cfg.processor.crp_ee.end_effector_step_sizes,
                    use_gripper=use_gripper,
                    use_latched_reference=cfg.processor.crp_ee.use_latched_reference,
                    ee_delta_max=cfg.processor.crp_ee.ee_delta_max,
                ),
            ]
        )
    # SO InverseKinematics path
    elif cfg.processor.inverse_kinematics is not None and kinematics_solver is not None:
        inverse_kinematics_steps = [
            MapTensorToDeltaActionDictStep(use_gripper=use_gripper),
            MapDeltaActionToRobotActionStep(),
            EEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes=cfg.processor.inverse_kinematics.end_effector_step_sizes,
                motor_names=motor_names,
                use_latched_reference=False,
                use_ik_solution=True,
            ),
            EEBoundsAndSafety(
                end_effector_bounds=cfg.processor.inverse_kinematics.end_effector_bounds,
            ),
            GripperVelocityToJoint(
                clip_max=cfg.processor.max_gripper_pos,
                speed_factor=1.0,
                discrete_gripper=True,
            ),
            InverseKinematicsRLStep(
                kinematics=kinematics_solver, motor_names=motor_names, initial_guess_current_joints=False
            ),
        ]
        action_pipeline_steps.extend(inverse_kinematics_steps)
        action_pipeline_steps.append(RobotActionToPolicyActionProcessorStep(motor_names=motor_names))
    # Joint-space CRP: policy tensor is already j1..j6(+gripper); RobotEnv.send_action handles it.
    elif crp_action_space == "joint":
        pass
    else:
        logger.warning(
            "No CRP EE / joint / IK action chain configured; env.step will receive raw policy actions"
        )

    return DataProcessorPipeline(
        steps=env_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
    ), DataProcessorPipeline(
        steps=action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
    )


def step_env_and_process_transition(
    env: gym.Env,
    transition: EnvTransition,
    action: torch.Tensor,
    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
) -> EnvTransition:
    """
    Execute one step with processor pipeline.

    Args:
        env: The robot environment
        transition: Current transition state
        action: Action to execute
        env_processor: Environment processor
        action_processor: Action processor

    Returns:
        Processed transition with updated state.
    """

    # Create action transition
    transition[TransitionKey.ACTION] = action
    transition[TransitionKey.OBSERVATION] = (
        env.get_raw_joint_positions() if hasattr(env, "get_raw_joint_positions") else {}
    )
    processed_action_transition = action_processor(transition)
    processed_action = processed_action_transition[TransitionKey.ACTION]

    # Joint-space HIL: CRPJointInterventionGPAssistStep already issued send_GPs while intervening.
    # The action tensor is then replaced with CRP joints for RL labels — must not send_GJs too.
    complementary_pre = processed_action_transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}
    apply_action = not bool(complementary_pre.get("crp_gp_command_sent", False))
    obs, reward, terminated, truncated, info = env.step(processed_action, apply_action=apply_action)

    reward = reward + processed_action_transition[TransitionKey.REWARD]
    terminated = terminated or processed_action_transition[TransitionKey.DONE]
    truncated = truncated or processed_action_transition[TransitionKey.TRUNCATED]
    complementary_data = processed_action_transition[TransitionKey.COMPLEMENTARY_DATA].copy()

    if hasattr(env, "get_raw_joint_positions"):
        raw_joint_positions = env.get_raw_joint_positions()
        if raw_joint_positions is not None:
            complementary_data["raw_joint_positions"] = raw_joint_positions

    # Merge env and action-processor info: env wins for str keys, action-processor
    # wins for `TeleopEvents` enum keys
    action_info = processed_action_transition[TransitionKey.INFO]
    new_info = info.copy()
    for key, value in action_info.items():
        if isinstance(key, TeleopEvents):
            new_info[key] = value

    new_transition = create_transition(
        observation=obs,
        action=processed_action,
        reward=reward,
        done=terminated,
        truncated=truncated,
        info=new_info,
        complementary_data=complementary_data,
    )
    new_transition = env_processor(new_transition)

    return new_transition


def reset_and_build_transition(
    env: gym.Env,
    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
) -> EnvTransition:
    """Reset env + processors and return the first env-processed transition.

    Stop the HIL GP stream **before** ``env.reset()`` scene wait. Otherwise Space
    intervention keeps sending absolute GP (possibly huge OMY deltas) during the
    countdown — looks like unrestricted motion and can snap when GI56 re-opens.
    """
    try:
        action_processor.reset()
    except Exception:
        logging.exception("action_processor.reset before env.reset failed")
    obs, info = env.reset()
    env_processor.reset()
    action_processor.reset()
    complementary_data: dict[str, Any] = {}
    if hasattr(env, "get_raw_joint_positions"):
        raw_joint_positions = env.get_raw_joint_positions()
        if raw_joint_positions is not None:
            complementary_data["raw_joint_positions"] = raw_joint_positions
    transition = create_transition(observation=obs, info=info, complementary_data=complementary_data)
    return env_processor(data=transition)


def control_loop(
    env: gym.Env,
    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    teleop_device: Teleoperator,
    cfg: GymManipulatorConfig,
) -> None:
    """Main control loop for robot environment interaction.
    if cfg.mode == "record": then a dataset will be created and recorded

    Args:
     env: The robot environment
     env_processor: Environment processor
     action_processor: Action processor
     teleop_device: Teleoperator device
     cfg: gym_manipulator configuration
    """
    dt = 1.0 / cfg.env.fps

    print(f"Starting control loop at {cfg.env.fps} FPS")
    print("Controls:")
    print("- Use gamepad/teleop device for intervention")
    print("- When not intervening, robot will stay still")
    print("- Press Ctrl+C to exit")

    transition = reset_and_build_transition(env, env_processor, action_processor)

    # Determine if gripper is used
    use_gripper = cfg.env.processor.gripper.use_gripper if cfg.env.processor.gripper is not None else True

    dataset = None
    if cfg.mode == "record":
        if teleop_device:
            action_features = teleop_device.action_features
        else:
            action_features = {
                "dtype": "float32",
                "shape": (4,),
                "names": ["delta_x", "delta_y", "delta_z", "gripper"],
            }
        features = {
            ACTION: action_features,
            REWARD: {"dtype": "float32", "shape": (1,), "names": None},
            DONE: {"dtype": "bool", "shape": (1,), "names": None},
        }
        if use_gripper:
            features["complementary_info.discrete_penalty"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": ["discrete_penalty"],
            }

        for key, value in transition[TransitionKey.OBSERVATION].items():
            if key == OBS_STATE:
                features[key] = {
                    "dtype": "float32",
                    "shape": value.squeeze(0).shape,
                    "names": None,
                }
            if "image" in key:
                features[key] = {
                    "dtype": "video",
                    "shape": value.squeeze(0).shape,
                    "names": ["channels", "height", "width"],
                }

        # Create dataset
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.env.fps,
            root=cfg.dataset.root,
            use_videos=True,
            image_writer_threads=4,
            image_writer_processes=0,
            features=features,
        )

    episode_idx = 0
    episode_step = 0
    episode_start_time = time.perf_counter()

    try:
        while episode_idx < cfg.dataset.num_episodes_to_record:
            step_start_time = time.perf_counter()

            # Create a neutral action (no movement)
            neutral_action = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
            if use_gripper:
                neutral_action = torch.cat([neutral_action, torch.tensor([1.0])])  # Gripper stay

            observation = {
                k: v.squeeze(0).cpu()
                for k, v in transition[TransitionKey.OBSERVATION].items()
                if isinstance(v, torch.Tensor)
            }

            transition = step_env_and_process_transition(
                env=env,
                transition=transition,
                action=neutral_action,
                env_processor=env_processor,
                action_processor=action_processor,
            )
            terminated = transition.get(TransitionKey.DONE, False)
            truncated = transition.get(TransitionKey.TRUNCATED, False)

            if cfg.mode == "record":
                action_to_record = transition[TransitionKey.COMPLEMENTARY_DATA].get(
                    "teleop_action", transition[TransitionKey.ACTION]
                )
                frame = {
                    **observation,
                    ACTION: action_to_record.cpu(),
                    REWARD: np.array([transition[TransitionKey.REWARD]], dtype=np.float32),
                    DONE: np.array([terminated or truncated], dtype=bool),
                }
                if use_gripper:
                    discrete_penalty = transition[TransitionKey.COMPLEMENTARY_DATA].get(
                        "discrete_penalty", 0.0
                    )
                    frame["complementary_info.discrete_penalty"] = np.array(
                        [discrete_penalty], dtype=np.float32
                    )

                if dataset is not None:
                    frame["task"] = cfg.dataset.task
                    dataset.add_frame(frame)

            episode_step += 1

            # Handle episode termination
            if terminated or truncated:
                episode_time = time.perf_counter() - episode_start_time
                logging.info(
                    f"Episode ended after {episode_step} steps in {episode_time:.1f}s with reward {transition[TransitionKey.REWARD]}"
                )
                episode_step = 0
                episode_idx += 1

                if dataset is not None:
                    if transition[TransitionKey.INFO].get(TeleopEvents.RERECORD_EPISODE, False):
                        logging.info(f"Re-recording episode {episode_idx}")
                        dataset.clear_episode_buffer()
                        episode_idx -= 1
                    else:
                        logging.info(f"Saving episode {episode_idx}")
                        dataset.save_episode()

                # Reset for new episode
                transition = reset_and_build_transition(env, env_processor, action_processor)

            # Maintain fps timing
            precise_sleep(max(dt - (time.perf_counter() - step_start_time), 0.0))
    finally:
        if dataset is not None and dataset.writer is not None and dataset.writer.image_writer is not None:
            logging.info("Waiting for image writer to finish...")
            dataset.writer.image_writer.stop()

    if dataset is not None and cfg.dataset.push_to_hub:
        logging.info("Finalizing dataset before pushing to hub")
        dataset.finalize()
        logging.info("Pushing dataset to hub")
        dataset.push_to_hub()


def replay_trajectory(
    env: gym.Env, action_processor: DataProcessorPipeline, cfg: GymManipulatorConfig
) -> None:
    """Replay recorded trajectory on robot environment."""
    assert cfg.dataset.replay_episode is not None, "Replay episode must be provided for replay"

    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=[cfg.dataset.replay_episode],
        download_videos=False,
    )
    actions = dataset.select_columns(ACTION)

    _, info = env.reset()

    for action_data in actions:
        start_time = time.perf_counter()
        transition = create_transition(
            observation=env.get_raw_joint_positions() if hasattr(env, "get_raw_joint_positions") else {},
            action=action_data[ACTION],
        )
        transition = action_processor(transition)
        env.step(transition[TransitionKey.ACTION])
        precise_sleep(max(1 / cfg.env.fps - (time.perf_counter() - start_time), 0.0))


@parser.wrap()
def main(cfg: GymManipulatorConfig) -> None:
    """Main entry point for gym manipulator script."""
    env, teleop_device = make_robot_env(cfg.env)
    env_processor, action_processor = make_processors(env, teleop_device, cfg.env, cfg.device)

    print("Environment observation space:", env.observation_space)
    print("Environment action space:", env.action_space)
    print("Environment processor:", env_processor)
    print("Action processor:", action_processor)

    if cfg.mode == "replay":
        replay_trajectory(env, action_processor, cfg)
        exit()

    control_loop(env, env_processor, action_processor, teleop_device, cfg)


if __name__ == "__main__":
    main()
