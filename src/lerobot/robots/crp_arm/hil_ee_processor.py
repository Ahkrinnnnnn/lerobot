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

"""HIL-SERL action processors for CRP end-effector GP control (no SO IK)."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.robots.crp_arm.ee_gp import (
    EE_OMY_DELTA_SCALE,
    MpGpStreamHandle,
    apply_xyz_delta_to_endpose,
    omy_rh_r1_to_got0,
    read_omy_ee_xyz,
    resolve_ee_delta_scale,
    resolve_ee_step_sizes,
    run_omy_relative_gp_stream,
    send_gp_endpose6,
    stop_omy_relative_gp_mp_stream,
    wrap_angle_delta_deg,
)
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.tools import TrajectoryProcessor
from lerobot.types import EnvTransition, PolicyAction, TransitionKey
from lerobot.utils.constants import ACTION

if TYPE_CHECKING:
    from lerobot.robots.crp_arm.crp_arm import CRPArm

logger = logging.getLogger(__name__)

# Complementary-data keys consumed by gym_manipulator / actor.
CRP_GP_COMMAND_SENT_KEY = "crp_gp_command_sent"
EXCLUDE_FROM_REPLAY_KEY = "exclude_from_replay"


def _transition_with_complementary(
    transition: EnvTransition, complementary: dict[str, Any]
) -> EnvTransition:
    out = dict(transition)
    out[TransitionKey.COMPLEMENTARY_DATA] = complementary
    return out


@ProcessorStepRegistry.register("crp_delta_ee_to_absolute_gp")
@dataclass
class CRPDeltaEEToAbsoluteGPStep(ProcessorStep):
    """Map ``delta_x/y/z/roll/pitch/yaw`` (+ optional gripper) to absolute CRP GP.

    Output action is a float tensor of shape ``(6,)`` or ``(7,)``:
    ``[x, y, z, roll, pitch, yaw]`` (+ ``gripper`` GOT0 when ``use_gripper``).

    Accepts 7D policy tensors (xyz+rpy+grip) or legacy 4D (xyz+grip, rpy held).
    """

    robot: CRPArm
    end_effector_step_sizes: dict[str, float] = field(
        default_factory=lambda: {"x": 1.0, "y": 1.0, "z": 1.0}
    )
    use_gripper: bool = True
    use_latched_reference: bool = False
    # Per-step clamp on |δxyz| after step-size scale (mm). ``None`` = no clamp.
    ee_delta_max: float | None = None
    reference_ee_pose: list[float] | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = dict(transition)
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            raise ValueError("Action is required for CRPDeltaEEToAbsoluteGPStep")

        delta_roll = delta_pitch = delta_yaw = 0.0
        if isinstance(action, PolicyAction):
            if action.dim() > 1:
                action = action.squeeze(0)
            delta_x = float(action[0].item())
            delta_y = float(action[1].item())
            delta_z = float(action[2].item())
            if action.numel() >= 6:
                delta_roll = float(action[3].item())
                delta_pitch = float(action[4].item())
                delta_yaw = float(action[5].item())
                gripper = float(action[6].item()) if self.use_gripper and action.numel() > 6 else 1.0
            else:
                gripper = float(action[3].item()) if self.use_gripper and action.numel() > 3 else 1.0
        elif isinstance(action, dict):
            delta_x = float(action.get("delta_x", 0.0))
            delta_y = float(action.get("delta_y", 0.0))
            delta_z = float(action.get("delta_z", 0.0))
            delta_roll = float(action.get("delta_roll", 0.0))
            delta_pitch = float(action.get("delta_pitch", 0.0))
            delta_yaw = float(action.get("delta_yaw", 0.0))
            gripper = float(action.get("gripper", 1.0))
        else:
            raise ValueError(f"Unsupported action type for CRP EE step: {type(action)}")

        # Prefer last commanded pose only while MOVE gate is open (stream / armed).
        # Before GI56 arm-ready, never trust ``_last_ee_cache`` (stale episode-start pose
        # would preload wrong GP and snap when the gate opens — same rule as recording).
        gate_open = True
        if hasattr(self.robot, "is_motion_enabled"):
            gate_open = bool(self.robot.is_motion_enabled())
        cached = getattr(self.robot, "_last_ee_cache", None) or {}
        keys = ("ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw")
        if gate_open and all(k in cached for k in keys):
            current = [float(cached[k]) for k in keys]
        else:
            current = list(self.robot.get_current_endpose(allow_cache_fallback=False))
        if self.use_latched_reference:
            if self.reference_ee_pose is None:
                self.reference_ee_pose = current
            ref = self.reference_ee_pose
        else:
            ref = current

        dxyz = [
            delta_x * float(self.end_effector_step_sizes.get("x", 1.0)),
            delta_y * float(self.end_effector_step_sizes.get("y", 1.0)),
            delta_z * float(self.end_effector_step_sizes.get("z", 1.0)),
        ]
        if self.ee_delta_max is not None:
            cap = float(self.ee_delta_max)
            dxyz = [max(-cap, min(cap, float(v))) for v in dxyz]
        gp6 = apply_xyz_delta_to_endpose(ref, dxyz)
        gp6[3] = float(ref[3]) + delta_roll
        gp6[4] = float(ref[4]) + delta_pitch
        gp6[5] = float(ref[5]) + delta_yaw

        values = list(gp6)
        if self.use_gripper:
            values.append(gripper)
        new_transition[TransitionKey.ACTION] = torch.tensor(values, dtype=torch.float32)
        return new_transition

    def reset(self) -> None:
        self.reference_ee_pose = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        dim = 7 if self.use_gripper else 6
        features[PipelineFeatureType.ACTION][ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(dim,))
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "end_effector_step_sizes": dict(self.end_effector_step_sizes),
            "use_gripper": self.use_gripper,
            "use_latched_reference": self.use_latched_reference,
            "ee_delta_max": self.ee_delta_max,
        }


@ProcessorStepRegistry.register("crp_joint_intervention_gp_assist")
@dataclass
class CRPJointInterventionGPAssistStep(ProcessorStep):
    """Drive CRP with OMY EE→GP during HIL intervention (Space hold).

    Control law matches recording:
    ``crp_xyz = p0 + scale * step_sizes * (omy_now - omy_ref)``, orientation = latched CRP rpy.

    ``rl_label_space``:
      - ``joint``: RL labels are CRP joints (joint HIL); release does GJ hold.
      - ``ee_delta``: RL labels are δxyz[+δrpy]+GOT (4D/7D via ``include_rpy``);
        release cuts with GI56=0 (no pose re-read / no GP·GJ hold re-send).
    """

    robot: CRPArm
    teleop: Any | None = None
    end_effector_step_sizes: dict[str, float] = field(
        default_factory=lambda: {"x": 1.0, "y": 1.0, "z": 1.0}
    )
    ee_delta_scale: float | None = None
    use_gripper: bool = True
    gp_start_index: int = 10
    gp_group_size: int = 5
    gp_secondary_index: int | None = 20
    omy_ee_ready_timeout_s: float = 5.0
    gp_stream_hz: float = 100.0
    ee_delta_log_interval_s: float = 0.5
    # Kept for config compatibility; unused (release is one-shot handoff, not multi-frame stall).
    # joint release: GJ hold; ee_delta release: GI56=0 only (see ``_handoff_to_policy``).
    post_release_hold_steps: int = 0
    # ``joint`` (default) or ``ee_delta`` for RL action labels written into the transition.
    rl_label_space: str = "joint"
    # When ``rl_label_space="ee_delta"``, include δrpy in labels (7D) or only δxyz+grip (4D).
    include_rpy: bool = True

    _trajectory_processor: Any = field(default=None, init=False, repr=False)
    _intervening: bool = field(default=False, init=False, repr=False)
    _crp_p0_xyz: tuple[float, float, float] | None = field(default=None, init=False, repr=False)
    _hold_rpy: tuple[float, float, float] | None = field(default=None, init=False, repr=False)
    _omy_ref_xyz: tuple[float, float, float] | None = field(default=None, init=False, repr=False)
    _resolved_step_sizes: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _resolved_delta_scale: float = field(default=EE_OMY_DELTA_SCALE, init=False, repr=False)
    _gp_cmd_lock: Any = field(default=None, init=False, repr=False)
    _latest_got0: int = field(default=0, init=False, repr=False)
    _gp_stream_stop: Any = field(default=None, init=False, repr=False)
    _gp_stream_thread: Any = field(default=None, init=False, repr=False)
    _mp_stream: MpGpStreamHandle | None = field(default=None, init=False, repr=False)
    # Space-held arming gap (like recording EE wait): no send until OMY ready + GP armed.
    _arming_deadline: float | None = field(default=None, init=False, repr=False)
    _arming_wait_printed: bool = field(default=False, init=False, repr=False)
    _prev_cmd_pose6: tuple[float, float, float, float, float, float] | None = field(
        default=None, init=False, repr=False
    )
    _cut_cb_registered: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._trajectory_processor = TrajectoryProcessor()
        self._gp_cmd_lock = threading.Lock()
        self._gp_stream_stop = threading.Event()
        self._resolved_step_sizes = resolve_ee_step_sizes(self.end_effector_step_sizes)
        self._resolved_delta_scale = resolve_ee_delta_scale(self.ee_delta_scale)
        self._cut_cb_registered = False
        self._register_omy_cut_on_space_release()

    def _register_omy_cut_on_space_release(self) -> None:
        """Space up → immediately signal GP stream stop (do not wait for next env step)."""
        if self._cut_cb_registered or self.teleop is None:
            return
        kb = getattr(self.teleop, "_hil_keyboard", None)
        if kb is None or not hasattr(kb, "add_intervene_end_listener"):
            return

        def _on_space_release() -> None:
            self._signal_omy_cut()

        kb.add_intervene_end_listener(_on_space_release)
        self._cut_cb_registered = True

    def _signal_omy_cut(self) -> None:
        """Async cut: stop GP stream + GI56=0 (safe from keyboard thread; join later).

        Cut means **freeze MOVE** (``set_motion_enabled(False)`` / GI56 off). It does
        **not** re-read pose or re-send a GP/GJ hold command.
        """
        if self._gp_stream_stop is not None:
            self._gp_stream_stop.set()
        handle = self._mp_stream
        if handle is not None:
            for ev_name in ("omy_action_stop", "gp_sender_stop"):
                ev = getattr(handle, ev_name, None)
                if ev is not None and hasattr(ev, "set"):
                    try:
                        ev.set()
                    except Exception:
                        pass
        if hasattr(self.robot, "set_motion_enabled"):
            try:
                self.robot.set_motion_enabled(False)
            except Exception:
                logger.exception("HIL cut: set_motion_enabled(False) / GI56=0 failed")
        logger.info("HIL Space released: OMY cut — GI56=0 (no pose re-read / no hold re-send)")

    def _clear_latches(self) -> None:
        self._intervening = False
        self._crp_p0_xyz = None
        self._hold_rpy = None
        self._omy_ref_xyz = None
        self._prev_cmd_pose6 = None

    def _clear_arming_wait(self) -> None:
        self._arming_deadline = None
        self._arming_wait_printed = False

    def _mark_switch_gap(self, complementary: dict[str, Any]) -> None:
        """Like recording pre-arm gap: do not apply env action and do not write replay."""
        complementary[EXCLUDE_FROM_REPLAY_KEY] = True
        complementary[CRP_GP_COMMAND_SENT_KEY] = True

    def _current_cmd_pose6_grip(self) -> tuple[tuple[float, float, float, float, float, float], float]:
        cached = getattr(self.robot, "_last_ee_cache", None) or {}
        keys = ("ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw")
        if all(k in cached for k in keys):
            pose = tuple(float(cached[k]) for k in keys)
        elif self._crp_p0_xyz is not None and self._hold_rpy is not None:
            pose = (*self._crp_p0_xyz, *self._hold_rpy)
        else:
            p = list(self.robot.get_current_endpose())
            pose = (float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5]))
        grip = float(self._latest_got0)
        if self.use_gripper:
            try:
                joints = (
                    self.robot.get_last_cached_joints()
                    if hasattr(self.robot, "get_last_cached_joints")
                    else {}
                )
                if "gripper.pos" in joints:
                    grip = float(joints["gripper.pos"])
            except Exception:
                pass
        return pose, grip  # type: ignore[return-value]

    def _ee_delta_action_tensor(self) -> torch.Tensor:
        """δxyz[+δrpy] vs previous GP command + absolute GOT."""
        pose, grip = self._current_cmd_pose6_grip()
        if self._prev_cmd_pose6 is None:
            dpose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        else:
            prev = self._prev_cmd_pose6
            dpose = (
                pose[0] - prev[0],
                pose[1] - prev[1],
                pose[2] - prev[2],
                wrap_angle_delta_deg(pose[3], prev[3]),
                wrap_angle_delta_deg(pose[4], prev[4]),
                wrap_angle_delta_deg(pose[5], prev[5]),
            )
        self._prev_cmd_pose6 = pose
        if self.include_rpy:
            values = list(dpose)
        else:
            values = [dpose[0], dpose[1], dpose[2]]
        if self.use_gripper:
            values.append(grip)
        return torch.tensor(values, dtype=torch.float32)

    def _label_action_tensor(self) -> torch.Tensor:
        if self.rl_label_space == "ee_delta":
            return self._ee_delta_action_tensor()
        return self._joint_action_tensor()

    def _joint_action_tensor(self) -> torch.Tensor:
        proprio = self._current_joint_action_dict()
        values = [float(proprio[f"j{i}.pos"]) for i in range(1, 7)]
        if self.use_gripper and "gripper.pos" in proprio:
            values.append(float(proprio["gripper.pos"]))
        return torch.tensor(values, dtype=torch.float32)

    def _gj_hold_current(self, *, reason: str, joints: list[float] | None = None) -> None:
        """Write joints into GJ registers then GI→GJ (both primary + secondary)."""
        if hasattr(self.robot, "hold_current_joints_gj"):
            held = self.robot.hold_current_joints_gj(joints)
            logger.info(
                "HIL %s: GJ hold j=[%.3f %.3f %.3f %.3f %.3f %.3f]",
                reason,
                *held[:6],
            )
            return
        hold = self._current_joint_action_dict()
        self.robot.send_action(hold)
        logger.info(
            "HIL %s: GJ hold j=[%.3f %.3f %.3f %.3f %.3f %.3f]",
            reason,
            hold["j1.pos"],
            hold["j2.pos"],
            hold["j3.pos"],
            hold["j4.pos"],
            hold["j5.pos"],
            hold["j6.pos"],
        )

    def _snapshot_joints_for_handoff(self) -> list[float] | None:
        """Grab j1..j6 while GP stream / proprio cache is still warm (no new SDK race)."""
        if hasattr(self.robot, "get_last_cached_joints"):
            cache = self.robot.get_last_cached_joints()
            try:
                return [float(cache[f"j{i}.pos"]) for i in range(1, 7)]
            except Exception:
                pass
        try:
            proprio = self.robot._read_proprio_observation()
            return [float(proprio[f"j{i}.pos"]) for i in range(1, 7)]
        except Exception:
            return None

    def _read_omy_xyz(self) -> tuple[float, float, float] | None:
        return read_omy_ee_xyz(self.teleop)

    def _wait_omy_xyz(self, timeout_s: float) -> tuple[float, float, float] | None:
        if timeout_s <= 0:
            return self._read_omy_xyz()
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            xyz = self._read_omy_xyz()
            if xyz is not None:
                return xyz
            time.sleep(0.01)
        return None

    def _init_pose6(self) -> list[float]:
        assert self._crp_p0_xyz is not None and self._hold_rpy is not None
        return [
            float(self._crp_p0_xyz[0]),
            float(self._crp_p0_xyz[1]),
            float(self._crp_p0_xyz[2]),
            float(self._hold_rpy[0]),
            float(self._hold_rpy[1]),
            float(self._hold_rpy[2]),
        ]

    def _current_joint_action_dict(self) -> dict[str, float]:
        proprio = self.robot._read_proprio_observation()
        out = {f"j{i}.pos": float(proprio[f"j{i}.pos"]) for i in range(1, 7)}
        if self.use_gripper and "gripper.pos" in proprio:
            out["gripper.pos"] = float(proprio["gripper.pos"])
        return out

    def _init_gp_registers(self, pose6: list[float]) -> None:
        """Preload GP registers at current pose, then switch GI→GP (no stale-register snap)."""
        send_gp_endpose6(
            self.robot,
            self._trajectory_processor,
            pose6,
            start_index=self.gp_start_index,
            group_size=self.gp_group_size,
            switch_to_gp_mode=False,
        )
        if self.gp_secondary_index is not None:
            send_gp_endpose6(
                self.robot,
                self._trajectory_processor,
                pose6,
                start_index=int(self.gp_secondary_index),
                group_size=self.gp_group_size,
                switch_to_gp_mode=False,
            )
        if hasattr(self.robot, "ensure_gp_mode"):
            self.robot.ensure_gp_mode()
        # Arm-ready: open teach-pendant MOVE gate (GI56) with GP mode.
        if hasattr(self.robot, "set_motion_enabled"):
            self.robot.set_motion_enabled(True)

    def _update_gripper_target(self) -> None:
        if not self.use_gripper or self.teleop is None or not hasattr(self.teleop, "get_gripper_raw"):
            return
        got0 = int(omy_rh_r1_to_got0(float(self.teleop.get_gripper_raw())))
        with self._gp_cmd_lock:
            self._latest_got0 = got0

    def _start_gp_stream(self, pose6: list[float]) -> None:
        """Start in-process EE→GP thread using the **same** teleop that latched ``omy_ref``.

        Do **not** use fork/spawn under ``python -m lerobot.rl.actor``: spawn re-imports
        ``lerobot.rl`` (torch) and dies, so the red "ready" banner appears while OMY
        never drives GP. Recording keeps its own MP path; HIL shares parent ROS+SDK.
        """
        self._stop_gp_stream()
        assert self._crp_p0_xyz is not None and self._omy_ref_xyz is not None and self._hold_rpy is not None
        if self.teleop is None:
            raise RuntimeError("HIL GP stream requires teleop (same process as CRP SDK)")
        self._update_gripper_target()
        if hasattr(self.robot, "enable_proprio_cache"):
            self.robot.enable_proprio_cache()
            if hasattr(self.robot, "update_ee_cache_from_pose6"):
                self.robot.update_ee_cache_from_pose6(pose6)
            if hasattr(self.robot, "refresh_proprio_cache"):
                self.robot.refresh_proprio_cache()

        self._gp_stream_stop.clear()
        got_holder = [int(self._latest_got0)]
        p0 = self._crp_p0_xyz
        ref = self._omy_ref_xyz
        hold = self._hold_rpy
        pose = list(pose6)
        steps = dict(self._resolved_step_sizes)
        scale = float(self._resolved_delta_scale)

        def _loop() -> None:
            run_omy_relative_gp_stream(
                robot=self.robot,
                teleop=self.teleop,
                trajectory_processor=self._trajectory_processor,
                stop_event=self._gp_stream_stop,
                p0_xyz=p0,
                hold_rpy=hold,
                omy_ref_xyz=ref,
                init_pose6=pose,
                step_sizes=steps,
                delta_scale=scale,
                start_index=self.gp_start_index,
                group_size=self.gp_group_size,
                stream_hz=self.gp_stream_hz,
                use_gripper=self.use_gripper,
                latest_got0_holder=got_holder,
                log_prefix="HIL OMY EE→CRP",
                log_interval_s=self.ee_delta_log_interval_s,
                relatch_omy_ref_on_start=True,
            )

        self._gp_stream_thread = threading.Thread(
            target=_loop, name="hil_crp_gp_stream", daemon=True
        )
        self._gp_stream_thread.start()
        logger.info(
            "HIL takeover: in-process GP thread @ %.0f Hz (parent OMY+CRP)",
            self.gp_stream_hz,
        )

    def _stop_gp_stream(self) -> None:
        stop_omy_relative_gp_mp_stream(self._mp_stream)
        self._mp_stream = None
        if self._gp_stream_stop is not None:
            self._gp_stream_stop.set()
        thread = self._gp_stream_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._gp_stream_thread = None
        if self._gp_stream_stop is not None:
            self._gp_stream_stop.clear()
        if hasattr(self.robot, "disable_proprio_cache"):
            self.robot.disable_proprio_cache()

    def _try_arm_intervention(self) -> bool:
        """Recording-style arming: wait OMY with **no** send_GPs/send_GJs until ready.

        Called once per env step while Space is held and not yet intervening. Returns
        True only after GP registers are loaded, GI→GP, and the stream thread is alive.
        """
        if self._arming_deadline is None:
            self._arming_deadline = time.monotonic() + float(self.omy_ee_ready_timeout_s)
            self._arming_wait_printed = False

        if not self._arming_wait_printed:
            self._arming_wait_printed = True
            print(
                "\033[1;93m"
                "========================================================================\n"
                "  ◆ HIL Space 切换中：等待 OMY EE（此段不发 GP/GJ、不写入 replay）\n"
                "========================================================================"
                "\033[0m",
                flush=True,
            )
            logger.info(
                "HIL EE arming: waiting for OMY EE (timeout=%.2fs); no motion cmds until armed",
                self.omy_ee_ready_timeout_s,
            )

        omy_xyz = self._read_omy_xyz()
        if omy_xyz is None:
            if time.monotonic() >= float(self._arming_deadline):
                logger.warning(
                    "HIL EE arming: OMY EE still None after %.2fs — keep holding Space",
                    self.omy_ee_ready_timeout_s,
                )
                self._arming_deadline = time.monotonic() + float(self.omy_ee_ready_timeout_s)
            return False

        # OMY ready → clear stale EE cache → latch fresh SDK pose → preload GP (no GI) →
        # GI→GP → start stream (same order as recording; GI56 stays off until preload done).
        if hasattr(self.robot, "clear_ee_cache"):
            self.robot.clear_ee_cache()
        p0 = list(self.robot.get_current_endpose(allow_cache_fallback=False))
        self._crp_p0_xyz = (float(p0[0]), float(p0[1]), float(p0[2]))
        self._hold_rpy = (float(p0[3]), float(p0[4]), float(p0[5]))
        fresh = self._read_omy_xyz()
        self._omy_ref_xyz = fresh if fresh is not None else omy_xyz
        logger.info(
            "HIL EE arming: co-latched p0=%s hold_rpy=%s omy_ref=%s",
            self._crp_p0_xyz,
            self._hold_rpy,
            self._omy_ref_xyz,
        )

        init_pose = self._init_pose6()
        self._init_gp_registers(init_pose)
        if self.teleop is not None and hasattr(self.teleop, "reset_reference"):
            self.teleop.reset_reference()

        self._start_gp_stream(init_pose)
        self._register_omy_cut_on_space_release()
        thread = self._gp_stream_thread
        if thread is None or not thread.is_alive():
            logger.error("HIL EE arming: GP stream thread failed to start")
            self._stop_gp_stream()
            self._clear_latches()
            self._clear_arming_wait()
            return False

        self._intervening = True
        self._clear_arming_wait()
        logger.info(
            "HIL EE armed ACTIVE — CRP holds p0 until OMY moves relative to omy_ref",
        )
        print(
            "\033[1;91m"
            "========================================================================\n"
            "  ★★★  HIL Space 介入已就绪：可以动 OMY  ★★★\n"
            "========================================================================"
            "\033[0m",
            flush=True,
        )
        return True

    def _handoff_to_policy(self) -> None:
        """Space release: stop OMY stream + GI56=0; this gap sends nothing."""
        if self.rl_label_space != "ee_delta":
            snap = self._snapshot_joints_for_handoff()
            if snap is not None:
                logger.info(
                    "HIL release: joint snapshot before stop GP j=[%.3f %.3f %.3f %.3f %.3f %.3f]",
                    *snap,
                )
            self._stop_gp_stream()
            self._gj_hold_current(reason="release→policy @ current", joints=snap)
            self._clear_latches()
            self._clear_arming_wait()
            logger.info("HIL release: handoff complete — next step policy acts from current obs")
            print(
                "\033[1;96m"
                "========================================================================\n"
                "  ◆ HIL 松 Space：已切断 OMY，GJ hold，下步起策略接管\n"
                "========================================================================"
                "\033[0m",
                flush=True,
            )
            return

        # EE path: cut = GI56=0 + stop stream. No pose re-read, no GP/GJ hold re-send.
        self._signal_omy_cut()
        self._stop_gp_stream()
        if hasattr(self.robot, "set_motion_enabled"):
            self.robot.set_motion_enabled(False)
        self._clear_latches()
        self._clear_arming_wait()
        logger.info("HIL release (ee_delta): GI56=0 + stream stopped — no hold re-send (no-op gap)")
        print(
            "\033[1;96m"
            "========================================================================\n"
            "  ◆ HIL 松 Space：GI56=0 切断运动，不重读/重发 hold，本段不做任何动作\n"
            "========================================================================"
            "\033[0m",
            flush=True,
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        info = transition.get(TransitionKey.INFO, {})
        is_intervention = bool(info.get(TeleopEvents.IS_INTERVENTION, False))
        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {})
        was_intervening = self._intervening
        new_transition = dict(transition)

        complementary.pop(EXCLUDE_FROM_REPLAY_KEY, None)
        complementary.pop(CRP_GP_COMMAND_SENT_KEY, None)

        # Space released while armed → handoff (ee: GI56=0; joint: GJ hold); gap not in replay.
        if was_intervening and not is_intervention:
            self._handoff_to_policy()
            self._mark_switch_gap(complementary)
            label = self._label_action_tensor()
            new_transition[TransitionKey.ACTION] = label
            complementary[TELEOP_ACTION_KEY] = label
            return _transition_with_complementary(new_transition, complementary)

        # Space released while still arming (never got ready) → cancel wait, no sends.
        if not is_intervention and self._arming_deadline is not None:
            self._clear_arming_wait()
            self._clear_latches()
            self._mark_switch_gap(complementary)
            return _transition_with_complementary(new_transition, complementary)

        if not is_intervention:
            return _transition_with_complementary(new_transition, complementary)

        # Space held, not yet intervening → arming gap (no send until OMY+GP ready).
        if (
            not self._intervening
            or self._crp_p0_xyz is None
            or self._omy_ref_xyz is None
            or self._hold_rpy is None
        ):
            ready = self._try_arm_intervention()
            self._mark_switch_gap(complementary)
            if not ready:
                self._clear_latches()
            else:
                # Seed prev command so first labeled delta starts from latched pose.
                assert self._crp_p0_xyz is not None and self._hold_rpy is not None
                self._prev_cmd_pose6 = (*self._crp_p0_xyz, *self._hold_rpy)
            label = self._label_action_tensor()
            new_transition[TransitionKey.ACTION] = label
            complementary[TELEOP_ACTION_KEY] = label
            return _transition_with_complementary(new_transition, complementary)

        # Active intervention: stream owns GP; skip env send; record transitions normally.
        self._update_gripper_target()
        complementary[CRP_GP_COMMAND_SENT_KEY] = True
        label = self._label_action_tensor()
        new_transition[TransitionKey.ACTION] = label
        complementary[TELEOP_ACTION_KEY] = label
        return _transition_with_complementary(new_transition, complementary)

    def reset(self) -> None:
        self._stop_gp_stream()
        self._clear_latches()
        self._clear_arming_wait()

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "end_effector_step_sizes": dict(self.end_effector_step_sizes),
            "ee_delta_scale": self.ee_delta_scale,
            "use_gripper": self.use_gripper,
            "gp_start_index": self.gp_start_index,
            "gp_group_size": self.gp_group_size,
            "gp_secondary_index": self.gp_secondary_index,
            "omy_ee_ready_timeout_s": self.omy_ee_ready_timeout_s,
            "gp_stream_hz": self.gp_stream_hz,
            "ee_delta_log_interval_s": self.ee_delta_log_interval_s,
            "post_release_hold_steps": self.post_release_hold_steps,
            "rl_label_space": self.rl_label_space,
            "include_rpy": self.include_rpy,
        }
