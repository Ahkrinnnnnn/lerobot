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

"""Unit / concurrency tests for CRP OMY EE→GP shared path (recording + HIL)."""

from __future__ import annotations

import inspect
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from lerobot.robots.crp_arm.crp_arm import CRPArm
from lerobot.robots.crp_arm import ee_gp
from lerobot.robots.crp_arm.ee_gp import (
    EE_OMY_DELTA_SCALE,
    apply_xyz_delta_to_endpose,
    omy_relative_xyz_to_gp6,
    omy_rh_r1_to_got0,
    read_omy_ee_xyz,
    run_omy_relative_gp_stream,
    start_omy_relative_gp_mp_stream,
)
from lerobot.robots.crp_arm.hil_ee_processor import (
    CRP_GP_COMMAND_SENT_KEY,
    CRPDeltaEEToAbsoluteGPStep,
    CRPJointInterventionGPAssistStep,
    EXCLUDE_FROM_REPLAY_KEY,
)
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.tools import TrajectoryProcessor
from lerobot.types import TransitionKey


class _FakeTeleop:
    def __init__(self, xyz=(0.0, 0.0, 0.0), grip_raw: float = 0.0):
        self.xyz = tuple(float(x) for x in xyz)
        self.grip_raw = float(grip_raw)
        self.reset_calls = 0

    def get_ros_end_effector_xyz_rpy_deg(self):
        return (self.xyz, (0.0, 0.0, 0.0))

    def get_gripper_raw(self) -> float:
        return self.grip_raw

    def reset_reference(self) -> None:
        self.reset_calls += 1


class _RaceDetectingRobot:
    """Minimal robot: detects overlapping SDK calls (simulates unprotected concurrent access)."""

    def __init__(self):
        self.is_connected = True
        self._inflight = 0
        self._max_inflight = 0
        self._gate = threading.Lock()
        self._motion_cmd_lock = threading.RLock()
        self.sent_gps: list[tuple[int, list]] = []
        self.got_values: list[int] = []
        self.endpose = [100.0, 200.0, 300.0, 1.5, -0.5, 2.0]
        self.joints = {f"j{i}": float(i) for i in range(1, 7)}
        self.force_gp_fail = False
        self.send_action_calls = 0

    def _enter_sdk(self) -> None:
        with self._gate:
            self._inflight += 1
            self._max_inflight = max(self._max_inflight, self._inflight)
        # Hold briefly so a racing caller can overlap if unlocked.
        time.sleep(0.002)

    def _leave_sdk(self) -> None:
        with self._gate:
            self._inflight -= 1

    def send_GPs(self, start_index: int, GPs, *, switch_to_gp_mode: bool = True) -> None:
        with self._motion_cmd_lock:
            self._enter_sdk()
            try:
                if self.force_gp_fail:
                    raise RuntimeError("simulated send_GPs failure")
                self.sent_gps.append((int(start_index), list(GPs)))
                self.last_switch_to_gp_mode = bool(switch_to_gp_mode)
            finally:
                self._leave_sdk()

    def ensure_gp_mode(self) -> None:
        self.gp_mode_ensured = getattr(self, "gp_mode_ensured", 0) + 1

    def ensure_gj_mode(self) -> None:
        self.gj_mode_ensured = getattr(self, "gj_mode_ensured", 0) + 1

    def set_motion_enabled(self, enabled: bool) -> None:
        self.motion_enabled = bool(enabled)
        self.motion_enable_calls = getattr(self, "motion_enable_calls", []) + [bool(enabled)]

    def is_motion_enabled(self) -> bool:
        return bool(getattr(self, "motion_enabled", True))

    def set_GOT(self, index: int, value: int) -> bool:
        with self._motion_cmd_lock:
            self._enter_sdk()
            try:
                self.got_values.append(int(value))
                return True
            finally:
                self._leave_sdk()

    def get_current_endpose(self, *, allow_cache_fallback: bool = False) -> list[float]:
        del allow_cache_fallback  # mock has no EE cache; accept CRPArm signature
        with self._motion_cmd_lock:
            self._enter_sdk()
            try:
                return list(self.endpose)
            finally:
                self._leave_sdk()

    def _read_proprio_observation(self) -> dict[str, float]:
        with self._motion_cmd_lock:
            self._enter_sdk()
            try:
                out = {f"{k}.pos": float(v) for k, v in self.joints.items()}
                out["gripper.pos"] = 0.0
                return out
            finally:
                self._leave_sdk()

    def send_action(self, action: dict) -> dict:
        with self._motion_cmd_lock:
            self._enter_sdk()
            try:
                self.send_action_calls += 1
                return dict(action)
            finally:
                self._leave_sdk()

    def enable_proprio_cache(self) -> None:
        pass

    def disable_proprio_cache(self) -> None:
        pass

    def update_ee_cache_from_pose6(self, pose6) -> None:
        pass

    def refresh_proprio_cache(self) -> bool:
        return True


def _first_gp_vec(sent_row) -> list[float]:
    """``send_GPs`` stores either a flat list or list-of-rows from TrajectoryProcessor."""
    mat = sent_row[1]
    if not mat:
        return []
    if isinstance(mat[0], (list, tuple)):
        return [float(x) for x in mat[0]]
    # Flat: take first 6
    return [float(x) for x in mat[:6]]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_no_handwritten_fixed_rpy_constants():
    assert not hasattr(ee_gp, "FIXED_ROLL_DEG")
    assert not hasattr(ee_gp, "FIXED_PITCH_DEG")
    assert not hasattr(ee_gp, "FIXED_YAW_DEG")
    src = Path(ee_gp.__file__).read_text(encoding="utf-8")
    assert "FIXED_ROLL_DEG" not in src
    assert "FIXED_ROLL" not in src
    assert "hold_rpy" in src


def test_omy_relative_zero_delta_holds_pose():
    p0 = (10.0, 20.0, 30.0)
    hold = (-12.0, 3.0, 4.0)
    ref = (1.0, 2.0, 3.0)
    gp6 = omy_relative_xyz_to_gp6(p0, ref, ref, hold_rpy=hold)
    assert gp6 == pytest.approx([10.0, 20.0, 30.0, -12.0, 3.0, 4.0])


def test_omy_relative_applies_xyz_only_with_step_sizes():
    p0 = (0.0, 0.0, 0.0)
    hold = (7.0, 8.0, 9.0)
    ref = (0.0, 0.0, 0.0)
    now = (1.0, 2.0, 3.0)
    gp6 = omy_relative_xyz_to_gp6(
        p0,
        now,
        ref,
        hold_rpy=hold,
        step_sizes={"x": -1.0, "y": 1.0, "z": -1.0},
        scale=1.0,
    )
    assert gp6[:3] == pytest.approx([-1.0, 2.0, -3.0])
    assert gp6[3:] == pytest.approx([7.0, 8.0, 9.0])


def test_apply_xyz_delta_defaults_to_p0_rpy():
    p0 = [1.0, 2.0, 3.0, 11.0, 22.0, 33.0]
    out = apply_xyz_delta_to_endpose(p0, [0.5, 0.0, 0.0])
    assert out == pytest.approx([1.5, 2.0, 3.0, 11.0, 22.0, 33.0])


def test_ee_omy_delta_scale_matches_fork():
    assert EE_OMY_DELTA_SCALE == 1.5


def test_read_omy_ee_xyz_none_safe():
    assert read_omy_ee_xyz(None) is None
    assert read_omy_ee_xyz(SimpleNamespace()) is None
    teleop = _FakeTeleop(xyz=(4.0, 5.0, 6.0))
    assert read_omy_ee_xyz(teleop) == pytest.approx((4.0, 5.0, 6.0))


def test_omy_rh_r1_to_got0_bounds():
    assert 0 <= omy_rh_r1_to_got0(0.0) <= 1000
    lo = omy_rh_r1_to_got0(-100.0)
    hi = omy_rh_r1_to_got0(100.0)
    assert lo == 0
    assert hi == 1000


# ---------------------------------------------------------------------------
# Shared stream safety
# ---------------------------------------------------------------------------


def test_stream_stationary_omy_keeps_init_pose():
    robot = _RaceDetectingRobot()
    teleop = _FakeTeleop(xyz=(1.0, 2.0, 3.0))
    traj = TrajectoryProcessor()
    stop = threading.Event()
    p0 = (100.0, 200.0, 300.0)
    hold = (1.5, -0.5, 2.0)
    ref = (1.0, 2.0, 3.0)
    init = [100.0, 200.0, 300.0, 1.5, -0.5, 2.0]

    t = threading.Thread(
        target=run_omy_relative_gp_stream,
        kwargs={
            "robot": robot,
            "teleop": teleop,
            "trajectory_processor": traj,
            "stop_event": stop,
            "p0_xyz": p0,
            "hold_rpy": hold,
            "omy_ref_xyz": ref,
            "init_pose6": init,
            "stream_hz": 50.0,
            "use_gripper": False,
            "log_interval_s": 10.0,
        },
        daemon=True,
    )
    t.start()
    time.sleep(0.12)
    stop.set()
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert len(robot.sent_gps) >= 2
    for row in robot.sent_gps:
        vec = _first_gp_vec(row)
        assert vec[:3] == pytest.approx([100.0, 200.0, 300.0], abs=1e-6)
        assert vec[3:] == pytest.approx([1.5, -0.5, 2.0], abs=1e-6)


def test_stream_follows_omy_delta_without_rotating():
    robot = _RaceDetectingRobot()
    teleop = _FakeTeleop(xyz=(0.0, 0.0, 0.0))
    traj = TrajectoryProcessor()
    stop = threading.Event()
    p0 = (0.0, 0.0, 0.0)
    hold = (9.0, 8.0, 7.0)
    ref = (0.0, 0.0, 0.0)
    init = [0.0, 0.0, 0.0, 9.0, 8.0, 7.0]

    t = threading.Thread(
        target=run_omy_relative_gp_stream,
        kwargs={
            "robot": robot,
            "teleop": teleop,
            "trajectory_processor": traj,
            "stop_event": stop,
            "p0_xyz": p0,
            "hold_rpy": hold,
            "omy_ref_xyz": ref,
            "init_pose6": init,
            "stream_hz": 40.0,
            "use_gripper": False,
            "log_interval_s": 10.0,
            "relatch_omy_ref_on_start": False,
            "delta_scale": 1.0,
        },
        daemon=True,
    )
    t.start()
    time.sleep(0.05)
    teleop.xyz = (0.1, -0.2, 0.3)
    time.sleep(0.1)
    stop.set()
    t.join(timeout=2.0)

    last = _first_gp_vec(robot.sent_gps[-1])
    assert last[:3] == pytest.approx([0.1, -0.2, 0.3], abs=1e-5)
    assert last[3:] == pytest.approx([9.0, 8.0, 7.0], abs=1e-5)


def test_cross_teleop_ref_mismatch_is_the_old_bug():
    """Document why spawn teleop B + main ref A caused drift with OMY still."""
    p0 = (0.0, 0.0, 0.0)
    hold = (0.0, 0.0, 0.0)
    main_ref = (10.0, 20.0, 30.0)
    spawn_now = (10.5, 20.0, 30.0)  # tiny subscriber mismatch
    gp6 = omy_relative_xyz_to_gp6(p0, spawn_now, main_ref, hold_rpy=hold)
    assert abs(gp6[0]) > 0.4  # fake delta — must not happen when ref==same teleop


def test_stream_stops_on_disconnect():
    robot = _RaceDetectingRobot()
    teleop = _FakeTeleop()
    traj = TrajectoryProcessor()
    stop = threading.Event()
    init = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    t = threading.Thread(
        target=run_omy_relative_gp_stream,
        kwargs={
            "robot": robot,
            "teleop": teleop,
            "trajectory_processor": traj,
            "stop_event": stop,
            "p0_xyz": (0.0, 0.0, 0.0),
            "hold_rpy": (0.0, 0.0, 0.0),
            "omy_ref_xyz": (0.0, 0.0, 0.0),
            "init_pose6": init,
            "stream_hz": 50.0,
            "use_gripper": False,
            "log_interval_s": 10.0,
        },
        daemon=True,
    )
    t.start()
    time.sleep(0.05)
    robot.is_connected = False
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert stop.is_set()


def test_stream_stops_on_send_gps_failure():
    robot = _RaceDetectingRobot()
    robot.force_gp_fail = True
    teleop = _FakeTeleop()
    traj = TrajectoryProcessor()
    stop = threading.Event()
    t = threading.Thread(
        target=run_omy_relative_gp_stream,
        kwargs={
            "robot": robot,
            "teleop": teleop,
            "trajectory_processor": traj,
            "stop_event": stop,
            "p0_xyz": (0.0, 0.0, 0.0),
            "hold_rpy": (0.0, 0.0, 0.0),
            "omy_ref_xyz": (0.0, 0.0, 0.0),
            "init_pose6": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "stream_hz": 50.0,
            "use_gripper": False,
            "log_interval_s": 10.0,
        },
        daemon=True,
    )
    t.start()
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert stop.is_set()


def test_stream_and_proprio_under_robot_lock_no_overlap():
    """GP stream + observation reads serialize via robot._motion_cmd_lock."""
    robot = _RaceDetectingRobot()
    teleop = _FakeTeleop(xyz=(0.0, 0.0, 0.0))
    traj = TrajectoryProcessor()
    stop = threading.Event()
    t = threading.Thread(
        target=run_omy_relative_gp_stream,
        kwargs={
            "robot": robot,
            "teleop": teleop,
            "trajectory_processor": traj,
            "stop_event": stop,
            "p0_xyz": (0.0, 0.0, 0.0),
            "hold_rpy": (0.0, 0.0, 0.0),
            "omy_ref_xyz": (0.0, 0.0, 0.0),
            "init_pose6": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "stream_hz": 80.0,
            "use_gripper": True,
            "log_interval_s": 10.0,
        },
        daemon=True,
    )
    t.start()

    def _hammer_proprio():
        for _ in range(40):
            robot._read_proprio_observation()
            time.sleep(0.001)

    readers = [threading.Thread(target=_hammer_proprio, daemon=True) for _ in range(3)]
    for r in readers:
        r.start()
    time.sleep(0.15)
    stop.set()
    t.join(timeout=2.0)
    for r in readers:
        r.join(timeout=2.0)
    assert robot._max_inflight == 1


def test_got0_holder_updated_under_lock():
    robot = _RaceDetectingRobot()
    teleop = _FakeTeleop(xyz=(0.0, 0.0, 0.0), grip_raw=0.05)
    traj = TrajectoryProcessor()
    stop = threading.Event()
    holder = [0]
    lock = threading.Lock()
    t = threading.Thread(
        target=run_omy_relative_gp_stream,
        kwargs={
            "robot": robot,
            "teleop": teleop,
            "trajectory_processor": traj,
            "stop_event": stop,
            "p0_xyz": (0.0, 0.0, 0.0),
            "hold_rpy": (0.0, 0.0, 0.0),
            "omy_ref_xyz": (0.0, 0.0, 0.0),
            "init_pose6": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "stream_hz": 40.0,
            "use_gripper": True,
            "latest_got0_holder": holder,
            "got0_lock": lock,
            "log_interval_s": 10.0,
        },
        daemon=True,
    )
    t.start()
    time.sleep(0.1)
    with lock:
        got = holder[0]
    stop.set()
    t.join(timeout=2.0)
    assert got == omy_rh_r1_to_got0(0.05)
    assert robot.got_values


# ---------------------------------------------------------------------------
# HIL processor collaboration
# ---------------------------------------------------------------------------


def test_hil_latches_current_rpy_not_fixed_constant():
    robot = _RaceDetectingRobot()
    robot.endpose = [1.0, 2.0, 3.0, 12.0, -3.0, 4.0]
    teleop = _FakeTeleop(xyz=(5.0, 6.0, 7.0))
    step = CRPJointInterventionGPAssistStep(robot=robot, teleop=teleop, use_gripper=True)
    step.gp_stream_hz = 30.0
    ok = step._try_arm_intervention()
    try:
        assert ok
        assert step._crp_p0_xyz == pytest.approx((1.0, 2.0, 3.0))
        assert step._hold_rpy == pytest.approx((12.0, -3.0, 4.0))
        assert step._omy_ref_xyz == pytest.approx((5.0, 6.0, 7.0))
        assert teleop.reset_calls == 1
        time.sleep(0.08)
        assert robot.sent_gps
        vec = _first_gp_vec(robot.sent_gps[-1])
        # OMY still at ref → xyz stays at p0, rpy at latched hold
        assert vec[:3] == pytest.approx([1.0, 2.0, 3.0], abs=1e-5)
        assert vec[3:] == pytest.approx([12.0, -3.0, 4.0], abs=1e-5)
    finally:
        step.reset()


def test_hil_intervention_flags_and_skips_gj():
    robot = _RaceDetectingRobot()
    teleop = _FakeTeleop(xyz=(0.0, 0.0, 0.0))
    step = CRPJointInterventionGPAssistStep(robot=robot, teleop=teleop, use_gripper=False)
    step.gp_stream_hz = 25.0
    transition = {
        TransitionKey.INFO: {TeleopEvents.IS_INTERVENTION: True},
        TransitionKey.COMPLEMENTARY_DATA: {},
        TransitionKey.ACTION: torch.zeros(7),
    }
    out = step(transition)
    try:
        assert out[TransitionKey.COMPLEMENTARY_DATA][CRP_GP_COMMAND_SENT_KEY] is True
        assert out[TransitionKey.COMPLEMENTARY_DATA][EXCLUDE_FROM_REPLAY_KEY] is True
        assert step._intervening
    finally:
        step.reset()


def test_hil_release_stops_stream():
    robot = _RaceDetectingRobot()
    teleop = _FakeTeleop()
    step = CRPJointInterventionGPAssistStep(robot=robot, teleop=teleop, use_gripper=False)
    step.gp_stream_hz = 25.0
    assert step._try_arm_intervention()
    assert step._gp_stream_thread is not None  # in-process thread
    step._handoff_to_policy()
    assert step._gp_stream_thread is None
    assert not step._intervening
    assert robot.send_action_calls >= 1


def test_hil_space_release_cuts_stream_before_next_env_step():
    """pynput/tty intervene-end must stop OMY→GP immediately (not wait for actor step)."""
    from lerobot.teleoperators.keyboard_hil_events import KeyboardHilEvents

    robot = _RaceDetectingRobot()
    teleop = _FakeTeleop(xyz=(0.0, 0.0, 0.0))
    kb = KeyboardHilEvents(use_pynput=False, use_tty=False)
    teleop._hil_keyboard = kb
    kb._intervening = True  # armed Space hold

    step = CRPJointInterventionGPAssistStep(
        robot=robot, teleop=teleop, use_gripper=False, rl_label_space="ee_delta"
    )
    step.gp_stream_hz = 50.0
    assert step._try_arm_intervention()
    thread = step._gp_stream_thread
    assert thread is not None and thread.is_alive()
    n_before = len(robot.sent_gps)

    # Simulate Space up without going through action_processor / env step.
    with kb._lock:
        notify = kb._clear_intervening_locked(notify=True)
    if notify:
        kb._notify_intervene_end()

    assert step._gp_stream_stop.is_set()
    assert robot.motion_enabled is False  # cut = GI56=0, not re-send hold
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    n_after = len(robot.sent_gps)
    time.sleep(0.05)
    assert len(robot.sent_gps) == n_after  # no further OMY→GP after cut
    assert n_after >= n_before

    # Handoff gap: join + GI56 stays off; no GJ/GP hold re-send.
    n_gps = len(robot.sent_gps)
    robot.send_action_calls = 0
    out = step(
        {
            TransitionKey.INFO: {TeleopEvents.IS_INTERVENTION: False},
            TransitionKey.COMPLEMENTARY_DATA: {},
            TransitionKey.ACTION: torch.zeros(4),
        }
    )
    assert out[TransitionKey.COMPLEMENTARY_DATA][CRP_GP_COMMAND_SENT_KEY] is True
    assert out[TransitionKey.COMPLEMENTARY_DATA][EXCLUDE_FROM_REPLAY_KEY] is True
    assert robot.send_action_calls == 0
    assert len(robot.sent_gps) == n_gps
    assert robot.motion_enabled is False
    step.reset()


def test_hil_stream_uses_shared_ee_gp_function():
    src = inspect.getsource(CRPJointInterventionGPAssistStep._start_gp_stream)
    # HIL must use in-process thread + parent teleop (spawn under rl.actor reimports torch and dies).
    assert "run_omy_relative_gp_stream" in src
    assert "start_omy_relative_gp_mp_stream" not in src
    assert "hil_crp_gp_stream" in src


def test_hil_switch_gap_no_send_until_ready():
    """Arming must not send GJ; joint release GJ-holds; ee_delta cut = GI56=0 (no hold re-send)."""
    arm_src = inspect.getsource(CRPJointInterventionGPAssistStep._try_arm_intervention)
    assert "hold_current_joints_gj" not in arm_src
    assert "send_action" not in arm_src
    assert "_mark_switch_gap" in inspect.getsource(CRPJointInterventionGPAssistStep.__call__)
    rel_src = inspect.getsource(CRPJointInterventionGPAssistStep._handoff_to_policy)
    assert "_gj_hold_current" in rel_src
    assert "_stop_gp_stream" in rel_src
    assert "release→policy" in rel_src
    assert "GI56=0" in rel_src
    assert "_signal_omy_cut" in rel_src
    cut_src = inspect.getsource(CRPJointInterventionGPAssistStep._signal_omy_cut)
    assert "set_motion_enabled(False)" in cut_src


def test_hil_release_handoff_holds_current_then_policy():
    robot = _RaceDetectingRobot()
    teleop = _FakeTeleop()
    step = CRPJointInterventionGPAssistStep(robot=robot, teleop=teleop, use_gripper=False)
    step.gp_stream_hz = 25.0
    assert step._try_arm_intervention()
    robot.send_action_calls = 0
    out = step(
        {
            TransitionKey.INFO: {TeleopEvents.IS_INTERVENTION: False},
            TransitionKey.COMPLEMENTARY_DATA: {},
            TransitionKey.ACTION: torch.zeros(7),
        }
    )
    assert out[TransitionKey.COMPLEMENTARY_DATA][EXCLUDE_FROM_REPLAY_KEY] is True
    assert out[TransitionKey.COMPLEMENTARY_DATA][CRP_GP_COMMAND_SENT_KEY] is True
    assert robot.send_action_calls >= 1
    assert not step._intervening
    # Next non-intervention frame: no exclude — policy may send_GJs
    out2 = step(
        {
            TransitionKey.INFO: {TeleopEvents.IS_INTERVENTION: False},
            TransitionKey.COMPLEMENTARY_DATA: {},
            TransitionKey.ACTION: torch.zeros(7),
        }
    )
    assert EXCLUDE_FROM_REPLAY_KEY not in out2[TransitionKey.COMPLEMENTARY_DATA]
    assert CRP_GP_COMMAND_SENT_KEY not in out2[TransitionKey.COMPLEMENTARY_DATA]
    step.reset()

def test_recording_script_uses_inline_fork_spawn():
    """Recording uses inline fork GP + spawn OMY with ready/armed protocol and hold_rpy latch."""
    import lerobot.scripts.crp_record_omy_ee_inc as rec

    src = Path(rec.__file__).read_text(encoding="utf-8")
    assert "gp_sender_process.start()" in src
    assert "omy_action_update_process.start()" in src
    assert "_spawn_omy_ee_gp_action_worker" in src
    assert "enable_proprio_cache" in src
    assert "hold_rpy" in src
    assert "shared_hold_rpy" in src
    assert "omy_stream_ready" in src
    assert "omy_gp_armed" in src
    assert "wait_stable_omy_ee_xyz" in src
    assert "resolve_ee_step_sizes" in src
    assert "hold_current_joints_gj" not in src or "hold_current_joints_gj()" not in src
    # Teardown must not send GJ / switch GI (that is motion before next ready).
    assert "GJ hold current" not in src
    assert "switch_to_gp_mode=False" in src
    assert "ensure_gp_mode" in src
    # Fork must stream init pose before omy_gp_armed releases OMY deltas.
    assert src.index("gp_sender_process.start()") < src.index("omy_gp_armed.set()")
    assert "re-latched omy_ref" in src
    assert "get_gripper_raw" in src
    assert "GOT_FOLLOW_DELTA" in src
    assert "last_got0_sent" in src
    assert "_print_teleop_ready_banner" in src
    # Shared control helpers come from ee_gp (single source of truth with HIL).
    assert "from lerobot.robots.crp_arm.ee_gp import" in src
    assert "omy_relative_xyz_to_gp6" in src
    assert "send_gp_endpose6" in src


def test_hold_current_joints_gj_helper_exists_for_hil():
    """Helper remains on CRPArm for HIL release; recording must not call it pre-ready."""
    src = inspect.getsource(CRPArm.hold_current_joints_gj)
    assert "read_joints" in src
    assert "send_GJs" in src
    assert "write_joint" in src
    assert "switch_to_gj_mode=False" in src
    assert "ensure_gj_mode" in src
    assert "joints" in src  # optional snapshot arg
    import lerobot.scripts.crp_record_omy_ee_inc as rec

    rec_src = Path(rec.__file__).read_text(encoding="utf-8")
    assert "robot.hold_current_joints_gj()" not in rec_src


def test_hil_handoff_snapshots_before_stop():
    src = inspect.getsource(CRPJointInterventionGPAssistStep._handoff_to_policy)
    assert "_snapshot_joints_for_handoff" in src
    assert src.index("_snapshot_joints_for_handoff") < src.index("_stop_gp_stream")


def test_mp_stream_matches_fork_stop_event_split():
    """GP sender stop must be fork Event; OMY stop must be Manager Event (not shared SemLock)."""
    src = inspect.getsource(start_omy_relative_gp_mp_stream)
    assert "omy_action_stop = manager.Event()" in src
    assert "gp_sender_stop = fork_ctx.Event()" in src
    assert "fork_ctx.Process" in src
    assert "spawn_ctx.Process" in src
    assert "_spawn_omy_gp6_worker" in src


def test_hil_init_gp_registers_preload_then_ensure_gp_mode():
    src = inspect.getsource(CRPJointInterventionGPAssistStep._init_gp_registers)
    assert "switch_to_gp_mode=False" in src
    assert "ensure_gp_mode" in src
    assert "set_motion_enabled" in src
    assert "send_gp_endpose6" in src


def test_record_arm_ready_opens_motion_enable_gi():
    """Recording: GI56 ON, then fork, then omy_gp_armed (deltas after sender is live)."""
    rec = Path(__file__).resolve().parents[2] / "src/lerobot/scripts/crp_record_omy_ee_inc.py"
    src = rec.read_text(encoding="utf-8")
    assert "set_motion_enabled(True)" in src
    assert "omy_gp_armed.set()" in src
    assert src.index("set_motion_enabled(True)") < src.index("gp_sender_process.start()")
    assert src.index("gp_sender_process.start()") < src.index("omy_gp_armed.set()")
    assert "set_motion_enabled(False)" in src
    # Teardown must stop fork before writing GI56 (SDK race → left-key abort).
    tear = src.index("Stop EE/GP workers FIRST")
    stop_idx = src.index("gp_sender_stop.set()", tear)
    disable_idx = src.index("set_motion_enabled(False)", tear)
    assert stop_idx < disable_idx
    assert "got_sent != last_got0_sent" in src


def test_crp_arm_set_motion_enabled_writes_gi56(monkeypatch):
    """CRPArm.set_motion_enabled maps to motion_enable_gi_index (default 56)."""
    from lerobot.robots.crp_arm.config_crp_arm import CRPArmConfig

    calls: list[tuple[int, int]] = []

    class _Sdk:
        def set_GI(self, index, value):
            calls.append((int(index), int(value)))

    arm = CRPArm.__new__(CRPArm)
    arm.config = CRPArmConfig(port="dummy", motion_enable_gi_index=56)
    arm.crp_arm_robot = _Sdk()
    arm._motion_cmd_lock = __import__("threading").RLock()
    arm._motion_enable_gi_cached = None

    arm.set_motion_enabled(False)
    arm.set_motion_enabled(True)
    arm.set_motion_enabled(True)  # cached no-op
    assert calls == [(56, 0), (56, 1)]


def test_ee_gp_exports_shared_helpers():
    from lerobot.robots.crp_arm import ee_gp

    assert hasattr(ee_gp, "omy_relative_xyz_to_gp6")
    assert hasattr(ee_gp, "resolve_ee_step_sizes")
    assert hasattr(ee_gp, "wait_stable_omy_ee_xyz")
    assert hasattr(ee_gp, "start_omy_relative_gp_mp_stream")
    assert not hasattr(ee_gp, "start_omy_relative_gp_forklike_stream")
    assert ee_gp.resolve_ee_step_sizes({"x": -1.0})["x"] == -1.0
    assert ee_gp.resolve_ee_step_sizes({"x": -1.0})["y"] == 1.0
    assert ee_gp.resolve_ee_delta_scale(None) == ee_gp.EE_OMY_DELTA_SCALE



def test_delta_ee_step_holds_reference_orientation():
    robot = MagicMock()
    robot.get_current_endpose.return_value = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]
    step = CRPDeltaEEToAbsoluteGPStep(
        robot=robot,
        end_effector_step_sizes={"x": 1.0, "y": 1.0, "z": 1.0},
        use_gripper=False,
        use_latched_reference=True,
    )
    transition = {TransitionKey.ACTION: torch.tensor([0.5, 0.0, 0.0], dtype=torch.float32)}
    out = step(transition)
    act = out[TransitionKey.ACTION]
    assert act[0].item() == pytest.approx(1.5)
    assert act[3].item() == pytest.approx(10.0)
    assert act[4].item() == pytest.approx(20.0)
    assert act[5].item() == pytest.approx(30.0)
    assert not hasattr(step, "fixed_rpy")
