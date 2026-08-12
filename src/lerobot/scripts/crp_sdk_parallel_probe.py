#!/usr/bin/env python
"""Safety-first probe: can CRP SDK tolerate concurrent parent+fork calls?

Default is **read-only** (no ``set_GPs`` / ``set_GJs`` / motion). Optional hold-GP
writes require an explicit risk flag and only rewrite the **current** measured pose.

Safety rules baked in:
  - Never command zeros / unknown poses.
  - Never ``movej`` / ``movel`` / open MOVE gate (GI56 left OFF when writable).
  - Cap duration; require interactive confirmation for any write mode.
  - Ctrl+C / errors → stop child, disconnect.

```bash
# Safest: concurrent reads only (recommended first)
python -m lerobot.scripts.crp_sdk_parallel_probe --port=192.168.0.100

# Same + parent also reads EE
python -m lerobot.scripts.crp_sdk_parallel_probe --port=192.168.0.100 --also_ee

# Opt-in: fork rewrites current EE into GP registers (hold). Pendant must NOT be
# chasing a distant GP; keep e-stop ready. Requires typing YES.
python -m lerobot.scripts.crp_sdk_parallel_probe --port=192.168.0.100 \\
  --writes=hold_gp --i-understand-risk
```
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any

from lerobot.robots.crp_arm._sdk import import_crp_robot_py
from lerobot.utils.robot_utils import precise_sleep

# Hard caps (cannot raise via CLI above these).
_MAX_SECONDS = 10.0
_MAX_GP_HZ = 100.0
_MAX_PARENT_HZ = 30.0
_DEFAULT_SPEED_RATIO = 5  # very low if any write path is used
_MOTION_ENABLE_GI = 56
_MOTION_ENABLE_OFF = 0


@dataclass
class ProbeStats:
    gp_ok: int = 0
    gp_fail: int = 0
    joint_ok: int = 0
    joint_fail: int = 0
    ee_ok: int = 0
    ee_fail: int = 0
    last_err: str = ""


def _try_set_gi(robot: Any, index: int, value: int) -> None:
    try:
        robot.set_GI(int(index), int(value))
    except Exception as exc:  # noqa: BLE001
        print(f"[safety] set_GI({index})={value} skipped/failed: {exc!r}")


def _try_stop(robot: Any) -> None:
    fn = getattr(robot, "stop_move", None)
    if callable(fn):
        try:
            fn()
            print("[safety] called stop_move()")
        except Exception as exc:  # noqa: BLE001
            print(f"[safety] stop_move() failed: {exc!r}")


def _read_pose6(robot: Any) -> list[float]:
    pose = robot.read_end_pose_user()
    if pose is None or len(pose) < 6:
        raise RuntimeError(f"read_end_pose_user returned invalid pose: {pose!r}")
    out = [float(pose[i]) for i in range(6)]
    if any(abs(v) > 1e6 for v in out):
        raise RuntimeError(f"pose looks corrupt (huge values): {out}")
    return out


def _fork_loop(
    robot: Any,
    stop: Any,
    hz: float,
    *,
    writes: str,
    pose6: list[float] | None,
    out: dict,
) -> None:
    """Child: either read-only or rewrite the latched hold pose into GP (no motion intent)."""
    period = 1.0 / max(1.0, hz)
    t_next = time.perf_counter()
    ok = fail = 0
    mat = None
    if writes == "hold_gp":
        assert pose6 is not None and len(pose6) == 6
        mat = [list(pose6) for _ in range(5)]
    while not stop.is_set():
        try:
            if writes == "hold_gp":
                robot.set_GPs(10, mat)
            else:
                # read-only contention from the child as well
                _ = robot.read_joints()
            ok += 1
        except Exception as exc:  # noqa: BLE001
            fail += 1
            out["child_err"] = repr(exc)
        t_next += period
        dt = t_next - time.perf_counter()
        if dt > 0:
            precise_sleep(dt)
        else:
            t_next = time.perf_counter()
    out["child_ok"] = ok
    out["child_fail"] = fail


def _confirm_writes() -> bool:
    print(
        "\n"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║  WRITE MODE: will call set_GPs with the CURRENT EE pose     ║\n"
        "║  (hold only). Arm must be clear; e-stop within reach.       ║\n"
        "║  Pendant should NOT execute a distant GP trajectory.        ║\n"
        "║  GI56 will be forced OFF (if set_GI works).                 ║\n"
        "╚══════════════════════════════════════════════════════════════╝\n"
        "Type YES (uppercase) to continue, anything else aborts: ",
        end="",
        flush=True,
    )
    try:
        return sys.stdin.readline().strip() == "YES"
    except Exception:
        return False


def main() -> None:
    p = argparse.ArgumentParser(
        description="CRP SDK concurrency probe (defaults to read-only / no motion)."
    )
    p.add_argument("--port", default="192.168.0.100")
    p.add_argument("--seconds", type=float, default=3.0, help=f"Duration (max {_MAX_SECONDS}s)")
    p.add_argument("--gp_hz", type=float, default=50.0, help=f"Child loop Hz (max {_MAX_GP_HZ})")
    p.add_argument("--parent_hz", type=float, default=30.0, help=f"Parent loop Hz (max {_MAX_PARENT_HZ})")
    p.add_argument("--also_ee", action="store_true", help="Parent also calls read_end_pose_user")
    p.add_argument(
        "--writes",
        choices=("none", "hold_gp"),
        default="none",
        help="none=reads only (safe default); hold_gp=set_GPs(current pose) only",
    )
    p.add_argument(
        "--layout",
        choices=("fork_share", "parent_only"),
        default="fork_share",
        help="fork_share=child+parent like ~/lerobot; parent_only=single process",
    )
    p.add_argument(
        "--i-understand-risk",
        action="store_true",
        dest="i_understand_risk",
        help="Required together with interactive YES for --writes=hold_gp",
    )
    p.add_argument(
        "--no-auto-mode",
        action="store_true",
        help="Do not switch_work_mode(Auto); leave pendant mode unchanged",
    )
    p.add_argument(
        "--speed-ratio",
        type=int,
        default=_DEFAULT_SPEED_RATIO,
        help=f"Speed ratio if hold_gp (default {_DEFAULT_SPEED_RATIO}, clamped 1..20)",
    )
    args = p.parse_args()

    seconds = min(max(0.5, float(args.seconds)), _MAX_SECONDS)
    child_hz = min(max(1.0, float(args.gp_hz)), _MAX_GP_HZ)
    parent_hz = min(max(1.0, float(args.parent_hz)), _MAX_PARENT_HZ)
    speed = max(1, min(20, int(args.speed_ratio)))

    if args.writes == "hold_gp":
        if not args.i_understand_risk:
            print("Refusing hold_gp without --i-understand-risk", file=sys.stderr)
            sys.exit(2)
        if not sys.stdin.isatty():
            print("Refusing hold_gp without an interactive TTY for YES confirm", file=sys.stderr)
            sys.exit(2)
        if not _confirm_writes():
            print("Aborted.")
            sys.exit(1)
    else:
        print(
            "[safety] writes=none — no set_GPs/set_GJs/move*; "
            "only read_joints / optional read_end_pose (safe default)."
        )

    CrpRobotPy, RobotMode = import_crp_robot_py()
    robot = CrpRobotPy()
    print(f"[safety] connect({args.port}) ...")
    robot.connect(args.port)

    # Keep servos as-is if already on; do not force aggressive mode changes for reads.
    try:
        if not args.no_auto_mode:
            robot.switch_work_mode(RobotMode.Auto)
            print("[safety] work_mode=Auto")
        else:
            print("[safety] left work_mode unchanged (--no-auto-mode)")
    except Exception as exc:  # noqa: BLE001
        print(f"[safety] switch_work_mode skipped: {exc!r}")

    # Always try to close MOVE gate so pendant cannot chase GP while we poke registers.
    _try_set_gi(robot, _MOTION_ENABLE_GI, _MOTION_ENABLE_OFF)
    print(f"[safety] requested GI{_MOTION_ENABLE_GI}={_MOTION_ENABLE_OFF} (MOVE gate OFF)")

    try:
        pose = _read_pose6(robot)
    except Exception as exc:
        print(f"[safety] ABORT: cannot latch current EE pose ({exc})")
        try:
            robot.disconnect()
        except Exception:
            pass
        sys.exit(3)

    print(f"[safety] latched hold pose xyz=({pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f}) "
          f"rpy=({pose[3]:.3f},{pose[4]:.3f},{pose[5]:.3f})")

    if args.writes == "hold_gp":
        try:
            robot.set_speed_ratio(speed)
            print(f"[safety] set_speed_ratio({speed})")
        except Exception as exc:  # noqa: BLE001
            print(f"[safety] set_speed_ratio failed: {exc!r}")
        # Preload current pose once (still with MOVE gate OFF).
        try:
            mat = [list(pose) for _ in range(5)]
            robot.set_GPs(10, mat)
            print("[safety] preloaded GP10 with hold pose (gate OFF → should not move)")
        except Exception as exc:  # noqa: BLE001
            print(f"[safety] ABORT: set_GPs preload failed: {exc!r}")
            try:
                robot.disconnect()
            except Exception:
                pass
            sys.exit(3)

    print(
        f"layout={args.layout} writes={args.writes} "
        f"child_hz={child_hz} parent_hz={parent_hz} also_ee={args.also_ee} seconds={seconds}"
    )

    stop = mp.Event()
    mgr = mp.Manager()
    shared = mgr.dict()
    proc = None
    stats = ProbeStats()

    def _shutdown(*_args: object) -> None:
        print("\n[safety] interrupt — stopping ...")
        stop.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        if args.layout == "fork_share":
            ctx = mp.get_context("fork")
            proc = ctx.Process(
                target=_fork_loop,
                kwargs={
                    "robot": robot,
                    "stop": stop,
                    "hz": child_hz,
                    "writes": args.writes,
                    "pose6": list(pose) if args.writes == "hold_gp" else None,
                    "out": shared,
                },
                daemon=True,
            )
            proc.start()
            time.sleep(0.15)

        parent_period = 1.0 / parent_hz
        t_end = time.perf_counter() + seconds
        t_next = time.perf_counter()
        while time.perf_counter() < t_end and not stop.is_set():
            try:
                _ = robot.read_joints()
                stats.joint_ok += 1
            except Exception as exc:  # noqa: BLE001
                stats.joint_fail += 1
                stats.last_err = f"read_joints: {exc!r}"
            if args.also_ee:
                try:
                    _ = robot.read_end_pose_user()
                    stats.ee_ok += 1
                except Exception as exc:  # noqa: BLE001
                    stats.ee_fail += 1
                    stats.last_err = f"read_end_pose: {exc!r}"
            if args.layout == "parent_only":
                try:
                    if args.writes == "hold_gp":
                        robot.set_GPs(10, [list(pose) for _ in range(5)])
                    else:
                        _ = robot.read_joints()
                    stats.gp_ok += 1
                except Exception as exc:  # noqa: BLE001
                    stats.gp_fail += 1
                    stats.last_err = f"parent_loop: {exc!r}"
            t_next += parent_period
            dt = t_next - time.perf_counter()
            if dt > 0:
                precise_sleep(dt)
            else:
                t_next = time.perf_counter()
    finally:
        stop.set()
        if proc is not None:
            proc.join(timeout=3.0)
            if proc.is_alive():
                print("[safety] child still alive — terminate")
                proc.terminate()
                proc.join(timeout=2.0)
        _try_set_gi(robot, _MOTION_ENABLE_GI, _MOTION_ENABLE_OFF)
        _try_stop(robot)
        try:
            robot.disconnect()
            print("[safety] disconnected")
        except Exception as exc:  # noqa: BLE001
            print(f"[safety] disconnect failed: {exc!r}")

    child_ok = int(shared.get("child_ok", stats.gp_ok))
    child_fail = int(shared.get("child_fail", stats.gp_fail))
    print("=== results ===")
    print(f"child: ok={child_ok} fail={child_fail}  ~Hz={child_ok / seconds:.1f}")
    print(f"joint: ok={stats.joint_ok} fail={stats.joint_fail}  ~Hz={stats.joint_ok / seconds:.1f}")
    if args.also_ee:
        print(f"ee:    ok={stats.ee_ok} fail={stats.ee_fail}  ~Hz={stats.ee_ok / seconds:.1f}")
    if shared.get("child_err"):
        print("child last err:", shared.get("child_err"))
    if stats.last_err:
        print("parent last err:", stats.last_err)
    print(
        "Note: even if Hz looks fine, shared-client fork+parent is still undefined for Thrift; "
        "use results only as empirical evidence."
    )


if __name__ == "__main__":
    main()
