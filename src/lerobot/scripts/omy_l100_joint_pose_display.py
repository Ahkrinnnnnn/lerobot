#!/usr/bin/env python

"""
Display OMY_L100 joint pose from ROS2 topics.

Expected robot launch:
    ros2 launch open_manipulator_bringup omy_l100_leader_ai.launch.py

This launch runs under `/leader` namespace and typically exposes:
    - /leader/joint_trajectory_command_broadcaster/joint_trajectory
    - /leader/joint_states
"""

import argparse
import math
import subprocess
import sys
import time
from typing import Iterable


DEFAULT_CANDIDATE_TOPICS = (
    "/leader/joint_states",
)


def print_table(topic: str, names: Iterable[str], positions: Iterable[float]) -> None:
    names = list(names)
    positions = list(positions)
    if not names or not positions:
        return

    t = time.strftime("%H:%M:%S")
    lines = [f"\n[{t}] OMY_L100 Joint Pose ({topic})", "-" * 64]
    lines.append(f"{'Joint':<24}{'rad':>14}{'deg':>14}")
    lines.append("-" * 64)
    for name, rad in zip(names, positions):
        deg = math.degrees(rad)
        lines.append(f"{name:<24}{rad:>14.6f}{deg:>14.2f}")
    lines.append("-" * 64)
    print("\n".join(lines), flush=True)


def _parse_scalar_list(lines: list[str], start_idx: int) -> tuple[list[str], int]:
    values: list[str] = []
    i = start_idx + 1
    while i < len(lines):
        raw = lines[i].strip()
        if not raw.startswith("- "):
            break
        values.append(raw[2:].strip())
        i += 1
    return values, i


def _parse_trajectory_positions(lines: list[str], point_index: int) -> list[float]:
    in_points = False
    point_count = -1
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s == "points:":
            in_points = True
            i += 1
            continue
        if in_points and s.startswith("- "):
            point_count += 1
            if point_count == point_index:
                j = i + 1
                while j < len(lines):
                    ss = lines[j].strip()
                    if ss == "positions:":
                        arr, _ = _parse_scalar_list(lines, j)
                        out: list[float] = []
                        for item in arr:
                            out.append(float(item))
                        return out
                    if ss.startswith("- ") and j > i + 1:
                        break
                    j += 1
                return []
        i += 1
    return []


def run_cli_backend(topic: str, timeout_s: float, point_index: int) -> int:
    cmd = ["ros2", "topic", "echo", topic]
    print(f"Using CLI backend: {' '.join(cmd)}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    assert proc.stdout is not None

    block: list[str] = []
    last_msg_ts = 0.0
    last_warn_ts = 0.0

    try:
        while True:
            line = proc.stdout.readline()
            now = time.time()

            if line == "":
                if proc.poll() is not None:
                    err = ""
                    if proc.stderr is not None:
                        err = proc.stderr.read().strip()
                    if err:
                        print(err, file=sys.stderr, flush=True)
                    return proc.returncode if proc.returncode is not None else 1
                if last_msg_ts > 0.0 and (now - last_msg_ts > timeout_s) and (now - last_warn_ts > 1.0):
                    print(f"[warn] No joint message for {now - last_msg_ts:.1f}s on {topic}", flush=True)
                    last_warn_ts = now
                time.sleep(0.05)
                continue

            s = line.rstrip("\n")
            if s.strip() == "---":
                names: list[str] = []
                positions: list[float] = []
                i = 0
                while i < len(block):
                    key = block[i].strip()
                    if key in ("name:", "joint_names:"):
                        arr, ni = _parse_scalar_list(block, i)
                        names = arr
                        i = ni
                        continue
                    if key == "position:":
                        arr, ni = _parse_scalar_list(block, i)
                        positions = [float(v) for v in arr]
                        i = ni
                        continue
                    i += 1

                if "joint_trajectory" in topic and not positions:
                    positions = _parse_trajectory_positions(block, point_index=point_index)

                if names and positions:
                    last_msg_ts = now
                    print_table(topic, names, positions)
                block = []
                continue

            block.append(s)
    except KeyboardInterrupt:
        return 0
    finally:
        try:
            proc.terminate()
        except Exception:
            pass


def run_node_backend(topic: str, timeout_s: float) -> int:
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from trajectory_msgs.msg import JointTrajectory
    except Exception as exc:
        raise RuntimeError(
            "ROS2 Python dependencies are required for --backend node "
            "(rclpy, sensor_msgs, trajectory_msgs)."
        ) from exc

    class _NodeDisplay(Node):
        def __init__(self) -> None:
            super().__init__("omy_l100_joint_pose_display")
            self.last_msg_ts = 0.0
            self.last_log_timeout_ts = 0.0
            if "joint_trajectory" in topic:
                self.create_subscription(JointTrajectory, topic, self._on_joint_trajectory, 10)
                self.get_logger().info(f"Subscribed to JointTrajectory topic: {topic}")
            else:
                self.create_subscription(JointState, topic, self._on_joint_state, 10)
                self.get_logger().info(f"Subscribed to JointState topic: {topic}")

        def _on_joint_trajectory(self, msg: JointTrajectory) -> None:
            if not msg.points or not msg.points[0].positions:
                return
            self.last_msg_ts = time.time()
            print_table(topic, msg.joint_names, msg.points[0].positions)

        def _on_joint_state(self, msg: JointState) -> None:
            if not msg.position:
                return
            self.last_msg_ts = time.time()
            print_table(topic, msg.name, msg.position)

        def check_timeout(self) -> None:
            if self.last_msg_ts <= 0.0:
                return
            now = time.time()
            if now - self.last_msg_ts > timeout_s and now - self.last_log_timeout_ts > 1.0:
                self.get_logger().warning(
                    f"No joint message received for {now - self.last_msg_ts:.1f}s on {topic}"
                )
                self.last_log_timeout_ts = now

    rclpy.init(args=None)
    node = _NodeDisplay()
    node.get_logger().info("Press Ctrl+C to stop.")
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)
            node.check_timeout()
    except KeyboardInterrupt:
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


def resolve_topic(explicit_topic: str | None, prefer_joint_trajectory: bool) -> str:
    if explicit_topic:
        return explicit_topic
    if prefer_joint_trajectory:
        return DEFAULT_CANDIDATE_TOPICS[0]
    return DEFAULT_CANDIDATE_TOPICS[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Display OMY_L100 joint pose from ROS2.")
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="ROS2 topic to subscribe. If omitted, uses default OMY_L100 leader topics.",
    )
    parser.add_argument(
        "--prefer-joint-states",
        action="store_true",
        help="Prefer /leader/joint_states instead of joint_trajectory broadcaster topic.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=3.0,
        help="Warn when no new message is received for this many seconds. Default: 3.0",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=("cli", "node"),
        default="cli",
        help="Data source backend: 'cli' uses `ros2 topic echo` (no rclpy node), 'node' uses rclpy subscriber.",
    )
    parser.add_argument(
        "--point-index",
        type=int,
        default=0,
        help="For JointTrajectory topic, which points[index].positions to display. Default: 0",
    )
    args = parser.parse_args()

    topic = resolve_topic(explicit_topic=args.topic, prefer_joint_trajectory=not args.prefer_joint_states)
    if args.backend == "cli":
        rc = run_cli_backend(topic=topic, timeout_s=args.timeout_s, point_index=args.point_index)
        sys.exit(rc)

    rc = run_node_backend(topic=topic, timeout_s=args.timeout_s)
    sys.exit(rc)


if __name__ == "__main__":
    main()
