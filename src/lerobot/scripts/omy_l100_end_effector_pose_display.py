#!/usr/bin/env python

"""
Display end-effector pose from ROS2 ``geometry_msgs/PoseStamped`` (e.g. ``/end_effector_pose``).

Typical open_manipulator / OMY stack exposes FK on ``/end_effector_pose`` or under a namespace
such as ``/leader/end_effector_pose``. Override with ``--topic``.
"""

import argparse
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass


DEFAULT_TOPIC = "/end_effector_pose"


@dataclass
class ParsedPoseStamped:
    frame_id: str
    stamp_sec: int
    stamp_nsec: int
    px: float
    py: float
    pz: float
    qx: float
    qy: float
    qz: float
    qw: float


def quat_to_rpy_rad(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float]:
    """Quaternion (x,y,z,w) to intrinsic roll, pitch, yaw (same convention as common ROS tools)."""
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def print_pose_table(topic: str, data: ParsedPoseStamped) -> None:
    roll, pitch, yaw = quat_to_rpy_rad(data.qx, data.qy, data.qz, data.qw)
    t = time.strftime("%H:%M:%S")
    lines = [
        f"\n[{t}] End-effector pose ({topic})",
        "-" * 72,
        f"  frame_id:     {data.frame_id}",
        f"  stamp:        {data.stamp_sec}.{data.stamp_nsec:09d}",
        "  position (m):",
        f"    x = {data.px: .6f}",
        f"    y = {data.py: .6f}",
        f"    z = {data.pz: .6f}",
        "  orientation (quat x,y,z,w):",
        f"    {data.qx: .6f}  {data.qy: .6f}  {data.qz: .6f}  {data.qw: .6f}",
        "  RPY (deg):",
        f"    roll = {math.degrees(roll):8.3f}   pitch = {math.degrees(pitch):8.3f}   yaw = {math.degrees(yaw):8.3f}",
        "-" * 72,
    ]
    print("\n".join(lines), flush=True)


_RE_STAMP = re.compile(
    r"stamp:\s*\n\s+sec:\s*(-?\d+)\s*\n\s+nanosec:\s*(-?\d+)",
    re.MULTILINE,
)
_RE_FRAME = re.compile(r"frame_id:\s*(.+)", re.MULTILINE)
_RE_POS = re.compile(
    r"position:\s*\n\s+x:\s*([-\d.eE+]+)\s*\n\s+y:\s*([-\d.eE+]+)\s*\n\s+z:\s*([-\d.eE+]+)",
    re.MULTILINE,
)
_RE_ORI = re.compile(
    r"orientation:\s*\n\s+x:\s*([-\d.eE+]+)\s*\n\s+y:\s*([-\d.eE+]+)\s*\n\s+z:\s*([-\d.eE+]+)\s*\n\s+w:\s*([-\d.eE+]+)",
    re.MULTILINE,
)


def parse_pose_stamped_echo_block(text: str) -> ParsedPoseStamped | None:
    """Parse a single ``ros2 topic echo`` YAML chunk for ``geometry_msgs/msg/PoseStamped``."""
    m_stamp = _RE_STAMP.search(text)
    m_frame = _RE_FRAME.search(text)
    m_pos = _RE_POS.search(text)
    m_ori = _RE_ORI.search(text)
    if not (m_pos and m_ori):
        return None

    stamp_sec = int(m_stamp.group(1)) if m_stamp else 0
    stamp_nsec = int(m_stamp.group(2)) if m_stamp else 0
    frame_raw = m_frame.group(1).strip() if m_frame else ""
    frame_id = frame_raw.strip("'\"")

    px, py, pz = float(m_pos.group(1)), float(m_pos.group(2)), float(m_pos.group(3))
    qx, qy, qz, qw = float(m_ori.group(1)), float(m_ori.group(2)), float(m_ori.group(3)), float(m_ori.group(4))
    return ParsedPoseStamped(
        frame_id=frame_id,
        stamp_sec=stamp_sec,
        stamp_nsec=stamp_nsec,
        px=px,
        py=py,
        pz=pz,
        qx=qx,
        qy=qy,
        qz=qz,
        qw=qw,
    )


def run_cli_backend(topic: str, timeout_s: float) -> int:
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
                    print(f"[warn] No pose message for {now - last_msg_ts:.1f}s on {topic}", flush=True)
                    last_warn_ts = now
                time.sleep(0.05)
                continue

            s = line.rstrip("\n")
            if s.strip() == "---":
                text = "\n".join(block)
                parsed = parse_pose_stamped_echo_block(text)
                if parsed is not None:
                    last_msg_ts = now
                    print_pose_table(topic, parsed)
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
        from geometry_msgs.msg import PoseStamped
        from rclpy.node import Node
    except Exception as exc:
        raise RuntimeError(
            "ROS2 Python dependencies are required for --backend node (rclpy, geometry_msgs)."
        ) from exc

    class _NodeDisplay(Node):
        def __init__(self) -> None:
            super().__init__("omy_l100_end_effector_pose_display")
            self.last_msg_ts = 0.0
            self.last_log_timeout_ts = 0.0
            self.create_subscription(PoseStamped, topic, self._on_pose, 10)
            self.get_logger().info(f"Subscribed to PoseStamped topic: {topic}")

        def _on_pose(self, msg: PoseStamped) -> None:
            self.last_msg_ts = time.time()
            h = msg.header
            p = msg.pose.position
            o = msg.pose.orientation
            data = ParsedPoseStamped(
                frame_id=h.frame_id,
                stamp_sec=int(h.stamp.sec),
                stamp_nsec=int(h.stamp.nanosec),
                px=float(p.x),
                py=float(p.y),
                pz=float(p.z),
                qx=float(o.x),
                qy=float(o.y),
                qz=float(o.z),
                qw=float(o.w),
            )
            print_pose_table(topic, data)

        def check_timeout(self) -> None:
            if self.last_msg_ts <= 0.0:
                return
            now = time.time()
            if now - self.last_msg_ts > timeout_s and now - self.last_log_timeout_ts > 1.0:
                self.get_logger().warning(
                    f"No pose message received for {now - self.last_msg_ts:.1f}s on {topic}"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Display end-effector PoseStamped from ROS2.")
    parser.add_argument(
        "--topic",
        type=str,
        default=DEFAULT_TOPIC,
        help=f"ROS2 topic (geometry_msgs/PoseStamped). Default: {DEFAULT_TOPIC}",
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
        help="Data source: 'cli' uses `ros2 topic echo`, 'node' uses rclpy subscriber.",
    )
    args = parser.parse_args()

    if args.backend == "cli":
        rc = run_cli_backend(topic=args.topic, timeout_s=args.timeout_s)
        sys.exit(rc)

    rc = run_node_backend(topic=args.topic, timeout_s=args.timeout_s)
    sys.exit(rc)


if __name__ == "__main__":
    main()
