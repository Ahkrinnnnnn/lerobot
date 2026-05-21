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

"""
Verify ``OMYL100.get_ros_end_effector_xyz_rpy_deg()`` (scaled xyz + RPY deg from ``Pose`` or ``PoseStamped``).

Run while your OMY / FK stack is publishing (same ``ROS_DOMAIN_ID`` as this process):

```bash
python -m lerobot.scripts.verify_omy_l100_get_ros_ee --timeout 30
```

Defaults match typical ``/end_effector_pose``: ``geometry_msgs/Pose`` + RELIABLE (see ``OMYL100Config``).
For ``PoseStamped`` + BEST_EFFORT publishers, pass ``--ee-pose-msg-type pose_stamped --ee-qos sensor``.

If readings stay ``None``, check ``ros2 topic info <topic> -v`` vs the printed config.
"""

from __future__ import annotations

import argparse
import sys
import time

from lerobot.teleoperators.OMY_L100.config_OMY_L100 import OMYL100Config
from lerobot.teleoperators.OMY_L100.OMY_L100 import EE_STATES_TOPIC, OMYL100


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--ee-topic",
        type=str,
        default="",
        help="``ros_end_effector_pose_topic`` (empty = OMY_L100 default / derive from joint topic).",
    )
    p.add_argument(
        "--ee-pose-msg-type",
        choices=("pose_stamped", "pose"),
        default="pose",
        help="``ros_end_effector_pose_msg_type``: must match ``ros2 topic info`` (``Pose`` vs ``PoseStamped``).",
    )
    p.add_argument(
        "--ee-qos",
        choices=("sensor", "reliable"),
        default="reliable",
        help="``ros_ee_pose_qos``: ``reliable`` for RELIABLE EE publishers; ``sensor`` for BEST_EFFORT.",
    )
    p.add_argument(
        "--joint-states-topic",
        type=str,
        default="",
        help="Override ``joint_states_topic`` (default from OMYL100Config).",
    )
    p.add_argument("--timeout", type=float, default=30.0, help="Seconds to poll before giving up.")
    p.add_argument("--poll-interval", type=float, default=0.05, help="Sleep between polls (s).")
    p.add_argument(
        "--stall-log-interval",
        type=float,
        default=2.0,
        help="Print a status line every N seconds while still None.",
    )
    args = p.parse_args()

    cfg_kw: dict = {"port": "dummy"}
    if args.ee_topic.strip():
        cfg_kw["ros_end_effector_pose_topic"] = args.ee_topic.strip()
    if args.joint_states_topic.strip():
        cfg_kw["joint_states_topic"] = args.joint_states_topic.strip()
    cfg_kw["ros_end_effector_pose_msg_type"] = args.ee_pose_msg_type
    cfg_kw["ros_ee_pose_qos"] = args.ee_qos

    cfg = OMYL100Config(**cfg_kw)
    teleop = OMYL100(cfg)

    print("Config:", flush=True)
    print(f"  joint_states_topic:        {cfg.joint_states_topic!r}", flush=True)
    print(f"  ros_end_effector_pose_topic: {cfg.ros_end_effector_pose_topic!r}", flush=True)
    print(f"  ros_end_effector_pose_msg_type: {cfg.ros_end_effector_pose_msg_type!r}", flush=True)
    print(f"  ros_ee_pose_qos: {cfg.ros_ee_pose_qos!r}", flush=True)
    print(f"  (empty EE topic → connect uses EE_STATES_TOPIC = {EE_STATES_TOPIC!r})", flush=True)
    print(f"  ros_ee_pose_position_scale: {cfg.ros_ee_pose_position_scale}", flush=True)
    print("", flush=True)

    try:
        print("Connecting OMYL100 (rclpy + subscriptions)…", flush=True)
        teleop.connect()
        print(f"  is_connected={teleop.is_connected}", flush=True)
        print("", flush=True)

        t0 = time.perf_counter()
        last_stall = t0
        n_none = 0
        first_xyz = None

        print(f"Polling get_ros_end_effector_xyz_rpy_deg() for up to {args.timeout:.1f}s…", flush=True)
        while time.perf_counter() - t0 < args.timeout:
            pair = teleop.get_ros_end_effector_xyz_rpy_deg()
            if pair is not None:
                first_xyz, rpy = pair
                print("", flush=True)
                print("OK — first non-None reading:", flush=True)
                print(f"  scaled xyz (after ros_ee_pose_position_scale): {first_xyz}", flush=True)
                print(f"  RPY (deg): {rpy}", flush=True)
                print(f"  elapsed: {time.perf_counter() - t0:.3f}s, polls where None before: {n_none}", flush=True)
                return 0

            n_none += 1
            now = time.perf_counter()
            if now - last_stall >= args.stall_log_interval:
                last_stall = now
                act = teleop.get_action()
                j1 = float(act.get("j1.pos", 0.0))
                print(
                    f"  … still None after {now - t0:.1f}s | joint j1.pos={j1:.4f} | "
                    f"polls={n_none} (EE pose not received — check topic / msg type / QoS vs ros2 topic info -v)",
                    flush=True,
                )
            time.sleep(args.poll_interval)

        print("", flush=True)
        print(
            "FAIL — get_ros_end_effector_xyz_rpy_deg() stayed None for entire timeout.",
            file=sys.stderr,
            flush=True,
        )
        print(
            "  Check: ros2 topic info <name> -v — type must match --ee-pose-msg-type; "
            "RELIABLE pub needs --ee-qos reliable.",
            file=sys.stderr,
            flush=True,
        )
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
        return 130
    finally:
        if teleop.is_connected:
            teleop.disconnect()
            print("Disconnected.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
