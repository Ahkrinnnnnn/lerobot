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

"""Synchronized camera frame + robot pose capture at save time."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from lerobot.cameras.camera import Camera

from .adapters import HandEyeRobot
from .scene import RobotPoseFrame
from .transforms import transform_from_xyz_rpy_deg


def read_pose_6d_if_available(robot: HandEyeRobot, frame: RobotPoseFrame) -> list[float] | None:
    read_fn = getattr(robot, "read_robot_pose_6d", None)
    if read_fn is None:
        return None
    return read_fn(frame)


@dataclass
class SyncCapture:
    """One save event: fresh image and (optionally) robot pose read back-to-back."""

    image: np.ndarray
    T_robot_to_ee: np.ndarray | None
    frame_read_ms: float
    pose_read_ms: float
    pose_after_frame_ms: float
    top_image: np.ndarray | None = None
    top_frame_read_ms: float = 0.0
    pose_6d: list[float] | None = None

    @property
    def total_sync_ms(self) -> float:
        return self.frame_read_ms + self.top_frame_read_ms + self.pose_read_ms


def capture_frame_and_pose(
    robot: HandEyeRobot,
    camera_name: str,
    robot_pose_frame: RobotPoseFrame,
    *,
    record_pose: bool = True,
    also_capture_top: str | None = None,
) -> SyncCapture:
    """Grab blocking camera frame(s), then immediately read the robot pose.

    Order: wrist (``camera_name``) → optional top → robot pose, so both images
    are taken while the arm is still and pose is read last.
    """
    wrist_cam: Camera = robot.cameras[camera_name]
    t0 = time.perf_counter()
    frame = wrist_cam.read()
    t1 = time.perf_counter()

    top_image = None
    top_ms = 0.0
    if also_capture_top is not None:
        if also_capture_top not in robot.cameras:
            raise KeyError(f"Top camera {also_capture_top!r} not in robot config.")
        top_cam = robot.cameras[also_capture_top]
        t_top0 = time.perf_counter()
        top_image = top_cam.read()
        t_top1 = time.perf_counter()
        top_ms = (t_top1 - t_top0) * 1e3

    T_robot_to_ee = None
    pose_6d = None
    if record_pose:
        pose_6d = read_pose_6d_if_available(robot, robot_pose_frame)
        if pose_6d is not None:
            T_robot_to_ee = transform_from_xyz_rpy_deg(*pose_6d)
        else:
            T_robot_to_ee = robot.read_robot_to_ee(robot_pose_frame)
    t2 = time.perf_counter()
    return SyncCapture(
        image=frame,
        T_robot_to_ee=T_robot_to_ee,
        frame_read_ms=(t1 - t0) * 1e3,
        pose_read_ms=(t2 - t1) * 1e3 if record_pose else 0.0,
        pose_after_frame_ms=(t2 - t1) * 1e3,
        top_image=top_image,
        top_frame_read_ms=top_ms,
        pose_6d=pose_6d,
    )
