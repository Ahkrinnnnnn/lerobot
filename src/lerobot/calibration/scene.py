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

"""Scene calibration: intrinsics, extrinsics, table grid in robot frame."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .hand_eye import CameraMount
from .transforms import transform_from_list, transform_to_list


class RobotPoseFrame(str, Enum):
    WORLD = "world"
    USER = "user"


@dataclass
class CameraCalibration:
    mount: CameraMount
    width: int
    height: int
    camera_matrix: list[list[float]]
    dist_coeffs: list[float]
    reprojection_error: float | None = None
    T_robot_to_camera: list[list[float]] | None = None
    T_ee_to_camera: list[list[float]] | None = None

    def camera_matrix_np(self) -> np.ndarray:
        return np.array(self.camera_matrix, dtype=np.float64)

    def dist_coeffs_np(self) -> np.ndarray:
        return np.array(self.dist_coeffs, dtype=np.float64)


@dataclass
class TableCalibration:
    T_robot_to_table: list[list[float]]
    grid_points_robot_mm: list[list[float]] = field(default_factory=list)
    notes: str = ""


@dataclass
class SceneCalibration:
    """Camera + table calibration saved as one JSON artifact."""

    version: int = 2
    robot_type: str = ""
    robot_id: str = ""
    robot_pose_frame: RobotPoseFrame = RobotPoseFrame.WORLD
    length_unit: str = "mm"
    angle_unit: str = "deg"
    charuco: dict[str, Any] | None = None
    cameras: dict[str, CameraCalibration] = field(default_factory=dict)
    table: TableCalibration | None = None

    def to_dict(self) -> dict[str, Any]:
        intrinsics: dict[str, Any] = {}
        T_robot_to_camera: dict[str, list[list[float]]] = {}
        T_ee_to_camera: dict[str, list[list[float]]] = {}
        mounts: dict[str, str] = {}

        for name, cam in self.cameras.items():
            mounts[name] = cam.mount.value
            intrinsics[name] = {
                "width": cam.width,
                "height": cam.height,
                "camera_matrix": cam.camera_matrix,
                "dist_coeffs": cam.dist_coeffs,
                "reprojection_error": cam.reprojection_error,
            }
            if cam.T_robot_to_camera is not None:
                T_robot_to_camera[name] = cam.T_robot_to_camera
            if cam.T_ee_to_camera is not None:
                T_ee_to_camera[name] = cam.T_ee_to_camera

        out: dict[str, Any] = {
            "version": self.version,
            "robot_type": self.robot_type,
            "robot_id": self.robot_id,
            "robot_pose_frame": self.robot_pose_frame.value,
            "length_unit": self.length_unit,
            "angle_unit": self.angle_unit,
            "intrinsics": intrinsics,
            "camera_mounts": mounts,
        }
        if self.charuco is not None:
            out["charuco"] = self.charuco
        if T_robot_to_camera:
            out["T_robot_to_camera"] = T_robot_to_camera
        if T_ee_to_camera:
            out["T_ee_to_camera"] = T_ee_to_camera
        if self.table is not None:
            out["T_robot_to_table"] = self.table.T_robot_to_table
            out["table_grid_points_robot_mm"] = self.table.grid_points_robot_mm
            if self.table.notes:
                out["table_notes"] = self.table.notes
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneCalibration:
        version = int(data.get("version", 1))
        cameras: dict[str, CameraCalibration] = {}

        if version >= 2 and "intrinsics" in data:
            mounts = data.get("camera_mounts", {})
            T_rtc = data.get("T_robot_to_camera", {})
            T_eec = data.get("T_ee_to_camera", {})
            for name, intr in data["intrinsics"].items():
                cameras[name] = CameraCalibration(
                    mount=CameraMount(mounts.get(name, CameraMount.EYE_TO_HAND.value)),
                    width=int(intr["width"]),
                    height=int(intr["height"]),
                    camera_matrix=intr["camera_matrix"],
                    dist_coeffs=intr["dist_coeffs"],
                    reprojection_error=intr.get("reprojection_error"),
                    T_robot_to_camera=T_rtc.get(name),
                    T_ee_to_camera=T_eec.get(name),
                )
        else:
            for name, cam in data.get("cameras", {}).items():
                cameras[name] = CameraCalibration(
                    mount=CameraMount(cam["mount"]),
                    width=int(cam["width"]),
                    height=int(cam["height"]),
                    camera_matrix=cam["camera_matrix"],
                    dist_coeffs=cam["dist_coeffs"],
                    reprojection_error=cam.get("reprojection_error"),
                    T_robot_to_camera=cam.get("T_robot_to_camera"),
                    T_ee_to_camera=cam.get("T_ee_to_camera"),
                )

        table = None
        if "T_robot_to_table" in data:
            table = TableCalibration(
                T_robot_to_table=data["T_robot_to_table"],
                grid_points_robot_mm=data.get("table_grid_points_robot_mm", []),
                notes=data.get("table_notes", data.get("table", {}).get("notes", "")),
            )
        elif table_data := data.get("table"):
            table = TableCalibration(
                T_robot_to_table=table_data["T_robot_to_table"],
                grid_points_robot_mm=table_data.get("grid_points_robot_mm", []),
                notes=table_data.get("notes", ""),
            )

        charuco = data.get("charuco")
        if charuco is None and "target" in data:
            charuco = data["target"]

        return cls(
            version=max(version, 2),
            robot_type=data.get("robot_type", ""),
            robot_id=data.get("robot_id", ""),
            robot_pose_frame=RobotPoseFrame(data.get("robot_pose_frame", RobotPoseFrame.WORLD.value)),
            length_unit=data.get("length_unit", "mm"),
            angle_unit=data.get("angle_unit", "deg"),
            charuco=charuco,
            cameras=cameras,
            table=table,
        )

    def get_T_robot_to_camera(self, camera_name: str, T_robot_to_ee: np.ndarray | None = None) -> np.ndarray:
        cam = self.cameras[camera_name]
        if cam.mount == CameraMount.EYE_TO_HAND:
            if cam.T_robot_to_camera is None:
                raise ValueError(f"Camera {camera_name!r} missing T_robot_to_camera.")
            return transform_from_list(cam.T_robot_to_camera)
        if cam.T_ee_to_camera is None:
            raise ValueError(f"Camera {camera_name!r} missing T_ee_to_camera.")
        if T_robot_to_ee is None:
            raise ValueError("eye_in_hand camera requires T_robot_to_ee at capture time.")
        return T_robot_to_ee @ transform_from_list(cam.T_ee_to_camera)

    def get_T_robot_to_table(self) -> np.ndarray:
        if self.table is None:
            raise ValueError("Table calibration missing.")
        return transform_from_list(self.table.T_robot_to_table)

    def export_summary(self, T_robot_to_ee: np.ndarray | None = None) -> dict[str, Any]:
        """Flattened result: intrinsics, T_robot_to_camera, table grid."""
        data = self.to_dict()
        resolved: dict[str, list[list[float]]] = {}
        for name, cam in self.cameras.items():
            try:
                resolved[name] = transform_to_list(self.get_T_robot_to_camera(name, T_robot_to_ee))
            except ValueError:
                continue
        if resolved:
            data["T_robot_to_camera_resolved"] = resolved
        return data
