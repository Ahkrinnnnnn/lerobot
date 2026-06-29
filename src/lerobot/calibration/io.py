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

"""Load / save scene calibration as NPZ."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS

from .scene import SceneCalibration


def default_scene_calibration_path(robot_type: str, robot_id: str) -> Path:
    return HF_LEROBOT_CALIBRATION / ROBOTS / robot_type / f"{robot_id}_calibration.npz"


def _str_array(value: str) -> np.ndarray:
    return np.array(value, dtype=object)


def scene_to_npz_arrays(scene: SceneCalibration, T_robot_to_ee: np.ndarray | None = None) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "version": np.array([scene.version], dtype=np.int32),
        "robot_type": _str_array(scene.robot_type),
        "robot_id": _str_array(scene.robot_id),
        "robot_pose_frame": _str_array(scene.robot_pose_frame.value),
        "length_unit": _str_array(scene.length_unit),
        "angle_unit": _str_array(scene.angle_unit),
        "camera_names": _str_array(",".join(scene.cameras.keys())),
    }

    if scene.charuco is not None:
        for key, value in scene.charuco.items():
            if isinstance(value, (int, float)):
                arrays[f"charuco_{key}"] = np.array([value], dtype=np.float64)
            else:
                arrays[f"charuco_{key}"] = _str_array(str(value))

    for name, cam in scene.cameras.items():
        prefix = f"{name}__"
        arrays[f"{prefix}camera_matrix"] = cam.camera_matrix_np()
        arrays[f"{prefix}dist_coeffs"] = cam.dist_coeffs_np()
        arrays[f"{prefix}width"] = np.array([cam.width], dtype=np.int32)
        arrays[f"{prefix}height"] = np.array([cam.height], dtype=np.int32)
        arrays[f"{prefix}mount"] = _str_array(cam.mount.value)
        if cam.reprojection_error is not None:
            arrays[f"{prefix}reprojection_error"] = np.array([cam.reprojection_error], dtype=np.float64)
        if cam.T_robot_to_camera is not None:
            arrays[f"{prefix}T_robot_to_camera"] = np.asarray(cam.T_robot_to_camera, dtype=np.float64)
        if cam.T_ee_to_camera is not None:
            arrays[f"{prefix}T_ee_to_camera"] = np.asarray(cam.T_ee_to_camera, dtype=np.float64)
        try:
            T_resolved = scene.get_T_robot_to_camera(name, T_robot_to_ee)
            arrays[f"{prefix}T_robot_to_camera_resolved"] = T_resolved
        except ValueError:
            pass

    if scene.table is not None:
        arrays["T_robot_to_table"] = np.asarray(scene.table.T_robot_to_table, dtype=np.float64)
        if scene.table.grid_points_robot_mm:
            arrays["table_grid_points_robot_mm"] = np.asarray(scene.table.grid_points_robot_mm, dtype=np.float64)
        if scene.table.notes:
            arrays["table_notes"] = _str_array(scene.table.notes)

    return arrays


def scene_from_npz_arrays(arrays: dict[str, np.ndarray]) -> SceneCalibration:
    from .hand_eye import CameraMount

    camera_names = str(arrays["camera_names"].item()).split(",") if arrays["camera_names"].item() else []
    cameras = {}
    for name in camera_names:
        if not name:
            continue
        prefix = f"{name}__"
        cam_data = {
            "mount": str(arrays[f"{prefix}mount"].item()),
            "width": int(arrays[f"{prefix}width"][0]),
            "height": int(arrays[f"{prefix}height"][0]),
            "camera_matrix": arrays[f"{prefix}camera_matrix"].tolist(),
            "dist_coeffs": arrays[f"{prefix}dist_coeffs"].reshape(-1).tolist(),
        }
        if f"{prefix}reprojection_error" in arrays:
            cam_data["reprojection_error"] = float(arrays[f"{prefix}reprojection_error"][0])
        if f"{prefix}T_robot_to_camera" in arrays:
            cam_data["T_robot_to_camera"] = arrays[f"{prefix}T_robot_to_camera"].tolist()
        if f"{prefix}T_ee_to_camera" in arrays:
            cam_data["T_ee_to_camera"] = arrays[f"{prefix}T_ee_to_camera"].tolist()
        cameras[name] = cam_data

    data: dict = {
        "version": int(arrays["version"][0]),
        "robot_type": str(arrays["robot_type"].item()),
        "robot_id": str(arrays["robot_id"].item()),
        "robot_pose_frame": str(arrays["robot_pose_frame"].item()),
        "length_unit": str(arrays["length_unit"].item()),
        "angle_unit": str(arrays["angle_unit"].item()),
        "intrinsics": {},
        "camera_mounts": {},
        "T_robot_to_camera": {},
        "T_ee_to_camera": {},
    }
    for name, cam in cameras.items():
        data["camera_mounts"][name] = cam["mount"]
        data["intrinsics"][name] = {
            "width": cam["width"],
            "height": cam["height"],
            "camera_matrix": cam["camera_matrix"],
            "dist_coeffs": cam["dist_coeffs"],
            "reprojection_error": cam.get("reprojection_error"),
        }
        if "T_robot_to_camera" in cam:
            data["T_robot_to_camera"][name] = cam["T_robot_to_camera"]
        if "T_ee_to_camera" in cam:
            data["T_ee_to_camera"][name] = cam["T_ee_to_camera"]

    charuco: dict = {}
    for key, value in arrays.items():
        if not key.startswith("charuco_"):
            continue
        field = key.removeprefix("charuco_")
        item = value.item() if value.dtype == object else value.reshape(-1)[0]
        charuco[field] = item
    if charuco:
        data["charuco"] = charuco

    if "T_robot_to_table" in arrays:
        data["T_robot_to_table"] = arrays["T_robot_to_table"].tolist()
        if "table_grid_points_robot_mm" in arrays:
            data["table_grid_points_robot_mm"] = arrays["table_grid_points_robot_mm"].tolist()
        if "table_notes" in arrays:
            data["table_notes"] = str(arrays["table_notes"].item())

    return SceneCalibration.from_dict(data)


def load_scene_calibration(path: Path | str) -> SceneCalibration:
    path = Path(path)
    if path.suffix == ".json":
        import json

        return SceneCalibration.from_dict(json.loads(path.read_text(encoding="utf-8")))
    with np.load(path, allow_pickle=True) as data:
        return scene_from_npz_arrays(dict(data))


def save_scene_calibration(
    calibration: SceneCalibration,
    path: Path | str,
    *,
    T_robot_to_ee: np.ndarray | None = None,
) -> Path:
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = scene_to_npz_arrays(calibration, T_robot_to_ee)
    np.savez(path, **arrays)
    return path


def load_or_create_scene_calibration(
    robot_type: str,
    robot_id: str,
    path: Path | str | None = None,
) -> SceneCalibration:
    fpath = Path(path) if path is not None else default_scene_calibration_path(robot_type, robot_id)
    if fpath.is_file():
        calib = load_scene_calibration(fpath)
        calib.robot_type = robot_type
        calib.robot_id = robot_id
        return calib
    return SceneCalibration(robot_type=robot_type, robot_id=robot_id)


def describe_npz(path: Path | str) -> dict[str, str]:
    """Human-readable summary of arrays stored in a calibration NPZ."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        summary: dict[str, str] = {}
        for key in sorted(data.files):
            arr = data[key]
            summary[key] = f"shape={arr.shape} dtype={arr.dtype}"
        return summary
