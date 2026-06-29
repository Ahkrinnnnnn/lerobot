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

"""Interactive hand-eye / scene calibration runner (robot-agnostic via adapters)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .adapters import HandEyeRobot
from .chessboard import CameraIntrinsics
from .hand_eye import CameraMount, HandEyeSample, hand_eye_target_origin_std, solve_hand_eye
from .io import default_scene_calibration_path, load_or_create_scene_calibration, save_scene_calibration
from .landmark import (
    camera_extrinsic_from_landmark,
    fuse_robot_to_landmark,
    get_eye_in_hand_extrinsic,
    intrinsics_from_scene,
    landmark_consistency_report,
    robot_to_landmark_from_eye_in_hand,
    save_landmark_as_table,
)
from .scene import (
    CameraCalibration,
    RobotPoseFrame,
    SceneCalibration,
)
from .target import CalibrationTargetConfig, calibrate_intrinsics, detect_target, draw_target, estimate_target_to_camera
from .transforms import transform_to_list

logger = logging.getLogger(__name__)


@dataclass
class CalibrationRunConfig:
    target: CalibrationTargetConfig
    robot_pose_frame: RobotPoseFrame = RobotPoseFrame.WORLD
    min_hand_eye_samples: int = 12
    min_landmark_samples: int = 6
    output_path: Path | None = None


def _resolve_output_path(robot: HandEyeRobot, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path.expanduser()
    return default_scene_calibration_path(robot.robot_type, robot.robot_id)


def _get_camera(robot: HandEyeRobot, camera_name: str):
    if camera_name not in robot.cameras:
        raise KeyError(f"Camera {camera_name!r} not in robot config. Available: {list(robot.cameras)}")
    return robot.cameras[camera_name]


def _intrinsics_to_camera_calib(intrinsics: CameraIntrinsics, mount: CameraMount) -> CameraCalibration:
    return CameraCalibration(
        mount=mount,
        width=intrinsics.width,
        height=intrinsics.height,
        camera_matrix=intrinsics.camera_matrix.tolist(),
        dist_coeffs=intrinsics.dist_coeffs.reshape(-1).tolist(),
        reprojection_error=intrinsics.reprojection_error,
    )


def _load_scene(robot: HandEyeRobot, cfg: CalibrationRunConfig) -> SceneCalibration:
    scene = load_or_create_scene_calibration(robot.robot_type, robot.robot_id, cfg.output_path)
    scene.robot_type = robot.robot_type
    scene.robot_id = robot.robot_id
    scene.robot_pose_frame = cfg.robot_pose_frame
    scene.charuco = cfg.target.to_dict()
    return scene


def _target_label(_target: CalibrationTargetConfig) -> str:
    return "charuco"


def collect_intrinsics_images(
    robot: HandEyeRobot,
    camera_name: str,
    target: CalibrationTargetConfig,
    *,
    stop_key: str = "q",
) -> list[np.ndarray]:
    cam = _get_camera(robot, camera_name)
    images: list[np.ndarray] = []
    window = f"hand_eye_intrinsics_{camera_name}"
    use_gui = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if use_gui:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    print(
        f"\n[Intrinsics] camera={camera_name!r} target={_target_label(target)}\n"
        f"  Move calibration target; Enter=capture, {stop_key}=done (need >= 3 views, recommend 15–25)."
    )
    while True:
        frame = cam.read()
        found = detect_target(frame, target)
        if use_gui:
            vis, found = draw_target(frame, target)
            cv2.putText(
                vis,
                f"captured={len(images)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if found else (0, 0, 255),
                2,
            )
            cv2.imshow(window, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        cmd = input(f"  [{len(images)} views] Enter=save, {stop_key}=done: ").strip().lower()
        if cmd == stop_key:
            break
        if cmd not in ("", "c"):
            continue
        if found:
            images.append(frame.copy())
            print(f"  saved view #{len(images)}")
        else:
            print("  skip: target not detected")

    if use_gui:
        cv2.destroyWindow(window)
    return images


def run_intrinsics_calibration(
    robot: HandEyeRobot,
    camera_name: str,
    mount: CameraMount,
    cfg: CalibrationRunConfig,
    *,
    images: list[np.ndarray] | None = None,
) -> CameraIntrinsics:
    if images is None:
        images = collect_intrinsics_images(robot, camera_name, cfg.target)
    intrinsics = calibrate_intrinsics(images, cfg.target)

    scene = _load_scene(robot, cfg)
    cam_calib = _intrinsics_to_camera_calib(intrinsics, mount)
    prev = scene.cameras.get(camera_name)
    if prev is not None:
        cam_calib.T_robot_to_camera = prev.T_robot_to_camera
        cam_calib.T_ee_to_camera = prev.T_ee_to_camera
    scene.cameras[camera_name] = cam_calib

    out = save_scene_calibration(scene, _resolve_output_path(robot, cfg.output_path))
    logger.info("Saved intrinsics %r → %s (RMS=%.4f)", camera_name, out, intrinsics.reprojection_error)
    return intrinsics


def collect_hand_eye_samples(
    robot: HandEyeRobot,
    camera_name: str,
    intrinsics: CameraIntrinsics,
    target: CalibrationTargetConfig,
    mount: CameraMount,
    robot_pose_frame: RobotPoseFrame,
    *,
    min_samples: int,
    stop_key: str = "q",
) -> list[HandEyeSample]:
    cam = _get_camera(robot, camera_name)
    samples: list[HandEyeSample] = []

    if mount == CameraMount.EYE_IN_HAND:
        setup = "ChArUco fixed on table; move arm so wrist camera sees it from many angles."
    else:
        setup = "ChArUco fixed on gripper; move arm so fixed camera sees it from many poses."

    print(
        f"\n[Hand-eye] camera={camera_name!r} mount={mount.value} target={_target_label(target)}\n"
        f"  {setup}\n"
        f"  Enter=capture, {stop_key}=done (need >= {min_samples})."
    )

    window = f"hand_eye_{camera_name}"
    use_gui = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if use_gui:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        frame = cam.read()
        T_target_to_cam = estimate_target_to_camera(frame, intrinsics, target)
        found = T_target_to_cam is not None
        if use_gui:
            vis, found = draw_target(frame, target)
            cv2.putText(
                vis,
                f"samples={len(samples)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if found else (0, 0, 255),
                2,
            )
            cv2.imshow(window, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        cmd = input(f"  [{len(samples)} samples] Enter=capture, {stop_key}=done: ").strip().lower()
        if cmd == stop_key:
            break
        if cmd not in ("", "c"):
            continue
        if T_target_to_cam is None:
            print("  skip: target not detected")
            continue
        T_robot_to_ee = robot.read_robot_to_ee(robot_pose_frame)
        samples.append(HandEyeSample(T_robot_to_ee=T_robot_to_ee, T_target_to_camera=T_target_to_cam))
        print(f"  captured sample #{len(samples)}")

    if use_gui:
        cv2.destroyWindow(window)
    if len(samples) < min_samples:
        raise RuntimeError(f"Collected {len(samples)} samples; need >= {min_samples}.")
    return samples


def run_hand_eye_calibration(
    robot: HandEyeRobot,
    camera_name: str,
    mount: CameraMount,
    cfg: CalibrationRunConfig,
    *,
    samples: list[HandEyeSample] | None = None,
) -> np.ndarray:
    scene = _load_scene(robot, cfg)
    if camera_name not in scene.cameras:
        raise RuntimeError(f"Run intrinsics for {camera_name!r} first (same output JSON).")
    cam_calib = scene.cameras[camera_name]
    if cam_calib.mount != mount:
        cam_calib.mount = mount

    intrinsics = CameraIntrinsics(
        width=cam_calib.width,
        height=cam_calib.height,
        camera_matrix=cam_calib.camera_matrix_np(),
        dist_coeffs=cam_calib.dist_coeffs_np(),
        reprojection_error=cam_calib.reprojection_error,
    )

    if samples is None:
        samples = collect_hand_eye_samples(
            robot,
            camera_name,
            intrinsics,
            cfg.target,
            mount,
            cfg.robot_pose_frame,
            min_samples=cfg.min_hand_eye_samples,
        )

    T_solved = solve_hand_eye(samples, mount)
    consistency = hand_eye_target_origin_std(samples, mount, T_solved)

    if mount == CameraMount.EYE_IN_HAND:
        cam_calib.T_ee_to_camera = transform_to_list(T_solved)
        cam_calib.T_robot_to_camera = None
    else:
        cam_calib.T_robot_to_camera = transform_to_list(T_solved)
        cam_calib.T_ee_to_camera = None

    scene.cameras[camera_name] = cam_calib
    out = save_scene_calibration(scene, _resolve_output_path(robot, cfg.output_path))
    logger.info(
        "Saved hand-eye %r (%s) → %s; consistency std=%.3f mm",
        camera_name,
        mount.value,
        out,
        consistency,
    )
    return T_solved


def collect_landmark_transforms(
    robot: HandEyeRobot,
    camera_name: str,
    cfg: CalibrationRunConfig,
    *,
    min_samples: int,
    stop_key: str = "q",
) -> list[np.ndarray]:
    """Multi-view wrist (eye-in-hand) observations of a fixed table ChArUco → T_robot_to_landmark."""
    scene = _load_scene(robot, cfg)
    if camera_name not in scene.cameras:
        raise RuntimeError(f"Run intrinsics for {camera_name!r} first.")
    T_ee_to_camera = get_eye_in_hand_extrinsic(scene, camera_name)
    intrinsics = intrinsics_from_scene(scene, camera_name)
    cam = _get_camera(robot, camera_name)
    transforms: list[np.ndarray] = []

    print(
        f"\n[Landmark map] camera={camera_name!r} target={_target_label(cfg.target)}\n"
        "  Glue ChArUco on table; move arm so wrist camera sees it from many poses.\n"
        f"  Enter=capture, {stop_key}=done (need >= {min_samples})."
    )

    window = f"landmark_map_{camera_name}"
    use_gui = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if use_gui:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        frame = cam.read()
        T_landmark_to_camera = estimate_target_to_camera(frame, intrinsics, cfg.target)
        found = T_landmark_to_camera is not None
        if use_gui:
            vis, found = draw_target(frame, cfg.target)
            cv2.putText(
                vis,
                f"samples={len(transforms)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if found else (0, 0, 255),
                2,
            )
            cv2.imshow(window, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        cmd = input(f"  [{len(transforms)} views] Enter=capture, {stop_key}=done: ").strip().lower()
        if cmd == stop_key:
            break
        if cmd not in ("", "c"):
            continue
        if T_landmark_to_camera is None:
            print("  skip: target not detected")
            continue
        T_robot_to_ee = robot.read_robot_to_ee(cfg.robot_pose_frame)
        T_robot_to_landmark = robot_to_landmark_from_eye_in_hand(
            T_robot_to_ee, T_ee_to_camera, T_landmark_to_camera
        )
        transforms.append(T_robot_to_landmark)
        print(f"  captured view #{len(transforms)}")

    if use_gui:
        cv2.destroyWindow(window)
    if len(transforms) < min_samples:
        raise RuntimeError(f"Collected {len(transforms)} views; need >= {min_samples}.")
    return transforms


def run_landmark_map_calibration(
    robot: HandEyeRobot,
    camera_name: str,
    cfg: CalibrationRunConfig,
    *,
    transforms: list[np.ndarray] | None = None,
) -> np.ndarray:
    if transforms is None:
        transforms = collect_landmark_transforms(
            robot, camera_name, cfg, min_samples=cfg.min_landmark_samples
        )
    T_fused, std_mm = fuse_robot_to_landmark(transforms)
    report = landmark_consistency_report(transforms)
    scene = _load_scene(robot, cfg)
    save_landmark_as_table(
        scene,
        T_fused,
        cfg.target,
        notes=f"{len(transforms)} wrist views, position_std={std_mm:.2f}mm",
    )
    out = save_scene_calibration(scene, _resolve_output_path(robot, cfg.output_path))
    logger.info(
        "Saved landmark/table map from %r → %s; std=%.2f mm max_pair=%.2f mm",
        camera_name,
        out,
        report["position_std_mm"],
        report["max_pairwise_mm"],
    )
    return T_fused


def collect_camera_via_landmark_transforms(
    robot: HandEyeRobot,
    camera_name: str,
    cfg: CalibrationRunConfig,
    *,
    min_samples: int = 1,
    stop_key: str = "q",
) -> list[np.ndarray]:
    """Fixed camera sees the same table ChArUco; infer T_robot_to_camera from saved table map."""
    scene = _load_scene(robot, cfg)
    if scene.table is None:
        raise RuntimeError("Run landmark_map first to define T_robot_to_table.")
    if camera_name not in scene.cameras:
        raise RuntimeError(f"Run intrinsics for {camera_name!r} first.")
    T_robot_to_table = np.array(scene.table.T_robot_to_table, dtype=np.float64)
    intrinsics = intrinsics_from_scene(scene, camera_name)
    cam = _get_camera(robot, camera_name)
    transforms: list[np.ndarray] = []

    print(
        f"\n[Camera via landmark] camera={camera_name!r} target={_target_label(cfg.target)}\n"
        "  Keep table ChArUco fixed; fixed camera should see it clearly.\n"
        f"  Enter=capture, {stop_key}=done (recommend >= 3 views)."
    )

    window = f"camera_via_landmark_{camera_name}"
    use_gui = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if use_gui:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        frame = cam.read()
        T_landmark_to_camera = estimate_target_to_camera(frame, intrinsics, cfg.target)
        found = T_landmark_to_camera is not None
        if use_gui:
            vis, found = draw_target(frame, cfg.target)
            cv2.putText(
                vis,
                f"samples={len(transforms)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if found else (0, 0, 255),
                2,
            )
            cv2.imshow(window, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        cmd = input(f"  [{len(transforms)} views] Enter=capture, {stop_key}=done: ").strip().lower()
        if cmd == stop_key:
            break
        if cmd not in ("", "c"):
            continue
        if T_landmark_to_camera is None:
            print("  skip: target not detected")
            continue
        T_robot_to_camera = camera_extrinsic_from_landmark(T_robot_to_table, T_landmark_to_camera)
        transforms.append(T_robot_to_camera)
        print(f"  captured view #{len(transforms)}")

    if use_gui:
        cv2.destroyWindow(window)
    if len(transforms) < min_samples:
        raise RuntimeError(f"Collected {len(transforms)} views; need >= {min_samples}.")
    return transforms


def run_camera_via_landmark_calibration(
    robot: HandEyeRobot,
    camera_name: str,
    mount: CameraMount,
    cfg: CalibrationRunConfig,
    *,
    transforms: list[np.ndarray] | None = None,
) -> np.ndarray:
    if transforms is None:
        transforms = collect_camera_via_landmark_transforms(robot, camera_name, cfg, min_samples=1)

    T_fused, std_mm = fuse_robot_to_landmark(transforms)
    scene = _load_scene(robot, cfg)
    if camera_name not in scene.cameras:
        raise RuntimeError(f"Run intrinsics for {camera_name!r} first.")
    cam_calib = scene.cameras[camera_name]
    cam_calib.mount = mount
    if mount == CameraMount.EYE_TO_HAND:
        cam_calib.T_robot_to_camera = transform_to_list(T_fused)
        cam_calib.T_ee_to_camera = None
    else:
        raise ValueError("camera_via_landmark expects eye_to_hand fixed camera.")

    scene.cameras[camera_name] = cam_calib
    out = save_scene_calibration(scene, _resolve_output_path(robot, cfg.output_path))
    logger.info(
        "Saved %r extrinsic via landmark → %s; fused position std=%.2f mm (%d views)",
        camera_name,
        out,
        std_mm,
        len(transforms),
    )
    return T_fused


def validate_scene_calibration(robot: HandEyeRobot, cfg: CalibrationRunConfig) -> dict[str, Any]:
    scene = _load_scene(robot, cfg)
    T_robot_to_ee = robot.read_robot_to_ee(cfg.robot_pose_frame)
    out_path = save_scene_calibration(
        scene,
        _resolve_output_path(robot, cfg.output_path),
        T_robot_to_ee=T_robot_to_ee,
    )
    from .io import describe_npz

    return {
        "output_path": str(out_path),
        "format": "npz",
        "arrays": describe_npz(out_path),
    }


def print_methods_help() -> None:
    print(
        """
ChArUco 视觉标定
--------------
配置文件:
  crp.json              机器人 + board_config_path
  charuco_board.json    标定板尺寸 (squares_x/y, square_size_mm, marker_size_mm, aruco_dict)

输出 NPZ (~/.cache/.../<robot_id>_calibration.npz):
  {camera}__camera_matrix, {camera}__dist_coeffs     内参
  {camera}__T_robot_to_camera                        固定相机外参 (top)
  {camera}__T_ee_to_camera                           腕部相机外参 (wrist)
  {camera}__T_robot_to_camera_resolved             validate 时写入的合成外参
  T_robot_to_table                                   桌面坐标系
  table_grid_points_robot_mm                         桌面 ChArUco 角点 (机器人系, mm)
  charuco_*                                          标定板参数

流程:

  lerobot-calibrate-hand-eye --config_path=examples/hand_eye_calibration/crp.json \\
      --phase=intrinsics --camera=top --mount=eye_to_hand
  lerobot-calibrate-hand-eye --config_path=... --phase=intrinsics --camera=wrist --mount=eye_in_hand
  lerobot-calibrate-hand-eye --config_path=... --phase=hand_eye --camera=wrist --mount=eye_in_hand
  lerobot-calibrate-hand-eye --config_path=... --phase=landmark_map --camera=wrist
  lerobot-calibrate-hand-eye --config_path=... --phase=camera_via_landmark --camera=top --mount=eye_to_hand
  lerobot-calibrate-hand-eye --config_path=... --phase=validate
"""
    )
