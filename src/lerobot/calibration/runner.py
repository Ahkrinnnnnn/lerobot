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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .adapters import HandEyeRobot
from .chessboard import CameraIntrinsics
from .hand_eye import (
    CameraMount,
    HandEyeSample,
    hand_eye_motion_residual_mm,
    hand_eye_target_origin_std,
    per_sample_board_origin_mm,
    solve_hand_eye,
)
from .experiment_paths import (
    DEFAULT_OUTPUTS_BASE,
    ExperimentPaths,
    archive_existing_phase_dir,
    ensure_experiment_meta,
    load_experiment_intrinsics,
    record_phase_run,
    save_experiment_intrinsics,
)
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
from .interactive import LiveSessionConfig, default_help_lines, run_live_session
from .session_output import CalibrationSessionOutput
from .sync_capture import SyncCapture, capture_frame_and_pose

logger = logging.getLogger(__name__)


def _debug():
    from . import debug as debug_mod

    return debug_mod


@dataclass
class CalibrationRunConfig:
    target: CalibrationTargetConfig
    robot_pose_frame: RobotPoseFrame = RobotPoseFrame.WORLD
    min_hand_eye_samples: int = 12
    min_landmark_samples: int = 6
    also_collect_top: bool = True
    top_camera: str = "top"
    wrist_camera: str = "wrist"
    calibrate_top_intrinsics: bool = False
    debug: bool = True
    experiment_dir: Path | None = None
    output_path: Path | None = None


def _experiment_paths(cfg: CalibrationRunConfig) -> ExperimentPaths | None:
    if cfg.experiment_dir is None:
        return None
    return ExperimentPaths(root=cfg.experiment_dir.expanduser())


def _hand_eye_min_kept(cfg: CalibrationRunConfig) -> int:
    return max(cfg.min_hand_eye_samples, 3)


def _artifacts_base(cfg: CalibrationRunConfig) -> Path:
    if cfg.experiment_dir is not None:
        return cfg.experiment_dir.expanduser()
    return DEFAULT_OUTPUTS_BASE.expanduser()


def _open_session(
    robot: HandEyeRobot,
    cfg: CalibrationRunConfig,
    phase: str,
    camera_name: str,
) -> CalibrationSessionOutput:
    paths = _experiment_paths(cfg)
    if paths is not None:
        archive_existing_phase_dir(paths, phase, camera_name)
        phase_dir = paths.phase_dir(phase, camera_name)
        session = CalibrationSessionOutput.open(phase_dir)
        ensure_experiment_meta(
            paths,
            robot_id=robot.robot_id,
            robot_type=robot.robot_type,
            experiment_name=paths.root.name,
        )
        if cfg.debug:
            print(f"[experiment] dir → {paths.root.resolve()}", flush=True)
            print(f"[experiment] phase → {phase_dir.relative_to(paths.root)}", flush=True)
        return session
    return CalibrationSessionOutput.create(_artifacts_base(cfg), robot.robot_id, phase, camera_name)


def _session_log_header(
    session: CalibrationSessionOutput,
    robot: HandEyeRobot,
    cfg: CalibrationRunConfig,
    *,
    phase: str,
    camera: str,
    **extra: object,
) -> tuple[str, ...]:
    lines = [
        f"session_dir={session.root}",
        f"phase={phase!r}",
        f"camera={camera!r}",
        f"robot_id={robot.robot_id!r}",
        f"robot_type={robot.robot_type!r}",
        f"robot_pose_frame={cfg.robot_pose_frame.value!r}",
        f"debug={cfg.debug}",
        f"also_collect_top={cfg.also_collect_top}",
        f"top_camera={cfg.top_camera!r}",
        f"target={cfg.target.to_dict()}",
    ]
    for key, value in extra.items():
        lines.append(f"{key}={value!r}")
    return tuple(lines)


def _begin_session_log(
    session: CalibrationSessionOutput,
    robot: HandEyeRobot,
    cfg: CalibrationRunConfig,
    *,
    phase: str,
    camera: str,
    **extra: object,
) -> None:
    session.begin_log(
        header_lines=_session_log_header(session, robot, cfg, phase=phase, camera=camera, **extra)
    )


def _resolve_output_path(robot: HandEyeRobot, cfg: CalibrationRunConfig) -> Path:
    if cfg.output_path is not None:
        return cfg.output_path.expanduser()
    if cfg.experiment_dir is not None:
        return cfg.experiment_dir.expanduser() / "calibration.npz"
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
    scene = load_or_create_scene_calibration(
        robot.robot_type, robot.robot_id, _resolve_output_path(robot, cfg)
    )
    scene.robot_type = robot.robot_type
    scene.robot_id = robot.robot_id
    scene.robot_pose_frame = cfg.robot_pose_frame
    scene.charuco = cfg.target.to_dict()
    return scene


def _resolve_top_camera(robot: HandEyeRobot, cfg: CalibrationRunConfig) -> str | None:
    if not cfg.also_collect_top:
        return None
    if cfg.top_camera not in robot.cameras:
        logger.warning("Top camera %r not found; skipping dual capture.", cfg.top_camera)
        return None
    return cfg.top_camera


def _hand_eye_sample_diagnostics(
    samples: list[HandEyeSample],
    T_ee_to_camera: np.ndarray,
    captures: list[SyncCapture],
) -> dict[str, Any]:
    origins = per_sample_board_origin_mm(samples, CameraMount.EYE_IN_HAND, T_ee_to_camera)
    mean_origin = origins.mean(axis=0)
    dists = np.linalg.norm(origins - mean_origin, axis=1)
    motion = hand_eye_motion_residual_mm(samples, T_ee_to_camera)
    per_sample = []
    for i, (sample, cap, dist) in enumerate(zip(samples, captures, dists, strict=True)):
        per_sample.append(
            {
                "index": i,
                "board_origin_mm": origins[i].tolist(),
                "origin_deviation_mm": float(dist),
                "sync_ms": cap.total_sync_ms,
                "wrist_frame_ms": cap.frame_read_ms,
                "top_frame_ms": cap.top_frame_read_ms,
                "pose_ms": cap.pose_read_ms,
            }
        )
    return {
        "board_origin_mean_mm": mean_origin.tolist(),
        "board_origin_spread_mm": (origins.max(axis=0) - origins.min(axis=0)).tolist(),
        "hand_eye_motion_residual_mm": motion,
        "per_sample": per_sample,
        "likely_causes_if_high_std": [
            "Robot pose noise or wrong frame (try --robot_pose_frame=user)",
            "Arm moved during capture (keep still until save completes)",
            "Board not rigidly fixed on table",
            "Large sync latency (see per_sample sync_ms)",
        ],
    }


def _solve_top_extrinsic_via_board(
    captures: list[SyncCapture],
    samples: list[HandEyeSample],
    T_ee_to_wrist: np.ndarray,
    top_intrinsics: CameraIntrinsics,
    target: CalibrationTargetConfig,
) -> tuple[np.ndarray, float, list[np.ndarray], list[dict[str, Any]]]:
    """Fuse ``T_robot_to_top`` from wrist board chain + top PnP per snapshot."""
    transforms: list[np.ndarray] = []
    board_transforms: list[np.ndarray] = []
    diags: list[dict[str, Any]] = []
    for i, (cap, sample) in enumerate(zip(captures, samples, strict=True)):
        if cap.top_image is None:
            continue
        T_board_to_top = estimate_target_to_camera(cap.top_image, top_intrinsics, target)
        if T_board_to_top is None:
            continue
        T_robot_to_board = robot_to_landmark_from_eye_in_hand(
            sample.T_robot_to_ee, T_ee_to_wrist, sample.T_target_to_camera
        )
        T_robot_to_top = camera_extrinsic_from_landmark(T_robot_to_board, T_board_to_top)
        transforms.append(T_robot_to_top)
        board_transforms.append(T_robot_to_board)
        diags.append(
            {
                "index": i,
                "T_robot_to_top_translation_mm": T_robot_to_top[:3, 3].tolist(),
                "T_robot_to_board_translation_mm": T_robot_to_board[:3, 3].tolist(),
            }
        )
    if not transforms:
        raise RuntimeError("No valid dual-camera samples (top must see ChArUco in each saved frame).")
    T_top, std_mm = fuse_robot_to_landmark(transforms)
    return T_top, std_mm, board_transforms, diags


def _load_or_calibrate_top_intrinsics(
    robot: HandEyeRobot,
    cfg: CalibrationRunConfig,
    top_images: list[np.ndarray],
) -> CameraIntrinsics:
    if not cfg.calibrate_top_intrinsics:
        from_exp = _load_intrinsics_for_camera(robot, cfg, cfg.top_camera)
        if from_exp is not None:
            return from_exp
    scene = _load_scene(robot, cfg)
    if (
        not cfg.calibrate_top_intrinsics
        and cfg.top_camera in scene.cameras
        and scene.cameras[cfg.top_camera].camera_matrix
    ):
        return intrinsics_from_scene(scene, cfg.top_camera)
    if len(top_images) < 3:
        raise RuntimeError(
            f"Top intrinsics missing and only {len(top_images)} top images collected; need >= 3."
        )
    intrinsics = calibrate_intrinsics(top_images, cfg.target)
    cam_calib = _intrinsics_to_camera_calib(intrinsics, CameraMount.EYE_TO_HAND)
    prev = scene.cameras.get(cfg.top_camera)
    if prev is not None:
        cam_calib.T_robot_to_camera = prev.T_robot_to_camera
    scene.cameras[cfg.top_camera] = cam_calib
    save_scene_calibration(scene, _resolve_output_path(robot, cfg))
    _persist_canonical_intrinsics(cfg, cfg.top_camera, intrinsics, phase="top_from_dual_capture")
    return intrinsics


def _target_label(_target: CalibrationTargetConfig) -> str:
    return "charuco"


def collect_intrinsics_images(
    robot: HandEyeRobot,
    camera_name: str,
    target: CalibrationTargetConfig,
    cfg: CalibrationRunConfig | None = None,
    *,
    stop_key: str = "q",
) -> tuple[list[np.ndarray], CalibrationSessionOutput | None]:
    images: list[np.ndarray] = []
    session = _open_session(robot, cfg, "intrinsics", camera_name) if cfg is not None else None
    if session is not None:
        _begin_session_log(session, robot, cfg, phase="intrinsics", camera=camera_name)

    try:
        def process_frame(frame: np.ndarray):
            vis, found = draw_target(frame, target)
            return vis, found, None

        def on_save():
            cap = capture_frame_and_pose(
                robot, camera_name, RobotPoseFrame.WORLD, record_pose=False
            )
            if not detect_target(cap.image, target):
                return False
            images.append(cap.image.copy())
            if session is not None:
                session.save_capture(len(images) - 1, cap, target)
            return True

        session_cfg = LiveSessionConfig(
            window_name=f"hand_eye_intrinsics_{camera_name}",
            title="Intrinsics",
            help_lines=(
                *default_help_lines(),
                "Need >= 3 views (recommend 15–25)",
            ),
            counter_label="views",
            stop_key=stop_key,
        )

        print(
            f"\n[Intrinsics] camera={camera_name!r} target={_target_label(target)}\n"
            f"  Move calibration target; live preview in OpenCV window."
        )

        count = run_live_session(
            robot,
            camera_name,
            session_cfg,
            process_frame=process_frame,
            on_save=on_save,
        )
        if count < 3:
            raise RuntimeError(f"Captured {count} views; need >= 3 for intrinsics.")
        logger.info("Session artifacts → %s", session.root if session else "n/a")
        return images, session
    finally:
        if session is not None:
            session.end_log()


def run_intrinsics_calibration(
    robot: HandEyeRobot,
    camera_name: str,
    mount: CameraMount,
    cfg: CalibrationRunConfig,
    *,
    images: list[np.ndarray] | None = None,
) -> CameraIntrinsics:
    session: CalibrationSessionOutput | None = None
    if images is None:
        images, session = collect_intrinsics_images(robot, camera_name, cfg.target, cfg)
    intrinsics = calibrate_intrinsics(images, cfg.target)

    scene = _load_scene(robot, cfg)
    cam_calib = _intrinsics_to_camera_calib(intrinsics, mount)
    prev = scene.cameras.get(camera_name)
    if prev is not None:
        cam_calib.T_robot_to_camera = prev.T_robot_to_camera
        cam_calib.T_ee_to_camera = prev.T_ee_to_camera
    scene.cameras[camera_name] = cam_calib

    out = save_scene_calibration(scene, _resolve_output_path(robot, cfg))
    canonical = _persist_canonical_intrinsics(cfg, camera_name, intrinsics, phase="intrinsics")
    if session is not None:
        session.write_intrinsics(intrinsics, camera_name)
        report: dict[str, Any] = {
            "phase": "intrinsics",
            "camera": camera_name,
            "num_views": len(images),
            "intrinsics_rms_px": intrinsics.reprojection_error,
            "intrinsics": session.root.joinpath("intrinsics.json").name,
            "npz_output": str(out),
            "log_file": session.log_path.name,
        }
        if canonical is not None:
            paths = _experiment_paths(cfg)
            rel = canonical.relative_to(paths.root) if paths else canonical
            report["intrinsics_canonical"] = str(rel)
        session.write_report(report)
        _record_phase_if_experiment(
            cfg,
            session,
            phase="intrinsics",
            camera=camera_name,
            extra={"intrinsics_rms_px": intrinsics.reprojection_error},
        )
    logger.info("Saved intrinsics %r → %s (RMS=%.4f)", camera_name, out, intrinsics.reprojection_error)
    return intrinsics


def _debug_intrinsics_for_capture(
    robot: HandEyeRobot,
    cfg: CalibrationRunConfig,
    camera_name: str,
    captures: list[SyncCapture],
    target: CalibrationTargetConfig,
    *,
    scene_intrinsics: CameraIntrinsics | None,
) -> CameraIntrinsics | None:
    """Intrinsics for live vision debug: reuse scene K/dist or provisional from >=3 views."""
    if scene_intrinsics is not None:
        return scene_intrinsics
    if len(captures) < 3:
        return None
    return calibrate_intrinsics([cap.image for cap in captures], target)


def _try_scene_intrinsics(
    robot: HandEyeRobot,
    cfg: CalibrationRunConfig,
    camera_name: str,
) -> CameraIntrinsics | None:
    scene = _load_scene(robot, cfg)
    if camera_name not in scene.cameras:
        return None
    cam = scene.cameras[camera_name]
    if not cam.camera_matrix:
        return None
    return intrinsics_from_scene(scene, camera_name)


def _load_intrinsics_for_camera(
    robot: HandEyeRobot,
    cfg: CalibrationRunConfig,
    camera_name: str,
) -> CameraIntrinsics | None:
    """Prefer ``intrinsics/{camera}.json`` in the active experiment folder."""
    paths = _experiment_paths(cfg)
    if paths is not None:
        from_exp = load_experiment_intrinsics(paths, camera_name)
        if from_exp is not None:
            return from_exp
    return _try_scene_intrinsics(robot, cfg, camera_name)


def _persist_canonical_intrinsics(
    cfg: CalibrationRunConfig,
    camera_name: str,
    intrinsics: CameraIntrinsics,
    *,
    phase: str,
) -> Path | None:
    paths = _experiment_paths(cfg)
    if paths is None:
        return None
    return save_experiment_intrinsics(paths, camera_name, intrinsics, phase=phase)


def _record_phase_if_experiment(
    cfg: CalibrationRunConfig,
    session: CalibrationSessionOutput | None,
    *,
    phase: str,
    camera: str,
    extra: dict[str, Any] | None = None,
) -> None:
    if session is None:
        return
    paths = _experiment_paths(cfg)
    if paths is None:
        return
    record_phase_run(paths, phase=phase, camera=camera, phase_dir=session.root, extra=extra)


def collect_wrist_sync_captures(
    robot: HandEyeRobot,
    camera_name: str,
    target: CalibrationTargetConfig,
    robot_pose_frame: RobotPoseFrame,
    cfg: CalibrationRunConfig,
    *,
    min_samples: int,
    session: CalibrationSessionOutput,
    stop_key: str = "q",
) -> list[SyncCapture]:
    """Collect synchronized (image, pose) pairs for combined intrinsics + hand-eye."""
    captures: list[SyncCapture] = []
    top_camera = _resolve_top_camera(robot, cfg)
    prev_pose_6d: list[float] | None = None
    prev_T_target: np.ndarray | None = None
    scene_intrinsics = _load_intrinsics_for_camera(robot, cfg, camera_name)

    setup_lines = [
        "ChArUco fixed on table; drag arm on teach pendant, then save.",
        f"Collect >= {min_samples} saves (intrinsics + hand-eye).",
        f"q works only after >= {min_samples} saves.",
    ]
    if top_camera is not None:
        setup_lines.append(f"Also captures {top_camera!r} each save → T_robot_to_top via static board.")
    if cfg.debug:
        if scene_intrinsics is not None:
            rms = scene_intrinsics.reprojection_error
            rms_s = f"{rms:.4f}px" if rms is not None else "from scene"
            print(
                f"[debug] live vision PnP: using wrist intrinsics from scene (RMS={rms_s})",
                flush=True,
            )
        else:
            print(
                "[debug] live vision PnP: will start after >=3 saved views"
                " (or load prior intrinsics from scene npz)",
                flush=True,
            )

    print(
        f"\n[Intrinsics + Hand-eye] camera={camera_name!r} target={_target_label(target)}\n"
        + "\n".join(f"  {line}" for line in setup_lines)
    )
    if cfg.debug:
        _debug().debug_print_session_header(
            robot_pose_frame=robot_pose_frame,
            camera=camera_name,
            top_camera=top_camera,
        )

    def process_frame(frame: np.ndarray):
        vis, found = draw_target(frame, target)
        return vis, found, None

    def on_save():
        nonlocal prev_pose_6d, prev_T_target
        cap = capture_frame_and_pose(
            robot,
            camera_name,
            robot_pose_frame,
            also_capture_top=top_camera,
        )
        if not detect_target(cap.image, target):
            print("  skip: wrist camera — ChArUco not detected", flush=True)
            return False
        if top_camera is not None:
            if cap.top_image is None or not detect_target(cap.top_image, target):
                print("  skip: top camera — ChArUco not detected", flush=True)
                return False
        captures.append(cap)
        session.save_capture(len(captures) - 1, cap, target)
        idx = len(captures) - 1
        debug_intrinsics = _debug_intrinsics_for_capture(
            robot, cfg, camera_name, captures, target, scene_intrinsics=scene_intrinsics
        )
        T_target_to_cam = None
        if debug_intrinsics is not None:
            T_target_to_cam = estimate_target_to_camera(cap.image, debug_intrinsics, target)
        if cfg.debug:
            _debug().debug_print_capture(
                idx,
                cap,
                target,
                robot_pose_frame=robot_pose_frame,
                pose_6d=cap.pose_6d,
                prev_pose_6d=prev_pose_6d,
                T_target_to_camera=T_target_to_cam,
                prev_T_target_to_camera=prev_T_target,
                intrinsics=debug_intrinsics,
                image=cap.image,
            )
            if T_target_to_cam is None:
                print(
                    f"        vision: pending (saved {len(captures)}/3 views for provisional intrinsics)",
                    flush=True,
                )
        if cap.pose_6d is not None:
            prev_pose_6d = cap.pose_6d
        if T_target_to_cam is not None:
            prev_T_target = T_target_to_cam
        print(
            f"    sync: wrist={cap.frame_read_ms:.1f}ms"
            + (f" top={cap.top_frame_read_ms:.1f}ms" if top_camera else "")
            + f" pose={cap.pose_read_ms:.1f}ms",
            flush=True,
        )
        return True

    session_cfg = LiveSessionConfig(
        window_name=f"wrist_calib_{camera_name}",
        title="Wrist intrinsics + hand-eye",
        help_lines=default_help_lines() + (f"collect >= {min_samples}",),
        counter_label="samples",
        stop_key=stop_key,
        **(
            {
                "secondary_camera": top_camera,
                "secondary_window_name": f"top_calib_{top_camera}",
                "secondary_title": f"Top ({top_camera})",
                "secondary_help_lines": (
                    "green = ChArUco detected in top view",
                    "save requires wrist + top both green",
                ),
            }
            if top_camera is not None
            else {}
        ),
    )

    count = run_live_session(
        robot,
        camera_name,
        session_cfg,
        process_frame=process_frame,
        on_save=on_save,
        min_count=min_samples,
    )
    if count < min_samples:
        raise RuntimeError(f"Collected {count} samples; need >= {min_samples}.")
    return captures


def run_intrinsics_and_hand_eye_calibration(
    robot: HandEyeRobot,
    camera_name: str,
    mount: CameraMount,
    cfg: CalibrationRunConfig,
) -> tuple[CameraIntrinsics, np.ndarray]:
    """Single wrist session: calibrate intrinsics then hand-eye on the same images."""
    if mount != CameraMount.EYE_IN_HAND:
        raise ValueError("intrinsics_and_hand_eye is intended for wrist eye_in_hand cameras.")

    min_samples = _hand_eye_min_kept(cfg)
    session = _open_session(robot, cfg, "intrinsics_and_hand_eye", camera_name)
    _begin_session_log(
        session,
        robot,
        cfg,
        phase="intrinsics_and_hand_eye",
        camera=camera_name,
        mount=mount.value,
    )
    try:
        captures = collect_wrist_sync_captures(
            robot,
            camera_name,
            cfg.target,
            cfg.robot_pose_frame,
            cfg,
            min_samples=min_samples,
            session=session,
        )

        images = [cap.image for cap in captures]
        intrinsics = calibrate_intrinsics(images, cfg.target)
        if cfg.debug:
            print(f"[debug] wrist intrinsics RMS={intrinsics.reprojection_error:.4f}px", flush=True)

        samples: list[HandEyeSample] = []
        for i, cap in enumerate(captures):
            T_target_to_cam = estimate_target_to_camera(cap.image, intrinsics, cfg.target)
            if T_target_to_cam is None:
                raise RuntimeError(f"PnP failed on saved sample {i} after intrinsics calibration.")
            samples.append(
                HandEyeSample(T_robot_to_ee=cap.T_robot_to_ee, T_target_to_camera=T_target_to_cam)
            )
            session.save_capture(
                i, cap, cfg.target, intrinsics=intrinsics, T_target_to_camera=T_target_to_cam
            )

        if cfg.debug:
            _debug().debug_print_samples_robot_vs_vision(
                captures,
                samples,
                cfg.target,
                intrinsics,
                robot_pose_frame=cfg.robot_pose_frame,
            )
        T_solved = solve_hand_eye(samples, mount)
        consistency = hand_eye_target_origin_std(samples, mount, T_solved)
        if cfg.debug:
            _debug().debug_print_hand_eye_summary(
                samples,
                T_solved,
                mount=mount,
                intrinsics=intrinsics,
                target=cfg.target,
                images=[cap.image for cap in captures],
                captures=captures,
            )
        wrist_diag = _hand_eye_sample_diagnostics(samples, T_solved, captures)

        scene = _load_scene(robot, cfg)
        cam_calib = _intrinsics_to_camera_calib(intrinsics, mount)
        cam_calib.T_ee_to_camera = transform_to_list(T_solved)
        cam_calib.T_robot_to_camera = None
        scene.cameras[camera_name] = cam_calib

        report: dict[str, Any] = {
            "phase": "intrinsics_and_hand_eye",
            "camera": camera_name,
            "mount": mount.value,
            "num_samples": len(samples),
            "min_hand_eye_samples": min_samples,
            "intrinsics_rms_px": intrinsics.reprojection_error,
            "intrinsics_source": {
                "type": "same_session",
                "note": "K/dist calibrated from the same saved images used for hand-eye.",
            },
            "robot_pose_frame": cfg.robot_pose_frame.value,
            "wrist_consistency_std_mm": consistency,
            "consistency_std_mm": consistency,
            "T_ee_to_camera": T_solved.tolist(),
            "wrist_diagnostics": wrist_diag,
            "log_file": session.log_path.name,
            "diagnosis": (
                "PnP reprojection is usually <1px when intrinsics_rms_px is low. "
                "If wrist_consistency_std_mm is still >>20mm, the mismatch is in the "
                "robot pose chain (CRP frame / TCP / arm flex), not ChArUco detection."
            ),
        }

        top_camera = _resolve_top_camera(robot, cfg)
        if top_camera is not None:
            top_images = [cap.top_image for cap in captures if cap.top_image is not None]
            top_intrinsics = _load_or_calibrate_top_intrinsics(robot, cfg, top_images)
            T_robot_to_top, top_std_mm, board_transforms, top_diags = _solve_top_extrinsic_via_board(
                captures,
                samples,
                T_solved,
                top_intrinsics,
                cfg.target,
            )
            T_robot_to_board, board_std_mm = fuse_robot_to_landmark(board_transforms)
            save_landmark_as_table(
                scene,
                T_robot_to_board,
                cfg.target,
                notes=f"{len(board_transforms)} dual-camera snapshots, board_std={board_std_mm:.2f}mm",
            )
            top_calib = scene.cameras.get(top_camera) or _intrinsics_to_camera_calib(
                top_intrinsics, CameraMount.EYE_TO_HAND
            )
            top_calib.mount = CameraMount.EYE_TO_HAND
            top_calib.T_robot_to_camera = transform_to_list(T_robot_to_top)
            top_calib.T_ee_to_camera = None
            scene.cameras[top_camera] = top_calib
            report["top_camera"] = top_camera
            report["top_extrinsic_std_mm"] = top_std_mm
            report["top_T_robot_to_camera"] = T_robot_to_top.tolist()
            report["T_robot_to_board"] = T_robot_to_board.tolist()
            report["board_fusion_std_mm"] = board_std_mm
            report["top_dual_camera_samples"] = top_diags
            if cfg.debug:
                print(
                    f"[debug] top T_robot_to_camera fusion std={top_std_mm:.2f}mm"
                    f" board_via_wrist std={board_std_mm:.2f}mm ({len(top_diags)} views)",
                    flush=True,
                )
            logger.info(
                "Top %r via static board: fused std=%.2f mm (%d views)",
                top_camera,
                top_std_mm,
                len(top_diags),
            )

        out = save_scene_calibration(scene, _resolve_output_path(robot, cfg))
        report["npz_output"] = str(out)
        canonical = _persist_canonical_intrinsics(
            cfg, camera_name, intrinsics, phase="intrinsics_and_hand_eye"
        )
        if canonical is not None:
            paths = _experiment_paths(cfg)
            report["intrinsics_canonical"] = str(
                canonical.relative_to(paths.root) if paths else canonical
            )
        session.write_intrinsics(intrinsics, camera_name)
        session.write_samples_npz(captures, samples=samples)
        report["intrinsics"] = "intrinsics.json"
        session.write_report(report)
        _record_phase_if_experiment(
            cfg,
            session,
            phase="intrinsics_and_hand_eye",
            camera=camera_name,
            extra={
                "consistency_std_mm": consistency,
                "intrinsics_rms_px": intrinsics.reprojection_error,
            },
        )
        logger.info(
            "Saved wrist intrinsics + hand-eye %r → %s; RMS=%.4f consistency std=%.3f mm",
            camera_name,
            out,
            intrinsics.reprojection_error,
            consistency,
        )
        logger.info("Session artifacts → %s", session.root)
        return intrinsics, T_solved
    finally:
        session.end_log()


def collect_hand_eye_samples(
    robot: HandEyeRobot,
    camera_name: str,
    intrinsics: CameraIntrinsics,
    target: CalibrationTargetConfig,
    mount: CameraMount,
    robot_pose_frame: RobotPoseFrame,
    cfg: CalibrationRunConfig,
    *,
    min_samples: int,
    stop_key: str = "q",
    session: CalibrationSessionOutput | None = None,
) -> tuple[list[HandEyeSample], CalibrationSessionOutput | None]:
    if session is None:
        session = _open_session(robot, cfg, "hand_eye", camera_name)

    owns_log = not session.log_active
    if owns_log:
        _begin_session_log(
            session,
            robot,
            cfg,
            phase="hand_eye",
            camera=camera_name,
            mount=mount.value,
        )

    try:
        samples: list[HandEyeSample] = []
        prev_pose_6d: list[float] | None = None
        prev_T_target: np.ndarray | None = None

        if mount == CameraMount.EYE_IN_HAND:
            setup = "ChArUco fixed on table; move arm so wrist camera sees it from many angles."
        else:
            setup = "ChArUco fixed on gripper; move arm so fixed camera sees it from many poses."

        print(
            f"\n[Hand-eye] camera={camera_name!r} mount={mount.value} target={_target_label(target)}\n"
            f"  {setup}\n"
            f"  Collect >= {min_samples}; q works only after minimum reached."
        )
        if cfg.debug:
            _debug().debug_print_session_header(
                robot_pose_frame=robot_pose_frame,
                camera=camera_name,
                top_camera=None,
            )

        def process_frame(frame: np.ndarray):
            T_target_to_cam = estimate_target_to_camera(frame, intrinsics, target)
            vis, found = draw_target(
                frame, target, intrinsics=intrinsics, T_target_to_camera=T_target_to_cam
            )
            return vis, found, T_target_to_cam

        def on_save():
            nonlocal prev_pose_6d, prev_T_target
            cap = capture_frame_and_pose(robot, camera_name, robot_pose_frame)
            T_target_to_cam = estimate_target_to_camera(cap.image, intrinsics, target)
            if T_target_to_cam is None:
                return False
            samples.append(
                HandEyeSample(T_robot_to_ee=cap.T_robot_to_ee, T_target_to_camera=T_target_to_cam)
            )
            idx = len(samples) - 1
            session.save_capture(
                idx, cap, target, intrinsics=intrinsics, T_target_to_camera=T_target_to_cam
            )
            if cfg.debug:
                _debug().debug_print_capture(
                    idx,
                    cap,
                    target,
                    robot_pose_frame=robot_pose_frame,
                    pose_6d=cap.pose_6d,
                    prev_pose_6d=prev_pose_6d,
                    T_target_to_camera=T_target_to_cam,
                    prev_T_target_to_camera=prev_T_target,
                    intrinsics=intrinsics,
                    image=cap.image,
                )
            if cap.pose_6d is not None:
                prev_pose_6d = cap.pose_6d
            prev_T_target = T_target_to_cam
            print(
                f"    sync: frame={cap.frame_read_ms:.1f}ms pose={cap.pose_read_ms:.1f}ms",
                flush=True,
            )
            return True

        session_cfg = LiveSessionConfig(
            window_name=f"hand_eye_{camera_name}",
            title="Hand-eye",
            help_lines=default_help_lines() + (f"collect >= {min_samples}",),
            counter_label="samples",
            stop_key=stop_key,
        )

        count = run_live_session(
            robot,
            camera_name,
            session_cfg,
            process_frame=process_frame,
            on_save=on_save,
            min_count=min_samples,
        )

        if count < min_samples:
            raise RuntimeError(f"Collected {count} samples; need >= {min_samples}.")
        logger.info("Session artifacts → %s", session.root)
        return samples, session
    finally:
        if owns_log:
            session.end_log()


def run_hand_eye_calibration(
    robot: HandEyeRobot,
    camera_name: str,
    mount: CameraMount,
    cfg: CalibrationRunConfig,
    *,
    samples: list[HandEyeSample] | None = None,
) -> np.ndarray:
    out_path = _resolve_output_path(robot, cfg)
    paths = _experiment_paths(cfg)
    intrinsics = _load_intrinsics_for_camera(robot, cfg, camera_name)
    if intrinsics is None:
        json_hint = paths.intrinsics_json(camera_name) if paths else out_path
        raise RuntimeError(
            f"No intrinsics for {camera_name!r}. Run --phase=intrinsics in the same experiment first.\n"
            f"Expected: {json_hint}"
        )

    scene = _load_scene(robot, cfg)
    cam_calib = scene.cameras.get(camera_name)
    if cam_calib is None:
        cam_calib = _intrinsics_to_camera_calib(intrinsics, mount)
        scene.cameras[camera_name] = cam_calib
    elif cam_calib.mount != mount:
        cam_calib.mount = mount

    if paths is not None and paths.intrinsics_json(camera_name).is_file():
        src_type = "experiment_json"
        src_path = paths.intrinsics_json(camera_name)
        src_note = (
            "PnP uses K/dist from intrinsics/{camera}.json in the active experiment folder."
        )
    else:
        src_type = "calibration_npz"
        src_path = out_path
        src_note = "PnP uses K/dist from calibration.npz in the experiment folder."

    if cfg.debug:
        rms = intrinsics.reprojection_error
        rms_s = f"{rms:.4f}px" if rms is not None else "unknown"
        exp_line = f"        experiment={paths.root}\n" if paths else ""
        print(
            f"[debug] hand_eye PnP intrinsics: {src_type}\n"
            f"{exp_line}"
            f"        path={src_path}\n"
            f"        camera={camera_name!r}  RMS={rms_s}  "
            f"size={intrinsics.width}x{intrinsics.height}",
            flush=True,
        )

    session: CalibrationSessionOutput | None = None
    if samples is None:
        session = _open_session(robot, cfg, "hand_eye", camera_name)
        _begin_session_log(
            session,
            robot,
            cfg,
            phase="hand_eye",
            camera=camera_name,
            mount=mount.value,
        )

    try:
        if samples is None:
            assert session is not None
            samples, session = collect_hand_eye_samples(
                robot,
                camera_name,
                intrinsics,
                cfg.target,
                mount,
                cfg.robot_pose_frame,
                cfg,
                min_samples=_hand_eye_min_kept(cfg),
                session=session,
            )
        T_solved = solve_hand_eye(samples, mount)
        consistency = hand_eye_target_origin_std(samples, mount, T_solved)
        if cfg.debug:
            _debug().debug_print_hand_eye_summary(
                samples,
                T_solved,
                mount=mount,
                intrinsics=intrinsics,
                target=cfg.target,
            )

        if session is not None:
            session.write_report(
                {
                    "phase": "hand_eye",
                    "camera": camera_name,
                    "mount": mount.value,
                    "num_samples": len(samples),
                    "min_hand_eye_samples": _hand_eye_min_kept(cfg),
                    "consistency_std_mm": consistency,
                    "T_solved": T_solved.tolist(),
                    "intrinsics_source": {
                        "type": src_type,
                        "path": str(src_path),
                        "experiment_dir": str(paths.root) if paths else None,
                        "camera": camera_name,
                        "intrinsics_rms_px": intrinsics.reprojection_error,
                        "note": src_note,
                    },
                    "robot_pose_frame": cfg.robot_pose_frame.value,
                    "log_file": session.log_path.name,
                }
            )

        if mount == CameraMount.EYE_IN_HAND:
            cam_calib.T_ee_to_camera = transform_to_list(T_solved)
            cam_calib.T_robot_to_camera = None
        else:
            cam_calib.T_robot_to_camera = transform_to_list(T_solved)
            cam_calib.T_ee_to_camera = None

        scene.cameras[camera_name] = cam_calib
        out = save_scene_calibration(scene, out_path)
        _record_phase_if_experiment(
            cfg,
            session,
            phase="hand_eye",
            camera=camera_name,
            extra={"consistency_std_mm": consistency},
        )
        logger.info(
            "Saved hand-eye %r (%s) → %s; consistency std=%.3f mm",
            camera_name,
            mount.value,
            out,
            consistency,
        )
        return T_solved
    finally:
        if session is not None:
            session.end_log()


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
    transforms: list[np.ndarray] = []
    session = _open_session(robot, cfg, "landmark_map", camera_name)
    _begin_session_log(session, robot, cfg, phase="landmark_map", camera=camera_name)

    try:
        print(
            f"\n[Landmark map] camera={camera_name!r} target={_target_label(cfg.target)}\n"
            "  Glue ChArUco on table; move arm so wrist camera sees it from many poses.\n"
            f"  Live preview; save grabs fresh frame + pose. Need >= {min_samples}."
        )

        def process_frame(frame: np.ndarray):
            T_landmark_to_camera = estimate_target_to_camera(frame, intrinsics, cfg.target)
            vis, found = draw_target(
                frame, cfg.target, intrinsics=intrinsics, T_target_to_camera=T_landmark_to_camera
            )
            return vis, found, T_landmark_to_camera

        def on_save():
            cap = capture_frame_and_pose(robot, camera_name, cfg.robot_pose_frame)
            T_landmark_to_camera = estimate_target_to_camera(cap.image, intrinsics, cfg.target)
            if T_landmark_to_camera is None:
                return False
            T_robot_to_landmark = robot_to_landmark_from_eye_in_hand(
                cap.T_robot_to_ee, T_ee_to_camera, T_landmark_to_camera
            )
            transforms.append(T_robot_to_landmark)
            session.save_capture(
                len(transforms) - 1,
                cap,
                cfg.target,
                intrinsics=intrinsics,
                T_target_to_camera=T_landmark_to_camera,
            )
            return True

        session_cfg = LiveSessionConfig(
            window_name=f"landmark_map_{camera_name}",
            title="Landmark map",
            help_lines=default_help_lines() + (f"need >= {min_samples}",),
            counter_label="views",
            stop_key=stop_key,
        )

        count = run_live_session(
            robot,
            camera_name,
            session_cfg,
            process_frame=process_frame,
            on_save=on_save,
        )

        if count < min_samples:
            raise RuntimeError(f"Collected {count} views; need >= {min_samples}.")
        session.write_report(
            {
                "phase": "landmark_map",
                "camera": camera_name,
                "num_views": count,
                "log_file": session.log_path.name,
            }
        )
        logger.info("Session artifacts → %s", session.root)
        return transforms
    finally:
        session.end_log()


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
    out = save_scene_calibration(scene, _resolve_output_path(robot, cfg))
    logger.info(
        "Saved landmark/table map from %r → %s; std=%.2f mm max_pair=%.2f mm",
        camera_name,
        out,
        report["position_std_mm"],
        report["max_pairwise_mm"],
    )
    return T_fused


def collect_dual_top_extrinsic_captures(
    robot: HandEyeRobot,
    wrist_camera_name: str,
    top_camera_name: str,
    target: CalibrationTargetConfig,
    robot_pose_frame: RobotPoseFrame,
    cfg: CalibrationRunConfig,
    *,
    min_samples: int,
    session: CalibrationSessionOutput,
    stop_key: str = "q",
) -> list[SyncCapture]:
    """Wrist + top synchronized captures; handheld board visible in both cameras."""
    if wrist_camera_name not in robot.cameras:
        raise KeyError(f"Wrist camera {wrist_camera_name!r} not in robot config.")
    if top_camera_name not in robot.cameras:
        raise KeyError(f"Top camera {top_camera_name!r} not in robot config.")

    captures: list[SyncCapture] = []
    wrist_intrinsics = _load_intrinsics_for_camera(robot, cfg, wrist_camera_name)
    if wrist_intrinsics is None:
        raise RuntimeError(
            f"No intrinsics for wrist camera {wrist_camera_name!r}. "
            f"Run --phase=intrinsics --camera={wrist_camera_name} first."
        )

    print(
        f"\n[Top extrinsic dual] wrist={wrist_camera_name!r} top={top_camera_name!r} "
        f"target={_target_label(target)}\n"
        "  Hold ChArUco so BOTH wrist and top see it; move arm to varied poses, then save.\n"
        f"  Board may move between saves; need >= {min_samples} dual-camera samples."
    )
    if cfg.debug:
        _debug().debug_print_session_header(
            robot_pose_frame=robot_pose_frame,
            camera=wrist_camera_name,
            top_camera=top_camera_name,
        )

    def process_frame(frame: np.ndarray):
        vis, found = draw_target(frame, target, intrinsics=wrist_intrinsics)
        return vis, found, None

    def on_save():
        cap = capture_frame_and_pose(
            robot,
            wrist_camera_name,
            robot_pose_frame,
            also_capture_top=top_camera_name,
        )
        if not detect_target(cap.image, target):
            print("  skip: wrist camera — ChArUco not detected", flush=True)
            return False
        if cap.top_image is None or not detect_target(cap.top_image, target):
            print("  skip: top camera — ChArUco not detected", flush=True)
            return False
        captures.append(cap)
        session.save_capture(len(captures) - 1, cap, target, intrinsics=wrist_intrinsics)
        print(
            f"    sync: wrist={cap.frame_read_ms:.1f}ms"
            f" top={cap.top_frame_read_ms:.1f}ms pose={cap.pose_read_ms:.1f}ms",
            flush=True,
        )
        return True

    session_cfg = LiveSessionConfig(
        window_name=f"top_dual_{wrist_camera_name}",
        title="Top extrinsic (dual cam)",
        help_lines=default_help_lines()
        + (
            f"hold board for wrist + {top_camera_name}",
            f"collect >= {min_samples}",
        ),
        counter_label="samples",
        stop_key=stop_key,
        secondary_camera=top_camera_name,
        secondary_window_name=f"top_dual_{top_camera_name}",
        secondary_title=f"Top ({top_camera_name})",
        secondary_help_lines=(
            "green = ChArUco detected in top view",
            "save requires wrist + top both green",
        ),
    )

    count = run_live_session(
        robot,
        wrist_camera_name,
        session_cfg,
        process_frame=process_frame,
        on_save=on_save,
        min_count=min_samples,
    )
    if count < min_samples:
        raise RuntimeError(f"Collected {count} dual-camera samples; need >= {min_samples}.")
    return captures


def run_top_extrinsic_dual_calibration(
    robot: HandEyeRobot,
    top_camera_name: str,
    mount: CameraMount,
    cfg: CalibrationRunConfig,
) -> np.ndarray:
    """Infer ``T_robot_to_top`` from wrist hand-eye chain + top PnP (handheld board)."""
    if mount != CameraMount.EYE_TO_HAND:
        raise ValueError("top_extrinsic_dual expects eye_to_hand fixed camera.")

    wrist_camera_name = cfg.wrist_camera
    min_samples = max(cfg.min_landmark_samples, 3)

    scene = _load_scene(robot, cfg)
    if wrist_camera_name not in scene.cameras:
        raise RuntimeError(f"Run hand_eye for {wrist_camera_name!r} first.")
    T_ee_to_wrist = get_eye_in_hand_extrinsic(scene, wrist_camera_name)

    wrist_intrinsics = _load_intrinsics_for_camera(robot, cfg, wrist_camera_name)
    if wrist_intrinsics is None:
        raise RuntimeError(f"No intrinsics for {wrist_camera_name!r}.")
    top_intrinsics = _load_intrinsics_for_camera(robot, cfg, top_camera_name)
    if top_intrinsics is None:
        raise RuntimeError(
            f"No intrinsics for {top_camera_name!r}. "
            f"Run --phase=intrinsics --camera={top_camera_name} first."
        )

    session = _open_session(robot, cfg, "top_extrinsic_dual", top_camera_name)
    _begin_session_log(
        session,
        robot,
        cfg,
        phase="top_extrinsic_dual",
        camera=top_camera_name,
        wrist_camera=wrist_camera_name,
        mount=mount.value,
    )
    try:
        captures = collect_dual_top_extrinsic_captures(
            robot,
            wrist_camera_name,
            top_camera_name,
            cfg.target,
            cfg.robot_pose_frame,
            cfg,
            min_samples=min_samples,
            session=session,
        )

        samples: list[HandEyeSample] = []
        for i, cap in enumerate(captures):
            T_target_to_cam = estimate_target_to_camera(cap.image, wrist_intrinsics, cfg.target)
            if T_target_to_cam is None:
                raise RuntimeError(f"PnP failed on wrist sample {i}.")
            if cap.T_robot_to_ee is None:
                raise RuntimeError(f"Missing robot pose on sample {i}.")
            samples.append(
                HandEyeSample(T_robot_to_ee=cap.T_robot_to_ee, T_target_to_camera=T_target_to_cam)
            )
            session.save_capture(
                i,
                cap,
                cfg.target,
                intrinsics=wrist_intrinsics,
                T_target_to_camera=T_target_to_cam,
            )

        T_robot_to_top, std_mm, board_transforms, top_diags = _solve_top_extrinsic_via_board(
            captures,
            samples,
            T_ee_to_wrist,
            top_intrinsics,
            cfg.target,
        )

        if top_camera_name not in scene.cameras:
            scene.cameras[top_camera_name] = _intrinsics_to_camera_calib(
                top_intrinsics, CameraMount.EYE_TO_HAND
            )
        top_calib = scene.cameras[top_camera_name]
        top_calib.mount = CameraMount.EYE_TO_HAND
        top_calib.T_robot_to_camera = transform_to_list(T_robot_to_top)
        top_calib.T_ee_to_camera = None
        scene.cameras[top_camera_name] = top_calib

        out = save_scene_calibration(scene, _resolve_output_path(robot, cfg))
        report: dict[str, Any] = {
            "phase": "top_extrinsic_dual",
            "camera": top_camera_name,
            "wrist_camera": wrist_camera_name,
            "mount": mount.value,
            "num_samples": len(samples),
            "top_extrinsic_std_mm": std_mm,
            "T_robot_to_camera": T_robot_to_top.tolist(),
            "dual_camera_samples": top_diags,
            "npz_output": str(out),
            "log_file": session.log_path.name,
            "note": (
                "T_robot_to_top fused from wrist board chain + top PnP per save. "
                "Board may be handheld; does not require landmark_map."
            ),
        }
        session.write_report(report)
        _record_phase_if_experiment(
            cfg,
            session,
            phase="top_extrinsic_dual",
            camera=top_camera_name,
            extra={"top_extrinsic_std_mm": std_mm},
        )
        logger.info(
            "Saved top %r via dual capture → %s; fused std=%.2f mm (%d views)",
            top_camera_name,
            out,
            std_mm,
            len(board_transforms),
        )
        logger.info("Session artifacts → %s", session.root)
        return T_robot_to_top
    finally:
        session.end_log()


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
    transforms: list[np.ndarray] = []
    session = _open_session(robot, cfg, "camera_via_landmark", camera_name)
    _begin_session_log(session, robot, cfg, phase="camera_via_landmark", camera=camera_name)

    try:
        print(
            f"\n[Camera via landmark] camera={camera_name!r} target={_target_label(cfg.target)}\n"
            "  Keep table ChArUco fixed; fixed camera should see it clearly.\n"
            "  Live preview; save grabs fresh frame. Recommend >= 3 captures."
        )

        def process_frame(frame: np.ndarray):
            T_landmark_to_camera = estimate_target_to_camera(frame, intrinsics, cfg.target)
            vis, found = draw_target(
                frame, cfg.target, intrinsics=intrinsics, T_target_to_camera=T_landmark_to_camera
            )
            return vis, found, T_landmark_to_camera

        def on_save():
            cap = capture_frame_and_pose(
                robot, camera_name, cfg.robot_pose_frame, record_pose=False
            )
            T_landmark_to_camera = estimate_target_to_camera(cap.image, intrinsics, cfg.target)
            if T_landmark_to_camera is None:
                return False
            T_robot_to_camera = camera_extrinsic_from_landmark(T_robot_to_table, T_landmark_to_camera)
            transforms.append(T_robot_to_camera)
            session.save_capture(
                len(transforms) - 1,
                cap,
                cfg.target,
                intrinsics=intrinsics,
                T_target_to_camera=T_landmark_to_camera,
            )
            return True

        session_cfg = LiveSessionConfig(
            window_name=f"camera_via_landmark_{camera_name}",
            title="Camera via landmark",
            help_lines=default_help_lines() + ("recommend >= 3 views",),
            counter_label="views",
            stop_key=stop_key,
        )

        count = run_live_session(
            robot,
            camera_name,
            session_cfg,
            process_frame=process_frame,
            on_save=on_save,
        )

        if count < min_samples:
            raise RuntimeError(f"Collected {count} views; need >= {min_samples}.")
        session.write_report(
            {
                "phase": "camera_via_landmark",
                "camera": camera_name,
                "num_views": count,
                "log_file": session.log_path.name,
            }
        )
        logger.info("Session artifacts → %s", session.root)
        return transforms
    finally:
        session.end_log()


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
    out = save_scene_calibration(scene, _resolve_output_path(robot, cfg))
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
        _resolve_output_path(robot, cfg),
        T_robot_to_ee=T_robot_to_ee,
    )
    from .io import describe_npz

    return {
        "output_path": str(out_path),
        "experiment_dir": str(cfg.experiment_dir) if cfg.experiment_dir else None,
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
      --phase=intrinsics_and_hand_eye --camera=wrist --mount=eye_in_hand
  lerobot-calibrate-hand-eye --config_path=examples/hand_eye_calibration/crp.json \\
      --phase=intrinsics --camera=top --mount=eye_to_hand
  lerobot-calibrate-hand-eye --config_path=... --phase=intrinsics --camera=wrist --mount=eye_in_hand
  lerobot-calibrate-hand-eye --config_path=... --phase=hand_eye --camera=wrist --mount=eye_in_hand
  lerobot-calibrate-hand-eye --config_path=... --phase=landmark_map --camera=wrist
  lerobot-calibrate-hand-eye --config_path=... --phase=top_extrinsic_dual --camera=top --mount=eye_to_hand
  lerobot-calibrate-hand-eye --config_path=... --phase=camera_via_landmark --camera=top --mount=eye_to_hand
  lerobot-calibrate-hand-eye --config_path=... --phase=validate
"""
    )
