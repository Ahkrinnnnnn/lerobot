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

"""Terminal debug helpers for hand-eye calibration troubleshooting."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .chessboard import CameraIntrinsics
from .hand_eye import (
    CameraMount,
    HandEyeSample,
    hand_eye_motion_residual_mm,
    per_sample_board_origin_mm,
)
from .scene import RobotPoseFrame
from .sync_capture import SyncCapture
from .target import CalibrationTargetConfig
from .transforms import invert_transform, rotation_translation_from_transform


def _fmt_pose_6d(pose: list[float] | None) -> str:
    if pose is None or len(pose) < 6:
        return "n/a"
    x, y, z, r, p, yw = pose
    return f"xyz=({x:.2f},{y:.2f},{z:.2f}) rpy=({r:.2f},{p:.2f},{yw:.2f})"


def _charuco_corner_count(image: np.ndarray, target: CalibrationTargetConfig) -> int:
    from .charuco import detect_charuco

    det = detect_charuco(image, target.charuco())
    if det is None:
        return 0
    return int(len(det[1]))


def pnp_reprojection_px(
    image: np.ndarray,
    intrinsics: CameraIntrinsics,
    target: CalibrationTargetConfig,
    T_target_to_camera: np.ndarray,
) -> float | None:
    from .charuco import detect_charuco, make_charuco_board

    det = detect_charuco(image, target.charuco())
    if det is None:
        return None
    corners, ids = det
    board = make_charuco_board(target.charuco())
    obj_pts, img_pts = board.matchImagePoints(corners, ids)
    if obj_pts is None or len(obj_pts) < 4:
        return None
    r, t = rotation_translation_from_transform(T_target_to_camera)
    rvec, _ = cv2.Rodrigues(r)
    proj, _ = cv2.projectPoints(obj_pts, rvec, t.reshape(3, 1), intrinsics.camera_matrix, intrinsics.dist_coeffs)
    return float(np.linalg.norm(proj.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1).mean())


def debug_print_session_header(*, robot_pose_frame: RobotPoseFrame, camera: str, top_camera: str | None) -> None:
    print(
        f"\n[debug] robot_pose_frame={robot_pose_frame.value!r} wrist={camera!r}"
        + (f" top={top_camera!r}" if top_camera else " top=off"),
        flush=True,
    )


def debug_print_capture(
    index: int,
    cap: SyncCapture,
    target: CalibrationTargetConfig,
    *,
    robot_pose_frame: RobotPoseFrame,
    pose_6d: list[float] | None = None,
    prev_pose_6d: list[float] | None = None,
    T_target_to_camera: np.ndarray | None = None,
    prev_T_target_to_camera: np.ndarray | None = None,
    intrinsics: CameraIntrinsics | None = None,
    image: np.ndarray | None = None,
) -> None:
    n_corners = _charuco_corner_count(cap.image, target)
    lines = [
        f"[debug] sample #{index:03d} pose_frame={robot_pose_frame.value}",
        f"        robot CRP {_fmt_pose_6d(pose_6d)}",
        f"        robot T_robot_to_ee t(mm)={_fmt_t(cap.T_robot_to_ee)}",
        f"        wrist corners={n_corners} sync={cap.total_sync_ms:.1f}ms"
        f" (wrist={cap.frame_read_ms:.1f} top={cap.top_frame_read_ms:.1f} pose={cap.pose_read_ms:.1f})",
    ]
    if cap.top_image is not None:
        lines.append(f"        top corners={_charuco_corner_count(cap.top_image, target)}")
    if prev_pose_6d is not None and pose_6d is not None:
        dxyz = np.array(pose_6d[:3]) - np.array(prev_pose_6d[:3])
        drpy = np.array(pose_6d[3:]) - np.array(prev_pose_6d[3:])
        lines.append(
            f"        robot Δworld: dxyz(mm)={_fmt_vec3(dxyz)}"
            f" drpy(deg)=({drpy[0]:.2f},{drpy[1]:.2f},{drpy[2]:.2f})"
        )
    if T_target_to_camera is not None:
        t = T_target_to_camera[:3, 3]
        R = T_target_to_camera[:3, :3]
        reproj_s = "n/a"
        if intrinsics is not None and image is not None:
            reproj = pnp_reprojection_px(image, intrinsics, target, T_target_to_camera)
            if reproj is not None:
                reproj_s = f"{reproj:.3f}px"
        lines.append(
            f"        vision board_in_camera t(mm)={_fmt_vec3(t)}"
            f" RPY(deg)={_fmt_rpy(R)} reproj={reproj_s}"
        )
        if prev_T_target_to_camera is not None:
            d_t = invert_transform(prev_T_target_to_camera) @ T_target_to_camera
            dt_cam_naive = t - prev_T_target_to_camera[:3, 3]
            ang = _rotation_angle_deg(d_t[:3, :3])
            lines.append(
                f"        vision Δ vs prev: Δt_board(mm)={_fmt_vec3(d_t[:3, 3])} ∠={ang:.1f}°"
                f"  Δt_camera_naive(mm)={_fmt_vec3(dt_cam_naive)}"
            )
    print("\n".join(lines), flush=True)


def _fmt_t(T: np.ndarray | None) -> str:
    if T is None:
        return "n/a"
    t = T[:3, 3]
    return f"({t[0]:.2f},{t[1]:.2f},{t[2]:.2f})"


def _fmt_vec3(v: np.ndarray) -> str:
    return f"({v[0]:.2f},{v[1]:.2f},{v[2]:.2f})"


def _rpy_deg_from_rotation(R: np.ndarray) -> tuple[float, float, float]:
    """Extract roll/pitch/yaw (deg) for R = Rz @ Ry @ Rx (CRP convention)."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def _fmt_rpy(R: np.ndarray) -> str:
    r, p, y = _rpy_deg_from_rotation(R)
    return f"({r:.2f},{p:.2f},{y:.2f})"


def _robot_world_delta_mm(samples: list[HandEyeSample], i: int) -> np.ndarray:
    """Naive world-frame translation change between consecutive robot poses."""
    return samples[i].T_robot_to_ee[:3, 3] - samples[i - 1].T_robot_to_ee[:3, 3]


def _vision_camera_t_delta_mm(samples: list[HandEyeSample], i: int) -> np.ndarray:
    """Naive camera-frame board-origin shift (valid when rotation is small)."""
    return samples[i].T_target_to_camera[:3, 3] - samples[i - 1].T_target_to_camera[:3, 3]


def debug_print_samples_robot_vs_vision(
    captures: list[SyncCapture],
    samples: list[HandEyeSample],
    target: CalibrationTargetConfig,
    intrinsics: CameraIntrinsics,
    *,
    robot_pose_frame: RobotPoseFrame,
) -> None:
    """Print robot + vision side-by-side for every saved sample (after PnP)."""
    print("\n[debug] all samples — robot vs vision:", flush=True)
    prev_pose_6d: list[float] | None = None
    prev_T: np.ndarray | None = None
    for i, (cap, sample) in enumerate(zip(captures, samples, strict=True)):
        debug_print_capture(
            i,
            cap,
            target,
            robot_pose_frame=robot_pose_frame,
            pose_6d=cap.pose_6d,
            prev_pose_6d=prev_pose_6d,
            T_target_to_camera=sample.T_target_to_camera,
            prev_T_target_to_camera=prev_T,
            intrinsics=intrinsics,
            image=cap.image,
        )
        prev_pose_6d = cap.pose_6d
        prev_T = sample.T_target_to_camera


def debug_print_pnp_sample(
    index: int,
    image: np.ndarray,
    intrinsics: CameraIntrinsics,
    target: CalibrationTargetConfig,
    T_target_to_camera: np.ndarray,
    *,
    prev_T_target_to_camera: np.ndarray | None = None,
) -> None:
    reproj = pnp_reprojection_px(image, intrinsics, target, T_target_to_camera)
    t = T_target_to_camera[:3, 3]
    R = T_target_to_camera[:3, :3]
    dist = float(np.linalg.norm(t))
    reproj_s = f"{reproj:.3f}px" if reproj is not None else "n/a"
    lines = [
        f"[debug] sample #{index:03d} vision (board in camera frame):",
        f"        t(mm)={_fmt_vec3(t)}  RPY(deg)={_fmt_rpy(R)}  dist={dist:.1f}mm  reproj={reproj_s}",
    ]
    if prev_T_target_to_camera is not None:
        d_t = invert_transform(prev_T_target_to_camera) @ T_target_to_camera
        dt_board = d_t[:3, 3]
        dt_cam_naive = t - prev_T_target_to_camera[:3, 3]
        ang = _rotation_angle_deg(d_t[:3, :3])
        lines.append(
            f"        Δ vs prev: Δt_board(mm)={_fmt_vec3(dt_board)}  ∠={ang:.1f}°"
            f"  Δt_camera_naive(mm)={_fmt_vec3(dt_cam_naive)}"
        )
    print("\n".join(lines), flush=True)


def debug_print_motion_pairs(samples: list[HandEyeSample], T_ee_to_camera: np.ndarray) -> None:
    print(
        "[debug] consecutive motion (robot vs vision); compare magnitudes and dominant axis:",
        flush=True,
    )
    print(
        "        Δt_robot_ee = relative translation in ee_{i-1} frame (for AX=XB)",
        flush=True,
    )
    print(
        "        Δt_vision_board = relative translation in board_{i-1} frame (for AX=XB)",
        flush=True,
    )
    print(
        "        Δt_world / Δt_camera_naive = easy-to-read axis checks (not identical when rotating)",
        flush=True,
    )
    for i in range(1, len(samples)):
        d_g = invert_transform(samples[i - 1].T_robot_to_ee) @ samples[i].T_robot_to_ee
        d_t = invert_transform(samples[i - 1].T_target_to_camera) @ samples[i].T_target_to_camera
        dt_robot_ee = d_g[:3, 3]
        dt_vision_board = d_t[:3, 3]
        dt_robot_world = _robot_world_delta_mm(samples, i)
        dt_vision_cam_naive = _vision_camera_t_delta_mm(samples, i)
        t_g = float(np.linalg.norm(dt_robot_ee))
        t_t = float(np.linalg.norm(dt_vision_board))
        ang_g = _rotation_angle_deg(d_g[:3, :3])
        ang_t = _rotation_angle_deg(d_t[:3, :3])
        lhs = d_g @ T_ee_to_camera
        rhs = T_ee_to_camera @ d_t
        dt_lhs = lhs[:3, 3]
        dt_rhs = rhs[:3, 3]
        dt_diff = dt_lhs - dt_rhs
        trans_err = float(np.linalg.norm(dt_diff))
        rot_err = float(np.linalg.norm(lhs[:3, :3] - rhs[:3, :3]))
        residual = trans_err + rot_err
        print(f"  pair {i - 1:02d}->{i:02d}:", flush=True)
        print(
            f"    robot  |Δt|={t_g:.1f}mm  Δt_ee(mm)={_fmt_vec3(dt_robot_ee)}"
            f"  Δt_world(mm)={_fmt_vec3(dt_robot_world)}  ∠={ang_g:.1f}°",
            flush=True,
        )
        print(
            f"    vision |Δt|={t_t:.1f}mm  Δt_board(mm)={_fmt_vec3(dt_vision_board)}"
            f"  Δt_camera_naive(mm)={_fmt_vec3(dt_vision_cam_naive)}  ∠={ang_t:.1f}°",
            flush=True,
        )
        print(
            f"    AX=XB  lhs_t(mm)={_fmt_vec3(dt_lhs)}  rhs_t(mm)={_fmt_vec3(dt_rhs)}"
            f"  diff(mm)={_fmt_vec3(dt_diff)}  residual={residual:.1f}"
            f"  (trans={trans_err:.1f} rot={rot_err:.1f})",
            flush=True,
        )


def debug_print_translation_chain(
    samples: list[HandEyeSample],
    T_ee_to_camera: np.ndarray,
    mount: CameraMount,
    *,
    kept_indices: list[int] | None = None,
    captures: list[SyncCapture] | None = None,
    pure_trans_deg: float = 3.0,
) -> None:
    """Highlight translation-chain issues: board-origin steps vs robot motion on pure-T pairs."""
    if len(samples) < 2:
        return

    origins = per_sample_board_origin_mm(samples, mount, T_ee_to_camera)
    mean_o = origins.mean(axis=0)

    def label(j: int) -> str:
        if kept_indices is not None and j < len(kept_indices):
            return f"#{kept_indices[j]:03d}"
        return f"#{j:03d}"

    print(
        "\n[debug] translation-chain check (board fixed on table → "
        "T_robot_to_board should be constant):",
        flush=True,
    )
    print(
        "        board_step = |origin_i - origin_{i-1}|  (0 = consistent chain for that motion)",
        flush=True,
    )
    print(
        "        trans_axxb = ||lhs_t - rhs_t|| only  (pure-T pairs: if high → robot FK/TCP or PnP depth)",
        flush=True,
    )

    pure_trans: list[dict[str, float]] = []
    mixed: list[dict[str, float]] = []

    for i in range(1, len(samples)):
        d_g = invert_transform(samples[i - 1].T_robot_to_ee) @ samples[i].T_robot_to_ee
        d_t = invert_transform(samples[i - 1].T_target_to_camera) @ samples[i].T_target_to_camera
        ang_g = _rotation_angle_deg(d_g[:3, :3])
        ang_t = _rotation_angle_deg(d_t[:3, :3])
        dt_world = float(np.linalg.norm(_robot_world_delta_mm(samples, i)))
        board_step = float(np.linalg.norm(origins[i] - origins[i - 1]))
        lhs = d_g @ T_ee_to_camera
        rhs = T_ee_to_camera @ d_t
        trans_axxb = float(np.linalg.norm(lhs[:3, 3] - rhs[:3, 3]))
        rot_axxb = float(np.linalg.norm(lhs[:3, :3] - rhs[:3, :3]))
        dev_i = float(np.linalg.norm(origins[i] - mean_o))
        dev_im1 = float(np.linalg.norm(origins[i - 1] - mean_o))

        if ang_g < pure_trans_deg and ang_t < pure_trans_deg:
            kind = "pure-T"
            pure_trans.append({"trans_axxb": trans_axxb, "board_step": board_step, "dt_world": dt_world})
        elif ang_g > 15 or ang_t > 15:
            kind = "rot"
        else:
            kind = "mixed"
            mixed.append({"trans_axxb": trans_axxb, "board_step": board_step})

        flag = " ***" if kind == "pure-T" and trans_axxb > 30 else ""
        print(
            f"  {label(i - 1)}->{label(i)} [{kind}]"
            f" |Δworld|={dt_world:.0f} board_step={board_step:.0f}"
            f" trans_axxb={trans_axxb:.1f} rot_axxb={rot_axxb:.1f}"
            f" dev={dev_im1:.0f}/{dev_i:.0f}mm{flag}",
            flush=True,
        )

    if pure_trans:
        pt = np.array([p["trans_axxb"] for p in pure_trans])
        bs = np.array([p["board_step"] for p in pure_trans])
        print(
            f"  pure-T summary ({len(pure_trans)} pairs):"
            f" trans_axxb mean={pt.mean():.1f} max={pt.max():.1f}"
            f"  board_step mean={bs.mean():.1f} max={bs.max():.1f}",
            flush=True,
        )
        if pt.mean() > 40:
            print(
                "  → pure translation pairs still have large trans_axxb:"
                " suspect CRP TCP / world frame / arm flex, not ChArUco.",
                flush=True,
            )
    if mixed:
        mt = np.array([p["trans_axxb"] for p in mixed])
        print(
            f"  mixed-motion summary ({len(mixed)} pairs): trans_axxb mean={mt.mean():.1f}",
            flush=True,
        )

    if captures is not None and len(captures) == len(samples):
        print("  same-xyz saves (robot reports zero motion, vision should rotate only):", flush=True)
        for i in range(1, len(captures)):
            p0, p1 = captures[i - 1].pose_6d, captures[i].pose_6d
            if p0 is None or p1 is None:
                continue
            dxyz = np.linalg.norm(np.array(p1[:3]) - np.array(p0[:3]))
            if dxyz > 0.05:
                continue
            d_t = invert_transform(samples[i - 1].T_target_to_camera) @ samples[i].T_target_to_camera
            ang_t = _rotation_angle_deg(d_t[:3, :3])
            if ang_t < 5:
                continue
            lhs = (
                invert_transform(samples[i - 1].T_robot_to_ee) @ samples[i].T_robot_to_ee
            ) @ T_ee_to_camera
            rhs = T_ee_to_camera @ d_t
            trans_axxb = float(np.linalg.norm(lhs[:3, 3] - rhs[:3, 3]))
            print(
                f"    {label(i - 1)}->{label(i)}: ∠vision={ang_t:.1f}° trans_axxb={trans_axxb:.1f}"
                f" (high → TCP rotation center ≠ camera mount)",
                flush=True,
            )

    print(
        "\n[debug] manual translation-chain tests (offline / next session):",
        flush=True,
    )
    for line in (
        "1) Repeat save: same teach pose ×3 → |Δworld| should be <0.1mm; if vision jumps → PnP noise",
        "2) Single-axis jog: ~50mm X only (hold orientation) → watch pure-T trans_axxb above",
        "3) Compare --robot_pose_frame=user vs world on the same poses",
        "4) CRP teach pendant: confirm tool/TCP offset matches physical camera flange",
        "5) After move, wait 1s before save (FK settle)",
    ):
        print(f"        {line}", flush=True)


def _rotation_angle_deg(R: np.ndarray) -> float:
    trace = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(trace))


def debug_print_hand_eye_summary(
    samples: list[HandEyeSample],
    T_ee_to_camera: np.ndarray,
    *,
    mount: CameraMount,
    intrinsics: CameraIntrinsics | None = None,
    target: CalibrationTargetConfig | None = None,
    images: list[np.ndarray] | None = None,
    kept_indices: list[int] | None = None,
    captures: list[SyncCapture] | None = None,
) -> None:
    origins = per_sample_board_origin_mm(samples, mount, T_ee_to_camera)
    mean_o = origins.mean(axis=0)
    std_o = np.std(origins, axis=0)
    consistency = float(np.linalg.norm(std_o))
    motion = hand_eye_motion_residual_mm(samples, T_ee_to_camera)

    print(
        f"\n[debug] hand-eye summary: consistency_std={consistency:.2f}mm"
        f" motion_residual_mean={motion['mean_mm']:.1f}mm max={motion['max_mm']:.1f}mm",
        flush=True,
    )
    print(
        f"        board origin mean(mm)=({mean_o[0]:.1f},{mean_o[1]:.1f},{mean_o[2]:.1f})"
        f" std/mm=({std_o[0]:.1f},{std_o[1]:.1f},{std_o[2]:.1f})",
        flush=True,
    )
    print(f"        T_ee_to_camera t(mm)=({_fmt_t(T_ee_to_camera)})", flush=True)

    for i, origin in enumerate(origins):
        dev = float(np.linalg.norm(origin - mean_o))
        extra = ""
        T_vis = samples[i].T_target_to_camera
        t_vis = T_vis[:3, 3]
        vis_line = f" vision_t={_fmt_vec3(t_vis)}"
        if intrinsics is not None and target is not None and images is not None and i < len(images):
            reproj = pnp_reprojection_px(images[i], intrinsics, target, samples[i].T_target_to_camera)
            if reproj is not None:
                extra = f" reproj={reproj:.3f}px"
        print(
            f"  #{i:03d} board_origin=({origin[0]:.1f},{origin[1]:.1f},{origin[2]:.1f})"
            f" dev={dev:.1f}mm{vis_line}{extra}",
            flush=True,
        )

    debug_print_motion_pairs(samples, T_ee_to_camera)
    debug_print_translation_chain(
        samples,
        T_ee_to_camera,
        mount,
        kept_indices=kept_indices,
        captures=captures,
    )
