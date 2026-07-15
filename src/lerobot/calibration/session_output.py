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

"""Persist calibration session artifacts under ``outputs/``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .chessboard import CameraIntrinsics
from .hand_eye import HandEyeSample
from .session_log import SessionLog
from .sync_capture import SyncCapture
from .chessboard import CameraIntrinsics
from .target import CalibrationTargetConfig, draw_target


def intrinsics_to_dict(intrinsics: CameraIntrinsics, camera: str) -> dict[str, Any]:
    k = intrinsics.camera_matrix
    return {
        "camera": camera,
        "width": intrinsics.width,
        "height": intrinsics.height,
        "camera_matrix": k.tolist(),
        "dist_coeffs": intrinsics.dist_coeffs.reshape(-1).tolist(),
        "reprojection_error_px": intrinsics.reprojection_error,
        "fx": float(k[0, 0]),
        "fy": float(k[1, 1]),
        "cx": float(k[0, 2]),
        "cy": float(k[1, 2]),
    }


@dataclass
class CalibrationSessionOutput:
    root: Path
    images_dir: Path
    vis_dir: Path
    images_top_dir: Path | None = None
    _session_log: SessionLog | None = None

    @property
    def log_path(self) -> Path:
        return self.root / "calibration.log"

    @property
    def log_active(self) -> bool:
        return self._session_log is not None

    def begin_log(self, *, header_lines: tuple[str, ...] = ()) -> None:
        """Mirror stdout/stderr to ``calibration.log`` under this session."""
        if self._session_log is not None:
            return
        self._session_log = SessionLog(self.log_path, header_lines=header_lines)
        self._session_log.start()
        print(f"[session] log file → {self.log_path.resolve()}", flush=True)

    def end_log(self, *, exc: BaseException | None = None) -> None:
        if self._session_log is None:
            return
        self._session_log.stop(exc=exc)
        self._session_log = None

    @classmethod
    def create(
        cls,
        base_dir: Path,
        robot_id: str,
        phase: str,
        camera: str,
    ) -> CalibrationSessionOutput:
        """Legacy layout: ``{base}/{robot_id}/{camera}/{phase}_{timestamp}/``."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = base_dir.expanduser() / robot_id / camera / f"{phase}_{stamp}"
        return cls.open(root)

    @classmethod
    def open(cls, root: Path) -> CalibrationSessionOutput:
        """Open a fixed phase directory (no timestamp subfolder)."""
        root = root.expanduser()
        images_dir = root / "images"
        vis_dir = root / "vis"
        images_top_dir = root / "images_top"
        images_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)
        images_top_dir.mkdir(parents=True, exist_ok=True)
        out = cls(root=root, images_dir=images_dir, vis_dir=vis_dir)
        out.images_top_dir = images_top_dir
        return out

    def save_capture(
        self,
        index: int,
        capture: SyncCapture,
        target: CalibrationTargetConfig,
        *,
        intrinsics: CameraIntrinsics | None = None,
        T_target_to_camera: np.ndarray | None = None,
    ) -> dict[str, Any]:
        stem = f"sample_{index:03d}"
        image_path = self.images_dir / f"{stem}.jpg"
        vis_path = self.vis_dir / f"{stem}.jpg"
        bgr = cv2.cvtColor(capture.image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(image_path), bgr)
        if capture.top_image is not None and self.images_top_dir is not None:
            top_path = self.images_top_dir / f"{stem}.jpg"
            cv2.imwrite(str(top_path), cv2.cvtColor(capture.top_image, cv2.COLOR_RGB2BGR))
        vis, found = draw_target(
            capture.image,
            target,
            intrinsics=intrinsics,
            T_target_to_camera=T_target_to_camera,
        )
        if found:
            cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        meta: dict[str, Any] = {
            "index": index,
            "image": str(image_path.relative_to(self.root)),
            "vis": str(vis_path.relative_to(self.root)) if found else None,
            "frame_read_ms": capture.frame_read_ms,
            "pose_read_ms": capture.pose_read_ms,
            "total_sync_ms": capture.total_sync_ms,
            "top_frame_read_ms": capture.top_frame_read_ms,
        }
        if capture.top_image is not None and self.images_top_dir is not None:
            meta["image_top"] = str((self.images_top_dir / f"{stem}.jpg").relative_to(self.root))
        if capture.T_ee_to_robot is not None:
            meta["T_ee_to_robot"] = capture.T_ee_to_robot.tolist()
        if T_target_to_camera is not None:
            meta["T_target_to_camera"] = T_target_to_camera.tolist()
        return meta

    def write_samples_npz(
        self,
        captures: list[SyncCapture],
        *,
        samples: list[HandEyeSample] | None = None,
    ) -> Path:
        path = self.root / "samples.npz"
        arrays: dict[str, np.ndarray] = {"num_samples": np.array([len(captures)], dtype=np.int32)}
        for i, cap in enumerate(captures):
            arrays[f"image_{i:03d}"] = cap.image
            arrays[f"sync_ms_{i:03d}"] = np.array([cap.total_sync_ms], dtype=np.float64)
            if cap.top_image is not None:
                arrays[f"image_top_{i:03d}"] = cap.top_image
            if cap.T_ee_to_robot is not None:
                arrays[f"T_ee_to_robot_{i:03d}"] = cap.T_ee_to_robot
        if samples is not None:
            for i, sample in enumerate(samples):
                arrays[f"T_target_to_camera_{i:03d}"] = sample.T_target_to_camera
        np.savez(path, **arrays)
        return path

    def write_intrinsics(self, intrinsics: CameraIntrinsics, camera: str) -> Path:
        path = self.root / "intrinsics.json"
        path.write_text(
            json.dumps(intrinsics_to_dict(intrinsics, camera), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def write_report(self, report: dict[str, Any]) -> Path:
        path = self.root / "report.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
