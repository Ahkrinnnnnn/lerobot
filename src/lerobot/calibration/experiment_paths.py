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

"""Single experiment folder layout under ``outputs/``."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .chessboard import CameraIntrinsics
from .session_output import intrinsics_to_dict

DEFAULT_OUTPUTS_BASE = Path("outputs/hand_eye_calibration")
DEFAULT_EXPERIMENT_NAME = "current"


@dataclass(frozen=True)
class ExperimentPaths:
    """All artifacts for one calibration experiment live under ``root``."""

    root: Path

    @property
    def calibration_npz(self) -> Path:
        return self.root / "calibration.npz"

    @property
    def meta_json(self) -> Path:
        return self.root / "experiment_meta.json"

    @property
    def intrinsics_dir(self) -> Path:
        return self.root / "intrinsics"

    @property
    def phases_dir(self) -> Path:
        return self.root / "phases"

    @property
    def archive_dir(self) -> Path:
        return self.root / "archive"

    def intrinsics_json(self, camera: str) -> Path:
        return self.intrinsics_dir / f"{camera}.json"

    def phase_dir(self, phase: str, camera: str) -> Path:
        safe_phase = phase.replace("/", "_")
        return self.phases_dir / f"{safe_phase}_{camera}"


def resolve_experiment_root(
    robot_id: str,
    *,
    base_dir: Path | None = None,
    experiment_name: str | None = None,
    experiment_dir: Path | None = None,
    new_experiment: bool = False,
) -> ExperimentPaths:
    """Resolve the active experiment directory."""
    if experiment_dir is not None:
        root = experiment_dir.expanduser()
    else:
        base = (base_dir or DEFAULT_OUTPUTS_BASE).expanduser()
        name = experiment_name or DEFAULT_EXPERIMENT_NAME
        if new_experiment:
            name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        root = base / robot_id / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "intrinsics").mkdir(exist_ok=True)
    (root / "phases").mkdir(exist_ok=True)
    return ExperimentPaths(root=root)


def ensure_experiment_meta(
    paths: ExperimentPaths,
    *,
    robot_id: str,
    robot_type: str,
    experiment_name: str,
) -> None:
    if paths.meta_json.exists():
        return
    payload = {
        "robot_id": robot_id,
        "robot_type": robot_type,
        "experiment_name": experiment_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runs": [],
    }
    paths.meta_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def record_phase_run(
    paths: ExperimentPaths,
    *,
    phase: str,
    camera: str,
    phase_dir: Path,
    extra: dict[str, Any] | None = None,
) -> None:
    entry: dict[str, Any] = {
        "phase": phase,
        "camera": camera,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dir": str(phase_dir.relative_to(paths.root)),
    }
    if extra:
        entry.update(extra)
    if paths.meta_json.exists():
        meta = json.loads(paths.meta_json.read_text(encoding="utf-8"))
    else:
        meta = {"runs": []}
    meta.setdefault("runs", []).append(entry)
    paths.meta_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def archive_existing_phase_dir(paths: ExperimentPaths, phase: str, camera: str) -> Path | None:
    """If phase dir already has data, move it to ``archive/`` before a new run."""
    phase_dir = paths.phase_dir(phase, camera)
    if not phase_dir.exists():
        return None
    if not any(phase_dir.iterdir()):
        return None
    paths.archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived = paths.archive_dir / f"{phase}_{camera}_{stamp}"
    shutil.move(str(phase_dir), str(archived))
    return archived


def save_experiment_intrinsics(
    paths: ExperimentPaths,
    camera: str,
    intrinsics: CameraIntrinsics,
    *,
    phase: str,
) -> Path:
    """Write canonical K/dist for ``camera`` (used by later hand_eye in same experiment)."""
    paths.intrinsics_dir.mkdir(parents=True, exist_ok=True)
    payload = intrinsics_to_dict(intrinsics, camera)
    payload["saved_from_phase"] = phase
    payload["saved_utc"] = datetime.now(timezone.utc).isoformat()
    out = paths.intrinsics_json(camera)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def load_experiment_intrinsics(paths: ExperimentPaths, camera: str) -> CameraIntrinsics | None:
    path = paths.intrinsics_json(camera)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("camera") not in (camera, None):
        return None
    return CameraIntrinsics(
        width=int(data["width"]),
        height=int(data["height"]),
        camera_matrix=np.asarray(data["camera_matrix"], dtype=np.float64),
        dist_coeffs=np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1),
        reprojection_error=data.get("reprojection_error_px"),
    )
