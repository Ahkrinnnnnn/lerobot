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

import numpy as np

from lerobot.calibration.chessboard import CameraIntrinsics
from lerobot.calibration.experiment_paths import (
    archive_existing_phase_dir,
    load_experiment_intrinsics,
    resolve_experiment_root,
    save_experiment_intrinsics,
)


def test_experiment_intrinsics_roundtrip(tmp_path):
    paths = resolve_experiment_root("lab_arm_01", experiment_dir=tmp_path / "exp_a")
    intrinsics = CameraIntrinsics(
        width=640,
        height=480,
        camera_matrix=np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]]),
        dist_coeffs=np.zeros(5),
        reprojection_error=0.25,
    )
    save_experiment_intrinsics(paths, "wrist", intrinsics, phase="intrinsics")
    loaded = load_experiment_intrinsics(paths, "wrist")
    assert loaded is not None
    assert loaded.width == 640
    assert np.allclose(loaded.camera_matrix, intrinsics.camera_matrix)
    assert loaded.reprojection_error == 0.25


def test_phase_archive_on_rerun(tmp_path):
    paths = resolve_experiment_root("lab_arm_01", experiment_dir=tmp_path / "exp_b")
    phase_dir = paths.phase_dir("hand_eye", "wrist")
    phase_dir.mkdir(parents=True)
    (phase_dir / "report.json").write_text("{}", encoding="utf-8")
    archived = archive_existing_phase_dir(paths, "hand_eye", "wrist")
    assert archived is not None
    assert archived.is_dir()
    assert not phase_dir.exists()
