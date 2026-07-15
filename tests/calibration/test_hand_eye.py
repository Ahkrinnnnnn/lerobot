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

from lerobot.calibration.hand_eye import (
    CameraMount,
    HandEyeSample,
    per_sample_board_origin_mm,
    board_to_robot_from_eye_in_hand,
    solve_hand_eye,
)
from lerobot.calibration.landmark import landmark_to_robot_from_eye_in_hand
from lerobot.calibration.transforms import compose_transforms, invert_transform, make_transform


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=3)
    a /= np.linalg.norm(a) + 1e-12
    angle = rng.uniform(-np.pi, np.pi)
    kx, ky, kz = a
    c, s = np.cos(angle), np.sin(angle)
    k = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
    return np.eye(3) + s * k + (1 - c) * (k @ k)


def _synthesize_eye_in_hand_samples(n: int, seed: int = 0) -> tuple[np.ndarray, list[HandEyeSample]]:
    """Geometric synthesis: p_cam = T_ee_to_cam @ p_ee, p_cam = C @ p_board."""
    rng = np.random.default_rng(seed)
    t_ee_to_cam = make_transform(_random_rotation(rng), rng.normal(size=3) * 50)
    t_robot_to_target = make_transform(_random_rotation(rng), rng.uniform(-200, 200, size=3))

    samples: list[HandEyeSample] = []
    for _ in range(n):
        t_robot_to_ee = make_transform(_random_rotation(rng), rng.uniform(-400, 400, size=3))
        # C = T_ee_to_cam @ inv(G) @ T_board_to_robot
        t_target_to_cam = compose_transforms(
            t_ee_to_cam,
            compose_transforms(invert_transform(t_robot_to_ee), t_robot_to_target),
        )
        samples.append(HandEyeSample(T_ee_to_robot=t_robot_to_ee, T_target_to_camera=t_target_to_cam))
    return t_ee_to_cam, samples


def test_solve_eye_in_hand_recovers_extrinsics():
    gt, samples = _synthesize_eye_in_hand_samples(12)
    est = solve_hand_eye(samples, CameraMount.EYE_IN_HAND)
    assert np.allclose(gt, est, atol=1e-2)


def test_board_to_robot_three_transform_chain():
    gt, samples = _synthesize_eye_in_hand_samples(8, seed=1)
    t_robot_to_target = board_to_robot_from_eye_in_hand(
        samples[0].T_ee_to_robot, gt, samples[0].T_target_to_camera
    )
    for sample in samples:
        via_helper = board_to_robot_from_eye_in_hand(
            sample.T_ee_to_robot, gt, sample.T_target_to_camera
        )
        via_landmark = landmark_to_robot_from_eye_in_hand(
            sample.T_ee_to_robot, gt, sample.T_target_to_camera
        )
        assert np.allclose(via_helper, via_landmark)
        assert np.allclose(via_helper, t_robot_to_target, atol=1e-6)

    origins = per_sample_board_origin_mm(samples, CameraMount.EYE_IN_HAND, gt)
    assert np.allclose(origins, np.tile(t_robot_to_target[:3, 3], (len(samples), 1)), atol=1e-6)


def test_scene_npz_roundtrip(tmp_path):
    from lerobot.calibration.io import load_scene_calibration, save_scene_calibration
    from lerobot.calibration.scene import CameraCalibration, SceneCalibration, TableCalibration
    from lerobot.calibration.hand_eye import CameraMount

    scene = SceneCalibration(
        robot_type="crp_arm",
        robot_id="test",
        charuco={
            "squares_x": 8,
            "squares_y": 11,
            "square_size_mm": 15.0,
            "marker_size_mm": 11.0,
            "aruco_dict": "DICT_4X4_50",
        },
        cameras={
            "top": CameraCalibration(
                mount=CameraMount.EYE_TO_HAND,
                width=640,
                height=480,
                camera_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
                dist_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
                T_camera_to_robot=np.eye(4).tolist(),
            )
        },
        table=TableCalibration(
            T_table_to_robot=np.eye(4).tolist(),
            grid_points_robot_mm=[[0.0, 0.0, 0.0], [15.0, 0.0, 0.0]],
        ),
    )
    path = tmp_path / "scene.npz"
    save_scene_calibration(scene, path)
    loaded = load_scene_calibration(path)
    assert loaded.robot_id == "test"
    assert "top" in loaded.cameras
    assert loaded.table is not None
    assert len(loaded.table.grid_points_robot_mm) == 2

    from lerobot.calibration.io import describe_npz

    summary = describe_npz(path)
    assert "top__camera_matrix" in summary
    assert "table_grid_points_robot_mm" in summary
