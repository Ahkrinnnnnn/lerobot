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
import pytest

from lerobot.calibration.board_config import CharucoBoardConfig
from lerobot.calibration.charuco import CharucoConfig, charuco_corner_points_board_mm, table_grid_points_robot_mm
from lerobot.calibration.landmark import (
    camera_extrinsic_from_landmark,
    fuse_robot_to_landmark,
    robot_to_landmark_from_eye_in_hand,
)
from lerobot.calibration.transforms import compose_transforms, invert_transform, make_transform


def test_board_config_to_target():
    board = CharucoBoardConfig(
        squares_x=8,
        squares_y=11,
        square_size_mm=15.0,
        marker_size_mm=11.0,
        aruco_dict="DICT_4X4_50",
    )
    target = board.to_target()
    assert target.squares_x == 8
    assert target.squares_y == 11
    assert target.square_size_mm == 15.0
    assert CharucoBoardConfig.from_target(target) == board


def test_target_from_dict_requires_board_fields():
    from lerobot.calibration.target import CalibrationTargetConfig

    with pytest.raises(KeyError):
        CalibrationTargetConfig.from_dict({"squares_x": 11})


def test_target_from_dict_roundtrip_charuco_json():
    from pathlib import Path

    import json

    from lerobot.calibration.target import CalibrationTargetConfig

    data = json.loads(
        Path("examples/hand_eye_calibration/charuco_board.json").read_text(encoding="utf-8")
    )
    target = CalibrationTargetConfig.from_dict(data)
    assert target.squares_x == 11
    assert target.squares_y == 8
    assert target.square_size_mm == 20.0
    assert target.to_dict() == data


def test_charuco_grid_corners_8x11():
    cfg = CharucoConfig(squares_x=8, squares_y=11, square_size=15.0)
    pts = charuco_corner_points_board_mm(cfg)
    assert len(pts) == 7 * 10
    assert np.allclose(pts[0], [0.0, 0.0, 0.0])
    assert np.allclose(pts[1], [15.0, 0.0, 0.0])


def test_table_grid_in_robot_frame():
    cfg = CharucoConfig(squares_x=8, squares_y=11, square_size=15.0)
    T = make_transform(np.eye(3), np.array([100.0, 200.0, 50.0]))
    grid = table_grid_points_robot_mm(T, cfg)
    assert np.allclose(grid[0], [100.0, 200.0, 50.0])
    assert np.allclose(grid[1], [115.0, 200.0, 50.0])


def test_robot_to_landmark_chain():
    T_robot_to_ee = make_transform(np.eye(3), np.array([100.0, 0.0, 200.0]))
    T_ee_to_cam = make_transform(np.eye(3), np.array([0.0, 0.0, 50.0]))
    T_landmark_to_cam = make_transform(np.eye(3), np.array([10.0, 20.0, 300.0]))
    T = robot_to_landmark_from_eye_in_hand(T_robot_to_ee, T_ee_to_cam, T_landmark_to_cam)
    expected = compose_transforms(
        T_robot_to_ee,
        compose_transforms(T_ee_to_cam, invert_transform(T_landmark_to_cam)),
    )
    assert np.allclose(T, expected)
    assert np.allclose(T[:3, 3], [90.0, -20.0, -50.0])


def test_camera_extrinsic_from_landmark():
    T_robot_to_table = make_transform(np.eye(3), np.array([0.0, 0.0, 0.0]))
    T_table_to_cam = make_transform(np.eye(3), np.array([0.0, 0.0, 500.0]))
    T_robot_to_cam = camera_extrinsic_from_landmark(T_robot_to_table, T_table_to_cam)
    assert np.allclose(T_robot_to_cam[:3, 3], [0.0, 0.0, -500.0])


def test_fuse_robot_to_landmark():
    base = make_transform(np.eye(3), np.array([10.0, 20.0, 30.0]))
    noisy = [
        base,
        make_transform(np.eye(3), np.array([11.0, 19.0, 31.0])),
        make_transform(np.eye(3), np.array([9.0, 21.0, 29.0])),
    ]
    fused, std = fuse_robot_to_landmark(noisy)
    assert np.allclose(fused[:3, 3], [10.0, 20.0, 30.0], atol=1.0)
    assert std < 2.0
