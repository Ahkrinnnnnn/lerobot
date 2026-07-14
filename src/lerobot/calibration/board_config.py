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

"""ChArUco board configuration (physical print parameters)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2


ARUCO_DICT_BY_NAME: dict[str, int] = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
}


def resolve_aruco_dict(name: str) -> int:
    key = name.strip().upper()
    if key not in ARUCO_DICT_BY_NAME:
        supported = ", ".join(sorted(ARUCO_DICT_BY_NAME))
        raise ValueError(f"Unknown aruco_dict {name!r}. Supported: {supported}")
    return ARUCO_DICT_BY_NAME[key]


ARUCO_DICT_NAME_BY_ID: dict[int, str] = {v: k for k, v in ARUCO_DICT_BY_NAME.items()}


def aruco_dict_to_name(aruco_id: int) -> str:
    return ARUCO_DICT_NAME_BY_ID.get(aruco_id, str(aruco_id))


@dataclass
class CharucoBoardConfig:
    """Printed ChArUco board geometry."""

    squares_x: int = 8
    squares_y: int = 11
    square_size_mm: float = 15.0
    marker_size_mm: float = 11.0
    aruco_dict: str = "DICT_4X4_50"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CharucoBoardConfig:
        return cls(
            squares_x=int(data["squares_x"]),
            squares_y=int(data["squares_y"]),
            square_size_mm=float(data["square_size_mm"]),
            marker_size_mm=float(data["marker_size_mm"]),
            aruco_dict=str(data.get("aruco_dict", "DICT_4X4_50")),
        )

    @classmethod
    def from_json(cls, path: Path | str) -> CharucoBoardConfig:
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "squares_x": self.squares_x,
            "squares_y": self.squares_y,
            "square_size_mm": self.square_size_mm,
            "marker_size_mm": self.marker_size_mm,
            "aruco_dict": self.aruco_dict,
        }

    def to_target(self) -> CalibrationTargetConfig:
        from .target import CalibrationTargetConfig

        return CalibrationTargetConfig(
            squares_x=self.squares_x,
            squares_y=self.squares_y,
            square_size_mm=self.square_size_mm,
            marker_size_mm=self.marker_size_mm,
            aruco_dict=resolve_aruco_dict(self.aruco_dict),
        )

    @classmethod
    def from_target(cls, target: CalibrationTargetConfig) -> CharucoBoardConfig:
        return cls(
            squares_x=target.squares_x,
            squares_y=target.squares_y,
            square_size_mm=target.square_size_mm,
            marker_size_mm=target.marker_size_mm,
            aruco_dict=aruco_dict_to_name(target.aruco_dict),
        )
