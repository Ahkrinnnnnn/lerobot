#!/usr/bin/env python

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

"""Replay a CRP EE dataset episode with **absolute** GP poses (``action.ee.*``).

Pendant: Auto + program that tracks GP (e.g. ``mv`` on GP registers when GI55=GP).
PC writes absolute ``set_GPs`` each dataset frame (default 30 Hz).

```bash
python -m lerobot.scripts.crp_replay_ee_abs \
  --config_path=examples/hilserl/record/crp_omy_replay_ee.json
```
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

from lerobot.configs import parser
from lerobot.datasets import LeRobotDataset
from lerobot.robots import RobotConfig, make_robot_from_config  # noqa: F401 — registers subclasses
from lerobot.robots.crp_arm import CRPArm
from lerobot.robots.crp_arm.config_crp_arm import CRPArmConfig  # noqa: F401 — register crp_arm
from lerobot.robots.crp_arm.ee_gp import send_gp_endpose6
from lerobot.tools import TrajectoryProcessor
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say

_EE_NAMES = ("ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw")


@dataclass
class DatasetReplayConfig:
    repo_id: str
    episode: int = 0
    root: str | Path | None = None
    fps: int | None = None


@dataclass
class ReplayConfig:
    robot: RobotConfig
    dataset: DatasetReplayConfig
    play_sounds: bool = True
    speed_ratio: int | None = 80
    # If True, open GI56 after GP preload (needed when pendant gates ``mv`` on S56).
    enable_motion_gate: bool = True


def _action_dict(dataset: LeRobotDataset, row: Any) -> dict[str, float]:
    names = list(dataset.features[ACTION]["names"])
    arr = row[ACTION]
    return {names[i]: float(arr[i]) for i in range(len(names))}


def _pose6(action: dict[str, float]) -> list[float]:
    missing = [n for n in _EE_NAMES if n not in action]
    if missing:
        raise KeyError(f"action missing EE keys {missing}; have {sorted(action)}")
    return [float(action[n]) for n in _EE_NAMES]


@parser.wrap()
def replay(cfg: ReplayConfig) -> None:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)
    if not isinstance(robot, CRPArm):
        raise TypeError("crp_replay_ee_abs requires robot.type=crp_arm")

    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=[cfg.dataset.episode],
    )
    fps = float(cfg.dataset.fps if cfg.dataset.fps is not None else dataset.fps)
    actions = dataset.select_columns(ACTION)
    names = list(dataset.features[ACTION]["names"])
    if not all(n in names for n in _EE_NAMES):
        raise ValueError(f"Dataset action must include {_EE_NAMES}; got {names}")

    traj = TrajectoryProcessor()
    robot.connect()
    try:
        if cfg.speed_ratio is not None:
            robot.set_speed_ratio(int(cfg.speed_ratio))

        # Always start with MOVE gate closed until first absolute pose is preloaded.
        robot.set_motion_enabled(False)

        first = _action_dict(dataset, actions[0])
        pose0 = _pose6(first)
        logging.info(
            "Preload absolute GP pose0 xyz=(%.3f %.3f %.3f) rpy=(%.3f %.3f %.3f)",
            *pose0,
        )
        send_gp_endpose6(robot, traj, pose0, switch_to_gp_mode=False)
        send_gp_endpose6(robot, traj, pose0, start_index=20, switch_to_gp_mode=False)
        robot.ensure_gp_mode()
        if "gripper.pos" in first and robot.config.use_gripper_feature:
            robot.set_GOT(0, int(max(0, min(1000, round(float(first["gripper.pos"]))))))

        if cfg.enable_motion_gate:
            robot.set_motion_enabled(True)
            logging.info("GI→GP armed; GI56 ON — pendant may track absolute GP")
        else:
            logging.info("GI→GP armed; GI56 left OFF (enable_motion_gate=false)")

        n = dataset.num_frames
        log_say(f"Replaying EE absolute episode {cfg.dataset.episode}", cfg.play_sounds, blocking=True)
        logging.info("Replaying %d absolute EE frames @ %.1f fps", n, fps)

        last_got: int | None = None
        for idx in range(n):
            t0 = time.perf_counter()
            action = _action_dict(dataset, actions[idx])
            pose = _pose6(action)
            send_gp_endpose6(robot, traj, pose, switch_to_gp_mode=False)
            if "gripper.pos" in action and robot.config.use_gripper_feature:
                got = int(max(0, min(1000, round(float(action["gripper.pos"])))))
                if got != last_got:
                    robot.set_GOT(0, got)
                    last_got = got
            precise_sleep(max(1.0 / fps - (time.perf_counter() - t0), 0.0))
    finally:
        try:
            robot.set_motion_enabled(False)
        except Exception:
            logging.getLogger(__name__).debug("teardown GI56", exc_info=True)
        robot.disconnect()


def main() -> None:
    replay()


if __name__ == "__main__":
    main()
