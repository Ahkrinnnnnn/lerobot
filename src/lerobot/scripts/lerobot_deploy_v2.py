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

"""Fixed-rate policy deployment for ACT and VLA models (deploy v2).

Combines a multiprocess-safe control loop (stable ``send_action`` frequency)
with the rollout inference stack (sync + RTC) and multi-robot support.

Control frequency is set by :attr:`~lerobot.rollout.configs.DeployV2Config.fps`
(default 30 Hz) or ``--fps=...`` on the CLI — it is **not** read from
``train_config.json``.

Usage examples
--------------
::

    # ACT — sync inference in a spawn subprocess
    lerobot-deploy-v2 \\
        --policy.path=lerobot/act_koch_real \\
        --inference.type=sync \\
        --robot.type=koch_follower \\
        --robot.port=/dev/ttyACM0 \\
        --task="pick up cube" \\
        --fps=30

    # Pi0 / SmolVLA / XVLA — RTC in a spawn subprocess (isolated GIL, recommended)
    lerobot-deploy-v2 \\
        --policy.path=lerobot/pi0_base \\
        --inference.type=rtc \\
        --inference.rtc.execution_horizon=10 \\
        --robot.type=so100_follower \\
        --robot.port=/dev/ttyACM0 \\
        --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
        --task="pick up cube" \\
        --fps=15 \\
        --duration=60
"""

import logging

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_rebot_b601_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    omx_follower,
    openarm_follower,
    reachy2,
    rebot_b601_follower,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.rollout import DeployV2Config, build_rollout_context, create_strategy
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun

logger = logging.getLogger(__name__)


@parser.wrap()
def deploy_v2(cfg: DeployV2Config) -> None:
    """Run fixed-rate deployment until shutdown or duration expires."""
    init_logging()

    if cfg.display_data:
        logger.info("Initializing Rerun visualization (ip=%s, port=%s)", cfg.display_ip, cfg.display_port)
        init_rerun(session_name="deploy_v2", ip=cfg.display_ip, port=cfg.display_port)

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    logger.info("Building deploy context (control fps=%.2f)...", cfg.fps)
    ctx = build_rollout_context(cfg, shutdown_event)

    strategy = create_strategy(cfg.strategy)
    logger.info(
        "Deploy v2 | strategy=%s | inference=%s | robot=%s | fps=%.2f | duration=%s",
        cfg.strategy.type,
        cfg.inference.type,
        cfg.robot.type if cfg.robot else "?",
        cfg.fps,
        f"{cfg.duration}s" if cfg.duration > 0 else "infinite",
    )

    try:
        strategy.setup(ctx)
        logger.info("Deploy setup complete, starting control loop...")
        strategy.run(ctx)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        strategy.teardown(ctx)

    logger.info("Deploy v2 finished")


def main() -> None:
    register_third_party_plugins()
    deploy_v2()


if __name__ == "__main__":
    main()
