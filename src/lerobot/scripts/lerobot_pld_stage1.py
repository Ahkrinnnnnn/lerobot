#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""PLD Algorithm 1 Stage 1: offline collect, Cal-QL, residual SAC on CRP + Pi0.5."""

import logging

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.pld.configs import PLDStage1Config
from lerobot.pld.orchestrator import PLDStage1Orchestrator
from lerobot.robots import (  # noqa: F401
    crp_arm,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


@parser.wrap()
def pld_stage1(cfg: PLDStage1Config) -> None:
    init_logging()
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    orchestrator = PLDStage1Orchestrator(cfg, shutdown_event=signal_handler.shutdown_event)
    logger.info(
        "PLD Stage 1 | task=%s | base=%s | output=%s",
        cfg.task,
        cfg.base_policy.pretrained_path,
        cfg.output_dir,
    )
    try:
        orchestrator.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    logger.info("PLD Stage 1 finished")


def main() -> None:
    register_third_party_plugins()
    pld_stage1()


if __name__ == "__main__":
    main()
