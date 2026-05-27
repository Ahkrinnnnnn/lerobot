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

"""Deploy strategy: fixed-rate control loop with decoupled inference."""

from __future__ import annotations

import logging

from ..control import multiprocess_control_loop
from ..context import RolloutContext
from .core import RolloutStrategy

logger = logging.getLogger(__name__)


class DeployStrategy(RolloutStrategy):
    """Autonomous deployment with a multiprocess-safe fixed-rate control loop.

    Supports sync (spawn subprocess) and RTC inference backends for policies
    such as ACT, SmolVLA, XVLA, and Pi0.
    """

    def setup(self, ctx: RolloutContext) -> None:
        self._init_engine(ctx)
        logger.info("Deploy strategy ready")

    def run(self, ctx: RolloutContext) -> None:
        control_loop_stats: dict = {}
        multiprocess_control_loop(ctx, self, control_loop_stats=control_loop_stats)

        if control_loop_stats:
            frames = control_loop_stats.get("frames", 0)
            overruns = control_loop_stats.get("overruns", 0)
            sends = control_loop_stats.get("sends", 0)
            pct = (100.0 * overruns / frames) if frames else 0.0
            logger.info(
                "Deploy: %d control frames, %d sends, %d overruns (%.1f%%).",
                frames,
                sends,
                overruns,
                pct,
            )

    def teardown(self, ctx: RolloutContext) -> None:
        self._teardown_hardware(
            ctx.hardware,
            return_to_initial_position=ctx.runtime.cfg.return_to_initial_position,
        )
        logger.info("Deploy strategy teardown complete")
