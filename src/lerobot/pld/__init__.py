# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from .configs import PLDStage1Config, PLDStage2Config

__all__ = [
    "PLDStage1Config",
    "PLDStage1Orchestrator",
    "PLDStage2Config",
    "PLDStage2Orchestrator",
]


def __getattr__(name: str):
    """Lazy imports avoid circular import with ``lerobot.rollout`` during strategy load."""
    if name == "PLDStage1Orchestrator":
        from .orchestrator import PLDStage1Orchestrator

        return PLDStage1Orchestrator
    if name == "PLDStage2Orchestrator":
        from .orchestrator_stage2 import PLDStage2Orchestrator

        return PLDStage2Orchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
