# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Guardrails: source datasets are read-only; outputs must live elsewhere."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def assert_source_dataset_readonly(
    source_root: str | Path,
    *,
    output_dataset_root: str | Path | None = None,
    annotation_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> None:
    """Ensure generated files are never written into the source dataset directory."""
    source = Path(source_root).resolve()

    def _check_output(label: str, path: str | Path) -> None:
        target = Path(path).resolve()
        if target == source:
            raise ValueError(
                f"{label} ({target}) must not equal the source dataset root. "
                "The source dataset is read-only and is never modified."
            )
        if _is_within(target, source):
            raise ValueError(
                f"{label} ({target}) must not be inside the source dataset ({source}). "
                "Configure a separate directory under `output` in your config file."
            )

    if output_dataset_root is not None:
        _check_output("output.dataset_root", output_dataset_root)
    if annotation_path is not None:
        _check_output("output.annotation_path", annotation_path)
    if report_path is not None:
        _check_output("output.report_path", report_path)

    logger.info("Source dataset is read-only: %s", source)
