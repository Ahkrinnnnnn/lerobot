# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Interactive OpenCV tool to mark success/failure segments on dataset episode videos."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .annotations import EpisodeRewardAnnotation, RewardClassifierAnnotations

logger = logging.getLogger(__name__)

WINDOW_NAME = "Reward Classifier Annotator"


@dataclass
class AnnotatorConfig:
    source_repo_id: str
    source_root: str | Path
    annotation_path: str | Path
    episode_start: int = 0
    episode_indices: list[int] | None = None
    playback_fps: float | None = None
    display_camera_key: str | None = None


def _tensor_to_hwc_uint8(value) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def _compose_display_frame(frame_by_camera: dict[str, np.ndarray], camera_keys: list[str]) -> np.ndarray:
    panels = [_tensor_to_hwc_uint8(frame_by_camera[key]) for key in camera_keys if key in frame_by_camera]
    if not panels:
        raise ValueError("No camera frames available for display.")
    if len(panels) == 1:
        return panels[0]
    return np.concatenate(panels, axis=1)


def _draw_timeline(
    canvas: np.ndarray,
    *,
    ep_length: int,
    frame_idx: int,
    annotation: EpisodeRewardAnnotation,
    pending_success_start: int | None,
    pending_failure_start: int | None,
) -> np.ndarray:
    import cv2

    bar_h = 28
    h, w = canvas.shape[:2]
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    bar[:] = (40, 40, 40)

    def draw_segment(start: int, end: int, color: tuple[int, int, int]) -> None:
        if ep_length <= 1:
            return
        x0 = int(start / (ep_length - 1) * (w - 1))
        x1 = int(end / (ep_length - 1) * (w - 1))
        cv2.rectangle(bar, (x0, 4), (x1, bar_h - 4), color, -1)

    for seg in annotation.failure_segments:
        draw_segment(seg[0], seg[1], (60, 60, 220))
    for seg in annotation.success_segments:
        draw_segment(seg[0], seg[1], (60, 200, 60))
    if pending_failure_start is not None:
        draw_segment(pending_failure_start, frame_idx, (120, 120, 255))
    if pending_success_start is not None:
        draw_segment(pending_success_start, frame_idx, (120, 255, 120))

    if ep_length > 1:
        cursor_x = int(frame_idx / (ep_length - 1) * (w - 1))
        cv2.line(bar, (cursor_x, 0), (cursor_x, bar_h - 1), (255, 255, 255), 2)

    return np.vstack([canvas, bar])


def _overlay_help(
    canvas: np.ndarray,
    *,
    ep_idx: int,
    num_episodes: int,
    frame_idx: int,
    ep_length: int,
    playing: bool,
    annotation: EpisodeRewardAnnotation,
) -> np.ndarray:
    import cv2

    lines = [
        f"Episode {ep_idx + 1}/{num_episodes} | Frame {frame_idx + 1}/{ep_length} | {'PLAY' if playing else 'PAUSE'}",
        f"Success segments: {len(annotation.success_segments)} | Failure segments: {len(annotation.failure_segments)}",
        "[ mark success START | ] mark success END | f mark failure START | g mark failure END",
        "Space: play/pause | a/d: step frame | n/p: next/prev episode | u: undo | q: save & quit",
    ]
    y = 24
    for line in lines:
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return canvas


def _load_episode_frames(
    source_repo_id: str,
    source_root: str | Path,
    ep_idx: int,
    camera_keys: list[str],
) -> list[dict[str, np.ndarray]]:
    ep_dataset = LeRobotDataset(
        source_repo_id,
        root=source_root,
        episodes=[ep_idx],
        return_uint8=True,
    )
    frames: list[dict[str, np.ndarray]] = []
    for local_idx in range(len(ep_dataset)):
        item = ep_dataset[local_idx]
        frames.append({key: item[key] for key in camera_keys})
    return frames


def run_reward_classifier_annotator(cfg: AnnotatorConfig) -> RewardClassifierAnnotations:
    if not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "DISPLAY is not set. The annotator requires a graphical environment (local monitor or X forwarding)."
        )

    import cv2

    meta = LeRobotDatasetMetadata(cfg.source_repo_id, root=cfg.source_root)
    meta.ensure_readable()
    camera_keys = list(meta.camera_keys)
    if not camera_keys:
        raise ValueError("Dataset has no camera keys to annotate.")

    if cfg.episode_indices:
        episode_list = [idx for idx in cfg.episode_indices if 0 <= idx < meta.total_episodes]
    else:
        episode_list = list(range(cfg.episode_start, meta.total_episodes))

    if not episode_list:
        raise ValueError("No episodes selected for annotation.")

    annotations = RewardClassifierAnnotations.load_or_create(
        cfg.annotation_path,
        source_repo_id=cfg.source_repo_id,
        source_root=str(meta.root),
        fps=meta.fps,
        camera_keys=camera_keys,
    )

    playback_fps = cfg.playback_fps or float(meta.fps)
    delay_ms = max(1, int(1000 / playback_fps))

    ep_cursor = 0
    frame_idx = 0
    playing = False
    pending_success_start: int | None = None
    pending_failure_start: int | None = None
    cached_ep_idx: int | None = None
    cached_frames: list[dict[str, np.ndarray]] = []

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    logger.info(
        "Annotating %d episodes. Labels auto-save to %s on episode switch / quit.",
        len(episode_list),
        cfg.annotation_path,
    )

    while True:
        ep_idx = episode_list[ep_cursor]
        if cached_ep_idx != ep_idx:
            logger.info("Loading episode %d frames...", ep_idx)
            cached_frames = _load_episode_frames(cfg.source_repo_id, cfg.source_root, ep_idx, camera_keys)
            cached_ep_idx = ep_idx
            frame_idx = min(frame_idx, max(0, len(cached_frames) - 1))
            pending_success_start = None
            pending_failure_start = None

        ep_length = len(cached_frames)
        if ep_length == 0:
            logger.warning("Episode %d is empty, skipping.", ep_idx)
            ep_cursor = min(ep_cursor + 1, len(episode_list) - 1)
            continue

        frame_idx = max(0, min(frame_idx, ep_length - 1))
        ep_ann = annotations.get_episode(ep_idx)
        display = _compose_display_frame(cached_frames[frame_idx], camera_keys)
        display = _draw_timeline(
            display,
            ep_length=ep_length,
            frame_idx=frame_idx,
            annotation=ep_ann,
            pending_success_start=pending_success_start,
            pending_failure_start=pending_failure_start,
        )
        display = _overlay_help(
            display,
            ep_idx=ep_idx,
            num_episodes=len(episode_list),
            frame_idx=frame_idx,
            ep_length=ep_length,
            playing=playing,
            annotation=ep_ann,
        )
        cv2.imshow(WINDOW_NAME, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))

        wait = delay_ms if playing else 0
        key = cv2.waitKey(wait) & 0xFF

        if playing and key == 255:
            frame_idx = min(frame_idx + 1, ep_length - 1)
            if frame_idx >= ep_length - 1:
                playing = False
            continue

        if key in (ord("q"), 27):
            annotations.save(cfg.annotation_path)
            break
        if key == ord(" "):
            playing = not playing
        elif key == ord("a"):
            frame_idx = max(0, frame_idx - 1)
            playing = False
        elif key == ord("d"):
            frame_idx = min(ep_length - 1, frame_idx + 1)
            playing = False
        elif key == ord("["):
            pending_success_start = frame_idx
            pending_failure_start = None
            playing = False
        elif key == ord("]"):
            if pending_success_start is None:
                ep_ann.add_success_segment(frame_idx, frame_idx)
            else:
                ep_ann.add_success_segment(pending_success_start, frame_idx)
                pending_success_start = None
            playing = False
        elif key == ord("f"):
            pending_failure_start = frame_idx
            pending_success_start = None
            playing = False
        elif key == ord("g"):
            if pending_failure_start is None:
                ep_ann.add_failure_segment(frame_idx, frame_idx)
            else:
                ep_ann.add_failure_segment(pending_failure_start, frame_idx)
                pending_failure_start = None
            playing = False
        elif key == ord("u"):
            ep_ann.undo_last()
            pending_success_start = None
            pending_failure_start = None
        elif key == ord("n"):
            annotations.save(cfg.annotation_path)
            ep_cursor = min(ep_cursor + 1, len(episode_list) - 1)
            frame_idx = 0
            playing = False
        elif key == ord("p"):
            annotations.save(cfg.annotation_path)
            ep_cursor = max(ep_cursor - 1, 0)
            frame_idx = 0
            playing = False

    cv2.destroyAllWindows()
    logger.info("Saved annotations: %s", annotations.to_dict())
    return annotations
