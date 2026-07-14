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

"""Non-blocking live preview for hand-eye calibration."""

from __future__ import annotations

import os
import select
import sys
import time
import tty
import termios
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from lerobot.cameras.camera import Camera

from .adapters import HandEyeRobot

# Primary save key; Space/Enter kept as aliases (common habit).
_CAPTURE_KEYS = frozenset({ord("s"), ord("S"), ord(" "), 13, 10})


def gui_available() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def read_live_frame(cam: Camera, *, max_age_ms: int = 2000) -> np.ndarray:
    """Return the freshest available frame without blocking on ``input()``."""
    try:
        return cam.read_latest(max_age_ms=max_age_ms)
    except (TimeoutError, RuntimeError):
        pass
    try:
        return cam.async_read(timeout_ms=100)
    except TimeoutError:
        return cam.read()


@dataclass
class LiveSessionConfig:
    window_name: str
    title: str
    help_lines: tuple[str, ...]
    counter_label: str = "samples"
    stop_key: str = "q"
    capture_key: str = "s"
    preview_fps: float = 30.0
    secondary_camera: str | None = None
    secondary_window_name: str | None = None
    secondary_title: str = ""
    secondary_help_lines: tuple[str, ...] = ("green = ChArUco detected", "save requires wrist + top")


def default_help_lines() -> tuple[str, ...]:
    return (
        "s / Space / Enter = save (terminal or preview window)",
        "q = finish phase (only after minimum sample count)",
        "Drag arm on teach pendant, then save",
    )


def _draw_hud(
    vis: np.ndarray,
    *,
    count: int,
    found: bool,
    cfg: LiveSessionConfig,
    extra_lines: tuple[str, ...] = (),
) -> np.ndarray:
    out = vis.copy()
    color = (0, 255, 0) if found else (0, 0, 255)
    cv2.putText(out, f"{cfg.counter_label}={count}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    y = 56
    for line in (cfg.title, *cfg.help_lines, *extra_lines):
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20
    return out


class _TerminalKeyReader:
    """Non-blocking single-key reads from the terminal (works alongside OpenCV)."""

    def __init__(self) -> None:
        self._enabled = sys.stdin.isatty()
        self._old_term: list[Any] | None = None
        if self._enabled:
            self._old_term = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

    def close(self) -> None:
        if self._enabled and self._old_term is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_term)
            self._old_term = None

    def poll(self) -> int:
        if not self._enabled:
            return 0
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0)
        except Exception:
            return 0
        if not ready:
            return 0
        ch = sys.stdin.read(1)
        if ch in ("\n", "\r"):
            return 13
        return ord(ch)


def _is_capture_key(key: int, cfg: LiveSessionConfig) -> bool:
    if key == 0 or key == 255:
        return False
    if key in _CAPTURE_KEYS:
        return True
    return key == ord(cfg.capture_key) or key == ord(cfg.capture_key.upper())


def _poll_key(use_gui: bool, window_names: tuple[str, ...], wait_ms: int, term_reader: _TerminalKeyReader) -> int:
    """Read key from terminal and/or any OpenCV preview window."""
    key = term_reader.poll()
    if key:
        return key
    if not use_gui:
        return 0
    raw = cv2.waitKey(max(1, wait_ms))
    if raw == -1:
        return 0
    return raw & 0xFF


def _any_window_closed(window_names: tuple[str, ...]) -> bool:
    for name in window_names:
        try:
            if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                return True
        except cv2.error:
            return True
    return False


def run_live_session(
    robot: HandEyeRobot,
    camera_name: str,
    cfg: LiveSessionConfig,
    *,
    process_frame: Callable[[np.ndarray], tuple[np.ndarray, bool, Any | None]],
    on_save: Callable[[], bool],
    min_count: int | None = None,
    initial_count: int = 0,
    status_callback: Callable[[str], None] | None = None,
) -> int:
    """Real-time preview until the user finishes the phase.

    Keys (work in **terminal** or OpenCV preview window):
      s / Space / Enter — trigger ``on_save()`` (sync capture happens inside the callback)
      q — finish phase

    Preview frames are **not** passed to ``on_save``; callers must grab a fresh
    ``cam.read()`` and robot pose inside the callback to avoid desync.
    """
    cam = robot.cameras[camera_name]
    use_gui = gui_available()
    wait_ms = int(1000 / max(cfg.preview_fps, 1.0))
    stop_code = ord(cfg.stop_key)
    term_reader = _TerminalKeyReader()

    secondary_cam: Camera | None = None
    secondary_window: str | None = None
    enable_secondary = bool(
        cfg.secondary_camera
        and cfg.secondary_window_name
        and cfg.secondary_camera in robot.cameras
    )
    if enable_secondary:
        secondary_cam = robot.cameras[cfg.secondary_camera]
        secondary_window = cfg.secondary_window_name

    window_names: list[str] = []
    if use_gui:
        try:
            cv2.startWindowThread()
        except Exception:
            pass
        cv2.namedWindow(cfg.window_name, cv2.WINDOW_NORMAL)
        window_names.append(cfg.window_name)
        if secondary_cam is not None and secondary_window is not None:
            cv2.namedWindow(secondary_window, cv2.WINDOW_NORMAL)
            window_names.append(secondary_window)

    gui_windows = tuple(window_names)

    def emit(msg: str) -> None:
        print(msg, flush=True)
        if status_callback is not None:
            status_callback(msg)

    count = max(initial_count, 0)
    frame_interval = 1.0 / max(cfg.preview_fps, 1.0)
    secondary_cfg = LiveSessionConfig(
        window_name=secondary_window or "secondary",
        title=cfg.secondary_title or (cfg.secondary_camera or "secondary"),
        help_lines=cfg.secondary_help_lines,
        counter_label=cfg.counter_label,
        stop_key=cfg.stop_key,
        capture_key=cfg.capture_key,
    )

    emit_lines = [f"\n[{cfg.title}] camera={camera_name!r}"] + [f"  {line}" for line in cfg.help_lines]
    if secondary_cam is not None and secondary_window is not None:
        emit_lines.append(f"  secondary preview: {cfg.secondary_camera!r} → window {secondary_window!r}")
    emit("\n".join(emit_lines))

    try:
        while True:
            loop_start = time.perf_counter()
            frame = read_live_frame(cam)
            vis, found, payload = process_frame(frame)

            if use_gui:
                hud = _draw_hud(vis, count=count, found=found, cfg=cfg)
                cv2.imshow(cfg.window_name, cv2.cvtColor(hud, cv2.COLOR_RGB2BGR))
                if secondary_cam is not None and secondary_window is not None:
                    sec_frame = read_live_frame(secondary_cam)
                    sec_vis, sec_found, _ = process_frame(sec_frame)
                    sec_hud = _draw_hud(
                        sec_vis,
                        count=count,
                        found=sec_found,
                        cfg=secondary_cfg,
                    )
                    cv2.imshow(secondary_window, cv2.cvtColor(sec_hud, cv2.COLOR_RGB2BGR))

            key = _poll_key(use_gui, gui_windows, wait_ms, term_reader)

            if key == stop_code:
                if min_count is not None and count < min_count:
                    emit(
                        f"  need >= {min_count} saves before finishing (have {count}); keep collecting"
                    )
                else:
                    break
            if _is_capture_key(key, cfg):
                if on_save():
                    count += 1
                    emit(f"  saved #{count}")
                    if min_count is not None and count >= min_count:
                        emit(
                            f"  reached minimum ({min_count}); press {cfg.stop_key!r} to finish or keep saving"
                        )

            if use_gui and gui_windows and _any_window_closed(gui_windows):
                emit("  preview window closed")
                break

            elapsed = time.perf_counter() - loop_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
    finally:
        term_reader.close()
        if use_gui:
            for name in gui_windows:
                cv2.destroyWindow(name)

    return count
