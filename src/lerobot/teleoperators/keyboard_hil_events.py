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

"""Shared keyboard events for HIL-SERL (OMY leader + CRP).

Key semantics (HIL path only; does not change PLD defaults):
  - space (hold): intervention / takeover
  - s: success
  - f: failure / terminate
  - r: rerecord episode

During scene-reset countdown, the same space key toggles pause (PLD-style).

Input backends (both can run):
  1. pynput global listener (no suppress — Linux suppress often breaks keys)
  2. tty/stdin cbreak reader — works when the actor terminal has focus
"""

from __future__ import annotations

import logging
import math
import os
import atexit
import select
import sys
import termios
import threading
import time
import tty
from typing import Any
from collections.abc import Callable

from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.import_utils import _pynput_available
from lerobot.utils.robot_utils import precise_sleep

logger = logging.getLogger(__name__)

_OPERATOR_BANNER = "!" * 72

# If no Space char arrives within this window, treat hold as released (tty key-repeat path).
# Keep this generous: key-repeat gaps on Linux often exceed 0.3s and were falsely ending takeover.
_TTY_SPACE_HOLD_TIMEOUT_S = 1.0

_PYNPUT_KEYBOARD = None
if _pynput_available and os.environ.get("DISPLAY"):
    try:
        from pynput import keyboard as _pynput_keyboard_mod

        _PYNPUT_KEYBOARD = _pynput_keyboard_mod
    except ImportError:
        pass


def _key_matches(event_key, key_name: str) -> bool:
    if _PYNPUT_KEYBOARD is None:
        return False
    special_keys = {
        "space": _PYNPUT_KEYBOARD.Key.space,
        "tab": _PYNPUT_KEYBOARD.Key.tab,
        "enter": _PYNPUT_KEYBOARD.Key.enter,
    }
    if key_name in special_keys:
        if event_key == special_keys[key_name]:
            return True
        # Some layouts report space as a char KeyCode.
        if key_name == "space" and getattr(event_key, "char", None) == " ":
            return True
        return False
    char = getattr(event_key, "char", None)
    if char is None:
        return False
    return char.lower() == key_name.lower()


class KeyboardHilEvents:
    """HIL intervention + episode labels + reset pause (pynput and/or tty)."""

    def __init__(
        self,
        *,
        intervene_key: str = "space",
        success_key: str = "s",
        failure_key: str = "f",
        rerecord_key: str = "r",
        reset_pause_key: str | None = None,
        use_pynput: bool = True,
        use_tty: bool = True,
    ) -> None:
        self._intervene_key = intervene_key
        self._success_key = success_key
        self._failure_key = failure_key
        self._rerecord_key = rerecord_key
        self._reset_pause_key = reset_pause_key
        self._use_pynput = use_pynput
        self._use_tty = use_tty

        self._lock = threading.Lock()
        self._intervening = False
        self._success = False
        self._failure = False
        self._rerecord = False
        self._reset_paused = False
        self._reset_mode = False
        self._tty_space_last_mono = 0.0
        # When pynput reports Space down, tty timeout must not clear intervening.
        self._pynput_space_held = False
        # pynput and tty both see the same physical key; pynput sets this so tty skips once.
        self._skip_tty_pause_once = False

        self._listener = None
        self._tty_thread: threading.Thread | None = None
        self._tty_running = False
        self._tty_old_attrs = None
        self._tty_fd: int | None = None
        self._intervene_end_listeners: list[Callable[[], None]] = []

    def add_intervene_end_listener(self, callback: Callable[[], None]) -> None:
        """Notify when Space intervention ends (pynput release or tty hold timeout)."""
        if callback not in self._intervene_end_listeners:
            self._intervene_end_listeners.append(callback)

    def _notify_intervene_end(self) -> None:
        for cb in list(self._intervene_end_listeners):
            try:
                cb()
            except Exception:
                logger.exception("HIL intervene-end listener failed")

    def _clear_intervening_locked(self, *, notify: bool) -> bool:
        was = self._intervening
        self._intervening = False
        self._tty_space_last_mono = 0.0
        self._pynput_space_held = False
        return bool(was and notify)

    @property
    def intervening(self) -> bool:
        with self._lock:
            ended = self._refresh_tty_space_hold_locked()
            intervening = self._intervening
        if ended:
            self._notify_intervene_end()
        return intervening

    @property
    def reset_paused(self) -> bool:
        with self._lock:
            return self._reset_paused

    def clear_episode_flags(self) -> None:
        with self._lock:
            self._success = False
            self._failure = False
            self._rerecord = False

    def clear_reset_paused(self) -> None:
        with self._lock:
            self._reset_paused = False
            self._skip_tty_pause_once = False

    def set_reset_mode(self, enabled: bool) -> None:
        """When True, ``reset_pause_key`` toggles countdown pause instead of hold-intervene."""
        notify = False
        with self._lock:
            self._reset_mode = enabled
            if enabled:
                self._reset_paused = False
                self._skip_tty_pause_once = False
                notify = self._clear_intervening_locked(notify=True)
        if notify:
            self._notify_intervene_end()

    def _toggle_reset_pause_locked(self) -> bool | None:
        """Toggle pause; return new paused state, or None if not in reset mode."""
        if not self._reset_mode or not self._reset_pause_key:
            return None
        self._reset_paused = not self._reset_paused
        return self._reset_paused

    def _log_reset_pause_state(self, paused: bool) -> None:
        pause_key = self._reset_pause_key or "p"
        if paused:
            logger.warning("")
            logger.warning(_OPERATOR_BANNER)
            logger.warning(
                "!!! [MANUAL SCENE RESET] countdown PAUSED — press '%s' to resume !!!",
                pause_key,
            )
            logger.warning(_OPERATOR_BANNER)
            logger.warning("")
        else:
            logger.warning(
                ">>> [MANUAL SCENE RESET] countdown resumed — press '%s' to pause <<<",
                pause_key,
            )

    def _refresh_tty_space_hold_locked(self) -> bool:
        """Clear intervening when tty Space repeats stop (no key-up events on stdin).

        Returns True if intervention just ended (caller should notify listeners **outside** the lock).

        Do **not** clear while pynput still reports Space held — that was falsely ending takeover
        within ~0.35s even though the user kept Space down.
        """
        if self._reset_mode or self._pynput_space_held:
            return False
        if self._tty_space_last_mono <= 0:
            return False
        if time.monotonic() - self._tty_space_last_mono > _TTY_SPACE_HOLD_TIMEOUT_S:
            return self._clear_intervening_locked(notify=True)
        return False

    def consume_events(self) -> dict[str, Any]:
        """Return TeleopEvents-compatible dict; consume one-shot success/failure/rerecord."""
        ended = False
        with self._lock:
            ended = self._refresh_tty_space_hold_locked()
            is_intervention = self._intervening and not self._reset_mode
            success = self._success
            failure = self._failure
            rerecord = self._rerecord
            self._success = False
            self._failure = False
            self._rerecord = False

        if ended:
            self._notify_intervene_end()

        terminate = bool(failure or rerecord)
        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate,
            TeleopEvents.SUCCESS: bool(success),
            TeleopEvents.RERECORD_EPISODE: bool(rerecord),
        }

    def _apply_char(self, ch: str, *, from_tty: bool = False) -> None:
        if not ch:
            return
        key = ch.lower() if len(ch) == 1 else ch
        paused: bool | None = None
        with self._lock:
            if self._reset_mode and self._reset_pause_key:
                pause_name = self._reset_pause_key
                is_pause = (pause_name == "space" and ch == " ") or (key == pause_name)
                if is_pause:
                    # Same physical key already handled by pynput — do not toggle again.
                    if from_tty and self._skip_tty_pause_once:
                        self._skip_tty_pause_once = False
                        return
                    paused = self._toggle_reset_pause_locked()
            else:
                if (self._intervene_key == "space" and ch == " ") or key == self._intervene_key:
                    self._intervening = True
                    if from_tty:
                        self._tty_space_last_mono = time.monotonic()
                    return
                if key == self._success_key:
                    self._success = True
                    return
                if key == self._failure_key:
                    self._failure = True
                    return
                if key == self._rerecord_key:
                    self._rerecord = True
                    return

        if paused is not None:
            self._log_reset_pause_state(paused)

    def _start_pynput(self) -> bool:
        if not self._use_pynput or _PYNPUT_KEYBOARD is None:
            return False

        def on_press(key):
            try:
                paused: bool | None = None
                with self._lock:
                    if self._reset_mode and self._reset_pause_key and _key_matches(key, self._reset_pause_key):
                        paused = self._toggle_reset_pause_locked()
                        # Tell tty to ignore the duplicate char from the same keypress.
                        self._skip_tty_pause_once = True
                    elif (not self._reset_mode) and _key_matches(key, self._intervene_key):
                        self._intervening = True
                        self._pynput_space_held = True
                        return
                    elif _key_matches(key, self._success_key):
                        self._success = True
                        return
                    elif _key_matches(key, self._failure_key):
                        self._failure = True
                        return
                    elif _key_matches(key, self._rerecord_key):
                        self._rerecord = True
                        return
                    else:
                        return
                if paused is not None:
                    self._log_reset_pause_state(paused)
            except Exception:
                logger.exception("HIL pynput on_press error")

        def on_release(key):
            try:
                if (not self._reset_mode) and _key_matches(key, self._intervene_key):
                    notify = False
                    with self._lock:
                        # Trust pynput key-up immediately — cut OMY GP stream without waiting
                        # for tty Space-repeat timeout.
                        notify = self._clear_intervening_locked(notify=True)
                    if notify:
                        self._notify_intervene_end()
            except Exception:
                pass

        # Never use suppress=True on Linux — it frequently makes the keyboard unusable.
        self._listener = _PYNPUT_KEYBOARD.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()
        alive = bool(getattr(self._listener, "is_alive", lambda: True)())
        logger.info("HIL pynput listener started (alive=%s)", alive)
        return True

    def _start_tty(self) -> bool:
        if not self._use_tty:
            return False
        try:
            fd = sys.stdin.fileno()
        except Exception:
            return False
        if not os.isatty(fd):
            logger.info("HIL tty keyboard skipped (stdin is not a TTY)")
            return False

        try:
            self._tty_old_attrs = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except Exception:
            logger.exception("HIL tty keyboard: failed to set cbreak")
            return False

        self._tty_fd = fd
        self._tty_running = True

        def _loop() -> None:
            while self._tty_running:
                try:
                    ready, _, _ = select.select([fd], [], [], 0.05)
                    if not ready:
                        continue
                    ch = os.read(fd, 1).decode("utf-8", errors="ignore")
                    if ch:
                        self._apply_char(ch, from_tty=True)
                except Exception:
                    if self._tty_running:
                        logger.exception("HIL tty keyboard read error")
                    break

        self._tty_thread = threading.Thread(target=_loop, name="hil_tty_keyboard", daemon=True)
        self._tty_thread.start()
        logger.info(
            "HIL tty keyboard started — focus the actor terminal; hold Space / press s,f,r"
        )
        return True

    def start(self) -> None:
        if self._listener is not None or self._tty_thread is not None:
            return

        pynput_ok = self._start_pynput()
        tty_ok = self._start_tty()
        if not pynput_ok and not tty_ok:
            logger.warning(
                "KeyboardHilEvents disabled — neither pynput nor tty available. "
                "Need DISPLAY for pynput, or run actor in a real terminal for tty keys."
            )
            return

        # Ctrl+C often skips teleop.disconnect(); atexit still restores cooked tty for shell ↑.
        atexit.register(self.stop)

        logger.info(
            "HIL keyboard enabled: hold '%s'=intervene, '%s'=success, '%s'=failure, '%s'=rerecord "
            "(pynput=%s tty=%s)",
            self._intervene_key,
            self._success_key,
            self._failure_key,
            self._rerecord_key,
            pynput_ok,
            tty_ok,
        )

    def stop(self) -> None:
        """Stop listeners and restore stdin (cbreak → cooked) so shell history/arrows work."""
        # Idempotent: disconnect + atexit may both call this.
        had_work = (
            self._tty_thread is not None
            or self._tty_fd is not None
            or self._listener is not None
            or self._tty_running
        )
        if not had_work:
            return

        self._tty_running = False
        if self._tty_thread is not None:
            self._tty_thread.join(timeout=1.0)
            self._tty_thread = None
        self._restore_tty()

        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                logger.exception("HIL pynput listener stop failed")
            self._listener = None

    def _restore_tty(self) -> None:
        fd = self._tty_fd
        old = self._tty_old_attrs
        self._tty_fd = None
        self._tty_old_attrs = None
        if fd is None or old is None:
            return
        for when in (termios.TCSADRAIN, termios.TCSANOW):
            try:
                termios.tcsetattr(fd, when, old)
                return
            except Exception:
                continue
        logger.exception("HIL tty keyboard: failed to restore terminal (try: stty sane)")


def wait_for_manual_scene_reset(
    wait_s: float,
    *,
    episode_index: int = 0,
    pause_key: str = "space",
    keyboard_events: KeyboardHilEvents | None = None,
) -> None:
    """PLD-style scene-reset countdown; optional pause via ``KeyboardHilEvents``.

    While paused, the remaining time is **frozen** (deadline extended by pause duration).
    """
    if wait_s <= 0:
        return

    if keyboard_events is not None:
        keyboard_events.clear_reset_paused()
        keyboard_events.set_reset_mode(True)
        # Drop any s/f/r pressed during the previous episode end or while waiting —
        # otherwise the first step of the next episode can terminate immediately
        # (looks like a 1-frame episode with reward 0/1).
        keyboard_events.clear_episode_flags()

    effective_pause = pause_key
    if keyboard_events is not None:
        kb_pause = getattr(keyboard_events, "_reset_pause_key", None)
        if kb_pause:
            effective_pause = str(kb_pause)

    logger.warning("")
    logger.warning(_OPERATOR_BANNER)
    logger.warning(
        "!!! [MANUAL SCENE RESET] episode %s — %.1fs to rearrange scene "
        "(press '%s' to pause/resume countdown) !!!",
        episode_index,
        wait_s,
        effective_pause,
    )
    logger.warning(_OPERATOR_BANNER)
    logger.warning("")

    deadline = time.perf_counter() + wait_s
    last_announced = -1
    last_pause_reminder = 0.0
    while True:
        if keyboard_events is not None and keyboard_events.reset_paused:
            pause_t0 = time.perf_counter()
            now = pause_t0
            if now - last_pause_reminder >= 5.0:
                logger.warning(
                    ">>> [MANUAL SCENE RESET] countdown PAUSED — press '%s' to resume <<<",
                    effective_pause,
                )
                last_pause_reminder = now
            while keyboard_events.reset_paused:
                precise_sleep(0.2)
            # Freeze remaining time across the pause.
            deadline += time.perf_counter() - pause_t0
            last_announced = -1
            continue

        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        sec_left = int(math.ceil(remaining))
        if sec_left != last_announced:
            logger.warning(">>> [MANUAL SCENE RESET] episode %s — %ss left <<<", episode_index, sec_left)
            last_announced = sec_left
        precise_sleep(min(1.0, remaining))

    if keyboard_events is not None:
        keyboard_events.clear_reset_paused()
        keyboard_events.set_reset_mode(False)
        keyboard_events.clear_episode_flags()

    logger.warning(">>> [MANUAL SCENE RESET] episode %s — window closed, resuming <<<", episode_index)
