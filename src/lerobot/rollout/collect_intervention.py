# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Session-long operator keys for PLD collect (pause/resume like manual s/f reward keys)."""

from __future__ import annotations

import logging
import os
import threading

from lerobot.utils.import_utils import _pynput_available

logger = logging.getLogger(__name__)

_OPERATOR_BANNER = "!" * 72

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
        return event_key == special_keys[key_name]
    char = getattr(event_key, "char", None)
    return char == key_name


class CollectInterventionController:
    """Global pause/resume toggle during PLD collect (same listener model as s/f reward keys)."""

    def __init__(self, pause_key: str = "space") -> None:
        self._pause_key = pause_key
        self._paused = False
        self._lock = threading.Lock()
        self._listener = None
        self._engine_hold_paused = False

    @property
    def pause_key(self) -> str:
        return self._pause_key

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    @property
    def engine_hold_paused(self) -> bool:
        return self._engine_hold_paused

    @engine_hold_paused.setter
    def engine_hold_paused(self, value: bool) -> None:
        self._engine_hold_paused = value

    def clear_paused(self) -> None:
        with self._lock:
            self._paused = False

    def _toggle(self) -> None:
        with self._lock:
            self._paused = not self._paused
            paused = self._paused
        if paused:
            logger.warning("")
            logger.warning(_OPERATOR_BANNER)
            logger.warning(
                "!!! [OPERATOR PAUSE] Collect paused — press '%s' to resume (robot holds pose) !!!",
                self._pause_key,
            )
            logger.warning(_OPERATOR_BANNER)
            logger.warning("")
        else:
            logger.warning(
                ">>> [OPERATOR PAUSE] Collect resumed — press '%s' to pause again <<<",
                self._pause_key,
            )

    def start(self) -> None:
        if not self._pause_key:
            return
        if _PYNPUT_KEYBOARD is None:
            logger.warning(
                "Operator pause key '%s' disabled — pynput unavailable (need DISPLAY). "
                "Manual s/f keys use the same requirement.",
                self._pause_key,
            )
            return
        if self._listener is not None:
            return

        def on_press(key):
            try:
                if _key_matches(key, self._pause_key):
                    self._toggle()
            except Exception:
                pass

        self._listener = _PYNPUT_KEYBOARD.Listener(on_press=on_press)
        self._listener.start()
        logger.info(
            "Operator pause key enabled: '%s' toggles collect pause/resume (global, like s/f reward keys)",
            self._pause_key,
        )

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        self.clear_paused()
