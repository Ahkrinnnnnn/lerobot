# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

import logging
import os

from lerobot.utils.import_utils import _pynput_available

from .pipeline_config import RewardClassifierRuntimeConfig
from .runtime import RewardClassifierRuntime

logger = logging.getLogger(__name__)

PYNPUT_AVAILABLE = _pynput_available
keyboard = None
if PYNPUT_AVAILABLE:
    try:
        if os.environ.get("DISPLAY"):
            from pynput import keyboard as pynput_keyboard

            keyboard = pynput_keyboard
    except ImportError:
        pass


class RewardClassifierDetector:
    """Binary vision reward / episode termination with optional manual override keys."""

    def __init__(self, config: RewardClassifierRuntimeConfig):
        self.config = config
        self._runtime: RewardClassifierRuntime | None = None
        self._manual_success = False
        self._manual_failure = False
        self._listener = None
        if config.path:
            self._runtime = RewardClassifierRuntime.from_pretrained(
                config.path,
                device=config.device,
                success_threshold=config.success_threshold,
                success_reward=config.success_reward,
                terminate_on_success=config.terminate_on_success,
            )

        if config.manual_fallback and keyboard is not None:
            self._setup_keyboard()

    def _setup_keyboard(self) -> None:
        success_key = self.config.manual_success_key
        failure_key = self.config.manual_failure_key

        def on_press(key):
            try:
                char = getattr(key, "char", None)
                if char == success_key:
                    self._manual_success = True
                elif char == failure_key:
                    self._manual_failure = True
            except Exception:
                pass

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.start()
        logger.info(
            "Manual reward keys enabled: '%s'=success, '%s'=failure",
            success_key,
            failure_key,
        )

    def reset_manual(self) -> None:
        self._manual_success = False
        self._manual_failure = False

    def consume_manual_success(self) -> bool:
        val = self._manual_success
        self._manual_success = False
        return val

    def consume_manual_failure(self) -> bool:
        val = self._manual_failure
        self._manual_failure = False
        return val

    def predict(self, images: dict) -> tuple[float, bool]:
        """Return (reward, done) from the vision classifier only.

        Manual overrides are handled via :meth:`consume_manual_success` /
        :meth:`consume_manual_failure` in the rollout callback.
        """
        if self._runtime is None:
            return 0.0, False

        return self._runtime.predict_reward_and_done(images)

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
