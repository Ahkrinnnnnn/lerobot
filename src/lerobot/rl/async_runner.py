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

"""Reusable async actor-learner runner for single-machine real-robot RL.

Implements the "rollouts and learning asynchronously" pattern used by RLT
(paper §IV-B), PLD (via SERL, paper Appendix B.2) and HIL-SERL. A learner
thread runs continuous gradient updates on the GPU while a rollout thread
collects episodes on the robot in parallel. Actor weights flow learner→rollout
via a thread-safe :class:`WeightSlot`; both threads share a thread-safe
:class:`~lerobot.rl.buffer.ReplayBuffer`.

Design choice — threads, not processes:
    PyTorch CUDA kernels and robot I/O release the GIL, so on a single machine
    with one GPU a second process would not increase throughput (kernels on the
    same GPU are serialized anyway) and would force the actor process to load
    its own copy of the frozen base policy (extra VRAM). A process-based runner
    is the right choice for *cross-machine* deployment (actor on a robot PC,
    learner on a GPU server, as in openpi-RLT's Machine A/B) or for process-level
    fault isolation; this thread-based runner targets the common single-machine
    case and stays simple by sharing memory.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import torch

logger = logging.getLogger(__name__)


def _clone_tensor_tree(obj):
    """Deep-clone every torch.Tensor leaf in a nested dict/list structure."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()
    if isinstance(obj, dict):
        return {k: _clone_tensor_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        cls = type(obj)
        return cls(_clone_tensor_tree(v) for v in obj)
    return obj


class WeightSlot:
    """Thread-safe latest-actor-weight publisher (learner → rollout).

    The learner publishes a CPU clone of the actor state_dict after each actor
    update; the rollout thread pulls a clone at each decision boundary. This
    decouples the learner's in-place optimizer updates from the rollout's read,
    so the rollout never observes a half-updated tensor. The published tree is
    cloned on both ``publish`` and ``get`` so neither side can mutate the other's
    copy.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._weights: dict | None = None
        self.version: int = 0

    def publish(self, weights: dict) -> None:
        with self._lock:
            self._weights = _clone_tensor_tree(weights)
            self.version += 1

    def get(self) -> dict | None:
        with self._lock:
            if self._weights is None:
                return None
            return _clone_tensor_tree(self._weights)


class AsyncActorLearnerRunner:
    """Orchestrate a learner thread and a rollout thread on a single machine.

    The runner is algorithm- and strategy-agnostic: callers supply three
    callbacks and the runner handles thread lifecycle, stop coordination,
    periodic logging, and periodic + final checkpointing.

    Args:
        shutdown_event: external :class:`threading.Event` (e.g. from
            :class:`ProcessSignalHandler`) signalling the whole run to stop.
        weight_slot: optional shared :class:`WeightSlot`; created if omitted.
        name: prefix for thread names and log lines.
    """

    def __init__(
        self,
        shutdown_event: threading.Event,
        weight_slot: WeightSlot | None = None,
        name: str = "async-rl",
    ) -> None:
        self.shutdown_event = shutdown_event
        self.weight_slot = weight_slot or WeightSlot()
        self.name = name
        self._stop_all = threading.Event()

    def run(
        self,
        *,
        learner_fn: Callable[[int], object | None],
        rollout_fn: Callable[[], None],
        stop_rollout_fn: Callable[[], None],
        target_steps: int,
        checkpoint_every_steps: int = 500,
        log_every: int = 50,
        checkpoint_fn: Callable[[int], None] | None = None,
        log: logging.Logger | None = None,
    ) -> int:
        """Run learner and rollout concurrently until ``target_steps`` or shutdown.

        Args:
            learner_fn: ``learner_fn(train_steps) -> stats | None``. Performs one
                gradient step and returns its :class:`TrainingStats` (or any
                object with ``losses``/``extra`` attributes). Return ``None`` to
                signal a skipped step (e.g. buffer too small) — the runner will
                *not* increment the step counter. The callback is responsible for
                publishing actor weights to ``self.weight_slot``.
            rollout_fn: blocking rollout loop; must exit when the rollout's own
                stop flag is set (see ``stop_rollout_fn``) or ``shutdown_event``.
            stop_rollout_fn: signal the rollout to exit gracefully (e.g. set
                ``strategy._collect_stop = True``).
            target_steps: stop the learner once this many *real* gradient steps
                have been performed.
            checkpoint_every_steps: call ``checkpoint_fn`` every this many steps.
            log_every: emit a progress log line every this many steps.
            checkpoint_fn: ``checkpoint_fn(train_steps) -> None``; also called
                once at the very end with the final step count.

        Returns:
            The final number of gradient steps performed.
        """
        log = log or logger
        stop_all = self._stop_all
        train_steps = 0
        ckpt_counter = 0

        def learner_loop() -> None:
            nonlocal train_steps, ckpt_counter
            last_stats = None
            log.info("[%s] learner thread started — target=%d grad steps", self.name, target_steps)
            while (
                train_steps < target_steps
                and not self.shutdown_event.is_set()
                and not stop_all.is_set()
            ):
                try:
                    last_stats = learner_fn(train_steps)
                except Exception:
                    log.exception("[%s] learner thread failed; stopping", self.name)
                    break
                if last_stats is None:
                    # Skipped step (e.g. buffer below learning threshold).
                    continue
                train_steps += 1
                ckpt_counter += 1
                if train_steps % log_every == 0:
                    losses = getattr(last_stats, "losses", None)
                    extra = getattr(last_stats, "extra", None)
                    log.info(
                        "[%s] async learner — grad_steps=%d/%d losses=%s extra=%s",
                        self.name,
                        train_steps,
                        target_steps,
                        losses,
                        extra,
                    )
                if ckpt_counter >= checkpoint_every_steps:
                    ckpt_counter = 0
                    if checkpoint_fn is not None:
                        checkpoint_fn(train_steps)
            stop_all.set()
            stop_rollout_fn()
            log.info("[%s] learner thread done — grad_steps=%d", self.name, train_steps)

        def rollout_loop() -> None:
            log.info("[%s] rollout thread started", self.name)
            try:
                rollout_fn()
            except Exception:
                log.exception("[%s] rollout thread failed; stopping", self.name)
                stop_all.set()
            log.info("[%s] rollout thread done", self.name)

        learner_t = threading.Thread(target=learner_loop, name=f"{self.name}-learner", daemon=True)
        rollout_t = threading.Thread(target=rollout_loop, name=f"{self.name}-rollout", daemon=True)
        learner_t.start()
        rollout_t.start()

        # Wait until either thread exits (target reached / rollout died) or external shutdown.
        while (
            learner_t.is_alive()
            and rollout_t.is_alive()
            and not self.shutdown_event.is_set()
        ):
            learner_t.join(timeout=0.5)
            rollout_t.join(timeout=0.5)

        # One thread finished (or external shutdown) — tear down the other.
        stop_all.set()
        stop_rollout_fn()
        learner_t.join(timeout=15.0)
        rollout_t.join(timeout=15.0)

        if checkpoint_fn is not None:
            checkpoint_fn(train_steps)
        log.info(
            "[%s] async run finished — total grad_steps=%d", self.name, train_steps
        )
        return train_steps
