# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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

import logging
import threading
import time
from multiprocessing import Event, Queue
from typing import TYPE_CHECKING

from lerobot.utils.import_utils import _grpc_available

from .queue import get_last_item_from_queue

if TYPE_CHECKING or _grpc_available:
    import grpc

    from lerobot.transport import services_pb2, services_pb2_grpc
    from lerobot.transport.utils import receive_bytes_in_chunks, send_bytes_in_chunks

    _ServicerBase = services_pb2_grpc.LearnerServiceServicer
else:
    grpc = None
    services_pb2 = None
    services_pb2_grpc = None
    receive_bytes_in_chunks = None
    send_bytes_in_chunks = None
    _ServicerBase = object

# Ready + StreamParameters + SendTransitions + SendInteractions, with headroom for
# stale actor reconnects (dead StreamParameters used to exhaust a pool of 3 and block Ready).
MAX_WORKERS = 12
SHUTDOWN_TIMEOUT = 10


class LearnerService(_ServicerBase):
    """
    Implementation of the LearnerService gRPC service
    This service is used to send parameters to the Actor and receive transitions and interactions from the Actor
    check transport.proto for the gRPC service definition
    """

    def __init__(
        self,
        shutdown_event: Event,  # type: ignore
        parameters_queue: Queue,
        seconds_between_pushes: float,
        transition_queue: Queue,
        interaction_message_queue: Queue,
        queue_get_timeout: float = 0.001,
    ):
        self.shutdown_event = shutdown_event
        self.parameters_queue = parameters_queue
        self.seconds_between_pushes = seconds_between_pushes
        self.transition_queue = transition_queue
        self.interaction_message_queue = interaction_message_queue
        self.queue_get_timeout = queue_get_timeout
        # Serialize parameter-queue drains across StreamParameters workers.
        self._parameters_lock = threading.Lock()
        # Only the newest StreamParameters stream should keep pushing; older ones exit.
        self._stream_epoch = 0
        self._stream_epoch_lock = threading.Lock()

    def StreamParameters(  # noqa: N802
        self, request: "services_pb2.Empty", context: "grpc.ServicerContext"
    ):
        # TODO: authorize the request
        with self._stream_epoch_lock:
            self._stream_epoch += 1
            my_epoch = self._stream_epoch
        logging.info(
            "[LEARNER] Received request to stream parameters from the Actor (epoch=%s)", my_epoch
        )

        last_push_time = 0.0

        while not self.shutdown_event.is_set():
            # Drop streams from crashed / superseded actors so workers are not pinned forever.
            if not context.is_active():
                logging.info("[LEARNER] Actor disconnected, ending parameter stream (epoch=%s)", my_epoch)
                break
            with self._stream_epoch_lock:
                if my_epoch != self._stream_epoch:
                    logging.info(
                        "[LEARNER] Parameter stream superseded by newer actor (epoch=%s), ending",
                        my_epoch,
                    )
                    break

            time_since_last_push = time.time() - last_push_time
            if time_since_last_push < self.seconds_between_pushes:
                self.shutdown_event.wait(self.seconds_between_pushes - time_since_last_push)
                continue

            logging.info("[LEARNER] Push parameters to the Actor")
            try:
                with self._parameters_lock:
                    buffer = get_last_item_from_queue(
                        self.parameters_queue, block=True, timeout=self.queue_get_timeout
                    )
            except OSError:
                logging.info("[LEARNER] Parameters queue closed, ending parameter stream")
                break

            if buffer is None:
                continue

            try:
                yield from send_bytes_in_chunks(
                    buffer,
                    services_pb2.Parameters,
                    log_prefix="[LEARNER] Sending parameters",
                    silent=True,
                )
            except Exception:
                logging.info("[LEARNER] Failed to push parameters (actor likely gone), ending stream")
                break

            last_push_time = time.time()
            logging.info("[LEARNER] Parameters sent")

        logging.info("[LEARNER] Stream parameters finished (epoch=%s)", my_epoch)
        return services_pb2.Empty()

    def SendTransitions(self, request_iterator, context: "grpc.ServicerContext"):  # noqa: N802
        # TODO: authorize the request
        logging.info("[LEARNER] Received request to receive transitions from the Actor")

        receive_bytes_in_chunks(
            request_iterator,
            self.transition_queue,
            self.shutdown_event,
            log_prefix="[LEARNER] transitions",
        )

        logging.debug("[LEARNER] Finished receiving transitions")
        return services_pb2.Empty()

    def SendInteractions(self, request_iterator, context: "grpc.ServicerContext"):  # noqa: N802
        # TODO: authorize the request
        logging.info("[LEARNER] Received request to receive interactions from the Actor")

        receive_bytes_in_chunks(
            request_iterator,
            self.interaction_message_queue,
            self.shutdown_event,
            log_prefix="[LEARNER] interactions",
        )

        logging.debug("[LEARNER] Finished receiving interactions")
        return services_pb2.Empty()

    def Ready(self, request: "services_pb2.Empty", context: "grpc.ServicerContext"):  # noqa: N802
        return services_pb2.Empty()
