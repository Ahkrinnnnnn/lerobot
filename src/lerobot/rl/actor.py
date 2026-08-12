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
"""
Actor server runner for distributed HILSerl robot policy training.

This script implements the actor component of the distributed HILSerl architecture.
It executes the policy in the robot environment, collects experience,
and sends transitions to the learner server for policy updates.

Examples of usage:

- Start an actor server for real robot training with human-in-the-loop intervention:
```bash
python -m lerobot.rl.actor --config_path src/lerobot/configs/train_config_hilserl_so100.json
```

**NOTE**: The actor server requires a running learner server to connect to. Ensure the learner
server is started before launching the actor.

**NOTE**: Human intervention is key to HILSerl training. Press the upper right trigger button on the
gamepad to take control of the robot during training. Initially intervene frequently, then gradually
reduce interventions as the policy improves.

**WORKFLOW**:
1. Determine robot workspace bounds using `lerobot-find-joint-limits`
2. Record demonstrations with `gym_manipulator.py` in record mode
3. Process the dataset and determine camera crops with `crop_dataset_roi.py`
4. Start the learner server with the training configuration
5. Start this actor server with the same configuration
6. Use human interventions to guide policy learning

For more details on the complete HILSerl training workflow, see:
https://github.com/michel-aractingi/lerobot-hilserl-guide
"""

import logging
import os
import time
from collections.abc import Generator
from functools import lru_cache
from queue import Empty
from typing import TYPE_CHECKING, Any

from lerobot.utils.import_utils import _grpc_available, require_package

if TYPE_CHECKING or _grpc_available:
    import grpc

    from lerobot.transport import services_pb2, services_pb2_grpc
    from lerobot.transport.utils import (
        bytes_to_state_dict,
        grpc_channel_options,
        python_object_to_bytes,
        receive_bytes_in_chunks,
        send_bytes_in_chunks,
        transitions_to_bytes,
    )
else:
    grpc = None
    services_pb2 = None
    services_pb2_grpc = None
    bytes_to_state_dict = None
    grpc_channel_options = None
    python_object_to_bytes = None
    receive_bytes_in_chunks = None
    send_bytes_in_chunks = None
    transitions_to_bytes = None

import torch
from torch import nn
from torch.multiprocessing import Queue

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.processor import TransitionKey
from lerobot.robots import so_follower  # noqa: F401
from lerobot.robots.crp_arm.config_crp_arm import CRPArmConfig  # noqa: F401 — register crp_arm
from lerobot.teleoperators import gamepad, so_leader  # noqa: F401
from lerobot.teleoperators.OMY_L100.config_OMY_L100 import OMYL100Config  # noqa: F401 — register OMY_L100
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import ACTION
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.random_utils import set_seed
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.transition import (
    Transition,
    move_transition_to_device,
)
from lerobot.utils.utils import (
    TimerManager,
    init_logging,
)

from .algorithms.base import RLAlgorithm
from .algorithms.factory import make_algorithm
from .gym_manipulator import (
    make_processors,
    make_robot_env,
    reset_and_build_transition,
    step_env_and_process_transition,
)
from .queue import get_last_item_from_queue
from .train_rl import TrainRLServerPipelineConfig

# Main entry point


@parser.wrap()
def actor_cli(cfg: TrainRLServerPipelineConfig):
    # Fail fast with a friendly error if the optional ``hilserl`` extra is missing.
    require_package("grpcio", extra="hilserl", import_name="grpc")
    cfg.validate()
    display_pid = False
    if not use_threads(cfg):
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")
        display_pid = True

    # Create logs directory to ensure it exists
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"actor_{cfg.job_name}.log")

    # Initialize logging with explicit log file
    init_logging(log_file=log_file, display_pid=display_pid)
    logging.info(f"Actor logging initialized, writing to {log_file}")

    is_threaded = use_threads(cfg)
    shutdown_event = ProcessSignalHandler(is_threaded, display_pid=display_pid).shutdown_event

    learner_client, grpc_channel = learner_service_client(
        host=cfg.policy.actor_learner_config.learner_host,
        port=cfg.policy.actor_learner_config.learner_port,
    )

    logging.info("[ACTOR] Establishing connection with Learner")
    if not establish_learner_connection(learner_client, shutdown_event):
        logging.error("[ACTOR] Failed to establish connection with Learner")
        grpc_channel.close()
        return

    if not use_threads(cfg):
        # If we use multithreading, we can reuse the channel
        grpc_channel.close()
        grpc_channel = None

    logging.info("[ACTOR] Connection with Learner established")

    parameters_queue = Queue()
    transitions_queue = Queue()
    interactions_queue = Queue()

    concurrency_entity = None
    if use_threads(cfg):
        from threading import Thread

        concurrency_entity = Thread
    else:
        from multiprocessing import Process

        concurrency_entity = Process

    receive_policy_process = concurrency_entity(
        target=receive_policy,
        args=(cfg, parameters_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    transitions_process = concurrency_entity(
        target=send_transitions,
        args=(cfg, transitions_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    interactions_process = concurrency_entity(
        target=send_interactions,
        args=(cfg, interactions_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    transitions_process.start()
    interactions_process.start()
    receive_policy_process.start()

    try:
        act_with_policy(
            cfg=cfg,
            shutdown_event=shutdown_event,
            parameters_queue=parameters_queue,
            transitions_queue=transitions_queue,
            interactions_queue=interactions_queue,
        )
        logging.info("[ACTOR] Policy loop finished")
    except Exception:
        logging.exception("[ACTOR] Unhandled exception in act_with_policy")
    finally:
        # Signal workers first so they stop before we close shared queues/channels.
        shutdown_event.set()
        logging.info("[ACTOR] Waiting for worker threads/processes")
        transitions_process.join(timeout=10)
        logging.info("[ACTOR] Transitions process joined")
        interactions_process.join(timeout=10)
        logging.info("[ACTOR] Interactions process joined")
        receive_policy_process.join(timeout=10)
        logging.info("[ACTOR] Receive policy process joined")

        logging.info("[ACTOR] Closing queues")
        transitions_queue.close()
        interactions_queue.close()
        parameters_queue.close()

        transitions_queue.cancel_join_thread()
        interactions_queue.cancel_join_thread()
        parameters_queue.cancel_join_thread()

        if grpc_channel is not None:
            logging.info("[ACTOR] Closing gRPC channel")
            grpc_channel.close()

        logging.info("[ACTOR] Cleanup complete")


# Core algorithm functions


def _hil_intervention_held(teleop_device: Any) -> bool:
    """True while Space (or intervene key) is held — does not consume s/f/r one-shots."""
    if teleop_device is None:
        return False
    kb = getattr(teleop_device, "_hil_keyboard", None)
    if kb is None:
        return False
    try:
        return bool(kb.intervening)
    except Exception:
        return False


def act_with_policy(
    cfg: TrainRLServerPipelineConfig,
    shutdown_event: Any,  # Event
    parameters_queue: Queue,
    transitions_queue: Queue,
    interactions_queue: Queue,
):
    """
    Executes policy interaction within the environment.

    This function rolls out the policy in the environment, collecting interaction data and pushing it to a queue for streaming to the learner.
    Once an episode is completed, updated network parameters received from the learner are retrieved from a queue and loaded into the network.

    Args:
        cfg: Configuration settings for the interaction process.
        shutdown_event: Event to check if the process should shutdown.
        parameters_queue: Queue to receive updated network parameters from the learner.
        transitions_queue: Queue to send transitions to the learner.
        interactions_queue: Queue to send interactions to the learner.
    """
    # Initialize logging for multiprocessing
    if not use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_policy_{os.getpid()}.log")
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor policy process logging initialized")

    logging.info("make_env online")

    online_env, teleop_device = make_robot_env(cfg=cfg.env)
    env_processor, action_processor = make_processors(online_env, teleop_device, cfg.env, cfg.policy.device)

    try:
        set_seed(cfg.seed)
        device = get_safe_torch_device(cfg.policy.device, log=True)

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        logging.info("make_policy")

        ### Instantiate the policy in both the actor and learner processes
        ### To avoid sending a policy object through the port, we create a policy instance
        ### on both sides, the learner sends the updated parameters every n steps to update the actor's parameters
        if cfg.dataset is not None:
            from lerobot.datasets import LeRobotDatasetMetadata

            try:
                ds_meta = LeRobotDatasetMetadata(
                    cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
                )
                if ds_meta.stats:
                    from lerobot.datasets.utils import dataset_stats_to_policy_config

                    feature_keys = list(cfg.policy.input_features) + list(cfg.policy.output_features)
                    cfg.policy.dataset_stats = dataset_stats_to_policy_config(
                        ds_meta.stats, feature_keys=feature_keys
                    )
                    logging.info("Loaded dataset_stats from %s", cfg.dataset.repo_id)

                offline_override = None
                include_rpy = True
                if cfg.env is not None and getattr(cfg.env, "processor", None) is not None:
                    crp_ee = getattr(cfg.env.processor, "crp_ee", None)
                    if crp_ee is not None:
                        offline_override = getattr(crp_ee, "offline_ee_action", None)
                        include_rpy = bool(getattr(crp_ee, "include_rpy", True))
                if offline_override != "delta":
                    from lerobot.datasets import LeRobotDataset
                    from lerobot.rl.ee_abs_to_delta import maybe_override_policy_action_stats_from_abs_ee_dataset

                    _stats_ds = LeRobotDataset(
                        cfg.dataset.repo_id,
                        root=cfg.dataset.root,
                        episodes=cfg.dataset.episodes,
                        download_videos=False,
                    )
                    cfg.policy.dataset_stats = maybe_override_policy_action_stats_from_abs_ee_dataset(
                        cfg.policy.dataset_stats,
                        features=ds_meta.features,
                        hf_dataset=_stats_ds.hf_dataset,
                        override=offline_override,
                        include_rpy=include_rpy,
                    )
                    del _stats_ds
            except Exception:
                logging.exception("Failed to load dataset_stats; keeping policy defaults")

        from lerobot.datasets.utils import ensure_vector_stats_match_features

        cfg.policy.dataset_stats = ensure_vector_stats_match_features(
            cfg.policy.dataset_stats,
            {**cfg.policy.input_features, **cfg.policy.output_features},
        )

        policy = make_policy(
            cfg=cfg.policy,
            env_cfg=cfg.env,
        )
        policy = policy.to(device).eval()
        assert isinstance(policy, nn.Module)

        # Build the algorithm
        algorithm = make_algorithm(cfg=cfg.algorithm, policy=policy)

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            dataset_stats=cfg.policy.dataset_stats,
        )

        transition = reset_and_build_transition(online_env, env_processor, action_processor)

        # NOTE: For the moment we will solely handle the case of a single environment
        sum_reward_episode = 0
        list_transition_to_send_to_learner = []
        episode_intervention = False
        # Add counters for intervention rate calculation
        episode_intervention_steps = 0
        episode_total_steps = 0

        policy_timer = TimerManager("Policy inference", log=False)

        # When Space press/release excludes frames, drop the last kept transition once so
        # the episode does not keep a (s, a, s') that straddles the mode-switch time gap.
        excluding_gap = False
        # Reused when Space is held so we never call select_action / GPU during intervention.
        last_policy_action: torch.Tensor | None = None
        action_feature = (cfg.policy.output_features or {}).get(ACTION)
        default_action_dim = int(action_feature.shape[0]) if action_feature is not None else 4

        for interaction_step in range(cfg.policy.online_steps):
            start_time = time.perf_counter()
            if shutdown_event.is_set():
                logging.info("[ACTOR] Shutting down act_with_policy")
                return

            observation = {
                k: v for k, v in transition[TransitionKey.OBSERVATION].items() if k in cfg.policy.input_features
            }

            # Peek Space hold without consuming s/f/r (those are read in AddTeleopEventsAsInfoStep).
            skip_policy = _hil_intervention_held(teleop_device)
            if skip_policy:
                # Placeholder only: intervention processors override action / skip env send.
                if last_policy_action is not None:
                    action = last_policy_action
                else:
                    action = torch.zeros(1, default_action_dim, device=device, dtype=torch.float32)
            else:
                # Time policy inference and check if it meets FPS requirement
                with policy_timer:
                    normalized_observation = preprocessor.process_observation(observation)
                    action = policy.select_action(batch=normalized_observation)
                    # Unnormalize only the continuous part.
                    if cfg.policy.num_discrete_actions is not None:
                        continuous_action = postprocessor.process_action(action[..., :-1])
                        discrete_action = action[..., -1:].to(
                            device=continuous_action.device, dtype=continuous_action.dtype
                        )
                        action = torch.cat([continuous_action, discrete_action], dim=-1)
                    else:
                        action = postprocessor.process_action(action)
                last_policy_action = action
                log_policy_frequency_issue(
                    policy_fps=policy_timer.fps_last, cfg=cfg, interaction_step=interaction_step
                )

            # Use the new step function
            new_transition = step_env_and_process_transition(
                env=online_env,
                transition=transition,
                action=action,
                env_processor=env_processor,
                action_processor=action_processor,
            )

            # Extract values from processed transition
            next_observation = {
                k: v
                for k, v in new_transition[TransitionKey.OBSERVATION].items()
                if k in cfg.policy.input_features
            }

            # Teleop action is the action that was executed in the environment
            # It is either the action from the teleop device or the action from the policy
            complementary_data = new_transition[TransitionKey.COMPLEMENTARY_DATA]
            executed_action = complementary_data["teleop_action"]

            reward = new_transition[TransitionKey.REWARD]
            done = new_transition.get(TransitionKey.DONE, False)
            truncated = new_transition.get(TransitionKey.TRUNCATED, False)

            # Mode-switch / EE-ready latch frames (Space press/release) are not RL transitions.
            exclude_from_replay = bool(complementary_data.get("exclude_from_replay", False))

            # Check for intervention from transition info
            intervention_info = new_transition[TransitionKey.INFO]
            is_intervention = bool(intervention_info.get(TeleopEvents.IS_INTERVENTION, False))

            if exclude_from_replay:
                # Entering a switch gap: drop last transition so buffer segments stay continuous
                # (its next_state would not match the first state after the gap).
                if not excluding_gap and list_transition_to_send_to_learner:
                    list_transition_to_send_to_learner.pop()
                    logging.info(
                        "[ACTOR] Dropped last transition before intervention switch gap "
                        "(interaction_step=%s); gap frames not written to buffer",
                        interaction_step,
                    )
                excluding_gap = True
                logging.debug(
                    "[ACTOR] Skipping transition (exclude_from_replay) at interaction_step=%s",
                    interaction_step,
                )
            else:
                excluding_gap = False
                sum_reward_episode += float(reward)
                episode_total_steps += 1
                if is_intervention:
                    episode_intervention = True
                    episode_intervention_steps += 1

                complementary_info = {
                    "discrete_penalty": torch.tensor(
                        [complementary_data.get("discrete_penalty", 0.0)]
                    ),
                    TeleopEvents.IS_INTERVENTION.value: is_intervention,
                }
                list_transition_to_send_to_learner.append(
                    Transition(
                        state=observation,
                        action=executed_action,
                        reward=reward,
                        next_state=next_observation,
                        done=done,
                        truncated=truncated,
                        complementary_info=complementary_info,
                    )
                )

            # Update transition for next iteration
            transition = new_transition

            if done or truncated:
                logging.info(f"[ACTOR] Global step {interaction_step}: Episode reward: {sum_reward_episode}")

                # Stop GP stream immediately (before slow weight load / queue push / scene wait).
                try:
                    action_processor.reset()
                except Exception:
                    logging.exception("[ACTOR] action_processor.reset at episode end failed")

                update_policy_parameters(algorithm=algorithm, parameters_queue=parameters_queue, device=device)

                min_steps = 0
                if cfg.env is not None and getattr(cfg.env, "processor", None) is not None:
                    reset_cfg = getattr(cfg.env.processor, "reset", None)
                    if reset_cfg is not None:
                        min_steps = int(getattr(reset_cfg, "min_episode_steps_for_replay", 0) or 0)

                n_kept = len(list_transition_to_send_to_learner)
                rerecord = bool(intervention_info.get(TeleopEvents.RERECORD_EPISODE, False))
                if rerecord:
                    logging.warning(
                        "[ACTOR] Rerecord requested — discarding %s transitions "
                        "(not sent to learner; interaction_step=%s)",
                        n_kept,
                        interaction_step,
                    )
                    list_transition_to_send_to_learner = []
                elif n_kept > 0 and min_steps > 0 and n_kept < min_steps:
                    logging.warning(
                        "[ACTOR] Dropping short episode (%s transitions < min_episode_steps_for_replay=%s); "
                        "not sent to learner (interaction_step=%s, reward=%.3f)",
                        n_kept,
                        min_steps,
                        interaction_step,
                        sum_reward_episode,
                    )
                    list_transition_to_send_to_learner = []
                elif n_kept > 0:
                    push_transitions_to_transport_queue(
                        transitions=list_transition_to_send_to_learner,
                        transitions_queue=transitions_queue,
                    )
                    list_transition_to_send_to_learner = []

                stats = get_frequency_stats(policy_timer)
                policy_timer.reset()

                # Calculate intervention rate
                intervention_rate = 0.0
                if episode_total_steps > 0:
                    intervention_rate = episode_intervention_steps / episode_total_steps

                # Send episodic reward to the learner
                interactions_queue.put(
                    python_object_to_bytes(
                        {
                            "Episodic reward": sum_reward_episode,
                            "Interaction step": interaction_step,
                            "Episode intervention": int(episode_intervention),
                            "Intervention rate": intervention_rate,
                            **stats,
                        }
                    )
                )

                # Reset intervention counters and environment
                sum_reward_episode = 0.0
                episode_intervention = False
                episode_intervention_steps = 0
                episode_total_steps = 0
                excluding_gap = False

                transition = reset_and_build_transition(online_env, env_processor, action_processor)

            if cfg.env.fps is not None:
                dt_time = time.perf_counter() - start_time
                precise_sleep(max(1 / cfg.env.fps - dt_time, 0.0))

    finally:
        # Stop HIL GP stream before tearing down robot (avoids set_GOT/send_GPs vs disconnect races).
        logging.info("[ACTOR] Disconnecting teleop / closing env")
        try:
            action_processor.reset()
        except Exception:
            logging.exception("[ACTOR] action_processor.reset failed")
        try:
            if teleop_device is not None:
                teleop_device.disconnect()
        except Exception:
            logging.exception("[ACTOR] teleop disconnect failed")
        try:
            online_env.close()
        except Exception:
            logging.exception("[ACTOR] env close failed")


#  Communication Functions - Group all gRPC/messaging functions


def establish_learner_connection(
    stub: "services_pb2_grpc.LearnerServiceStub",
    shutdown_event: Any,  # Event
    attempts: int = 30,
) -> bool:
    """Establish a connection with the learner.

    Args:
        stub (services_pb2_grpc.LearnerServiceStub): The stub to use for the connection.
        shutdown_event (Event): The event to check if the connection should be established.
        attempts (int): The number of attempts to establish the connection.
    Returns:
        bool: True if the connection is established, False otherwise.
    """
    for _ in range(attempts):
        if shutdown_event.is_set():
            logging.info("[ACTOR] Shutting down establish_learner_connection")
            return False

        # Force a connection attempt and check state.
        # Use a short timeout so Ctrl+C / shutdown_event can interrupt between retries
        # (a hanging Ready with no deadline previously required a second Ctrl+C force exit).
        try:
            logging.info("[ACTOR] Send ready message to Learner")
            if stub.Ready(services_pb2.Empty(), timeout=2.0) == services_pb2.Empty():
                return True
        except grpc.RpcError as e:
            logging.error(f"[ACTOR] Waiting for Learner to be ready... {e}")
            time.sleep(2)
    return False


@lru_cache(maxsize=1)
def learner_service_client(
    host: str = "127.0.0.1",
    port: int = 50051,
) -> "tuple[services_pb2_grpc.LearnerServiceStub, grpc.Channel]":
    """Return a client for the learner service.

    GRPC uses HTTP/2, which is a binary protocol and multiplexes requests over a single connection.
    So we need to create only one client and reuse it.

    Returns:
        tuple[services_pb2_grpc.LearnerServiceStub, grpc.Channel]: The stub and the channel.
    """

    channel = grpc.insecure_channel(
        f"{host}:{port}",
        grpc_channel_options(),
    )
    stub = services_pb2_grpc.LearnerServiceStub(channel)
    logging.info("[ACTOR] Learner service client created")
    return stub, channel


def receive_policy(
    cfg: TrainRLServerPipelineConfig,
    parameters_queue: Queue,
    shutdown_event: Any,  # Event
    learner_client: "services_pb2_grpc.LearnerServiceStub | None" = None,
    grpc_channel: "grpc.Channel | None" = None,
) -> None:
    """Receive parameters from the learner.

    Args:
        cfg (TrainRLServerPipelineConfig): The configuration for the actor.
        parameters_queue (Queue): The queue to receive the parameters.
        shutdown_event (Event): The event to check if the process should shutdown.
        learner_client (services_pb2_grpc.LearnerServiceStub | None): Optional pre-created stub.
        grpc_channel (grpc.Channel | None): Optional pre-created channel.
    """
    logging.info("[ACTOR] Start receiving parameters from the Learner")
    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_receive_policy_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor receive policy process logging initialized")

        # Setup process handlers to handle shutdown signal
        # But use shutdown event from the main process
        _ = ProcessSignalHandler(use_threads=False, display_pid=True)

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        iterator = learner_client.StreamParameters(services_pb2.Empty())
        receive_bytes_in_chunks(
            iterator,
            parameters_queue,
            shutdown_event,
            log_prefix="[ACTOR] parameters",
        )

    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Received policy loop stopped")


def send_transitions(
    cfg: TrainRLServerPipelineConfig,
    transitions_queue: Queue,
    shutdown_event: Any,  # Event
    learner_client: "services_pb2_grpc.LearnerServiceStub | None" = None,
    grpc_channel: "grpc.Channel | None" = None,
) -> None:
    """Send transitions to the learner.

    This function continuously retrieves messages from the queue and processes:

    - Transition Data:
        - A batch of transitions (observation, action, reward, next observation) is collected.
        - Transitions are moved to the CPU and serialized using PyTorch.
        - The serialized data is wrapped in a `services_pb2.Transition` message and sent to the learner.

    Args:
        cfg (TrainRLServerPipelineConfig): The configuration for the actor.
        transitions_queue (Queue): The queue to receive the transitions.
        shutdown_event (Event): The event to check if the process should shutdown.
        learner_client (services_pb2_grpc.LearnerServiceStub | None): Optional pre-created stub.
        grpc_channel (grpc.Channel | None): Optional pre-created channel.
    """

    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_transitions_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor transitions process logging initialized")

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        learner_client.SendTransitions(
            transitions_stream(
                shutdown_event, transitions_queue, cfg.policy.actor_learner_config.queue_get_timeout
            )
        )
    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    logging.info("[ACTOR] Finished streaming transitions")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Transitions process stopped")


def send_interactions(
    cfg: TrainRLServerPipelineConfig,
    interactions_queue: Queue,
    shutdown_event: Any,  # Event
    learner_client: "services_pb2_grpc.LearnerServiceStub | None" = None,
    grpc_channel: "grpc.Channel | None" = None,
) -> None:
    """Send interactions to the learner.

    This function continuously retrieves messages from the queue and processes:

    - Interaction Messages:
        - Contains useful statistics about episodic rewards and policy timings.
        - The message is serialized using `pickle` and sent to the learner.

    Args:
        cfg (TrainRLServerPipelineConfig): The configuration for the actor.
        interactions_queue (Queue): The queue to receive the interactions.
        shutdown_event (Event): The event to check if the process should shutdown.
        learner_client (services_pb2_grpc.LearnerServiceStub | None): Optional pre-created stub.
        grpc_channel (grpc.Channel | None): Optional pre-created channel.
    """

    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_interactions_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor interactions process logging initialized")

        # Setup process handlers to handle shutdown signal
        # But use shutdown event from the main process
        _ = ProcessSignalHandler(use_threads=False, display_pid=True)

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        learner_client.SendInteractions(
            interactions_stream(
                shutdown_event, interactions_queue, cfg.policy.actor_learner_config.queue_get_timeout
            )
        )
    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    logging.info("[ACTOR] Finished streaming interactions")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Interactions process stopped")


def transitions_stream(
    shutdown_event: Any,  # Event
    transitions_queue: Queue,
    timeout: float,
) -> "Generator[Any, None, services_pb2.Empty]":
    while not shutdown_event.is_set():
        try:
            message = transitions_queue.get(block=True, timeout=timeout)
        except Empty:
            logging.debug("[ACTOR] Transition queue is empty")
            continue

        yield from send_bytes_in_chunks(
            message, services_pb2.Transition, log_prefix="[ACTOR] Send transitions"
        )

    return services_pb2.Empty()


def interactions_stream(
    shutdown_event: Any,  # Event
    interactions_queue: Queue,
    timeout: float,
) -> "Generator[Any, None, services_pb2.Empty]":
    while not shutdown_event.is_set():
        try:
            message = interactions_queue.get(block=True, timeout=timeout)
        except Empty:
            logging.debug("[ACTOR] Interaction queue is empty")
            continue

        yield from send_bytes_in_chunks(
            message,
            services_pb2.InteractionMessage,
            log_prefix="[ACTOR] Send interactions",
        )

    return services_pb2.Empty()


#  Policy functions


def update_policy_parameters(algorithm: RLAlgorithm, parameters_queue: Queue, device):
    """Drain the latest learner-pushed weights into ``algorithm.policy``."""
    bytes_state_dict = get_last_item_from_queue(parameters_queue, block=False)
    if bytes_state_dict is not None:
        logging.info("[ACTOR] Load new parameters from Learner.")
        state_dicts = bytes_to_state_dict(bytes_state_dict)

        # TODO: check encoder parameter synchronization possible issues:
        # 1. When shared_encoder=True, we're loading stale encoder params from actor's state_dict
        #    instead of the updated encoder params from critic (which is optimized separately)
        # 2. When freeze_vision_encoder=True, we waste bandwidth sending/loading frozen params
        # 3. Need to handle encoder params correctly for both actor and discrete_critic
        # Potential fixes:
        # - Send critic's encoder state when shared_encoder=True
        # - Skip encoder params entirely when freeze_vision_encoder=True
        # - Ensure discrete_critic gets correct encoder state (currently uses encoder_critic)
        algorithm.load_weights(state_dicts, device=device)


#  Utilities functions


def push_transitions_to_transport_queue(transitions: list, transitions_queue):
    """Send transitions to learner in smaller chunks to avoid network issues.

    Args:
        transitions: List of transitions to send
        message_queue: Queue to send messages to learner
        chunk_size: Size of each chunk to send
    """
    transition_to_send_to_learner = []
    for transition in transitions:
        tr = move_transition_to_device(transition=transition, device="cpu")
        for key, value in tr["state"].items():
            if torch.isnan(value).any():
                logging.warning(f"Found NaN values in transition {key}")

        transition_to_send_to_learner.append(tr)

    transitions_queue.put(transitions_to_bytes(transition_to_send_to_learner))


def get_frequency_stats(timer: TimerManager) -> dict[str, float]:
    """Get the frequency statistics of the policy.

    Args:
        timer (TimerManager): The timer with collected metrics.

    Returns:
        dict[str, float]: The frequency statistics of the policy.
    """
    stats = {}
    if timer.count > 1:
        avg_fps = timer.fps_avg
        p90_fps = timer.fps_percentile(90)
        logging.debug(f"[ACTOR] Average policy frame rate: {avg_fps}")
        logging.debug(f"[ACTOR] Policy frame rate 90th percentile: {p90_fps}")
        stats = {
            "Policy frequency [Hz]": avg_fps,
            "Policy frequency 90th-p [Hz]": p90_fps,
        }
    return stats


def log_policy_frequency_issue(policy_fps: float, cfg: TrainRLServerPipelineConfig, interaction_step: int):
    if policy_fps < cfg.env.fps:
        logging.warning(
            f"[ACTOR] Policy FPS {policy_fps:.1f} below required {cfg.env.fps} at step {interaction_step}"
        )


def use_threads(cfg: TrainRLServerPipelineConfig) -> bool:
    return cfg.policy.concurrency.actor == "threads"


if __name__ == "__main__":
    actor_cli()
