# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
Record a LeRobot dataset while a **humanoid is controlled by external ROS2 nodes**.

This script does **not** run teleoperation or low-level control (no ``send_action``, no GP
streaming, no trajectory processes). It only:

  - polls ``crp_humanoid`` observations (body/arm/hand joints via ti5 ROS + cameras),
  - reads robot command topics via ``crp_skeleton`` (``MotorCommand`` + ``SkillfulHandCommand``),
  - writes episodes via ``LeRobotDataset``.

For ``action_source=teleop``, ``crp_skeleton.get_action()`` is written **directly** to the
dataset — no ``teleop_action_processor`` (the external ROS stack is the processor).

ROS interfaces (ti5_interfaces):
  - Feedback: ``/multi_motor_state``, ``/joint_skillful_hand_state``
  - Commands (``send_action`` / deploy): ``/motor_command``, ``/skillfulHand_command``

Cameras (OrbbecSDK v2 via ``pyorbbecsdk2``, default 640x480 @ 30 FPS):
  - ``head``, ``left_wrist``, ``right_wrist`` — USB Orbbec (``OrbbecCameraConfig``)
  - Set a unique ``serial_number`` per camera when multiple devices are connected
  - Override with ``--robot.cameras`` or use ``OrbbecCamera.find_cameras()`` to list devices

``lerobot-record-humanoid`` never calls ``send_action``; deploy / control loops use
``CRPHumanoid.send_action()`` to publish the same command topics.

Prerequisites:
  - ROS2 stack and record/control nodes running independently.
  - ``ti5_interfaces`` available in the ROS workspace (``source install/setup.bash``).
  - ``pyorbbecsdk2`` for Orbbec cameras (``pip install pyorbbecsdk2`` in your conda env).
  - Orbbec udev rules installed; assign each camera a ``serial_number`` via ``--robot.cameras``.

Example:

```shell
lerobot-record-humanoid \\
    --robot.type=crp_humanoid \\
    --teleop.type=crp_skeleton \\
    --teleop.port=ros \\
    --dataset.repo_id=<user>/crp_humanoid_demo \\
    --dataset.num_episodes=5 \\
    --dataset.single_task="Walk forward" \\
    --dataset.fps=30 \\
    --action_source=teleop \\
    --display_data=false
```

Use ``--action_source=robot`` to log measured joint/hand state as ``action`` and omit teleop.
Disable default cameras with ``--robot.cameras='{}'``.

Episode timing (scheme B): ``--dataset.episode_time_s`` is the **effective** saved duration.
``--episode_start_delay_s`` / ``--episode_end_delay_s`` add pre/post countdowns where nothing
is written (safe with cameras and streaming encoding).
Disable ROS feedback stub warnings with empty topics, e.g.
``--robot.multi_motor_state_topic='' --robot.joint_skillful_hand_state_topic=''``.
"""

import logging
import math
import time
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import Literal

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.orbbec.configuration_orbbec import OrbbecCameraConfig  # noqa: F401
from lerobot.cameras.ros.configuration_ros import RosImageCameraConfig  # noqa: F401
from lerobot.common.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.configs import parser
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
    safe_stop_image_writer,
)
from lerobot.processor import (
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import Robot, RobotConfig, make_robot_from_config

import lerobot.robots.crp_humanoid  # noqa: F401 — registers crp_humanoid
from lerobot.robots.crp_humanoid import CRPHumanoid
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig, make_teleoperator_from_config

import lerobot.teleoperators.crp_skeleton  # noqa: F401 — registers crp_skeleton
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

ActionSource = Literal["teleop", "robot"]


def _values_for_dataset_action(
    dataset: LeRobotDataset, values: dict[str, float]
) -> dict[str, float]:
    """Map a flat ``{key: float}`` dict to ``dataset.features['action']['names']``."""
    spec = dataset.features.get("action") or {}
    names: list[str] = list(spec.get("names") or [])
    if not names:
        return dict(values)
    out: dict[str, float] = {}
    for name in names:
        if name in values:
            out[name] = float(values[name])
        elif name.endswith(".pos"):
            stem = name.removesuffix(".pos")
            out[name] = float(values.get(name, values.get(stem, 0.0)))
        else:
            out[name] = float(values.get(name, 0.0))
    return out


@dataclass
class HumanoidRecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    teleop: TeleoperatorConfig | None = None
    action_source: str = "teleop"
    # Seconds before each episode to poll hardware without saving (countdown logging).
    episode_start_delay_s: float = 2.0
    # Seconds after each episode's effective recording window without saving (countdown logging).
    episode_end_delay_s: float = 2.0
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False
    play_sounds: bool = True
    resume: bool = False

    def __post_init__(self) -> None:
        if self.action_source not in ("teleop", "robot"):
            raise ValueError(
                f"action_source must be 'teleop' or 'robot', got {self.action_source!r}."
            )
        if self.action_source == "teleop" and self.teleop is None:
            raise ValueError(
                "action_source=teleop requires --teleop.type=crp_skeleton (ROS pass-through actions). "
                "Use action_source=robot to log humanoid joint snapshots only."
            )
        if self.episode_start_delay_s < 0 or self.episode_end_delay_s < 0:
            raise ValueError("episode_start_delay_s and episode_end_delay_s must be >= 0.")


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | None = None,
    action_source: ActionSource = "teleop",
    control_time_s: int | float | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,
    *,
    save_frames: bool = True,
    countdown: bool = False,
    phase_label: str = "",
) -> bool:
    """Poll robot/teleop for ``control_time_s`` seconds. Returns True if ``exit_early`` was used."""
    if control_time_s is None or control_time_s <= 0:
        return False

    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    write_frames = dataset is not None and save_frames
    control_interval = 1 / fps
    timestamp = 0.0
    start_episode_t = time.perf_counter()
    last_countdown_s = -1

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            return True

        if countdown and phase_label:
            remaining_s = control_time_s - timestamp
            countdown_s = int(math.ceil(max(remaining_s, 0.0)))
            if countdown_s != last_countdown_s:
                logging.info("%s: %ds", phase_label, countdown_s)
                last_countdown_s = countdown_s

        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)

        if action_source == "teleop":
            if teleop is None:
                raise RuntimeError("action_source=teleop but teleop is None")
            action_values = teleop.get_action()
        elif action_source == "robot":
            if isinstance(robot, CRPHumanoid):
                action_values = robot.joint_positions_snapshot()
            else:
                action_values = {
                    k: float(v) for k, v in obs.items() if k.endswith(".pos") and isinstance(v, (int, float))
                }
        else:
            raise ValueError(f"Unknown action_source: {action_source}")

        if write_frames:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
            action_for_frame = _values_for_dataset_action(dataset, action_values)
            action_frame = build_dataset_frame(dataset.features, action_for_frame, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if display_data:
            log_rerun_data(
                observation=obs_processed,
                action=action_values,
                compress_images=display_compressed_images,
            )

        dt_s = time.perf_counter() - start_loop_t
        sleep_time_s = control_interval - dt_s
        if sleep_time_s < 0:
            logging.warning(
                "Record loop slower (%.1f Hz) than target %d Hz — frames may be irregular.",
                1.0 / dt_s if dt_s > 0 else 0.0,
                fps,
            )
        precise_sleep(max(sleep_time_s, 0.0))
        timestamp = time.perf_counter() - start_episode_t

    return False


def record_episode_with_gates(
    *,
    robot: Robot,
    events: dict,
    fps: int,
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    dataset: LeRobotDataset,
    teleop: Teleoperator | None,
    action_source: ActionSource,
    episode_time_s: float,
    episode_start_delay_s: float,
    episode_end_delay_s: float,
    single_task: str | None,
    display_data: bool,
    display_compressed_images: bool,
    episode_index: int,
) -> None:
    """Run start countdown → effective recording → end countdown for one dataset episode."""
    if episode_start_delay_s > 0:
        logging.info(
            "Episode %d: waiting %.1fs before saving (effective record length %.1fs).",
            episode_index,
            episode_start_delay_s,
            episode_time_s,
        )
        record_loop(
            robot=robot,
            events=events,
            fps=fps,
            robot_observation_processor=robot_observation_processor,
            teleop=teleop,
            action_source=action_source,
            dataset=dataset,
            control_time_s=episode_start_delay_s,
            single_task=single_task,
            display_data=display_data,
            display_compressed_images=display_compressed_images,
            save_frames=False,
            countdown=True,
            phase_label=f"Episode {episode_index}: recording starts in",
        )

    logging.info("Episode %d: saving data for %.1fs.", episode_index, episode_time_s)
    record_loop(
        robot=robot,
        events=events,
        fps=fps,
        robot_observation_processor=robot_observation_processor,
        teleop=teleop,
        action_source=action_source,
        dataset=dataset,
        control_time_s=episode_time_s,
        single_task=single_task,
        display_data=display_data,
        display_compressed_images=display_compressed_images,
        save_frames=True,
    )

    if episode_end_delay_s > 0:
        logging.info("Episode %d: saving stopped; cooldown %.1fs.", episode_index, episode_end_delay_s)
        record_loop(
            robot=robot,
            events=events,
            fps=fps,
            robot_observation_processor=robot_observation_processor,
            teleop=teleop,
            action_source=action_source,
            dataset=dataset,
            control_time_s=episode_end_delay_s,
            single_task=single_task,
            display_data=display_data,
            display_compressed_images=display_compressed_images,
            save_frames=False,
            countdown=True,
            phase_label=f"Episode {episode_index}: next phase in",
        )


@parser.wrap()
def record(cfg: HumanoidRecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="humanoid_recording", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    _, _, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )
    if cfg.action_source == "teleop" and teleop is not None:
        dataset_features = combine_feature_dicts(
            dataset_features,
            hw_to_dataset_features(teleop.action_features, ACTION, use_video=cfg.dataset.video),
        )
    elif cfg.action_source == "robot":
        dataset_features = combine_feature_dicts(
            dataset_features,
            hw_to_dataset_features(robot.action_features, ACTION, use_video=cfg.dataset.video),
        )

    dataset: LeRobotDataset | None = None
    listener = None

    try:
        if cfg.resume:
            num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                camera_encoder=cfg.dataset.camera_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras
                if num_cameras > 0
                else 0,
            )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            repo_name = cfg.dataset.repo_id.split("/", 1)[-1]
            if repo_name.startswith("eval_"):
                raise ValueError(
                    "Dataset names starting with 'eval_' are reserved. "
                    "Use lerobot-rollout for policy evaluation."
                )
            cfg.dataset.stamp_repo_id()
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera
                * len(getattr(robot, "cameras", {})),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                camera_encoder=cfg.dataset.camera_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            )

        robot.connect()
        if teleop is not None:
            teleop.connect()

        listener, events = init_keyboard_listener()

        if not cfg.dataset.streaming_encoding:
            logging.info(
                "Tip: enable faster saves with --dataset.streaming_encoding=true "
                "(see LeRobot streaming video encoding docs)."
            )

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
                record_episode_with_gates(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    robot_observation_processor=robot_observation_processor,
                    dataset=dataset,
                    teleop=teleop,
                    action_source=cfg.action_source,
                    episode_time_s=cfg.dataset.episode_time_s,
                    episode_start_delay_s=cfg.episode_start_delay_s,
                    episode_end_delay_s=cfg.episode_end_delay_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    display_compressed_images=display_compressed_images,
                    episode_index=dataset.num_episodes,
                )

                if not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", cfg.play_sounds)
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        action_source=cfg.action_source,
                        control_time_s=cfg.dataset.reset_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1
    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)

        if dataset:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop and teleop.is_connected:
            teleop.disconnect()

        if not is_headless() and listener:
            listener.stop()

        if cfg.dataset.push_to_hub:
            if dataset and dataset.num_episodes > 0:
                dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
            else:
                logging.warning("No episodes saved — skipping push to hub")

        log_say("Exiting", cfg.play_sounds)

    return dataset  # type: ignore[return-value]


def main() -> None:
    register_third_party_plugins()
    record()


if __name__ == "__main__":
    main()
