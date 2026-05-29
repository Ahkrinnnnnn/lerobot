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
Deploy a policy on the real robot: control loop at training dataset FPS, no dataset recording.

Policy I/O shapes and normalization stats come from the **training dataset** referenced in
``train_config.json`` next to the checkpoint (same directory as ``config.json``, or under
``pretrained_model/``). Two processes: main (observations, fixed-rate ``send_action``) and one
**inference** subprocess. Cameras are read **only** when a new frame can be queued for inference (queue size 1); other ticks
read proprio only for ``send_action``. The inference subprocess uses ``spawn`` (CUDA-safe vs fork)
and **never receives the robot handle**: the main process captures camera observations, builds
``observation_frame``, and enqueues it—only picklable payloads cross the process boundary (no
``CrpRobotPy`` in the child).

All ``robot.get_observation`` / ``robot.send_action`` calls share a **reentrant lock** so the main
process and the inference process never hit the robot SDK (e.g. CrpRobotPy + cameras) concurrently;
otherwise one side can block indefinitely while the other holds the device.

Example:

```shell
lerobot-deploy \
    --robot.type=crp_arm \
    --policy.path=/path/to/checkpoint/pretrained_model \
    --single_task="Pick the cube"
```

"""

import logging
import multiprocessing
import queue
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import torch

from lerobot.cameras import (  # noqa: F401
    CameraConfig,
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TRAIN_CONFIG_NAME, TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.processor.rename_processor import rename_stats
from lerobot.tools.lib_loader import load_CrpRobotPy
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    reachy2,
    so_follower,
    unitree_g1,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.common.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


@dataclass
class DeployConfig:
    robot: RobotConfig
    policy: PreTrainedConfig | None = None
    # Language instruction for policies that use a task string (e.g. SmolVLA).
    single_task: str | None = None
    # Passed to the policy preprocessor ``rename_observations_processor``.
    rename_map: dict[str, str] = field(default_factory=dict)
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False
    play_sounds: bool = True

    def __post_init__(self):
        # HACK: policy.* is stripped from argv before draccus.parse; reload from --policy.path like RecordConfig.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.policy is None:
            raise ValueError("Deploy requires --policy.path=... to load the pretrained policy.")


    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


""" --------------- deploy_loop (deploy-only, two processes) ---------------------------
  Main: fixed-rate tick -> (optional full camera read if inference queue not full)
        -> build_dataset_frame -> obs_queue (drop if full) -> policy action / hold
        -> robot_action_processor -> robot.send_action
  Side: inference process consumes obs_queue, runs predict_action, writes latest_action_ref
"""


def _deploy_format_action_dict(d: dict[str, Any] | None) -> str:
    """Compact string of action-like dicts for logs (sorted keys, numeric rounded)."""
    if not d:
        return "{}"
    parts: list[str] = []
    for k in sorted(d.keys()):
        v = d[k]
        try:
            parts.append(f"{k}={float(v):.5g}")
        except (TypeError, ValueError):
            parts.append(f"{k}={v!r}")
    return "{" + ", ".join(parts) + "}"


def _deploy_summarize_array_or_tensor(x: Any, max_head: int = 12) -> str:
    """One-line numeric summary for numpy arrays or torch tensors."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if not isinstance(x, np.ndarray):
        return repr(x)[:200]
    if x.size == 0:
        return f"empty(shape={x.shape}, dtype={x.dtype})"
    xf = np.asarray(x, dtype=np.float64).ravel()
    head = xf[:max_head].tolist()
    return (
        f"shape={x.shape} dtype={x.dtype} min={float(xf.min()):.5g} max={float(xf.max()):.5g} "
        f"mean={float(xf.mean()):.5g} head={head}"
    )


def _deploy_summarize_observation_frame_one_line(observation_frame: dict[str, Any], max_len: int = 2500) -> str:
    """Single log line: keys + per-key shape/stats (truncated)."""
    segs: list[str] = []
    for k in sorted(observation_frame.keys()):
        segs.append(f"{k}:{_deploy_summarize_array_or_tensor(observation_frame[k])}")
    out = " | ".join(segs)
    return out if len(out) <= max_len else out[: max_len - 3] + "..."


def _current_pose_as_action(
    obs_processed: RobotObservation,
    action_features: dict,
) -> RobotAction:
    """Build a hold-action from current joint positions (same keys as robot action)."""
    return {
        k: obs_processed[k]
        for k in action_features
        if k in obs_processed
    }


def _deploy_multiprocessing_context() -> "multiprocessing.context.BaseContext":
    """Return ``spawn`` so the inference child does not fork after CUDA init in the parent (PyTorch)."""
    return multiprocessing.get_context("spawn")


def _inference_worker(
    obs_queue: Any,
    latest_action_ref: dict,
    latest_action_lock: Any,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    dataset_features: dict,
    single_task: str | None,
    robot_type: str,
    use_amp: bool,
    device,
) -> None:
    """Runs in the inference subprocess: consume pre-built observation frames from main, run policy."""
    ok_count = 0
    while True:
        observation_frame = obs_queue.get()
        if observation_frame is None:
            break
        try:
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=use_amp,
                task=single_task,
                robot_type=robot_type,
            )
            action = make_robot_action(action_values, dataset_features)
            with latest_action_lock:
                latest_action_ref["action"] = action
            ok_count += 1
            # Always-on diagnostic (warning level): policy tensor + dict + obs summary (throttled).
            if ok_count <= 8 or ok_count % 30 == 0:
                action_names = (dataset_features.get(ACTION) or {}).get("names")
                logging.warning(
                    "Deploy[inference_ok n=%d] task=%r robot_type=%r dataset_action.names=%s "
                    "raw_action_tensor=%s policy_action_dict=%s obs_frame=%s",
                    ok_count,
                    single_task,
                    robot_type,
                    action_names,
                    _deploy_summarize_array_or_tensor(action_values),
                    _deploy_format_action_dict(action),
                    _deploy_summarize_observation_frame_one_line(observation_frame),
                )
        except Exception as e:
            logging.warning(
                "Deploy[inference_err] exc=%r task=%r robot_type=%r dataset_action.names=%s obs_frame=%s",
                e,
                single_task,
                robot_type,
                (dataset_features.get(ACTION) or {}).get("names"),
                _deploy_summarize_observation_frame_one_line(observation_frame)
                if isinstance(observation_frame, dict)
                else observation_frame,
            )


def _deploy_get_robot_observation(robot: Robot, *, include_images: bool = True) -> RobotObservation:
    """Call ``robot.get_observation``; pass ``include_images=False`` when supported (e.g. CRP arm)."""
    fn = robot.get_observation
    if not include_images:
        try:
            return fn(include_images=False)  # type: ignore[misc]
        except TypeError:
            return fn()
    return fn()


def _deploy_policy_dir_for_train_config(policy_pretrained: Path | str) -> Path:
    """Directory that contains ``train_config.json`` (checkpoint ``pretrained_model/`` or model root)."""
    p = Path(policy_pretrained).resolve()
    if (p / TRAIN_CONFIG_NAME).is_file():
        return p
    sub = p / "pretrained_model"
    if (sub / TRAIN_CONFIG_NAME).is_file():
        return sub
    raise FileNotFoundError(
        f"Could not find {TRAIN_CONFIG_NAME} under {p} or {sub}. "
        "Point --policy.path at a checkpoint directory that contains the saved training config "
        "(e.g. .../pretrained_model with train_config.json from lerobot_train)."
    )


def _deploy_training_dataset_meta(policy_pretrained: Path | str) -> LeRobotDatasetMetadata:
    """Load the training dataset metadata referenced by the policy checkpoint's ``train_config.json``."""
    policy_dir = _deploy_policy_dir_for_train_config(policy_pretrained)
    train_cfg = TrainPipelineConfig.from_pretrained(str(policy_dir))
    dcfg = train_cfg.dataset
    return LeRobotDatasetMetadata(dcfg.repo_id, root=dcfg.root, revision=dcfg.revision)


def deploy_loop(
    robot: Robot,
    events: dict,
    fps: int,
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    frame_features: dict,
    single_task: str | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,
    control_loop_stats: dict | None = None,
) -> None:
    """
    Fixed-rate control on the **main** process plus one **inference** subprocess (``spawn`` for CUDA).

    The main process enqueues ``build_dataset_frame`` outputs when the queue accepts a new item;
    the inference child never touches the robot (required for ``spawn`` + non-picklable SDK objects).
    """
    policy.reset()
    if preprocessor is not None and postprocessor is not None:
        preprocessor.reset()
        postprocessor.reset()

    # Manager dict shares ``latest_action_ref`` across processes (spawn-safe pickling vs plain dict).
    mp_ctx = _deploy_multiprocessing_context()
    with mp_ctx.Manager() as action_manager:
        latest_action_ref: Any = action_manager.dict()
        latest_action_ref["action"] = None
        latest_action_lock = mp_ctx.Lock()
        robot_io_lock = mp_ctx.RLock()
        obs_queue = mp_ctx.Queue(1)
        device = get_safe_torch_device(policy.config.device)
        period_s = 1.0 / fps
    
        def _compute_robot_action_for_send(
            act_to_send: RobotAction,
            obs_processed: RobotObservation,
        ) -> RobotAction | None:
            try:
                return robot_action_processor((act_to_send, obs_processed))
            except Exception as e:
                logging.warning(
                    "robot_action_processor failed (keeping previous send command): %s",
                    e,
                )
                return None
    
        inference_process = mp_ctx.Process(
            target=_inference_worker,
            kwargs=dict(
                obs_queue=obs_queue,
                latest_action_ref=latest_action_ref,
                latest_action_lock=latest_action_lock,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset_features=frame_features,
                single_task=single_task,
                robot_type=robot.robot_type,
                use_amp=policy.config.use_amp,
                device=device,
            ),
            daemon=True,
        )
        inference_process.start()
    
        with robot_io_lock:
            first_obs = _deploy_get_robot_observation(robot, include_images=True)
        first_obs_processed = robot_observation_processor(first_obs)
        first_observation_frame = build_dataset_frame(
            frame_features, dict(first_obs_processed), prefix=OBS_STR
        )
        try:
            obs_queue.put_nowait(first_observation_frame)
        except queue.Full:
            pass
        with latest_action_lock:
            first_policy_action = latest_action_ref["action"]
        if first_policy_action is not None and len(first_policy_action) == len(robot.action_features):
            first_act_to_send: RobotAction = first_policy_action
        else:
            first_act_to_send = _current_pose_as_action(first_obs_processed, robot.action_features)
            if first_policy_action is None:
                logging.warning(
                    "Deploy[first_tick] first_policy_action=None first_obs_frame=%s first_act_hold=%s",
                    _deploy_summarize_observation_frame_one_line(first_observation_frame),
                    _deploy_format_action_dict(first_act_to_send),
                )
            else:
                logging.warning(
                    "Deploy[first_tick] policy_rejected len(policy)=%d len(robot.action_features)=%d "
                    "policy_dict=%s expected_keys=%s dataset_action.names=%s first_act_hold=%s",
                    len(first_policy_action),
                    len(robot.action_features),
                    _deploy_format_action_dict(first_policy_action),
                    list(robot.action_features.keys()),
                    (frame_features.get(ACTION) or {}).get("names"),
                    _deploy_format_action_dict(first_act_to_send),
                )
    
        first_ra = _compute_robot_action_for_send(first_act_to_send, first_obs_processed)
        last_robot_action_sent: RobotAction | None = None
        if display_data:
            log_rerun_data(
                observation=first_obs_processed,
                action=first_act_to_send,
                compress_images=display_compressed_images,
            )
        if first_ra is not None:
            try:
                with robot_io_lock:
                    robot.send_action(first_ra)
                last_robot_action_sent = dict(first_ra)
                if control_loop_stats is not None:
                    control_loop_stats["sends"] = 1
            except Exception as e:
                logging.warning(
                    "Deploy[send_err first] exc=%r robot_action_to_send=%s",
                    e,
                    _deploy_format_action_dict(first_ra),
                )
    
        if control_loop_stats is not None:
            control_loop_stats["frames"] = control_loop_stats.get("frames", 0) + 1
    
        last_obs_with_images: RobotObservation = dict(first_obs)
        next_tick_t = time.perf_counter()
        hold_due_to_policy_len_mismatch = 0
        no_policy_action_loops = 0
        warn_policy_none_interval = max(1, int(fps * 2))
        main_loop_i = 0
    
        try:
            while True:
                if events.get("exit_early"):
                    break
    
                next_tick_t += period_s
                precise_sleep(max(0.0, next_tick_t - time.perf_counter()))
                start_loop_t = time.perf_counter()
                main_loop_i += 1
    
                # Main reads proprio for control; full camera read + dataset frame happen here so the
                # inference child only runs ``predict_action`` (spawn cannot unpickle CrpRobotPy).
                with robot_io_lock:
                    obs = _deploy_get_robot_observation(robot, include_images=False)
                obs_processed = robot_observation_processor(obs)
                try:
                    with robot_io_lock:
                        obs_for_policy = _deploy_get_robot_observation(robot, include_images=True)
                    obs_policy_processed = robot_observation_processor(obs_for_policy)
                    observation_frame_q = build_dataset_frame(
                        frame_features, dict(obs_policy_processed), prefix=OBS_STR
                    )
                    obs_queue.put_nowait(observation_frame_q)
                except queue.Full:
                    pass
    
                with latest_action_lock:
                    policy_action = latest_action_ref["action"]
                if policy_action is not None and len(policy_action) == len(robot.action_features):
                    act_to_send: RobotAction = policy_action
                    hold_due_to_policy_len_mismatch = 0
                    no_policy_action_loops = 0
                else:
                    act_to_send = _current_pose_as_action(obs_processed, robot.action_features)
                    if policy_action is not None:
                        hold_due_to_policy_len_mismatch += 1
                        if hold_due_to_policy_len_mismatch == 1 or hold_due_to_policy_len_mismatch % max(
                            1, int(fps * 5)
                        ) == 0:
                            logging.warning(
                                "Deploy[hold_mismatch repeat=%d] len(policy)=%d len(robot.action_features)=%d "
                                "policy_dict=%s expected_keys=%s dataset_action.names=%s hold_act=%s",
                                hold_due_to_policy_len_mismatch,
                                len(policy_action),
                                len(robot.action_features),
                                _deploy_format_action_dict(policy_action),
                                list(robot.action_features.keys()),
                                (frame_features.get(ACTION) or {}).get("names"),
                                _deploy_format_action_dict(act_to_send),
                            )
                    else:
                        no_policy_action_loops += 1
                        if no_policy_action_loops % warn_policy_none_interval == 0:
                            joint_obs = {
                                k: obs_processed[k]
                                for k in sorted(obs_processed)
                                if k.endswith(".pos") and k.startswith("j")
                            }
                            logging.warning(
                                "Deploy[no_policy main_i=%d] policy_action=None after %d iters (~%.1fs fps=%d) "
                                "hold_act=%s obs_joint_sample=%s obs_frame=%s",
                                main_loop_i,
                                no_policy_action_loops,
                                no_policy_action_loops / fps,
                                fps,
                                _deploy_format_action_dict(act_to_send),
                                _deploy_format_action_dict(joint_obs),
                                _deploy_format_action_dict(joint_obs),
                            )
                    if len(act_to_send) != len(robot.action_features):
                        logging.warning(
                            "Could not build full current-pose action (%d/%d keys), sending as-is.",
                            len(act_to_send),
                            len(robot.action_features),
                        )
    
                ra = _compute_robot_action_for_send(act_to_send, obs_processed)
                to_send = ra if ra is not None else last_robot_action_sent
                if to_send is not None:
                    try:
                        with robot_io_lock:
                            robot.send_action(to_send)
                        if ra is not None:
                            last_robot_action_sent = dict(ra)
                        if control_loop_stats is not None:
                            control_loop_stats["sends"] = control_loop_stats.get("sends", 0) + 1
                            sends = control_loop_stats["sends"]
                            if sends <= 8 or sends % 60 == 0:
                                logging.warning(
                                    "Deploy[send_ok n=%d] send_action_arg=%s",
                                    sends,
                                    _deploy_format_action_dict(to_send),
                                )
                    except Exception as e:
                        logging.warning(
                            "Deploy[send_err n=%d] exc=%r robot_action_to_send=%s",
                            control_loop_stats.get("sends", 0) if control_loop_stats else 0,
                            e,
                            _deploy_format_action_dict(to_send),
                        )
    
                if main_loop_i <= 10 or main_loop_i % max(1, int(fps * 2)) == 0:
                    src = "policy" if (
                        policy_action is not None and len(policy_action) == len(robot.action_features)
                    ) else "hold"
                    logging.warning(
                        "Deploy[main_loop n=%d src=%s] act_to_send=%s robot_action_to_send=%s obs_joint_sample=%s",
                        main_loop_i,
                        src,
                        _deploy_format_action_dict(act_to_send),
                        _deploy_format_action_dict(ra) if ra is not None else {},
                        _deploy_format_action_dict(
                            {
                                k: obs_processed[k]
                                for k in sorted(obs_processed)
                                if k.endswith(".pos") and k.startswith("j")
                            }
                        ),
                    )
    
                if display_data:
                    log_rerun_data(
                        observation=obs_processed,
                        action=act_to_send,
                        compress_images=display_compressed_images,
                    )
    
                dt_s = time.perf_counter() - start_loop_t
                if control_loop_stats is not None:
                    control_loop_stats["frames"] = control_loop_stats.get("frames", 0) + 1
                    if dt_s > period_s:
                        overruns = control_loop_stats.get("overruns", 0) + 1
                        control_loop_stats["overruns"] = overruns
                        if overruns <= 5 or overruns % 30 == 0:
                            logging.warning(
                                "Deploy[pipeline overrun]: step took %.1f ms (period %.1f ms); "
                                "control rate may slip. Total overruns: %d.",
                                dt_s * 1e3,
                                period_s * 1e3,
                                overruns,
                            )
        finally:
            obs_queue.put(None)
            inference_process.join(timeout=5.0)
    
    

@parser.wrap()
def deploy(cfg: DeployConfig) -> None:
    """Load policy, run on robot until exit. No dataset recording."""
    init_logging()
    assert cfg.policy is not None  # set in DeployConfig.__post_init__ from --policy.path
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="deploy", ip=cfg.display_ip, port=cfg.display_port)
        logging.warning(
            "Deploy: display_data=True may increase control loop latency (Rerun logging and optional "
            "image compression). For stable control rate, consider disabling with --display_data=false."
        )
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    assert cfg.policy.pretrained_path is not None
    ds_meta = _deploy_training_dataset_meta(cfg.policy.pretrained_path)
    fps = ds_meta.fps

    robot = make_robot_from_config(cfg.robot)
    _, robot_action_processor, robot_observation_processor = make_default_processors()

    ds_action_names = list((ds_meta.features.get(ACTION) or {}).get("names") or [])
    robot_action_keys = list(robot.action_features.keys())
    if set(ds_action_names) != set(robot_action_keys):
        logging.warning(
            "Deploy: training dataset action.names and robot.action_features keys differ. "
            "dataset.names=%s robot.keys=%s — policy dict / send_action may not align.",
            ds_action_names,
            robot_action_keys,
        )
    if len(ds_action_names) != len(robot_action_keys):
        logging.warning(
            "Deploy: action cardinality mismatch len(dataset.names)=%d vs len(robot.action_features)=%d.",
            len(ds_action_names),
            len(robot_action_keys),
        )

    policy = make_policy(cfg.policy, ds_meta=ds_meta, rename_map=cfg.rename_map)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=rename_stats(ds_meta.stats, cfg.rename_map),
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    listener = None
    try:
        robot.connect()
        listener, events = init_keyboard_listener()

        log_say("Deploy: run policy (no recording). Press exit key to stop.", cfg.play_sounds)

        control_loop_stats: dict = {}
        deploy_loop(
            robot=robot,
            events=events,
            fps=fps,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            frame_features=ds_meta.features,
            single_task=cfg.single_task,
            display_data=cfg.display_data,
            display_compressed_images=display_compressed_images,
            control_loop_stats=control_loop_stats,
        )

        if control_loop_stats:
            frames = control_loop_stats.get("frames", 0)
            overruns = control_loop_stats.get("overruns", 0)
            sends = control_loop_stats.get("sends", 0)
            pct = (100.0 * overruns / frames) if frames else 0
            logging.info(
                "Deploy: %d pipeline frames, %d sends to robot, %d pipeline overruns (%.1f%%).",
                frames,
                sends,
                overruns,
                pct,
            )
            if overruns > 0:
                logging.warning(
                    "Pipeline overruns may cause the control loop to miss ticks; "
                    "consider disabling display or reducing load."
                )
    finally:
        log_say("Stop deploy", cfg.play_sounds, blocking=True)

        if robot.is_connected:
            robot.disconnect()

        if not is_headless() and listener:
            listener.stop()

        log_say("Exiting", cfg.play_sounds)


def main():
    # CRP: configure ``sys.path`` / ``LD_LIBRARY_PATH`` for ``CrpRobotPy`` before any ``crp_arm`` import.
    load_CrpRobotPy()
    import lerobot.robots.crp_arm  # noqa: F401 — register ``crp_arm`` on ``RobotConfig``

    register_third_party_plugins()
    deploy()


if __name__ == "__main__":
    main()
