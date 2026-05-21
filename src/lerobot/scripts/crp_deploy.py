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
在 CRP 机械臂上部署策略：与 ``lerobot_deploy`` 相同的策略 / 数据集元信息管线，
但**仅支持** ``crp_arm``，且**单线程**主循环（每步：读相机 + 关节 → 推理 → ``send_action``），
不使用后台推理线程或队列。

控制周期仍按训练数据集的 ``fps`` 用 ``precise_sleep`` 对齐；若单步（相机 + GPU）
超过一个周期，会产生与 ``lerobot_deploy`` 类似的 overrun 日志。

示例：

```shell
python -m lerobot.scripts.crp_deploy \\
    --robot.port=/dev/ttyUSB0 \\
    --robot.cameras='{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \\
    --policy.path=/path/to/checkpoint/pretrained_model \\
    --single_task="Pick the cube"
```
"""

import logging
import time
from dataclasses import asdict, dataclass, field
from pprint import pformat
from typing import Any

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import build_dataset_frame
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
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.scripts.lerobot_deploy import (
    _current_pose_as_action,
    _deploy_format_action_dict,
    _deploy_get_robot_observation,
    _deploy_summarize_array_or_tensor,
    _deploy_summarize_observation_frame_one_line,
    _deploy_training_dataset_meta,
)
from lerobot.tools.lib_loader import load_CrpRobotPy
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


@dataclass
class CrpDeployConfig:
    robot: RobotConfig
    policy: PreTrainedConfig | None = None
    single_task: str | None = None
    rename_map: dict[str, str] = field(default_factory=dict)
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False
    play_sounds: bool = True

    def __post_init__(self) -> None:
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        if self.policy is None:
            raise ValueError("需要 --policy.path=... 以加载预训练策略。")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


def crp_deploy_loop(
    robot,
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
    """单线程：每周期全观测（含图像）→ 策略 → ``send_action``。"""
    policy.reset()
    if preprocessor is not None and postprocessor is not None:
        preprocessor.reset()
        postprocessor.reset()

    device = get_safe_torch_device(policy.config.device)
    period_s = 1.0 / fps
    ok_count = 0

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

    last_robot_action_sent: RobotAction | None = None
    hold_due_to_policy_len_mismatch = 0
    no_policy_action_loops = 0
    warn_policy_none_interval = max(1, int(fps * 2))
    main_loop_i = 0
    next_tick_t = time.perf_counter()

    while True:
        if events.get("exit_early"):
            break
        if main_loop_i > 0:
            next_tick_t += period_s
            precise_sleep(max(0.0, next_tick_t - time.perf_counter()))

        start_loop_t = time.perf_counter()
        main_loop_i += 1

        obs = _deploy_get_robot_observation(robot, include_images=True)
        obs_processed = robot_observation_processor(obs)
        observation_frame = build_dataset_frame(
            frame_features, dict(obs_processed), prefix=OBS_STR
        )

        policy_action: RobotAction | None = None
        action_values: Any = None
        try:
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
            )
            policy_action = make_robot_action(action_values, frame_features)
            ok_count += 1
            if ok_count <= 8 or ok_count % 30 == 0:
                logging.warning(
                    "CrpDeploy[inference_ok n=%d] task=%r robot_type=%r dataset_action.names=%s "
                    "raw_action_tensor=%s policy_action_dict=%s obs_frame=%s",
                    ok_count,
                    single_task,
                    robot.robot_type,
                    (frame_features.get(ACTION) or {}).get("names"),
                    _deploy_summarize_array_or_tensor(action_values),
                    _deploy_format_action_dict(policy_action),
                    _deploy_summarize_observation_frame_one_line(observation_frame),
                )
        except Exception as e:
            logging.warning(
                "CrpDeploy[inference_err] exc=%r task=%r robot_type=%r dataset_action.names=%s obs_frame=%s",
                e,
                single_task,
                robot.robot_type,
                (frame_features.get(ACTION) or {}).get("names"),
                _deploy_summarize_observation_frame_one_line(observation_frame),
            )

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
                        "CrpDeploy[hold_mismatch repeat=%d] len(policy)=%d len(robot.action_features)=%d "
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
                        "CrpDeploy[no_policy main_i=%d] policy_action=None after %d iters (~%.1fs fps=%d) "
                        "hold_act=%s obs_joint_sample=%s",
                        main_loop_i,
                        no_policy_action_loops,
                        no_policy_action_loops / fps,
                        fps,
                        _deploy_format_action_dict(act_to_send),
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
                robot.send_action(to_send)
                if ra is not None:
                    last_robot_action_sent = dict(ra)
                if control_loop_stats is not None:
                    control_loop_stats["sends"] = control_loop_stats.get("sends", 0) + 1
                    sends = control_loop_stats["sends"]
                    if sends <= 8 or sends % 60 == 0:
                        logging.warning(
                            "CrpDeploy[send_ok n=%d] send_action_arg=%s",
                            sends,
                            _deploy_format_action_dict(to_send),
                        )
            except Exception as e:
                logging.warning(
                    "CrpDeploy[send_err n=%d] exc=%r robot_action_to_send=%s",
                    control_loop_stats.get("sends", 0) if control_loop_stats else 0,
                    e,
                    _deploy_format_action_dict(to_send),
                )

        if main_loop_i <= 10 or main_loop_i % max(1, int(fps * 2)) == 0:
            src = "policy" if (
                policy_action is not None and len(policy_action) == len(robot.action_features)
            ) else "hold"
            logging.warning(
                "CrpDeploy[main_loop n=%d src=%s] act_to_send=%s robot_action_to_send=%s obs_joint_sample=%s",
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
                        "CrpDeploy[pipeline overrun]: step took %.1f ms (period %.1f ms); "
                        "control rate may slip. Total overruns: %d.",
                        dt_s * 1e3,
                        period_s * 1e3,
                        overruns,
                    )


@parser.wrap()
def crp_deploy(cfg: CrpDeployConfig) -> None:
    init_logging()
    assert cfg.policy is not None
    if cfg.robot.type != "crp_arm":
        raise ValueError("crp_deploy 仅支持 --robot.type=crp_arm")
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="crp_deploy", ip=cfg.display_ip, port=cfg.display_port)
        logging.warning(
            "CrpDeploy: display_data=True 可能增加单步耗时；需要稳定控制率时可关闭 --display_data=false。"
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
            "CrpDeploy: 数据集 action.names 与 robot.action_features 键不一致。"
            "dataset.names=%s robot.keys=%s",
            ds_action_names,
            robot_action_keys,
        )
    if len(ds_action_names) != len(robot_action_keys):
        logging.warning(
            "CrpDeploy: action 数量不一致 len(dataset.names)=%d vs len(robot.action_features)=%d.",
            len(ds_action_names),
            len(robot_action_keys),
        )

    policy = make_policy(cfg.policy, ds_meta=ds_meta)
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
        log_say("CrpDeploy: 单线程策略运行中，按退出键结束。", cfg.play_sounds)

        control_loop_stats: dict = {}
        crp_deploy_loop(
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
                "CrpDeploy: %d 帧, %d 次 send_action, %d 次超时 (%.1f%%).",
                frames,
                sends,
                overruns,
                pct,
            )
            if overruns > 0:
                logging.warning(
                    "单步耗时经常超过周期会导致控制节拍漂移；可关闭可视化或降低负载。"
                )
    finally:
        log_say("停止 CrpDeploy", cfg.play_sounds, blocking=True)
        if robot.is_connected:
            robot.disconnect()
        if not is_headless() and listener:
            listener.stop()
        log_say("退出", cfg.play_sounds)


def main() -> None:
    load_CrpRobotPy()
    import lerobot.robots.crp_arm  # noqa: F401

    register_third_party_plugins()
    crp_deploy()


if __name__ == "__main__":
    main()
