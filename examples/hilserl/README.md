# HIL-SERL：CRP + OMY_L100（EE-delta / 插管）

示例配置：[`crp_omy_inserting_rod.json`](./crp_omy_inserting_rod.json)（**EE-delta**；默认 `include_rpy=false` → action **4**；`true` → **7**）  
录制 / EE replay：[`../record/`](../record/)（[`crp_omy_record.json`](../record/crp_omy_record.json)、[`crp_omy_replay_ee.json`](../record/crp_omy_replay_ee.json)）  
HIL 控制约定：Space 介入（录制同款 100 Hz GP）；松手 **GI56=0** 切断；切换帧不进 replay。

当前 demo：`outputs/Insert_blue_tube_zyy_0807_01_ee`（磁盘仍为 **7D 绝对 EE+gripper**；HIL 装 offline 时按 `include_rpy` 内存转 delta，不改 parquet/`meta`）。

---

## 流程速览

### 录制（OMY → CRP GP → dataset）

```text
开 OMY bringup → 跑 crp_record_omy_ee_inc + crp_omy_record.json
  → connect：GI56=0（不 mv）→ EE 稳定 → latch → preload GP → GI55=GP → GI56=1 → 100Hz
  → 左右键 / 段结束：GI56=0，停流，保持末位 GP
  → 下一段重新 EE wait + latch + arm
  → 写完 episode 进 dataset
```

```bash
python -m lerobot.scripts.crp_record_omy_ee_inc \
  --config_path=examples/record/crp_omy_record.json
```

EE 绝对位姿 replay（验证录制）：

```bash
python -m lerobot.scripts.crp_replay_ee_abs \
  --config_path=examples/record/crp_omy_replay_ee.json
```

#### 注意：示教器追的是寄存器；PC 只改目标与使能

`send_GPs` / `send_GJs` / `set_GI` 只改寄存器；**真正 `mv` 的是示教器 Auto 循环**。主安全闸是 **GI56**：

1. **connect / teardown：`set_motion_enabled(False)`（GI56=0）** — 不 mv，残留 GP 无害。
2. **EE wait 零发送** — 不写 GP、不切 GI→GP、GI56 保持 0。
3. **arm ready**：4/4 latch preload → `ensure_gp_mode` → **`set_motion_enabled(True)`** → `omy_gp_armed` → fork。勿在 spawn 后再读 EE。

开局不写 GJ hold；安全靠 **GI56=0**（示教器须按上文门控改好）。

#### 示教器程序约定（请与 PC 对齐）

PC 写寄存器；示教器 Auto 循环才 `mv`。推荐逻辑（GI55=模式，GI56=是否允许 move）：

```text
循环:
  if GI56 == 0:
    ; 不 move（空转）
    if GI55 == 1:
      mv GP
    else:
      mv GJ
    endif
  endif
```

| 寄存器 | 含义 | PC 何时写 |
|--------|------|-----------|
| GI55 | GP(1) / GJ(0) | `ensure_gp_mode` / `ensure_gj_mode` / `send_GP*s` 切模式 |
| GI56 | move 使能 | connect/teardown → **0**；arm ready / 策略 `send_action` → **1** |

JSON：`robot.motion_enable_gi_index`（默认 56；`null` 关闭写入）。

GI56 时序（录制）：`connect→0` → EE wait 保持 0 → arm 时 `1` → 流存活期**不再改** → teardown **先停 fork** 再 `0`。左右键后下一段再走同样流程。流内存活时父进程禁止写 GI（会和 fork 抢 SDK）。

录制手感排查：

- **一卡一卡**：
  1. GP 指令用 **`Array` 共享内存**（勿 `Manager.list`）。
  2. fork **只发 GP**（流期间不 `read_joints` / `read_end_pose`）；obs = 上一帧 action。
  3. 若仍卡：检查示教器 `mv GP` 是否**等到位才循环**（应改为可被新目标打断的连续跟踪）。
- **遥操在动、CRP 愣住**：spawn 侧 EE 为 `None` 时不更新 GP（日志 `EE ... None streak`）。
- **就绪跳回开局**：teardown `clear_ee_cache`；示教器 `GI56==0` 时不能误走 GJ。

就绪提示（终端 ANSI 颜色；CRP SDK 无灯光接口）：

- **黄**：`EE wait 4/4` — 父进程 EE 已稳，还在等 spawn / arm，**先别动手**
- **红横幅**：录制阶段 fork GP 已起 — **可以开始遥操**
- **青横幅**：复位阶段遥操就绪（摆场景 / 回位，**不写 episode**）

左右键后会再进一段 `record_loop(phase=reset)`，因此可能再看到一次就绪提示——这是两段循环，不是重复 bug。

### HIL-SERL（learner + actor）

```text
终端1 OMY bringup
终端2 learner（GPU 训练 + gRPC）
终端3 actor（真机）：策略 δxyz(+GOT)→send_GPs ↔ Space 按住则 OMY→GP 介入（100Hz）
  → offline：磁盘 abs EE 在装 buffer 时按 include_rpy 转 4D/7D delta（不改磁盘）
  → 介入标签同维；松手 **GI56=0** 切断（不重发 GP/GJ hold）；切换/arming 帧不进 replay
```

架构与官方 LeRobot HIL-SERL 相同（异步 actor/learner、双 buffer、TeleopEvents）；CRP 只换了控制层。

**Load-time abs→delta（不改磁盘）**：action names 含 `ee.x` 且无 `delta_*` → 差分；`processor.crp_ee.include_rpy` 控制是否带 δrpy：

| `include_rpy` | action 维 | 内容 | JSON 还需改 |
|---------------|-----------|------|-------------|
| **`false`**（当前示例） | **4** | δxyz + 绝对 GOT；执行时 rpy 锁参考位姿 | `action.shape=[4]`，`target_entropy≈-2` |
| **`true`** | **7** | δxyz + δrpy + 绝对 GOT | `action.shape=[7]`，`target_entropy≈-3.5` |

---

### Obs / Action 与数据集对齐（已核对）

| 字段 | 磁盘 demo | 当前 HIL 配置 | 含义 |
|------|-----------|---------------|------|
| `observation.state` | 7：`ee.x..yaw` + `gripper.pos` | shape `[7]` | 指令 cache（obs=上一帧 action） |
| `action` | 7：绝对 EE+gripper | 策略 shape `[4]` 或 `[7]`（看 `include_rpy`） | load-time 转 delta |
| 图像 | `top`/`wrist` 480×640 | resize → 128×128 | 键名一致 |
| 控制频率 | demo **30 FPS** | env **30 Hz**（目标环路；跑不满时墙钟步数会少） | 标签按 env step 记 |

录制开关（`robot`）：至少一个为 true。OMY→GP 录制优先 `enable_ee=true`。

| 开关 | obs | action |
|------|-----|--------|
| `enable_joint` | 上一帧 action 关节 | 同上 |
| `enable_ee` | 上一帧 action EE | 当前 GP 指令 |

数值范围：EE 绝对约 mm 级；转 delta 后 `|δ|` 约相邻帧差；gripper GOT0∈[0,1000]。ACTION 归一化用 **转换后** 的 min/max（勿用磁盘 abs stats）。

### 超参（对齐 HIL-SERL / RLPD）

相对 LeRobot SAC 默认，示例已按论文附录 + 社区复现改过：

| 项 | 论文/复现 | 本配置 |
|----|-----------|--------|
| `batch_size` | 256 | 256 |
| `utd_ratio` | **20**（高 UTD） | **10**（可再升到 20；同卡 actor 时更吃 GPU） |
| `discount` | 0.97 | 0.97 |
| `actor/critic/temp_lr` | 3e-4 | 3e-4 |
| `temperature_init` | 0.01 | 0.01 |
| `target_entropy` | ≈ −\|A\|/2 | **−2.0**（当前 4 维；开 rpy 时改 −3.5） |
| `online_ratio` | 0.5（对称采 demo+online） | 0.5 |
| online buffer | ~2–3 万 transition | **32000** |
| offline buffer | 约 demo 全量 | capacity **8000**（~7k 帧 EE demo） |
| MLP / latent | 256×256 / 256 | 同 |
| proprio encoder | 64 | **64** |
| `env.fps` | 10（论文常见） | **30**（与 demo 对齐；实际常低于目标） |
| episode 时长 | 论文短任务 ~10–12 s | **`control_time_s=30`**（步数上限 ≈ fps×30；可提前 s/f/r） |
| `save_freq` | — | **5000**（`resume=true` 时以 checkpoint 内配置为准） |

与论文仍不同、但为本任务保留：夹爪为**连续** GOT0（数据集如此），未用论文的离散 gripper DQN；reward classifier 可先 `null` 用手标 `s`/`f`。

### 预计多久有效果？

论文：**1–2.5 小时**真机训练可达近完美（Franka + 好 classifier + 短时任务）。插 peg / 本机台更保守估计：

- **~0.5–1 h**：介入率应开始下降，策略不再完全乱晃（需短修正介入，勿整段代打成功）
- **~1.5–3 h**：可望看到零星自主成功；有可靠 reward classifier 时更接近论文节奏
- **~3–5 h**：较稳的成功率（取决于介入质量、复位是否与 demo 同分布、灯光）

有效 online 步数量级约 **2–3 万** transition；人手介入质量比堆时间更关键。

---

## EE vs 关节（配置开关）

```json
"processor": {
  "crp_action_space": "ee",
  "crp_ee": { "ee_delta_scale": 1.5, "ee_delta_max": 5.0, "offline_ee_action": null }
}
```

| `crp_action_space` | 策略学什么 | 自主执行 | Space 干预 | 与 0807 EE demo |
|--------------------|------------|----------|-----------|-----------------|
| **`ee`**（当前示例） | δxyz[+δrpy] + 绝对 GOT（`include_rpy`） | `send_GPs` + GI56；`ee_delta_max` 限步 | 录制同款 100Hz GP；标签同维；松手 **GI56=0** | load-time abs→delta |
| **`joint`** | CRP `j1..j6(+gripper)` | `send_GJs` | OMY→GP；标签读 CRP 关节；松手 GJ hold | 需关节 demo |

Space 接管：锁存 `p0_crp` + 当前 CRP rpy + `omy_ref` → 相对 xyz GP（100Hz）。姿态跟锁存 rpy。切换帧 `exclude_from_replay`。

---

## 启动（三个终端）

HIL-SERL 是异步架构：**learner 与 actor 必须分开开**；OMY 再占一个终端。两者共用同一 `output_dir`（先开 learner 再建目录是正常的，actor 不应因此 `FileExistsError`）。

```bash
# 终端 1 — OMY 主臂（非 conda 环境按你平时习惯）
cd ~/robotis/
source install/setup.sh
ros2 launch open_manipulator_bringup omy_l100_leader_ai.launch.py

# 终端 2 — learner（先开；训练 + gRPC :50051）
conda activate zyy_lerobot   # 或你的环境
cd lerobot/
python -m lerobot.rl.learner --config_path examples/hilserl/crp_omy_inserting_rod.json

# 终端 3 — actor（learner 起来后再开；真机采集）
python -m lerobot.rl.actor --config_path examples/hilserl/crp_omy_inserting_rod.json
```

同目录重跑：删掉 `output_dir` 或改名。

**继续训练（resume）**：示例配置里设 `"resume": true`，或命令行加 `--resume=true`。会自动读：

`{output_dir}/checkpoints/last/pretrained_model/`

```bash
# 与首次启动相同的 config；仅多 resume
python -m lerobot.rl.learner --config_path=examples/hilserl/crp_omy_inserting_rod.json --resume=true
python -m lerobot.rl.actor --config_path=examples/hilserl/crp_omy_inserting_rod.json --resume=true
```

注意：配置里是 `policy.type=gaussian_actor` + `algorithm.type=sac`，**不要**写 `policy.type=sac`（会报 ChoiceRegistry KeyError）。

## 键盘

| 键 | 作用 |
|----|------|
| Space（按住） | 接管（OMY→GP 100Hz；arming/松手切换帧不进 buffer） |
| s / f | 成功 / 失败（结束本集并推 learner） |
| r | 重录（结束本集；**整集不进** learner buffer） |
| p | 场景复位倒计时暂停/恢复（`hil_reset_pause_key` / `episode_reset_pause_key`） |

录制脚本（`crp_record_omy_ee_inc`）左右键：

| 键 | 流程 |
|----|------|
| **→ 右键** | 结束本段录制 → **复位循环**（可遥操摆位，青色提示）→ `save_episode` 写盘 → 下一段录制（红色提示） |
| **← 左键** | 结束本段录制 → **复位循环**（青色）→ 清空缓冲不写盘 → 再开录制（红色） |

所以左键会看到「青 + 红」两次就绪，不是 bug：一次复位、一次重录。右键先红（若本段刚 armed）或复位青，复位结束后才写盘。

复位倒计时用 **`p`** 暂停（与 Space 接管分离）；暂停期间剩余时间冻结。

接管时把焦点放在 **actor 终端**（现已加 tty 键盘兜底）；不要依赖 pynput suppress。启动日志应出现 `HIL tty keyboard started` 或 `HIL pynput listener started`。

发 **GP** 前会写 `set_GI(55)=1`，发 **GJ** 前写 `set_GI(55)=0`（示教器常标为 S55；仅在模式切换时写一次）。另写 **`set_GI(56)`** 作为 move 使能（见上文示教器约定）。可在 JSON 用 `robot.motion_mode_gi_index` / `robot.motion_enable_gi_index` 关掉（`null`）。

`teleop.ros_ee_pose_position_scale` 默认 **1000**（ROS 米 → CRP 毫米）；若接管时臂几乎不动，先查此项。

Policy FPS 若长期低于 `env.fps`（日志会警告），控制会变慢，属算力/相机瓶颈，不是按键问题。

## 必核对

- `dataset.repo_id` 必须是 `namespace/name` 两段（如 `3floor/inserting_rod`），不要写成 `dataset/3floor/...`
- `dataset.root` 用绝对路径（`~` 也可，代码会 expanduser）；指向含 `meta/info.json` 的本地目录
- 全量 inserting_rod ≈ 33 万帧；`offline_buffer_capacity` 较小时 learner 会自动抽若干 episode 填满 buffer（也可手动设 `dataset.episodes`）
- `policy.storage_device` 建议 `"cpu"`（replay 存图很占显存；offline 会 resize 到 `input_features` 的 128×128）
- 归一化统计从 `dataset.root` 的 `meta/stats.json` 自动加载（覆盖 gaussian_actor 默认的 2 维 state）
- 相机 index、`robot.port`
- `crp_action_space`：本示例为 **`ee`**（对齐 0807 EE demo）；关节 demo 才用 `"joint"`
- `teleop.hil_ee_delta=true`（关节模式下干预仍靠 OMY EE 带动）
- `reward_classifier.pretrained_path`（可先 `null`，用手标 s/f）
