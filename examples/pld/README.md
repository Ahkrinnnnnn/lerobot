# PLD（Pi0.5 + Residual SAC）— stack_the_base

论文 Algorithm 1 的真机实现：**Stage 1**（offline → Cal-QL → online SAC）→ **Stage 2**（hybrid 采集）→ **`lerobot-train`** 对 base policy 做 SFT。

Reward classifier 请先按 [../reward_classifier/README.md](../reward_classifier/README.md) 训好。

## 安装

```bash
cd /home/dingyuxuan/zyy/lerobot
pip install -e ".[dataset,training]"
(torch-cuda)
```

命令名为 **连字符**：`lerobot-pld-stage1`、`lerobot-pld-stage2`（不是下划线）。

## 前置条件

1. Pi0.5 SFT base policy（示例：`outputs/models/pi05_stack_the_base_all/checkpoints/080000/pretrained_model`）
2. Reward classifier 已训练并写入 PLD config 的 `reward_classifier.path`
3. CRP 机械臂 + 相机（示例：top=14, wrist=6）

## 操作键（collect 全程）

| 键 | 作用 |
|----|------|
| `s` | 手动标成功，结束 episode |
| `f` | 手动标失败，结束 episode |
| Space | 暂停 / 恢复 collect（全局 pynput，与 s/f 相同机制；臂保持姿态，不计 env step） |

episode 间黄色 `!!! [MANUAL SCENE RESET]` 为摆场景 15s 窗口；Space 也可暂停倒计时。

---

## Stage 1

Config：`crp_pi05_stage1_stack_the_base.json`

### 完整流程

```bash
HF_HUB_OFFLINE=1 lerobot-pld-stage1 \
  --config_path=examples/pld/crp_pi05_stage1_stack_the_base.json
```

`offline 采集 → Cal-QL → online SAC` 一条龙。

### 分步 / 续跑

```bash
# 仅 offline
lerobot-pld-stage1 --config_path=examples/pld/crp_pi05_stage1_stack_the_base.json \
  --skip_calql_pretrain=true --skip_rl_train=true

# 跳过 offline，从 offline buffer 继续 Cal-QL + RL
lerobot-pld-stage1 --config_path=examples/pld/crp_pi05_stage1_stack_the_base.json \
  --skip_offline_collect=true \
  --pld.resume_offline_buffer=outputs/pld_stage1_pi05_stack_the_base_all/offline_buffer.pt

# 续跑 online RL（跳过 offline + Cal-QL）
HF_HUB_OFFLINE=1 lerobot-pld-stage1 \
  --config_path=examples/pld/crp_pi05_stage1_stack_the_base.json \
  --skip_offline_collect=true \
  --skip_calql_pretrain=true \
  --pld.resume_offline_buffer=outputs/pld_stage1_pi05_stack_the_base_all/offline_buffer.pt \
  --pld.resume_online_buffer=outputs/pld_stage1_pi05_stack_the_base_all/online_buffer.pt
```

**Warmup**：`warmup_episodes × avg_episode_steps`（默认 5×500=2500）。`online_buffer` 已有 ≥2500 条时 **自动跳过 warmup**，直接 π_b+π_δ。

**续训权重**：自动加载 `residual_sac.pt`（优先于 `calql_critic.pt`）。**步数计数**：每轮 SAC 后写入 `rl_train_progress.json`；若无此文件则从 `train_steps=0` 计数（权重仍续上）。

### 常用 CLI 覆盖

```bash
lerobot-pld-stage1 --config_path=examples/pld/crp_pi05_stage1_stack_the_base.json \
  --base_policy.path=outputs/models/pi05_stack_the_base_all/checkpoints/080000/pretrained_model \
  --reward_classifier.path=outputs/reward_classifier/models/stack_the_base/checkpoints/003000/pretrained_model \
  --pld.rl_steps=5000 \
  --pld.collect_env_steps_per_iter=800 \
  --pld.train_iters_per_collect=50
```

### Stage 1 输出（`output_dir`）

| 文件 | 含义 |
|------|------|
| `offline_buffer.pt` | offline 成功轨迹 |
| `online_buffer.pt` | online 轨迹 |
| `calql_critic.pt` | Cal-QL critic |
| `residual_sac.pt` | SAC 完整权重 |
| `residual_policy_weights.pt` | **Stage 2 必需** 的 residual actor |
| `rl_train_progress.json` | 续跑用梯度步数 / 轮次（每轮 SAC 后更新） |

### 当前 config 要点

- `finish_episode_before_round_stop: true` — 到 collect 预算后 **跑完当前 episode** 再 SAC
- `episode_reset_time_s: 15` — homing 之后额外 15s 摆场景
- `xi: 0.05`，`collect_env_steps_per_iter: 800`，`train_iters_per_collect: 50` → 约 100 轮 SAC（5000 步）

---

## Replay Buffer 管理

`offline_buffer.pt` 与 `online_buffer.pt` **独立**；体积过大时用 `compact`：

```bash
# 查看 episode
python src/lerobot/scripts/lerobot_filter_replay_buffer.py list \
  --path outputs/pld_stage1_pi05_stack_the_base_all/offline_buffer.pt

# 导出首尾帧 PNG 检查误判
python src/lerobot/scripts/lerobot_filter_replay_buffer.py preview \
  --path outputs/pld_stage1_pi05_stack_the_base_all/offline_buffer.pt \
  --output-dir outputs/pld_stage1_pi05_stack_the_base_all/buffer_preview

# 删除误判 episode（Trial N → episode_index N-1）
python src/lerobot/scripts/lerobot_filter_replay_buffer.py filter \
  --path outputs/pld_stage1_pi05_stack_the_base_all/offline_buffer.pt \
  --exclude 1,3 \
  --output outputs/pld_stage1_pi05_stack_the_base_all/offline_buffer_clean.pt

# 压缩 capacity（推荐 online 前）
python src/lerobot/scripts/lerobot_filter_replay_buffer.py compact \
  --path outputs/pld_stage1_pi05_stack_the_base_all/offline_buffer.pt \
  --capacity 5000 --optimize-memory \
  --output outputs/pld_stage1_pi05_stack_the_base_all/offline_buffer_compact.pt
```

估算：`offline_capacity ≈ n_success_trials × max_episode_steps × 1.2`；`online_capacity ≈ (rl_steps / train_iters_per_collect) × collect_env_steps_per_iter × 1.2`。

---

## Stage 2（hybrid 采集）

Config：`crp_pi05_stage2_stack_the_base.json`  
需要 Stage 1 的 `residual_policy_weights.pt`（或 `--pld.stage1_output_dir` 自动解析）。

```bash
HF_HUB_OFFLINE=1 lerobot-pld-stage2 \
  --config_path=examples/pld/crp_pi05_stage2_stack_the_base.json
```

- 每 episode 随机 `T_base`：前段仅 π_b，后段 π_b+π_δ（冻结）
- **只保存成功 episode** 到 LeRobotDataset
- 数据集目录：`outputs/pld_stage2_datasets/stack_the_base/`
- 统计：`outputs/pld_stage2_pi05_stack_the_base/collection_stats.json`

跳过采集（调试）：`--skip_collection=true`（**没有** `--skip_sft_train`；Stage2 本身不做 SFT）。

首次创建 dataset 时 `repo_id` 会带时间戳，训练前请确认：

```bash
cat outputs/pld_stage2_datasets/stack_the_base/meta/info.json | grep repo_id
```

---

## Stage 2 之后：Pi0.5 SFT（`lerobot-train`）

Stage2 产出标准 **LeRobotDataset**，可直接 fine-tune base policy。

Config：`pi05_stage2_sft_stack_the_base.json`

```bash
# 将 dataset.repo_id 改成 info.json 中的实际值（若带时间戳）
HF_HUB_OFFLINE=1 lerobot-train \
  --config_path=examples/pld/pi05_stage2_sft_stack_the_base.json
```

输出示例：`outputs/models/pi05_stack_the_base_pld_stage2_sft/`。

---

## 推荐完整流程（stack_the_base）

```text
1. reward-classifier: inspect → annotate → export → train
2. lerobot-pld-stage1（完整或分步续跑）
3. lerobot-pld-stage2（hybrid 采集）
4. lerobot-train（pi05_stage2_sft_stack_the_base.json）
5. deploy 新 checkpoint
```

## 配置文件

| 文件 | 用途 |
|------|------|
| `crp_pi05_stage1_stack_the_base.json` | Stage 1 |
| `crp_pi05_stage2_stack_the_base.json` | Stage 2 采集 |
| `pi05_stage2_sft_stack_the_base.json` | Stage 2 后 SFT |

## 已知限制

- CRP 碰撞保护后 **无自动恢复**，需人工复位 / 降 `xi` / 降 `interpolation_multiplier`
- `rl_train_progress.json` 与 `output_dir` 绑定；若要 **从零计数梯度步** 但保留权重，删该文件即可
- Stage2 的 `README.md` 里 SVG 来自 dataset 的 Hub 模板徽章，可忽略
- 真机 RL 耗时长，可先用 `--pld.rl_steps=500` pilot
