# RL Token（RLT）Stage 1 — 训练 RL Token 与可选 VLA SFT

对应论文 [*RL Token: Bootstrapping Online RL with Vision-Language-Action Models*](https://arxiv.org/abs/2604.23073) 的 **Algorithm 1 第 1–3 步**：在预训练 VLA 上训练 RL token 重建模块，并可选地联合 fine-tune base policy。

本目录示例基于 **PI0.5** 作为 `base_policy`；RL token 本身是独立模块，通过 backbone 适配器可挂到其他 VLA（见「扩展 base policy」）。

> **说明**：本目录示例覆盖 **Stage 1（离线适配训练）** 与 **Stage 2（真机在线 RL）**。Stage 2 入口为 `lerobot-rlt-stage2`。

---

## 安装与环境

见仓库根目录；RLT Stage 1 至少需要 **dataset + training + pi** 三组 extra：

```bash
cd /path/to/lerobot
pip install -e ".[dataset,training,pi]"
```

| 依赖 extra | 用途 |
|------------|------|
| `dataset` | LeRobot 数据集加载、视频解码 |
| `training` | `lerobot-train`、`accelerate`、可选 `wandb` |
| `pi` | Pi0 / Pi0.5 所需的 `transformers` 等 |

**硬件**：Pi0.5 建议使用 **NVIDIA GPU**（示例为 `dtype=bfloat16`、`device=cuda`）。

**Hugging Face**（首次从 Hub 拉取 Pi0.5 权重时）：

```bash
huggingface-cli login
# 或 export HF_TOKEN=your_token
```

若 `base_policy.path` 已指向本地 `pretrained_model` 目录，可设 `HF_HUB_OFFLINE=1` 离线训练。

**验证安装**：

```bash
lerobot-train --help
python -c "from lerobot.policies.rl_token import RLTPolicy, get_registered_rlt_backbone_types; print(get_registered_rlt_backbone_types())"
# 期望输出包含 pi05
```

---

## 代码结构（简要）

```
src/lerobot/policies/rl_token/
├── rl_token.py           # RLTokenModel（encoder-decoder 瓶颈）
├── backbone.py           # 适配器协议 + 注册表
├── backbones/pi05.py     # Pi0.5 prefix embedding 提取
├── stage1.py             # L_ro + α·L_vla 训练逻辑
├── configuration_rlt.py  # RLTConfig
└── modeling_rlt.py       # RLTPolicy（组合 base_policy + rlt_module）
```

训练入口与普通 SFT 相同：**`lerobot-train`**。

---

## 前置条件

1. **LeRobot 格式演示数据集**：多相机图像、语言 instruction、state、action chunk 等（与 Pi0.5 SFT 相同）。即使 `rlt_alpha=0`，每个 batch 仍要过 frozen VLA 提取 prefix embedding。
2. **预训练 Pi0.5 checkpoint**（本地路径或 Hub）
3. 将示例 JSON 中的 **`dataset.root` / `dataset.repo_id`** 和 **`policy.base_policy.path`** 改成你的路径

---

## 两种训练模式

损失：**`L = L_ro + rlt_alpha × L_vla`**

| 模式 | 主控字段 | 行为 |
|------|----------|------|
| **仅训 RL Token** | `rlt_finetune_vla: false` | 冻结 base VLA，只更新 RL token |
| **联合 VLA SFT** | `rlt_finetune_vla: true` | 同时优化 RL token 与 Pi0.5 flow-matching |

| 文件 | 模式 |
|------|------|
| `rlt_pi05_token_only.json` | `rlt_finetune_vla=false` |
| `rlt_pi05_joint_sft.json` | `rlt_finetune_vla=true`（`rlt_alpha` 默认 `1.0`） |

**`rlt_alpha`**（论文 α）：`rlt_finetune_vla=false` 时自动置 `0.0`；联合 SFT 时省略则默认 `1.0`。

### 启动训练

```bash
# 仅 RL Token
lerobot-train --config_path=examples/rlt/rlt_pi05_token_only.json

# RL Token + VLA 联合 SFT
lerobot-train --config_path=examples/rlt/rlt_pi05_joint_sft.json
```

### 常用 CLI 覆盖

```bash
lerobot-train --config_path=examples/rlt/rlt_pi05_token_only.json \
  --dataset.root=/path/to/your/dataset \
  --dataset.repo_id=local/your_task \
  --policy.base_policy.path=/path/to/pi05/pretrained_model \
  --output_dir=outputs/models/my_rlt_run \
  --steps=20000 \
  --batch_size=4
```

---

## 配置字段

### 使用前必改

| 字段 | 说明 |
|------|------|
| `dataset.root` | 数据集根目录（含 `meta/`、`data/`） |
| `dataset.repo_id` | 逻辑名，与 `meta/info.json` 一致 |
| `policy.base_policy.path` | 预训练 Pi0.5 权重目录 |
| `output_dir` | checkpoint 输出目录 |

### 模式相关字段

| 字段 | token_only | joint_sft | 说明 |
|------|:----------:|:---------:|------|
| `rlt_finetune_vla` | ✓ | ✓ | 是否联合 SFT |
| `rlt_alpha` | — | ✓ | VLA 损失权重（默认 `1.0`） |
| `rlt_save_token_only` | ✓ | ✓ | checkpoint 只存 RL token（默认 `true`） |
| `rlt_num_tokens` / `rlt_num_layers` | ✓ | ✓ | token 数与层数（默认 `1` / `2`） |
| `rlt_embed_dim` / `rlt_input_dim` | ✓ | ✓ | 与 Pi0.5 hidden size 一致（`2048`） |
| `rlt_optimizer_lr` / `rlt_scheduler_*` | ✓ | — | 仅 token_only |
| `base_policy.dtype` / `gradient_checkpointing` | ✓ | ✓ | |
| `base_policy.freeze_vision_encoder` | — | ✓ | 示例默认 `true` |
| `base_policy.optimizer_lr` / `scheduler_*` | — | ✓ | VLA 学习率 |

联合 SFT 时 VLA 学习率走 **`base_policy.optimizer_lr`**；只训 RL token 时走 **`rlt_optimizer_lr`**。

#### Vision encoder（joint_sft）

| 设置 | 适用场景 |
|------|----------|
| `freeze_vision_encoder: true`（示例默认） | 已有任务微调 Pi0.5；主要调 language + action expert |
| `freeze_vision_encoder: false` | 对齐 openpi-RLT 全量 joint 微调 |
| `train_expert_only: true` | 只训 action expert，最省显存 |

### 与论文 / openpi-RLT 对齐

| 论文 / 参考 | 本实现 | 当前值 |
|-------------|--------|--------|
| 单个 RL token `e_rl` | `rlt_num_tokens` | `1` |
| encoder-decoder 层数 | `rlt_num_layers` | `2` |
| Pi0.5 hidden size | `rlt_embed_dim`, `rlt_input_dim` | `2048` |
| 损失 `L_ro + α·L_vla` | `rlt_loss` + `rlt_alpha` × `vla_loss` | — |
| 是否 fine-tune VLA | `rlt_finetune_vla` | — |

> **架构差异**：论文为 append `e_rl` + 自回归 decoder；本实现采用 **cross-attention** encoder-decoder（对齐 [openpi-RLT](https://github.com/Yyshadow/openpi-RLT)），损失形式与 Algorithm 1 一致。

---

## 训练日志与关注指标

| 日志键 | 论文符号 | 何时出现 | 关注什么 |
|--------|---------|----------|----------|
| `loss` | 总损失 | 始终 | 应整体下降 |
| `rlt_loss` / `rlt_mse` | **L_ro** | 始终 | 只训 token 时 **`rlt_mse` 应持续下降** |
| `vla_loss` | **L_vla** | `rlt_finetune_vla=true` | Pi0.5 flow-matching |
| `grad_norm` / `lr` | — | 始终 | 训练稳定性 |

建议开启 WandB（`wandb.enable: true`），否则终端只有总 `loss`。

---

## 输出与推理

- 默认 checkpoint（`rlt_save_token_only=true`）仅含 RL token encoder-decoder（MB 级）；`config.json` 中保留 `base_policy.path` 指向 frozen VLA。
- 加载时从 `base_policy.path` 读 VLA，再加载 RL token 权重。
- 提取 RL token：`RLTPolicy.extract_rl_token(batch)` → `[B, num_rl_tokens, embed_dim]`。
- 需保存 fine-tuned VLA 时设 `rlt_save_token_only=false`（体积很大）。

---

## 扩展 base policy

1. 在 base VLA 中提供 prefix 最终层 embedding 提取接口（Pi0.5 见 `PI05Pytorch.extract_prefix_embeddings`）。
2. 在 `policies/rl_token/backbones/` 下实现适配器，并用 `@register_rlt_backbone("your_type")` 注册。
3. 训练配置中设置 `base_policy.type`、`base_policy.path`、`rlt_input_dim` 与 base hidden size 一致。

当前已注册 backbone：`pi05`（`get_registered_rlt_backbone_types()` 查询）。

---

## 参考

- 论文：[RL Token (arXiv:2604.23073)](https://arxiv.org/abs/2604.23073)
- Pi0.5 文档：[`docs/source/pi05.mdx`](../../docs/source/pi05.mdx)
- 社区参考实现：[openpi-RLT](https://github.com/Yyshadow/openpi-RLT)

---

## Stage 2 — 真机在线 RL（TD3 + BC）

对应论文 **Algorithm 1 第 4–6 步**：在冻结的 RLT 增强型 VLA 之上，训练轻量级 chunked actor-critic（TD3 + Behavioral Cloning 正则），用真机交互数据在线微调动作 chunk。

### 代码结构

```
src/lerobot/
├── policies/rlt_actor/                      # 可训练 chunk actor（Gaussian, 固定 std）
│   ├── configuration_rlt_actor.py           # RLTActorConfig
│   └── modeling_rlt_actor.py                # ChunkActor / ChunkCriticEnsemble / RLTActorPolicy
├── rl/algorithms/rlt_td3/                   # TD3 + BC 算法
│   ├── configuration_rlt_td3.py             # RLTTD3AlgorithmConfig
│   └── rlt_td3_algorithm.py                 # chunked Bellman + BC + delta_penalty + warmup 调度
├── rollout/strategies/rlt_collect.py        # chunked 开环执行 + step-trace 收集
└── rlt/                                     # 编排
    ├── configs.py                           # RLTStage2Config
    ├── rlt_training_config.py               # 超参
    ├── replay_windows.py                    # step-trace → replay 窗口（纯逻辑，可单测）
    ├── policy_setup.py                      # 从 Stage-1 checkpoint 构建冻结 RLT + actor
    └── orchestrator.py                      # warmup → 异步 collect/train → eval
```

### 前置条件

1. **Stage-1 RLT checkpoint**（`lerobot-train --config_path=examples/rlt/rlt_pi05_token_only.json` 的输出，`config.json` + RL token 权重，并引用 frozen base VLA）。
2. **真机 + 相机**（示例用 `crp_arm` + 两个 OpenCV 相机）。
3. **奖励分类器**（成功检测；可启用 `manual_fallback` 用 `s`/`f` 键人工兜底）。

### 架构要点（与论文 / openpi-RLT 对齐）

- **冻结的 "Machine A"**：base VLA + RL token 在 **chunk 边界** in-process 运行，产出 `(z_rl, ã_{1:C})`。不使用 RTC 推理引擎——RLT 决策每 `C` tick 一次，延迟预算充足，且能直接拿到完整 reference chunk 与 `z_rl`，避免 VLA 重复计算。（说明：论文原文并未提及 RTC，这是 lerobot 自身的推理引擎概念；此处不使用是实现取舍，非论文要求。）
- **异步 rollout / learning（论文 §IV-B 原文："we perform the rollouts and learning asynchronously"）**：`async_training=true` 时，orchestrator 起两个线程——learner 线程在 GPU 上持续跑 TD3 更新，rollout 线程在机器人上持续采集 episode；actor 权重经线程安全的 `_WeightSlot`（learner 发布 CPU clone、rollout 拉取）单向流动，rollout 线程持有 actor 的独立 deepcopy（`rollout_actor`）避免与 learner 的 in-place 更新竞争；replay buffer 自带锁，add/sample 线程安全。`async_training=false` 时回退为顺序 collect→train 交替（PLD 风格，便于调试）。
- **可训练 actor**：以 `(z_rl, s_p, ã)` 为输入，固定 std 的高斯策略输出 refined chunk；训练时对 `ã` 做 **reference dropout**（论文 0.5）。
- **critic**：`Q(z_rl, s_p, a_{1:C})` 的 MLP ensemble（TD3 clipped double-Q），由算法持有，rollout 不需要。
- **TD3**：clipped double-Q Bellman backup、延迟策略更新、软目标 EMA；无熵正则。
- **BC 正则**：按 `source` 选 BC target——`BASE` 用 VLA reference chunk，`HUMAN`/`MIXED` 用人工动作 chunk，`RL` 不施加 BC；`bc_weight` 在 `bc_decay_steps` 内从 `bc_weight_max` 线性衰减到 `bc_weight_min`。
- **delta_penalty**（可选）：chunk 内相邻步动作差的平方，鼓励平滑。
- **replay 窗口**：每 episode 记录原始 per-tick step trace，episode 末切分为 replay 窗口。`replay_stride=0` 为 chunk-boundary 模式（每边界一窗）；`>0` 为 dense 模式（每 `stride` 步一窗，`z_rl`/`ref` 由所属 chunk 边界携带）。论文 §IV-B "Subsampling Action Chunks" 默认 `stride=2`。
- **warmup 门控**：buffer 达到 `warmup_min_buffer_size` 前只执行 VLA reference（`source=BASE`）；可选 `warmup_pretraining_updates` 做 BC-only 预训练。warmup 仍为顺序（需要机器人、尚无 actor 在线）。

### 启动

```bash
lerobot-rlt-stage2 --config_path=examples/rlt/rlt_pi05_stage2_insert_the_blue_tube.json
```

### 常用 CLI 覆盖

```bash
lerobot-rlt-stage2 --config_path=examples/rlt/rlt_pi05_stage2_insert_the_blue_tube.json \
  --rlt_checkpoint_path=outputs/models/rlt_pi05_token_only_xxx/checkpoints/last/pretrained_model \
  --training.rl_steps=10000 \
  --training.bc_weight_max=1.0 \
  --training.bc_decay_steps=20000 \
  --chunk_len=10 \
  --output_dir=outputs/rlt_stage2_xxx
```

### 仅评估

```bash
lerobot-rlt-stage2 --config_path=examples/rlt/rlt_pi05_stage2_insert_the_blue_tube.json \
  --eval_only=true \
  --training.resume_actor_checkpoint=outputs/rlt_stage2_xxx/actor_weights.pt
```

### 输出

| 文件 | 内容 |
|------|------|
| `online_buffer.pt` | 在线 replay buffer |
| `rlt_td3.pt` | 算法持有的 critic / target 网络 |
| `actor_weights.pt` | 可训练 actor 权重（`get_weights()`） |
| `rlt_actor/` | `RLTActorPolicy` checkpoint（仅 actor；`config.json` 引用 Stage-1 RLT） |
| `rlt_stage2_progress.json` | 训练进度（train_steps / collect_round） |

### 配置字段

| 字段 | 默认 | 说明 |
|------|------|------|
| `rlt_checkpoint_path` | — | Stage-1 RLT checkpoint 目录（必填） |
| `chunk_len` | `10` | 动作 chunk 长度 C |
| `proprio_keys` | `[]` | 本体感觉键（空则自动取 arm `.pos`） |
| `training.warmup_min_buffer_size` | `200` | actor 上线前最小 replay 窗口数 |
| `training.replay_stride` | `2` | 0=chunk-boundary；>0=dense（论文 §IV-B: stride=2） |
| `training.async_training` | `true` | 异步 rollout/learning（论文 §IV-B）；false=顺序交替 |
| `training.utd_ratio` | `5` | update-to-data ratio（论文 §IV-B: 5） |
| `training.checkpoint_every_steps` | `500` | 异步训练中周期性存盘间隔（grad steps） |
| `training.bc_weight_max` / `bc_decay_steps` | `1.0` / `10000` | BC 权重上限与线性衰减步数 |
| `training.ref_dropout_prob` | `0.5` | reference-action dropout |
| `training.policy_update_freq` | `2` | TD3 延迟策略更新频率 |
| `training.delta_penalty_weight` | `0.0` | chunk 平滑惩罚权重 |

### 与论文 / openpi-RLT 对齐

| 论文 / 参考 | 本实现 | 默认值 |
|-------------|--------|--------|
| chunk length C | `chunk_len` | `10` |
| 固定 std 高斯 actor | `ChunkActor`（`fixed_std`） | `0.1` |
| reference dropout | `ref_dropout_prob` | `0.5` |
| TD3 clipped double-Q | `ChunkCriticEnsemble` + min | `num_critics=2` |
| two critic updates per actor update | `policy_update_freq` | `2` |
| update-to-data ratio = 5 | `utd_ratio` | `5` |
| rollout/learning 异步 | `async_training` + `AsyncActorLearnerRunner`（双线程） | `true` |
| chunk subsampling stride=2 | `replay_stride` | `2` |
| BC 正则（Eq. 5） | 按 `source` 选 BC target | `bc_weight_max=1.0` |
| warmup（base-only） | `warmup_min_buffer_size` 门控 | `200` |
| `source` / `source_chunk` 标签 | `SOURCE_BASE/RL/HUMAN/MIXED` | — |
| step-trace → window | `replay_windows.build_replay_windows` | stride=2 |

> **限制**：当前 `_read_intervention_action` 为最小实现（仅当 teleop 连接时返回 None）；如需 DAgger 式人工纠错采集，可在此钩子接入 correction key + teleop action。真机部署相关集成（base VLA 直接构造、processor 复用）需在目标硬件上做端到端联调。

## 共享核心：lerobot ↔ RLinf 双框架协同（路径 1）

为避免在 lerobot（真机）和 RLinf（大规模仿真）两边各实现一遍 RLT 算法，RLT Stage-2 的**框架无关核心**被抽取为单一来源，由两个框架共同复用：

- **网络**：`ChunkActor`、`ChunkCriticEnsemble`（纯 `nn.Module`，`lerobot.policies.rlt_actor`）。
- **损失/目标数学**：`td3_critic_loss`、`rlt_actor_loss`、`polyak_update`、`bc_weight_schedule`、`ref_dropout_mask`（`lerobot.rl.algorithms.rlt_td3.losses`）——纯自由函数，只接受 `nn.Module` + plain `fb` dict，不依赖 `RLAlgorithm`/`PreTrainedPolicy`/硬件/Ray。
- **Replay 逻辑**：`StepRecord`、`ReplayWindow`、`build_replay_windows`、`SOURCE_*`（`lerobot.rlt.replay_windows`）。
- **统一入口**：`lerobot.rlt.shared` 汇集上述全部符号，并提供确定性 conformance fixture `make_rlt_tiny_fixture()` 与 `golden_loss_values()`。

`RLTTD3Algorithm`（lerobot 真机 learner）与 `rlinf.rlt.RLTTD3Driver`（RLinf 仿真 learner）**调用同一份损失函数**，因此数学上按构造等价；网络为同一类、同一初始化，checkpoint 可在两框架间直接 `load_state_dict` 互转——实现「仿真训练 → 真机部署」与「双框架协同作业」且无重实现漂移风险。

### 测试

- lerobot 侧单元测试：
  ```bash
  conda run -n zyy_lerobot python -m pytest tests/rl/test_rlt_shared_core.py -q
  ```
- RLinf 侧 conformance 测试（证明在 RLinf 环境内导入并运行共享核心，损失值与 lerobot `golden_loss_values()` bit 一致；driver 跑通 TD3 外层步；actor 权重 `torch.save`/`load` 往返；chunked rollout 产出 lerobot 兼容的 `ReplayWindow`）：
  ```bash
  cd RLinf && .venv/bin/python -m pytest tests/unit_tests/test_rlt_shared_core_conformance.py -q
  ```

### 环境要点

- lerobot fork 需在 RLinf 的 `.venv`（**Python 3.12+**，`install.sh` 默认 `3.12.3`）中以 editable 安装。`requirements/install.sh` 的 `install_lerobot()` 会优先检测 sibling `../lerobot` 或环境变量 `LEROBOT_PATH`，并以 `--no-deps -e` 安装 fork，再调用 `install_lerobot_runtime_deps` 补齐轻量依赖（规避 fork `pyproject` 中 `torch>=2.7` 与 RLinf `torch 2.6` 的冲突）。
- 全新环境：`bash requirements/install.sh embodied --model openpi --env isaaclab`（可选 `export LEROBOT_PATH=/path/to/lerobot`）。
- 后续集成计划见 RLinf 仓库 `rlinf/rlt/ROADMAP.md`（不在此重复 Phase 2 细节）。

