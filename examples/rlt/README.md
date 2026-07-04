# RL Token（RLT）Stage 1 — 训练 RL Token 与可选 VLA SFT

对应论文 [*RL Token: Bootstrapping Online RL with Vision-Language-Action Models*](https://arxiv.org/abs/2604.23073) 的 **Algorithm 1 第 1–3 步**：在预训练 VLA 上训练 RL token 重建模块，并可选地联合 fine-tune base policy。

本目录示例基于 **PI0.5** 作为 `base_policy`；RL token 本身是独立模块，通过 backbone 适配器可挂到其他 VLA（见「扩展 base policy」）。

> **说明**：当前实现仅覆盖 **Stage 1（离线适配训练）**。Stage 2 在线 Actor-Critic 尚未实现。

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
