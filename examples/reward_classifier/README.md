# Reward Classifier（stack_the_base）

真机 PLD 用的视觉成功/失败分类器。与 PLD **独立**：先完成本节，再在 [../pld/README.md](../pld/README.md) 跑 Stage 1/2。

## 前置条件

- 已安装 lerobot（在仓库根目录）：

```bash
cd /home/dingyuxuan/zyy/lerobot
pip install -e ".[dataset,training]"
```

- 源演示数据集（只读，不会被修改）：
  - 示例路径：`/media/dingyuxuan/My Passport/stack_the_base_all`
  - 在 `stack_the_base_inspect.json` / `annotate.json` / `export.json` 的 `source.root` 中改成你的路径

## 流程概览

```text
inspect → annotate → export → train → 链到 PLD config
```

## Step 0：检查源数据集

确认源数据含 `observation.images.top` / `wrist` 等字段。

```bash
lerobot-reward-classifier --config_path=examples/reward_classifier/stack_the_base_inspect.json
```

## Step 1：交互式标注

需要 `DISPLAY`（图形界面）。标注写入 `output.annotation_path`。

```bash
lerobot-reward-classifier --config_path=examples/reward_classifier/stack_the_base_annotate.json
```

| 键 | 作用 |
|----|------|
| `[` / `]` | 成功段起点 / 终点 |
| `f` / `g` | 失败段起点 / 终点 |
| Space | 播放 / 暂停 |
| `a` / `d` | 上一帧 / 下一帧 |
| `n` / `p` | 下一集 / 上一集 |
| `u` | 撤销 |
| `q` | 保存并退出 |

## Step 2：导出 train + eval 数据集

默认 20% hold-out 验证集。

```bash
lerobot-reward-classifier --config_path=examples/reward_classifier/stack_the_base_export.json
```

| 输出 | 路径（见 `stack_the_base_export.json` → `output`） |
|------|-----------------------------------------------------|
| 标注 JSON | `outputs/reward_classifier/annotations/stack_the_base.json` |
| 训练集 | `outputs/reward_classifier/datasets/stack_the_base_labeled` |
| 验证集 | `outputs/reward_classifier/datasets/stack_the_base_labeled_eval` |

### 导出进阶选项（`stack_the_base_export.json`）

- `class_ratio: [1, 5]` — 正负样本 1:5，减轻误报
- `camera_keys` — 显式导出双相机
- `image_preprocessing` — 导出时 crop + resize 到 128×128，省磁盘
- `use_videos: false` — 单帧 PNG，避免每帧 AV1 编码

## Step 3：训练分类器

```bash
lerobot-train --config_path=examples/reward_classifier/stack_the_base_train.json
```

- 日志会打印 `Eval accuracy: xx.xx%`
- 指标：`outputs/reward_classifier/models/stack_the_base/eval_accuracy.json`
- Checkpoint 示例：`outputs/reward_classifier/models/stack_the_base/checkpoints/003000/pretrained_model`

## Step 4：接到 PLD

**方式 A（软链接，推荐）**

```bash
ln -sf "$(pwd)/outputs/reward_classifier/models/stack_the_base/checkpoints/003000/pretrained_model" \
       outputs/reward_classifier/models/stack_the_base/pretrained_model
```

**方式 B（CLI 覆盖）**

```bash
--reward_classifier.path=outputs/reward_classifier/models/stack_the_base/checkpoints/003000/pretrained_model
```

PLD config 里 `reward_classifier.path` 需指向上述目录；`success_threshold` 建议与验证集表现一致（当前 stack_the_base 用 `0.99`）。

## 配置文件

| 文件 | 用途 |
|------|------|
| `stack_the_base_inspect.json` | 检查源数据 |
| `stack_the_base_annotate.json` | 交互标注 |
| `stack_the_base_export.json` | 导出 labeled 数据集 |
| `stack_the_base_train.json` | `lerobot-train` 训练 |

修改训练/导出路径：编辑对应 json 的 `output.*`、`dataset.root`、`output_dir`。

## 与 PLD 的关系

- **Offline / Online collect**：`RewardClassifierDetector` 每步推理；`manual_fallback=true` 时可用 `s`/`f` 手动标成功/失败
- **Stage 2 hybrid**：同样用该分类器判定 episode 是否保存

下一步 → [PLD Stage 1](../pld/README.md)
