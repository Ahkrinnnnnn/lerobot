# 手眼 / 场景标定

用 **ChArUco 视觉链式标定** 得到：

1. 各相机**内参**
2. 各相机在**机器人坐标系**下的变换矩阵
3. 桌面 ChArUco **角点网格**在机器人系下的坐标（mm）

全程交互式 CLI，每步结果增量写入同一个 NPZ 文件。

---

## 原理简述

```
腕部相机 (eye-in-hand)          固定 top 相机 (eye-to-hand)
        │                                │
        ├─ hand_eye → T_ee→wrist         │
        ├─ landmark_map → T_robot→table  │  (可选，要桌面网格时用)
        │         (同一块桌面 ChArUco)    │
        └─ top_extrinsic_dual ──────────┴→ T_robot→top
              (手持板，wrist+top 同时看到)
```

- **hand_eye**：桌面 ChArUco 不动，移动机械臂，求腕部相机相对末端的外参 `T_ee_to_camera`。
- **top_extrinsic_dual**（**推荐标 top**）：手持 ChArUco，每次 save 同时采 wrist + top；用 wrist 手眼链 + top PnP 融合 `T_robot_to_camera`。**不需要** `landmark_map`。
- **landmark_map**（可选）：桌面同一块板，腕部多视角融合 `T_robot_to_table` 与角点网格。
- **camera_via_landmark**（备选）：top 固定看桌面板，需先跑 `landmark_map`；工作台布局固定、top 能俯视桌面时可用。

---

## 标定前准备

### 1. 打印并固定 ChArUco 板

按 `charuco_board.json` 的参数打印（当前：**11×8 格，checker 20 mm，marker 14.67 mm，DICT_4X4_50**），贴在工作台可见区域：

- 贴牢、尽量平，后续 **hand_eye / landmark_map / camera_via_landmark 共用同一块板**
- 板子不要被夹爪、线缆长期遮挡
- 打印尺寸必须与 JSON 一致，否则 PnP 尺度会错

### 2. 编辑 `crp.json`

| 字段 | 说明 |
|------|------|
| `robot.port` | CRP 控制器 IP |
| `robot.cameras.top.index_or_path` | 固定相机 OpenCV 序号 |
| `robot.cameras.wrist.index_or_path` | 腕部相机序号 |
| `board_config_path` | 指向 `charuco_board.json` |
| `robot_pose_frame` | 与示教器读位姿一致：`world` 或 `user`（当前默认 `world`） |

**示教器建议（与 `robot_pose_frame` 一致）**

- 坐标系：**世界坐标** → `crp.json` 里设 `"robot_pose_frame": "world"`
- 工具 TCP：**设在夹爪 TCP 上**（代码读到的 `T_robot_to_ee` 即该 TCP 在世界/用户系下的位姿）
- 手眼求的是 **`T_ee_to_camera`**（TCP/末端 → 腕部相机），相机不在 TCP 上也没关系，只要相机与法兰刚性固定
- `world` / `user` 只影响绝对位姿；**相对运动**一致即可，但请全程不要改 frame

### 3. CRP 位姿方向：`CRP_POSE_INVERT=1`（CRP 必看）

CRP SDK 的 `read_end_pose_world` / `read_end_pose_user` 返回 `[X, Y, Z, Rx, Ry, Rz]`（mm + deg），但文档**未写明**这 6 个数对应哪方向的 4×4 齐次矩阵。

本仓库按默认 RPY（`xyz` 外旋）把它们拼成矩阵 `T` 时，在真机上验证结果是：**`T` 等价于 `T_ee_to_robot`（末端/TCP → 机器人基座/世界）**，而手眼标定链需要的是 **`T_robot_to_ee`**（基座 → 末端，即 OpenCV `gripper2base`）。二者互为逆矩阵；方向用反时 `consistency_std_mm` 可达上百 mm，PnP reproj 仍可 <1 px。

因此 **CRP 手眼标定请始终设置**：

```bash
export CRP_POSE_INVERT=1
```

或在单条命令前加前缀：

```bash
CRP_POSE_INVERT=1 lerobot-calibrate-hand-eye --config_path=examples/hand_eye_calibration/crp.json \
  --phase=hand_eye --camera=wrist --mount=eye_in_hand
```

**作用位置**（环境变量，不是 `crp.json` 字段）：`src/lerobot/calibration/transforms.py` 的 `transform_from_xyz_rpy_deg()` 在拼好 4×4 后取逆。凡把 CRP 位姿变成 `T_robot_to_ee` 的路径都会受影响，包括：

| 阶段 | 是否受 `CRP_POSE_INVERT` 影响 |
|------|--------------------------------|
| `hand_eye` / `intrinsics_and_hand_eye` | ✅ |
| `landmark_map` / `top_extrinsic_dual` / `validate` | ✅ |
| `intrinsics`（不读 pose）/ `camera_via_landmark` | ❌ |

同一实验文件夹内 **`hand_eye` / `top_extrinsic_dual` / `landmark_map` / `validate` 须全程用相同的 `CRP_POSE_INVERT`**（建议一直 `export CRP_POSE_INVERT=1`）。Debug 日志里的 `robot CRP xyz=... rpy=...` 仍是 SDK 原始 6D，仅显示用；参与求解的是取逆后的 `T_robot_to_ee`。

修对后 `consistency_std_mm` / `top_extrinsic_std_mm` 通常在 **数 mm 量级**（视采集与刚性而定）；仍偏高时再看 TCP、停稳、以及 top 阶段是否也设了 invert。

### 4. 实验文件夹（所有阶段共用）

默认实验目录：

```text
outputs/hand_eye_calibration/lab_arm_01/current/
```

| 路径 | 内容 |
|------|------|
| `calibration.npz` | 汇总标定（各相机 K/dist、外参、桌面网格） |
| `intrinsics/wrist.json` | **canonical 内参**（拆分流程时 hand_eye 读此文件） |
| `intrinsics/top.json` | top 内参 |
| `phases/intrinsics_wrist/` | 内参阶段过程文件（images、report、log） |
| `phases/hand_eye_wrist/` | 手眼阶段过程文件 |
| `phases/intrinsics_and_hand_eye_wrist/` | 合并阶段 |
| `archive/` | 同一 phase 重跑时，旧结果自动移入此处 |
| `experiment_meta.json` | 各 phase 运行记录 |

**多次实验**：改 `outputs.experiment` 名称，或 CLI：

```bash
# 新建 exp_YYYYMMDD_HHMMSS 文件夹
lerobot-calibrate-hand-eye --config_path=... --new_experiment=true ...

# 指定实验名（在 base_dir/robot_id/ 下）
lerobot-calibrate-hand-eye --config_path=... --experiment=run_tcp_world ...
```

**拆分内参 + 手眼**（同一 `experiment` 下顺序执行）：

```bash
# 1. 写 intrinsics/wrist.json + phases/intrinsics_wrist/
lerobot-calibrate-hand-eye ... --phase=intrinsics --camera=wrist --mount=eye_in_hand

# 2. 自动读 intrinsics/wrist.json 做 PnP
lerobot-calibrate-hand-eye ... --phase=hand_eye --camera=wrist --mount=eye_in_hand
```

### 5. 确认相机能出图

机械臂上电、相机 `index_or_path` 正确。有图形界面时会弹出检测预览窗口（绿=检测到 ChArUco，红=未检测到）。

---

## 配置文件

### `crp.json` — 机器人 + 输出路径

```json
{
  "robot": { "type": "crp_arm", "id": "lab_arm_01", "port": "...", "cameras": { ... } },
  "board_config_path": "examples/hand_eye_calibration/charuco_board.json",
  "robot_pose_frame": "world",
  "outputs": {
    "base_dir": "outputs/hand_eye_calibration",
    "experiment": "current"
  }
}
```

### `charuco_board.json` — 标定板（与实物一致）

```json
{
  "squares_x": 11,
  "squares_y": 8,
  "square_size_mm": 20.0,
  "marker_size_mm": 14.67,
  "aruco_dict": "DICT_4X4_50"
}
```

换板子只改此文件；`squares_x/y` 为 OpenCV ChArUco 的**方格数**（不是内角点数）。

---

## 命令速查

```bash
# 激活含 lerobot / OpenCV 的 Python 环境后：
cd /path/to/lerobot_handeye
export CRP_POSE_INVERT=1   # CRP 全程必需
```

共用参数：`--config_path=examples/hand_eye_calibration/crp.json`  
实验目录：`outputs/hand_eye_calibration/<robot_id>/<experiment>/`（见 `outputs.experiment`）  
建议调试：`--debug=true`（默认已开）

---

### 方案 A：全流程（推荐 wrist 内参 + 手眼一步完成）

**一次采集** → 用**同一批图**先标内参、再标手眼，PnP 与 hand-eye 不会出现「内参来自另一次采图」的偏差。

```bash
export CRP_POSE_INVERT=1

# ① wrist 内参 + 手眼（桌面板固定，≥12 次 save，含旋转）
lerobot-calibrate-hand-eye --config_path=examples/hand_eye_calibration/crp.json \
  --phase=intrinsics_and_hand_eye --camera=wrist --mount=eye_in_hand --debug=true

# ② top 内参（板可手持多视角，≥3 张；不读 pose，可不依赖 invert）
lerobot-calibrate-hand-eye --config_path=examples/hand_eye_calibration/crp.json \
  --phase=intrinsics --camera=top --mount=eye_to_hand

# ③ top 外参（手持板，wrist+top 双相机，≥6 次 save；须与 ① 相同 invert）
lerobot-calibrate-hand-eye --config_path=examples/hand_eye_calibration/crp.json \
  --phase=top_extrinsic_dual --camera=top --mount=eye_to_hand

# ④（可选）桌面系 + 角点网格 — 仅当需要 T_robot_to_table 时
# lerobot-calibrate-hand-eye --config_path=... --phase=landmark_map --camera=wrist

# ⑤ 校验 NPZ
lerobot-calibrate-hand-eye --config_path=examples/hand_eye_calibration/crp.json \
  --phase=validate
```

合并阶段若配置了 `top` 相机会同步采 top 图并尝试求 `T_robot_to_top`；不需要则加 `--also_collect_top=false`。

---

### 方案 B：分步（内参、外参分开采）

**外参阶段用哪套内参？**  
同一实验文件夹内：`--phase=hand_eye` 优先读 **`intrinsics/{camera}.json`**，其次 `calibration.npz`。

| 阶段 | 内参来源 |
|------|----------|
| `intrinsics` | 本次图 → 写 `intrinsics/{camera}.json` + `calibration.npz` |
| `hand_eye`（拆分） | **同实验** `intrinsics/{camera}.json` |
| `intrinsics_and_hand_eye` | 同一次 save 的图现场标 K |

分步命令（**共用同一 `outputs.experiment`，且全程 `export CRP_POSE_INVERT=1`**）：

```bash
export CRP_POSE_INVERT=1

# 1 top 内参
lerobot-calibrate-hand-eye --config_path=examples/hand_eye_calibration/crp.json \
  --phase=intrinsics --camera=top --mount=eye_to_hand

# 2 wrist 内参（板可手持；与 hand_eye 是否同一批图无关，但分辨率/焦距勿变）
lerobot-calibrate-hand-eye --config_path=examples/hand_eye_calibration/crp.json \
  --phase=intrinsics --camera=wrist --mount=eye_in_hand

# 3 wrist 手眼（桌面板固定；用步骤 2 写入的 wrist K/dist 做 PnP）
lerobot-calibrate-hand-eye --config_path=examples/hand_eye_calibration/crp.json \
  --phase=hand_eye --camera=wrist --mount=eye_in_hand --debug=true

# 4 top 外参（双相机手持板）+ 5 validate
lerobot-calibrate-hand-eye --config_path=examples/hand_eye_calibration/crp.json \
  --phase=top_extrinsic_dual --camera=top --mount=eye_to_hand
lerobot-calibrate-hand-eye --config_path=examples/hand_eye_calibration/crp.json \
  --phase=validate
```

**注意**

- 各步共用同一 `outputs.experiment`（或 `--experiment_dir`）
- 重跑同一 phase 时，旧 `phases/...` 会自动移入 `archive/`
- 新实验用 `--new_experiment=true` 或改 `outputs.experiment`

---

## 标定流程（分步说明）

所有命令共用同一份 `--config_path=examples/hand_eye_calibration/crp.json`，只改 `--phase` 等参数。

**交互约定**（各阶段相同）：

- **s**（或 Space / Enter）：保存样本（**终端或预览窗口**均可）
- **q**：结束本阶段并求解
- 在示教器上**手动拖动**机械臂改变姿态，再按 s 保存
- 保存时会 **阻塞采新图 + 立即读位姿**（不再使用预览缓冲帧），终端会打印 `sync: frame=…ms pose=…ms`

**过程文件**（图像、可视化、`report.json`、`calibration.log`）写入当前实验目录：

```text
outputs/hand_eye_calibration/<robot_id>/<experiment>/phases/<phase>_<camera>/
```

实验根目录见 `crp.json` → `outputs.base_dir` + `outputs.experiment`；CLI 覆盖：`--experiment=名称` / `--experiment_dir=路径` / `--new_experiment=true`

---

### 推荐：wrist 内参 + 手眼（一步完成）

**目的**：同一次采图同时求 `wrist__camera_matrix` / `dist_coeffs` 和 `wrist__T_ee_to_camera`，避免内外参样本不一致。

**操作**：ChArUco **固定在桌面**；示教器拖臂，多姿态保存（默认至少 **12** 组，建议 15–25）。

```bash
export CRP_POSE_INVERT=1
lerobot-calibrate-hand-eye --config_path=examples/hand_eye_calibration/crp.json \
  --phase=intrinsics_and_hand_eye --camera=wrist --mount=eye_in_hand
```

**写入 NPZ**：`wrist__camera_matrix`、`wrist__dist_coeffs`、`wrist__T_ee_to_camera`。

若机器人配置中有 `top` 相机（默认 `--also_collect_top=true`），每次保存会**同步采集 top 图像**，并弹出 **第二个预览窗口**（`top_calib_top`）显示 top 视角的 ChArUco 检测（绿=检测到）。两台相机都检测到板子才能保存。

关闭 top 同步采集：`--also_collect_top=false`

---

### 步骤 1：top 相机内参

**目的**：求 top 的 `camera_matrix`、`dist_coeffs`。

**操作**：手持或临时放置 ChArUco，在 top 视野内**不同距离、角度**移动（至少 3 张，建议 15–25 张）。

```bash
lerobot-calibrate-hand-eye --config_path=examples/hand_eye_calibration/crp.json \
  --phase=intrinsics --camera=top --mount=eye_to_hand
```

**写入 NPZ**：`top__camera_matrix`、`top__dist_coeffs`、`top__width/height` 等。

---

### 步骤 2–3（可选拆分）：wrist 内参 + 手眼外参

若未使用上面的合并阶段，可分开执行步骤 2（内参）和步骤 3（手眼）。**手眼阶段同样仅在保存时同步采图+读位姿。**

#### 步骤 2：wrist 相机内参

**目的**：求 wrist 内参（同上，换腕部相机采图）。

**操作**：用腕部相机看 ChArUco，多姿态采图。

```bash
lerobot-calibrate-hand-eye --config_path=examples/hand_eye_calibration/crp.json \
  --phase=intrinsics --camera=wrist --mount=eye_in_hand
```

**写入 NPZ**：`wrist__camera_matrix`、`wrist__dist_coeffs` 等。

---

#### 步骤 3：wrist 手眼外参

**目的**：求 `T_ee_to_camera`（腕部相机相对末端）。

**前提**：步骤 2 已完成；**ChArUco 固定在桌面**（与后续步骤同一块板）。

**操作**：在示教器上手动拖动机械臂，让 wrist 从多个角度看清桌面板（默认至少 **12** 组样本）。姿态要分散：不同高度、偏航、俯仰，避免全在同一平面。

```bash
lerobot-calibrate-hand-eye --config_path=examples/hand_eye_calibration/crp.json \
  --phase=hand_eye --camera=wrist --mount=eye_in_hand
```

**写入 NPZ**：`wrist__T_ee_to_camera`。

**可调**：`--min_hand_eye_samples=15`

---

### 步骤 4：top 相机外参（双相机 + 手持板，推荐）

**目的**：求 `T_robot_to_camera`（top 在机器人系下）。

**前提**：

- wrist **手眼**已完成（NPZ 里有 `wrist__T_ee_to_camera`）
- wrist、top **内参**已标定
- **不需要** `landmark_map`，板子可手持

**操作**：举着 ChArUco，让 **wrist 和 top 同时看到板子**（两个预览窗口都变绿再 save）。拖动机械臂换姿态，默认至少 **6** 次 save。各次之间板子位置可以变。

```bash
CRP_POSE_INVERT=1 lerobot-calibrate-hand-eye --config_path=examples/hand_eye_calibration/crp.json \
  --phase=top_extrinsic_dual --camera=top --mount=eye_to_hand
```

**原理**：每次 save 用 wrist 图 + 位姿 + 手眼外参得到 `T_robot_to_board`，top 图 PnP 得 `T_board_to_top`，合成 `T_robot_to_top`；多帧融合。

**写入 NPZ**：`top__T_robot_to_camera`

**可调**：`--min_landmark_samples=8`（本阶段复用该参数作为最少双相机样本数）

---

### 步骤 4b（可选）：桌面坐标系 + 角点网格

仅当需要 `T_robot_to_table` / `table_grid_points_robot_mm`（例如仿真桌面网格）时执行。桌面板需固定不动。

```bash
lerobot-calibrate-hand-eye --config_path=examples/hand_eye_calibration/crp.json \
  --phase=landmark_map --camera=wrist
```

---

### 步骤 4c（备选）：top 外参 via 固定桌面板

需先完成 **landmark_map**；top 相机固定且能俯视桌面 ChArUco。不适合 top 视野难覆盖工作台的情况。

```bash
lerobot-calibrate-hand-eye --config_path=examples/hand_eye_calibration/crp.json \
  --phase=camera_via_landmark --camera=top --mount=eye_to_hand
```

---

### 步骤 5：验证并导出

**目的**：检查 NPZ 是否齐全，写入当前位姿下的合成外参 `*_T_robot_to_camera_resolved`（含 wrist）。

```bash
lerobot-calibrate-hand-eye --config_path=examples/hand_eye_calibration/crp.json \
  --phase=validate
```

终端会打印 NPZ 路径及各数组的 `shape`；确认存在 top/wrist 内参、外参、`T_robot_to_table`、`table_grid_points_robot_mm` 即可用于仿真或 VLA 对齐。

---

## 输出 NPZ 字段

路径：`~/.cache/huggingface/lerobot/calibration/robots/crp_arm/lab_arm_01_calibration.npz`

| 数组名 | 形状 / 含义 |
|--------|-------------|
| `{cam}__camera_matrix` | (3, 3) 内参 K |
| `{cam}__dist_coeffs` | (5,) 畸变 |
| `{cam}__width`, `{cam}__height` | 分辨率 |
| `top__T_robot_to_camera` | (4, 4) top → 机器人系 |
| `wrist__T_ee_to_camera` | (4, 4) wrist → 末端 |
| `{cam}__T_robot_to_camera_resolved` | (4, 4) validate 时当前臂姿下的合成外参 |
| `T_robot_to_table` | (4, 4) 桌面系 → 机器人系 |
| `table_grid_points_robot_mm` | (N, 3) 桌面角点网格，单位 mm |
| `charuco_squares_x` 等 | 标定板参数 |

### Python 读取

```python
import numpy as np

path = "~/.cache/huggingface/lerobot/calibration/robots/crp_arm/lab_arm_01_calibration.npz"
data = np.load(path, allow_pickle=True)

K_top = data["top__camera_matrix"]
dist_top = data["top__dist_coeffs"]
T_robot_top = data["top__T_robot_to_camera"]

K_wrist = data["wrist__camera_matrix"]
T_ee_wrist = data["wrist__T_ee_to_camera"]

T_robot_table = data["T_robot_to_table"]
grid_robot_mm = data["table_grid_points_robot_mm"]  # (N, 3)；11×8 方格 → 内角点 10×7=70
```

---

## 常用 CLI 参数

| 参数 | 说明 |
|------|------|
| `--config_path` | 主配置（机器人 + board_config_path） |
| `--phase` | `intrinsics` / `hand_eye` / `intrinsics_and_hand_eye` / `top_extrinsic_dual` / `landmark_map` / `camera_via_landmark` / `validate` |
| `--robot_pose_frame` | 覆盖 `crp.json`：`world` 或 `user` |
| `--camera` | `top` 或 `wrist` |
| `--mount` | `eye_to_hand`（top）或 `eye_in_hand`（wrist）；内参 / 手眼 / top_extrinsic_dual / camera_via_landmark 必填 |
| `--wrist_camera` | 双相机标 top 时腕部相机名（默认 `wrist`） |
| `--experiment` | 覆盖 `outputs.experiment`（实验子目录名） |
| `--new_experiment` | 新建 `exp_YYYYMMDD_HHMMSS` 实验文件夹 |
| `--experiment_dir` | 完整实验目录（覆盖 base_dir/experiment） |
| `--output_path` | 覆盖 `calibration.npz` 路径 |
| `--min_hand_eye_samples` | 手眼最少样本数（默认 12） |
| `--min_landmark_samples` | landmark 最少视角数（默认 6） |
| `--debug` | 终端打印 CRP 原始位姿、PnP、运动残差等（默认 true；`--debug=false` 关闭） |
| `--show_methods_help` | 打印简要帮助 |

内联覆盖标定板参数（不改 JSON 文件时；数值须与实物一致）：

```bash
lerobot-calibrate-hand-eye --config_path=examples/hand_eye_calibration/crp.json \
  --board.squares_x=11 --board.squares_y=8 \
  --board.square_size_mm=20 --board.marker_size_mm=14.67 \
  --phase=intrinsics --camera=top --mount=eye_to_hand
```

---

## 常见问题

**`consistency_std_mm` 一直下不来（例如 >30 mm）？**

1. **CRP 机器人：确认已 `export CRP_POSE_INVERT=1`**。未设置时易出现 reproj 正常但 std >100 mm（位姿矩阵方向反了）。见上文「§3 CRP 位姿方向」。
2. **优先用** `--phase=intrinsics_and_hand_eye`，避免拆分内参/外参两次采图不一致。  
3. 看 `--debug=true` 末尾 **`translation-chain check`**：`pure-T` 行 `trans_axxb` 大 → 机器人平移链/TCP/位姿帧问题；`reproj` 仍 <1 px 则不是 ChArUco 检测问题。  
4. **采集**：至少 4 个差异大的位姿，**含明显旋转**；每位姿最多重复 2–3 次；不要同一位姿连 save 12 次（会解退化）。  
5. **`robot_pose_frame`** 与示教器一致（当前 `world`）；标定过程中不要切换 world/user。  
6. TCP 在夹爪上时，手眼解的是 **TCP→相机**；确认 save 时臂已停稳（看终端 `sync:` 与 `pose_ms`）。  
7. 拆分流程时确认 debug 打印：`hand_eye PnP intrinsics: loaded from scene NPZ` 路径正确，且 RMS 合理（通常 <1 px）。  
8. 不剔离群值：report 里的 consistency 反映**全部帧**真实误差，便于查 TCP/平移链，数值偏高是预期现象直到机械链对齐。

**检测一直是红色？**  
检查光照、焦距、板是否平整；确认 `charuco_board.json` 与打印板一致；ArUco 字典必须是 `DICT_4X4_50`（若打印工具用的是别的字典需同步修改）。

**某阶段报错 “Run intrinsics first”？**  
按顺序执行，并共用同一实验目录（`outputs.experiment` / `--experiment` / `--experiment_dir`）。

**landmark / top_extrinsic_dual 融合 std 很大？**  
确认全程 `CRP_POSE_INVERT=1`；补采差异大的位姿；`top_extrinsic_dual` 下手持板可变，但两侧相机须同时检测到板。

**hand_eye 与 top_extrinsic_dual 能否共用同一批数据？**  
不能自动共用，需分别运行。手眼阶段桌面板宜固定；top dual 可手持板。

**内参阶段板子必须贴桌面吗？**  
不必。内参 / top dual 可手持；**wrist hand_eye**（以及可选的 `landmark_map`）阶段板宜固定在桌面。

---

## 代码位置

```text
src/lerobot/calibration/       # 算法与交互 runner
src/lerobot/scripts/lerobot_calibrate_hand_eye.py
examples/hand_eye_calibration/
  crp.json                     # 机器人
  charuco_board.json             # 标定板
```
