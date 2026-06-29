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
        ├─ landmark_map → T_robot→table  │
        │         (同一块桌面 ChArUco)    │
        └────────────────────────────────┴─ camera_via_landmark → T_robot→top
```

- **hand_eye**：桌面 ChArUco 不动，移动机械臂，求腕部相机相对末端的外参 `T_ee_to_camera`。
- **landmark_map**：仍是桌面同一块板，腕部多视角观测，融合得到桌面坐标系 `T_robot_to_table`，并计算角点网格坐标。
- **camera_via_landmark**：top 相机看向**同一块**桌面 ChArUco，结合已知的 `T_robot_to_table` 反推 top 外参 `T_robot_to_camera`。

逻辑上等价于「末端 pin 点建桌面系」，只是把接触点换成视觉多点融合。

---

## 标定前准备

### 1. 打印并固定 ChArUco 板

按 `charuco_board.json` 的参数打印（当前：**8×11 格，checker 15 mm，marker 11 mm，DICT_4X4_50**），贴在工作台可见区域：

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

### 3. 确认相机能出图

机械臂上电、相机 index 正确。有图形界面时会弹出检测预览窗口（绿=检测到 ChArUco，红=未检测到）。

### 4. 输出路径

默认写入：

```text
~/.cache/huggingface/lerobot/calibration/robots/crp_arm/lab_arm_01_calibration.npz
```

自定义路径：`--output_path=/path/to/my_calibration.npz`

---

## 配置文件

### `crp.json` — 机器人

```json
{
  "robot": { "type": "crp_arm", "id": "lab_arm_01", "port": "...", "cameras": { ... } },
  "board_config_path": "examples/hand_eye_calibration/charuco_board.json"
}
```

### `charuco_board.json` — 标定板（与实物一致）

```json
{
  "squares_x": 8,
  "squares_y": 11,
  "square_size_mm": 15.0,
  "marker_size_mm": 11.0,
  "aruco_dict": "DICT_4X4_50"
}
```

换板子只改此文件；`squares_x/y` 为 OpenCV ChArUco 的**方格数**（不是内角点数）。

---

## 标定流程（按顺序执行）

所有命令共用同一份 `--config_path=examples/hand_eye_calibration/crp.json`，只改 `--phase` 等参数。

**交互约定**（各阶段相同）：

- **Enter**：当前帧检测到 ChArUco 则采集
- **q**：结束本阶段并求解
- 有 GUI 时窗口实时显示检测结果

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

### 步骤 2：wrist 相机内参

**目的**：求 wrist 内参（同上，换腕部相机采图）。

**操作**：用腕部相机看 ChArUco，多姿态采图。

```bash
lerobot-calibrate-hand-eye --config_path=examples/hand_eye_calibration/crp.json \
  --phase=intrinsics --camera=wrist --mount=eye_in_hand
```

**写入 NPZ**：`wrist__camera_matrix`、`wrist__dist_coeffs` 等。

---

### 步骤 3：wrist 手眼外参

**目的**：求 `T_ee_to_camera`（腕部相机相对末端）。

**前提**：步骤 2 已完成；**ChArUco 固定在桌面**（与后续步骤同一块板）。

**操作**：移动机械臂，让 wrist 从多个角度看清桌面板（默认至少 **12** 组样本）。姿态要分散：不同高度、偏航、俯仰，避免全在同一平面。

```bash
lerobot-calibrate-hand-eye --config_path=examples/hand_eye_calibration/crp.json \
  --phase=hand_eye --camera=wrist --mount=eye_in_hand
```

**写入 NPZ**：`wrist__T_ee_to_camera`。

**可调**：`--min_hand_eye_samples=15`

---

### 步骤 4：桌面坐标系 + 角点网格

**目的**：多视角融合 `T_robot_to_table`，并计算 `table_grid_points_robot_mm`（ChArUco 内角点在机器人系下的坐标）。

**前提**：步骤 3 已完成；**不要移动/更换**桌面 ChArUco。

**操作**：继续移动机械臂，wrist 从不同位姿看同一块板（默认至少 **6** 视角）。位姿差异越大，融合越稳。

```bash
lerobot-calibrate-hand-eye --config_path=examples/hand_eye_calibration/crp.json \
  --phase=landmark_map --camera=wrist
```

**写入 NPZ**：

- `T_robot_to_table`
- `table_grid_points_robot_mm`（形状 `(N, 3)`，N = (squares_x−1)×(squares_y−1)，当前 8×11 板为 **70** 个点）
- `charuco_*` 板参数

日志会打印融合一致性（position std）；std 过大建议补采几帧。

**可调**：`--min_landmark_samples=8`

---

### 步骤 5：top 相机外参

**目的**：求 `T_robot_to_camera`（top 在机器人系下）。

**前提**：步骤 1、4 已完成；桌面 ChArUco **仍在原处**；top 能清晰看到整块或大部分板。

**操作**：固定 top 相机，确保板在视野内；Enter 采 1 帧即可，建议多采 3 帧取平均。

```bash
lerobot-calibrate-hand-eye --config_path=examples/hand_eye_calibration/crp.json \
  --phase=camera_via_landmark --camera=top --mount=eye_to_hand
```

**写入 NPZ**：`top__T_robot_to_camera`

---

### 步骤 6：验证并导出

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
grid_robot_mm = data["table_grid_points_robot_mm"]  # (70, 3) for 8×11 board
```

---

## 常用 CLI 参数

| 参数 | 说明 |
|------|------|
| `--config_path` | 主配置（机器人 + board_config_path） |
| `--phase` | `intrinsics` / `hand_eye` / `landmark_map` / `camera_via_landmark` / `validate` |
| `--camera` | `top` 或 `wrist` |
| `--mount` | `eye_to_hand`（top）或 `eye_in_hand`（wrist）；内参 / 手眼 / camera_via_landmark 必填 |
| `--output_path` | 自定义 NPZ 路径 |
| `--min_hand_eye_samples` | 手眼最少样本数（默认 12） |
| `--min_landmark_samples` | landmark 最少视角数（默认 6） |
| `--show_methods_help` | 打印简要帮助 |

内联覆盖标定板参数（不改 JSON 文件时）：

```bash
lerobot-calibrate-hand-eye --config_path=examples/hand_eye_calibration/crp.json \
  --board.squares_x=8 --board.squares_y=11 \
  --board.square_size_mm=15 --board.marker_size_mm=11 \
  --phase=intrinsics --camera=top --mount=eye_to_hand
```

---

## 常见问题

**检测一直是红色？**  
检查光照、焦距、板是否平整；确认 `charuco_board.json` 与打印板一致；ArUco 字典必须是 `DICT_4X4_50`（若打印工具用的是别的字典需同步修改）。

**某阶段报错 “Run intrinsics first”？**  
按步骤 1→2→3→4→5 顺序执行，且共用同一 `--output_path`。

**landmark 融合 std 很大？**  
补采更多差异大的腕部位姿；确认桌面板未移动；ChArUco 未被遮挡。

**步骤 3 和 4 能否共用同一批数据？**  
不能自动共用，需分别运行；但可以使用**同一块固定板**，先 hand_eye 再 landmark_map，操作可连续完成。

**内参阶段板子必须贴桌面吗？**  
不必。内参阶段 ChArUco 可手持移动；从步骤 3 起板子需固定在桌面不动。

---

## 代码位置

```text
src/lerobot/calibration/       # 算法与交互 runner
src/lerobot/scripts/lerobot_calibrate_hand_eye.py
examples/hand_eye_calibration/
  crp.json                     # 机器人
  charuco_board.json             # 标定板
```
