# 手眼 / 场景标定

用 ChArUco 求：相机内参、腕部 `T_ee_to_camera`、固定相机 `T_camera_to_robot`，以及可选的桌面网格。

全程交互式 CLI；结果写入同一实验目录下的 `calibration.npz`。

---

## 变换命名约定（重要）

所有 `T_a_to_b` 表示**点坐标从 a 变到 b**：

```text
p_b = T_a_to_b @ p_a
```

| 名字 | 含义 | 点映射 |
|------|------|--------|
| `T_ee_to_robot` | 末端 TCP → 机器人基座 | `p_robot = T @ p_ee`（= OpenCV `gripper2base`） |
| `T_ee_to_camera` | 末端 → 腕部相机 | `p_cam = T @ p_ee` |
| `T_camera_to_robot` | 相机 → 机器人基座 | `p_robot = T @ p_cam`（= OpenCV `cam2base`） |
| `T_target_to_camera` | 标定板 → 相机 | `p_cam = T @ p_board`（= solvePnP） |
| `T_table_to_robot` | 桌面系 → 机器人基座 | `p_robot = T @ p_table` |

腕部安装偏置（相机原点在 TCP/EE 系）为：

```text
t_cam_in_ee = -Rᵀ @ t    # 对 T_ee_to_camera = [R|t]
```

**不要**把 `T_ee_to_camera` 的平移列直接当成「法兰到相机的物理偏置」。

眼在手上棋盘链：

```text
T_board_to_robot = T_ee_to_robot @ inv(T_ee_to_camera) @ T_target_to_camera
```

旧 NPZ 键名（`T_robot_to_ee` / `T_robot_to_camera` / `T_robot_to_table`）加载时仍兼容。

---

## 标定前准备

1. 按 `charuco_board.json` 打印板（当前 11×8 格，checker 20 mm，marker 14.67 mm，`DICT_4X4_50`），贴平贴牢。
2. 编辑 `crp.json`：控制器 IP、相机 `index_or_path`、`robot_pose_frame`（与示教器一致，默认 `world`）。
3. 示教器工具 TCP 设在夹爪 TCP；手眼解的是 **TCP → 腕部相机**。
4. SDK 位姿与示教器应一致：单位 mm + deg；`T_ee_to_robot = [R|xyz]`，固定 XYZ 只改姿态时 `|Δxyz|≈0`。

---

## 推荐流程（一条路径）

```bash
cd /path/to/lerobot_handeye
CFG=--config_path=examples/hand_eye_calibration/crp.json

# ① wrist 内参 + 手眼（桌面板固定，≥12 次 save，含明显旋转）
lerobot-calibrate-hand-eye $CFG \
  --phase=intrinsics_and_hand_eye --camera=wrist --mount=eye_in_hand --debug=true

# ② top 内参（板可手持，多视角）
lerobot-calibrate-hand-eye $CFG \
  --phase=intrinsics --camera=top --mount=eye_to_hand

# ③ top 外参（手持板，wrist+top 同时看到，≥6 次）
lerobot-calibrate-hand-eye $CFG \
  --phase=top_extrinsic_dual --camera=top --mount=eye_to_hand

# ④ 校验
lerobot-calibrate-hand-eye $CFG --phase=validate
```

**交互**：`s` / Space / Enter = 保存；`q` = 结束本阶段。保存时阻塞采图 + 读位姿。

**实验目录**：`outputs/hand_eye_calibration/<robot_id>/<experiment>/`  
（`crp.json` → `outputs`；可用 `--experiment=` / `--new_experiment=true`）

可选：

```bash
# 桌面系 + 角点网格（仅当需要 table 网格时）
lerobot-calibrate-hand-eye $CFG --phase=landmark_map --camera=wrist
```

无需合并采图时也可拆分：`--phase=intrinsics` 再 `--phase=hand_eye`（共用同一 experiment；手眼阶段读 `intrinsics/wrist.json`）。

合并阶段默认若配置了 `top` 会同步采 top；不需要加 `--also_collect_top=false`。

---

## 配置摘要

`crp.json`：机器人、相机、`board_config_path`、`robot_pose_frame`、`outputs`。  
`charuco_board.json`：必须与实物一致；`squares_x/y` 是方格数。

| 参数 | 说明 |
|------|------|
| `--phase` | `intrinsics` / `intrinsics_and_hand_eye` / `hand_eye` / `top_extrinsic_dual` / `landmark_map` / `camera_via_landmark` / `validate` |
| `--camera` / `--mount` | `wrist`+`eye_in_hand` 或 `top`+`eye_to_hand` |
| `--robot_pose_frame` | `world` / `user` |
| `--min_hand_eye_samples` | 默认 12 |
| `--debug` | 默认 true |

---

## 输出

路径示例：`outputs/hand_eye_calibration/lab_arm_01/<experiment>/calibration.npz`

| 键 | 含义 |
|----|------|
| `{cam}__camera_matrix` / `__dist_coeffs` | 内参 |
| `wrist__T_ee_to_camera` | 末端→腕部相机 |
| `top__T_camera_to_robot` | top→机器人基座 |
| `{cam}__T_camera_to_robot_resolved` | validate 时当前臂姿下合成的相机→基座 |
| `T_table_to_robot` / `table_grid_points_robot_mm` | 可选桌面 |

```python
import numpy as np

data = np.load("outputs/hand_eye_calibration/lab_arm_01/together/calibration.npz", allow_pickle=True)
T_ee_to_cam = data["wrist__T_ee_to_camera"]
R, t = T_ee_to_cam[:3, :3], T_ee_to_cam[:3, 3]
t_cam_in_ee = -R.T @ t  # 物理安装偏置 (mm)
```

过程文件在 `phases/<phase>_<camera>/`（images、`report.json`、log）；重跑同一 phase 时旧结果进 `archive/`。

---

## 常见问题

**`consistency_std_mm` > 30 mm？**  
优先用 `intrinsics_and_hand_eye`；多姿态含旋转；停稳再 save；看 debug 的 `translation-chain` / AX=XB；`camera origin in EE` 应接近物理安装（百毫米量级，不应接近 1 m）。

**检测一直红色？**  
光照、焦距、板参数 / `DICT_4X4_50` 是否与打印一致。

**top_extrinsic_dual std 高？**  
确认 wrist 手眼已写入且合理；两侧预览都绿再 save。

---

## 代码位置

```text
src/lerobot/calibration/
src/lerobot/scripts/lerobot_calibrate_hand_eye.py
examples/hand_eye_calibration/
  crp.json
  charuco_board.json
```
