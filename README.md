# Kevin 项目操作指南

## 项目简介

本项目包含 mjlab 机器人学习框架及相关数据文件，主要用于 Unitree G1 机器人的运动模仿和强化学习训练。

## 环境设置

### Conda 环境

创建并激活 conda 环境：

```bash
# 创建 conda 环境（Python 3.10-3.13）
conda create -n mjlab python=3.11
conda activate mjlab

# 安装依赖（如果使用 conda 而非 uv）
pip install git+https://github.com/google-deepmind/mujoco_warp@1dc288cf1fa819fc3346ec5c9546e2cc2b7be667
pip install mjlab
```

**注意**：
- 本项目推荐使用 `uv` 进行依赖管理
- 如果使用 conda，安装 mjlab 后可以直接使用 `train`、`play` 等命令（无需 `uv run`）
- 例如：`train Mjlab-Tracking-Flat-Unitree-G1 ...` 而不是 `uv run train ...`

## 目录结构

- `mjlab/` - mjlab 框架主目录（机器人学习框架）
- `g1_npz/` - G1 机器人动作数据（.npz 格式）
- `g1_csv/` - G1 机器人动作数据（.csv 格式）
- `g1_pkl/` - G1 机器人动作数据（.pkl 格式）
- `MimicKit_Data/` - 模型和动作数据资源
- `g1_spinkick_example/` - 示例项目

## 常用操作命令

**重要提示**：
- 所有命令需在激活 `mjlab` conda 环境后运行
- **所有仿真命令默认使用 GPU**（包含 `CUDA_VISIBLE_DEVICES=0` 和 `--device cuda:0`）

### 1. 训练模型

训练 G1 机器人进行动作跟踪（使用 cartwheel 动作）：

```bash
cd /home/wasabi/Kevin/mjlab
CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true uv run train Mjlab-Tracking-Flat-Unitree-G1 \
  --env.commands.motion.motion-file /home/wasabi/Kevin/g1_npz/g1_cartwheel.npz \
  --agent.logger tensorboard \
  --env.scene.num-envs 4096
```

### 2. 可视化播放

#### 方法 A：使用标准 Tracking 任务（推荐）

使用零策略（zero agent）可视化动作文件：

```bash
cd /home/wasabi/Kevin/mjlab
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run play Mjlab-Tracking-Flat-Unitree-G1 \
  --agent zero \
  --motion-file /home/wasabi/Kevin/g1_npz/g1_kick_combo.npz \
  --num-envs 1 \
  --no-terminations True \
  --device cuda:0 \
  --viewer native
```

#### 方法 B：使用 Spinkick 任务（在 g1_spinkick_example 目录下）

如果你想使用 `Mjlab-Spinkick-Unitree-G1` 任务，需要在 `g1_spinkick_example` 目录下运行：

```bash
cd /home/wasabi/Kevin/g1_spinkick_example
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run play Mjlab-Spinkick-Unitree-G1 \
  --agent zero \
  --motion-file /home/wasabi/Kevin/g1_npz/g1_kick_combo.npz \
  --num-envs 1 \
  --no-terminations True \
  --device cuda:0 \
  --viewer native
```

**注意**：
- `Mjlab-Spinkick-Unitree-G1` 是自定义任务，需要在 `g1_spinkick_example` 目录下运行
- 运行时会显示 `Environment device: cuda:0` 确认 GPU 使用
- 常见警告（`libEGL warning`、`GLFWError`）可忽略，不影响 GPU 执行

### 3. Sim2Sim 回放

使用训练好的模型回放动作：

```bash
cd /home/wasabi/Kevin/mjlab
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run play Mjlab-Tracking-Flat-Unitree-G1 \
  --checkpoint-file /home/wasabi/Kevin/mjlab/logs/rsl_rl/g1_tracking/2026-02-04_00-28-16/model_4500.pt \
  --motion-file /home/wasabi/Kevin/g1_npz/g1_cartwheel.npz \
  --num-envs 1 \
  --no-terminations True \
  --device cuda:0 \
  --viewer native
```

**注意**：如遇 CUDA 错误，可使用 CPU 模式（见"注意事项"部分）

## 数据集转换

### 1. PKL 转 CSV

将 MimicKit 的 PKL 格式动作文件转换为 CSV 格式：

```bash
cd /home/wasabi/Kevin/g1_spinkick_example
uv run pkl_to_csv.py \
  --pkl-file /home/wasabi/Kevin/g1_pkl/g1_cartwheel.pkl \
  --csv-file /home/wasabi/Kevin/g1_csv/g1_cartwheel.csv
```

可选参数：
- `--duration` - 指定输出时长（秒），如果比原始时长长，会循环播放
- `--pad-duration` - 在末尾保持最终姿态的时长（秒）
- `--transition-duration` - 过渡到/从安全站立姿态的时长（秒）
- `--add-start-transition` - 添加从安全站立姿态到动作开始的过渡
- `--add-end-transition` - 添加从动作结束到安全站立姿态的过渡

### 2. CSV 转 NPZ

将 CSV 格式的动作文件转换为 NPZ 格式（用于训练）：

```bash
cd /home/wasabi/Kevin/mjlab
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run src/mjlab/scripts/csv_to_npz.py \
  --input-file /home/wasabi/Kevin/g1_csv/g1_cartwheel.csv \
  --output-name g1_cartwheel \
  --input-fps 30 \
  --output-fps 50 \
  --device cuda:0 \
  --render  # 可选：生成预览视频
```

**注意**：转换后的文件会上传至 WandB registry，或保存在 `/tmp/motion.npz`

## 可用动作文件

以下动作文件位于 `g1_npz/` 目录：

- `g1_cartwheel.npz` - 侧手翻动作
- `g1_kick_combo.npz` - 踢腿组合动作
- `g1_spinkick.npz` - 旋转踢动作
- `g1_run.npz` - 跑步动作
- `g1_walk.npz` - 行走动作
- `g1_double_kong.npz` - 双空翻动作
- `g1_speed_vault.npz` - 快速跳跃动作

## 注意事项

1. **GPU 使用**：所有命令默认使用 GPU（包含 `CUDA_VISIBLE_DEVICES=0` 和 `--device cuda:0`）
2. **CPU 模式（GPU 不可用时）**：如遇 CUDA 错误 804，移除 `CUDA_VISIBLE_DEVICES=0` 并将 `--device cuda:0` 改为 `--device cpu`
   ```bash
   # 示例：CPU 模式
   cd /home/wasabi/Kevin/mjlab
   MUJOCO_GL=egl uv run play Mjlab-Tracking-Flat-Unitree-G1 \
     --agent zero \
     --motion-file /home/wasabi/Kevin/g1_npz/g1_kick_combo.npz \
     --num-envs 1 \
     --no-terminations True \
     --device cpu \
     --viewer native
   ```
3. **环境变量**：`MUJOCO_GL=egl` 用于无头渲染
4. **日志位置**：训练日志保存在 `mjlab/logs/rsl_rl/` 目录

## 快速参考

- **训练目录**：`/home/wasabi/Kevin/mjlab`
- **数据目录**：`/home/wasabi/Kevin/g1_npz/`
- **模型目录**：`/home/wasabi/Kevin/mjlab/logs/rsl_rl/`

更多详细信息请参考 `mjlab/README.md` 和官方文档。
