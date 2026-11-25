# 2PWBC 训练、续训与回放

## 配置

| 文件 | 内容 |
|---|---|
| `conf/wbc2p/config.yaml` | World Model、Replay、BPTT、双策略和运行配置 |
| `conf/wbc2p/task/go2/mujoco.yaml` | Go2 命令、控制和环境 |
| `conf/wbc2p/task/go2/rewards.yaml` | Go2 奖励权重、参数及 MuJoCo 和想象展开实现 |
| `conf/wbc2p/task/l1w/mujoco.yaml` | L1-W 命令、控制和环境 |
| `conf/wbc2p/task/l1w/rewards.yaml` | L1-W 奖励权重、参数及 MuJoCo 和想象展开实现 |

路径相对于 `src/legged_manipulation_unilab/`。使用 `uv run legged-train --task go2 --cfg stage=stage1` 打印最终配置。Go2 Stage 1 从任务文件读取 Raibert 权重，Stage 2 的权重为零。L1-W 的机身高度目标为 `0.30 m`，高度误差权重为 `-10`。

L1-W 步态频率为 1.5 Hz，任一指令超过阈值时启用步态，阈值为前进 1.0 m/s、横移 0.08 m/s、转向 0.2 rad/s。其余指令下相位归零，时钟输入为零，抬腿奖励关闭，接触奖励鼓励四轮支撑。

L1-W 的轮腿奖励参考本机 Uni-SuperMimic。支撑轮按
`(vx - yaw_rate * wheel_y) / wheel_radius` 跟踪轮速，使用归一化 Huber 奖励，
权重为 0.30，纯横移与零指令时关闭。传感器顺序为 FBL/FAR/RBL/RAR，
轮电机顺序为 FBL/FAR/RAR/RBL，支撑权重在计算轮速奖励前显式重排。
相位过渡宽度为 0.08，关节与轮子的动作变化率、加速度分别配置。
轮速静止惩罚仅在三维速度指令均不超过 0.01 时生效。

L1-W 使用加权奖励直接求和。当前 2PWBC 的双 Actor、Planner 姿态跟踪、
航向对齐末端任务和两阶段训练继续由当前工程实现。
想象轮地接触使用 FK 高度的连续接触估计，滚动奖励额外乘支撑置信度，
防止极小接触概率获得接近满分的奖励。抬腿高度核仍沿用参考工程的
`target_foot_height=0.10`、`swing_reward_sigma=0.01`，贴地时奖励偏高的局限仍存在。

## 从零开始

```bash
CUDA_VISIBLE_DEVICES=0 bash train_2pwbc.sh --task go2 \
  1024 5000 30000 training.device=cuda:0 training.compile=true algo.utd=0.4

CUDA_VISIBLE_DEVICES=0 bash train_2pwbc.sh --task l1w \
  1024 5000 30000 training.device=cuda:0 training.compile=true algo.utd=0.4
```

`CUDA_VISIBLE_DEVICES` 选择物理 GPU，CUDA 对进程可见的设备从 `cuda:0` 开始编号。三项训练参数依次是并行环境数、Stage 1 更新数和 Stage 2 追加更新数。省略时默认取 `1024 1000 3000`。

CUDA 训练用 BF16 计算 MLP，状态积分、FK、奖励和回报使用 FP32。`training.compile=true` 启用 Inductor。`algo.utd=0.4` 交替每个环境采集 2 步和 3 步。省略该选项时每轮采集 24 步。Replay 默认在 learner 设备上，`replay.storage_device=cpu` 可以改用 CPU。

## 单独训练或续训

下面用 L1-W 举例。Go2 用 `--task go2`。

```bash
uv run legged-train --task l1w stage=stage1 algo.max_iterations=5000
bash migration/resume_stage1.sh --task l1w "${STAGE1_RUN}" 1000 1024 -1
bash migration/train_stage2.sh --task l1w "${STAGE1_RUN}" 30000 1024 -1
bash migration/train_stage2.sh --task l1w "${STAGE2_RUN}" 3000 1024 -1
```

后面三个位置参数依次是追加更新数、环境数和 checkpoint 编号。`-1` 会选择目录下最大的 `model_<编号>.pt`。续训从训练 checkpoint 恢复模型、优化器和归一化统计，并重新采集 Replay。Stage 2 首次加载 Stage 1 时会新建机械臂 Actor 和 Critic。

日志默认写入 `logs/2pwbc-mujoco/Go2ArmManipLoco/` 或 `logs/2pwbc-mujoco/L1WArmManipLoco/`。一份训练目录包含完整 checkpoint、单独的 Actor 权重、实际运行配置和 TensorBoard 事件。

## 回放与 TensorBoard

```bash
uv run 2pwbc-play --run "${STAGE2_RUN}" --device cpu
uv run legged-eval --run "${STAGE2_RUN}" training.device=cpu interactive.keyboard=true
uv run tensorboard --logdir logs/2pwbc-mujoco --host 127.0.0.1 --port 16006
```

回放入口支持完整 checkpoint、单次运行目录和任务日志目录。阶段、机器人和观测结构来自 checkpoint。L1-W 当前底盘观测为 64 维，包含四维门控步态时钟。旧的 60 维无相位权重可回放，恢复相位后训练需从 Stage 1 重新开始。

方向键控制前后与转向，`7/8` 横移，`Enter` 停车。Stage 2 的 `U/J`、`H/K`、`Y/O` 移动航向对齐系下的末端目标，`L` 回到初始末端目标。`Backspace` 手动重置，空格暂停。

TensorBoard 的 `WorldModel/loss` 是多步动力学监督误差，`Loss/leg_actor` 和 `Loss/arm_actor` 是负的想象回报。可并排查看 `Real/reward_leg`、`Real/reward_arm`、`Real/falls` 和实际 Pitch/Roll 误差。L1-W 额外记录 `WorldModel/final_wheel_velocity_rmse_rad_s`。
