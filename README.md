# 多蛇互博贪吃蛇（Multi-Snake Arena）

一个“多蛇同场互博”的网格贪吃蛇环境 + 两套可训练基线算法：`Rainbow(DQN)` 与 `A2C`。

本仓库已将 **reward 计算固化在 env 层**（`env/multi_snake_env.py`），训练脚本不再二次拼 reward，避免训练发散或学会“转圈”。

## 亮点速览

- **Authoritative Server**：`network/server.py` 使用 asyncio 管理所有房间、成员与动作收集，统一判定胜负与对局结束，杜绝状态漂移。
- **单一 Online 模式**：移除历史多模式分支，房主只需配置网格、蛇数、食物、步数与 tick 间隔即可开局，真人与 AI 客户端共用协议。
- **PyQt 客户端**：`network/client.py` 集成房间列表、参数表单、成员管理、日志与渲染入口；CLI 模式（`--mode cli`）适合无桌面环境。
- **Rainbow 基线**：NoisyNet + C51 + n-step + 优先经验回放。
- **A2C 基线**：共享参数 Actor-Critic + GAE + 多环境并行 rollout。
- **更可训练的 reward**：默认使用 step penalty + 食物奖励 + 死亡惩罚 + 击杀奖励 + 距离塑形（差分）。
- **Pygame 渲染器**：`network/renderer.py` 展示房间号、用户-角色-得分与步数，支持真人按键或纯观战，退出时自动处理角色/槽位。
- **中文注释 & 教学友好**：主要模块均附中文说明，方便课堂讲解与二次开发。

## 目录速查

```
agent/
   a2c_*.py             # A2C: config/model/agent/trainer
   rainbow_*.py         # Rainbow: config/model/agent/trainer/replay
env/
   config.py           # 环境 + reward 配置
   multi_snake_env.py  # 多蛇对战环境（直接返回 reward）
network/
   server.py           # 云端服务器
   client.py           # PyQt/CLI 客户端
   renderer.py         # Pygame 渲染窗口
   utils.py            # 方向与端口辅助
a2c_train.py          # A2C 本地训练入口
a2c_curriculum_train.py # A2C 课程式训练入口（推荐）
a2c_eval.py           # A2C 本地评估/演示
rainbow_train.py      # 本地训练入口
rainbow_curriculum_train.py # Rainbow 课程式训练入口（推荐）
rainbow_eval.py       # 本地评估/演示
requirements.txt
README.md
```

## 快速开始（离线训练）

1. 安装依赖：
   ```bat
   cd /d d:\code\PythonProject\machineLearningCourseDevise
   pip install -r requirements.txt
   ```

   若你看到 `ModuleNotFoundError: No module named 'torch'`，说明当前 Python 环境没装上 PyTorch。
   - CPU 版本可直接：`pip install torch`
   - GPU 版本请按 PyTorch 官方安装指引选择对应 CUDA 轮子（不同机器/驱动不同）。
   
   **注意**：本项目使用 `pygame` 进行无需显示器的图像渲染，请确保已安装 `pygame`。

2. 训练 A2C (图像输入版)：
   ```bat
   python a2c_train.py
   :: 或双击 scripts\run_a2c_train.bat (需自行更新脚本内容)
   ```
   说明：现在训练默认使用 84x84 的 RGB 图像作为输入，不再使用网格特征。

3. 训练 Rainbow (图像输入版)：
   ```bat
   python rainbow_train.py
   ```
4. 本地评估/演示（会打开渲染器）：
   ```bat
   python a2c_eval.py --model agent\checkpoints\a2c_snake_latest.pth
   python rainbow_eval.py --model agent\checkpoints\rainbow_snake_latest.pth
   ```

> 服务器只负责“对局托管与广播”；训练建议离线跑 `a2c_train.py` / `rainbow_train.py`。

## 快速开始（联机对战：服务器 + 客户端）

1. 启动服务器：
   ```bat
   cd /d d:\code\PythonProject\machineLearningCourseDevise
   python network\server.py --host 0.0.0.0 --port 5555
   ```
2. 启动客户端：
   ```bat
   python network\client.py
   ```
3. 客户端创建房间 -> 加入/准备 -> 开始。

说明：当前服务器端 HUD 的 `score_board` 仍按 `info["events"]` 做展示性计分；训练用 reward 以 `env.step()` 返回的 `rewards` 为准（两者不必一致）。

## 房间与角色

- **角色规则**：`human` 和 `ai` 占用槽位并参与战斗；`spectator` 释放槽位。AI 角色在未加载模型前无法准备。
- **自动判负**：对局进行中退出渲染窗口或切换至 `spectator` 会立即判定蛇死亡，避免僵尸蛇；对局结束后自动恢复退出前的角色。
- **广播内容**：服务器会推送成员列表、蛇状态、得分、颜色与步数，客户端据此渲染 UI/HUD。

## 渲染与输入

- `network/renderer.py` 使用方向键控制（禁止立即调头），同时显示房间号、步数以及“用户名-角色：得分”。
- ESC 退出渲染窗口：如果仍在对局，会自动切换到观战；对局结束则保持原角色，方便下一局。

## 训练参数怎么调

- 训练入口脚本：`a2c_train.py` / `rainbow_train.py`
- 环境与 reward：`env/config.py`、以及两个 config（`agent/a2c_config.py`、`agent/rainbow_config.py`）中的 env 字段
- 若仍出现“转圈”，优先把 `step_penalty` 变得更负一些（例如 `-0.02`），并确保不要再加正向“存活奖励”。

## A6000 推荐设置

你是 A6000（显存非常充足），建议优先把并行度/批大小拉起来吃满 GPU：

- A2C：`num_envs=32`，`rollout_length=128`
- Rainbow：`num_envs=32`，`batch_size=512`，`replay_capacity>=1_000_000`

若显存不够或 GPU 利用率不高：

- 显存不够：先把 `batch_size` / `num_envs` 减半
- 吞吐不高：优先把 `num_envs` 加倍（例如 16→32→64），再调 `batch_size`

## 推荐起步参数（先跑通再加难度）

- `grid_size=10~14`，`num_snakes=2`，`num_envs=2~8`
- `food_reward=1.0`，`death_penalty=-1.0`，`step_penalty=-0.01~-0.02`

## 常见问题

- **是否支持离线单机？** 可运行 `rainbow_eval.py` 进行本地演示；标准流程仍推荐“服务器 + 客户端”以保持一致的时序与协议。
- **如何扩展参数/玩法？** 优先从 `env/multi_snake_env.py` 的动力学与 `env/config.py` 的 reward 入手；联机参数再对齐 `network/server.py` 的房间配置与前端表单。
- **渲染器需要哪些依赖？** Windows 默认可用；Linux/macOS 需安装 SDL 运行库及可用的显示环境。
- **AI 掉线后会怎样？** 服务器检测断开后即释放槽位并广播最新房间状态；如果该蛇在对局中，会被判定阵亡。

## 未来方向

- WebSocket/Web UI 版房间管理和观战
- 训练/评估指标上报（Prometheus、InfluxDB 等）
- Self-Play、Curriculum 等强化学习策略扩展
- WebGL/Three.js 渲染器，支持零客户端观看

欢迎在课程、讲座、社团活动中展示“多蛇云端生存战”，也欢迎 Issue/PR 共同完善！
