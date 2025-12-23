# 🐍 Multi-Snake Battle AI (Neon Edition)

一个基于强化学习（DQN / PPO）的多蛇对战环境，支持局域网联机与霓虹风格可视化。

![Neon Snake](https://via.placeholder.com/800x400?text=Snake+AI+Battle+Neon+Style)

## ✨ 核心特性

- **现代视觉体验**: 重构的 PyQt6 渲染引擎，支持霓虹辉光、径向渐变、抗锯齿绘图。
- **高内聚架构**:
    - **Unified Environment**: 单一环境 `BattleSnakeEnv` 同时支持单机练习与多蛇乱斗。
    - **Agent Abstraction**: 独立的 `agent/` 模块，封装 DQN/PPO 网络与推理逻辑。
    - **Shared Renderer**: 渲染器作为独立组件 (`utils/renderer.py`)，被 GUI 和 Client 复用。
- **双模训练**:
    - **DQN (Deep Q-Network)**: 支持 Off-policy 训练，适用于单蛇或多蛇。
    - **PPO (Proximal Policy Optimization)**: 支持 On-policy 高并行度训练，适用于多蛇博弈。
- **局域网联机**: 提供完整的 Server-Client 架构，支持人类玩家、AI 托管与观战模式混战。

## 📂 项目结构

```text
project/
  ├── agent/            # AI 模型抽象 (DQN/PPO)
  ├── env/              # 统一游戏环境 (BattleSnakeEnv)
  ├── net/              # 网络通信 (GameServer/QtClient)
  ├── utils/            # 通用工具 (GameRenderer)
  ├── gui_game.py       # 本地游戏入口 (单机版)
  ├── train_dqn.py      # 通用 DQN 训练脚本
  ├── train_ppo.py      # 通用 PPO 训练脚本
  └── requirements.txt  # 依赖列表
```

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```
*主要依赖: `torch`, `numpy`, `PyQt6`*

### 2. 单机试玩 (`gui_game.py`)
最简单的体验方式，无需启动服务器。
```bash
# 人工模式 (方向键控制 P0)
python gui_game.py --mode single --human

# 观看 DQN 模型演示 (单蛇)
python gui_game.py --mode single --algo dqn --model agent/checkpoints/dqn_best.pth

# 观看 PPO 混战 (4蛇互搏)
python gui_game.py --mode battle --algo ppo --model agent/checkpoints/ppo_battle_best.pth
```

### 3. 模型训练

**DQN (Deep Q-Network)**
```bash
# 训练单蛇 (生成 agent/checkpoints/dqn_best.pth)
python train_dqn.py --single

# 训练多蛇 (生成 agent/checkpoints/dqn_battle_best.pth)
python train_dqn.py
```

**PPO (Proximal Policy Optimization)**
```bash
# 默认开启 8 环境并行训练 (生成 agent/checkpoints/ppo_battle_final.pth)
python train_ppo.py
```

### 4. 局域网联机对战

**Step 1: 启动服务器**
```bash
python net/game_server.py
```
*默认监听 `0.0.0.0:5555`*

**Step 2: 启动客户端**
```bash
python net/game_client.py
```
*在图形界面中输入服务器 IP，选择模式 (Human/AI/Spectator) 进行连接。*

## 🎮 游戏模式

| 模式 | 描述 | 适用脚本 |
| :--- | :--- | :--- |
| **Single** | 经典的单蛇吃豆模式，撞墙或撞身即死。 | `train_dqn.py --single` |
| **Battle** | 2-4 条蛇的生存大乱斗。支持击杀奖励、碰撞判定。 | `train_dqn.py`, `train_ppo.py` |

## 🛠️ 技术细节

- **State Space (15-dim)**:
    - 4x Food Direction (One-hot)
    - 3x Immediate Danger (Straight, Left, Right)
    - 4x Current Direction (One-hot)
    - 4x Nearest Enemy Direction (One-hot)
- **Reward Function**:
    - `+10`: Eat Food
    - `-10`: Die (Wall/Collision)
    - `+20`: Kill Enemy (Battle only)
    - `+0.2 / -0.3`: Distance Shaping (Closer/Farther from food)

## 📝 开发指南

- **添加新算法**: 在 `agent/` 下新建文件，参考 `dqn.py` 实现 `Act/Load` 接口。
- **修改环境**: 编辑 `env/battle_snake_env.py`，它是所有模式的核心。
- **自定义 UI**: 编辑 `utils/renderer.py`，修改 `paintEvent` 即可同时改变本地和联机版的画风。

---
*Created for Machine Learning Course Project.*
