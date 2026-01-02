# 🐍 Multi-Snake Battle AI (V7.0 Champion Edition)

一个基于强化学习（DQN / PPO）的高性能多蛇对战环境，支持自动课程学习、自博弈对抗与冠军级模型优化。

![Neon Snake](https://via.placeholder.com/800x400?text=Snake+AI+Battle+V7.0+Champion+Edition)

## ✨ V7.0 核心突破 (Champion Patch)

- **性能革命 (Omni-Batch 架构)**: 自研全量批处理推理，对战模式 FPS 从 1.0 飙升至 **400-600**。
- **博弈多样性 (Chaos Sampling)**: 自博弈池引入 10% 混沌扰动（随机策略），强制 Agent 建立全向鲁棒性。
- **精细化收敛 (Multi-Stage LR)**: 训练后期（80% 后）自动 10 倍 LR 衰减，锁定最优动作，消除训练波动。
- **自博弈显存缓存**: 自动热加载对抗模型，磁盘 I/O 不再是训练瓶颈。

## 📂 项目结构

```text
project/
  ├── agent/            # AI 模型抽象 (DQN/PPO/Dueling/PER)
  ├── env/              # 统一游戏环境 (BattleSnakeEnv, 25D 向量 + 7x7 局部网格)
  ├── scripts/          # V7.0 冠军运行脚本 (.sh)
  ├── train_dqn_curriculum.py  # DQN 系自动课程学习脚本 (1蛇 -> 4蛇)
  ├── train_ppo_curriculum.py  # PPO 自动课程学习脚本
  ├── gui_game.py       # 本地可视化演示界面
  └── requirements.txt  # 依赖列表
```

## 🚀 快速开始

### 1. 安装环境
```bash
pip install -r requirements.txt
```

### 2. 启动冠军训练 (V7.0 脚本)
推荐使用 `Dueling-DQN` 或 `PPO` 变体，它们在 V7.0 中表现最强。
```bash
# 启动 DQN 变体课程训练 (默认 Dueling-DQN)
bash scripts/run_dqn_curriculum.sh [dqn|ddqn|per|dueling]

# 启动 PPO 课程训练
bash scripts/run_ppo_curriculum.sh

# 🛑 一键停止所有训练 (清理主进程与子进程)
bash scripts/stop_training.sh
```

### 2.1 常用性能/训练旋钮（建议先从这些开始）

```bash
# 限制 CPU 线程：减少 AsyncVectorEnv 多进程互抢
export DQN_CPU_THREADS=1

# 降低对手刷新频率：减少 I/O/加载抖动
export RIVAL_UPDATE_INTERVAL=100000

# 自博弈池大小：更大更稳，但别过大
export SELF_PLAY_POOL_SIZE=30
```

Battle 额外推荐（DQN/DDQN）：

```bash
# 直接走课程脚本，但只调 Phase 2（battle）相关参数
bash scripts/run_dqn_curriculum.sh dqn \
  --eps-start2 0.5 --eps-min2 0.05 \
  --sp-prob-start 0.7 --sp-prob-end 0.4 --sp-prob-frac 0.3 \
  --finetune-lr-mult2 0.5 \
  --save2 agent/checkpoints/dqn_battle_lr05.pth
```

### 3. 本地演示
```bash
# 观看 PPO 冠军模型博弈
python gui_game.py --mode battle --algo ppo --model agent/checkpoints/ppo_battle_best.pth
```

### 4. 离线评估（推荐用来判断“效果好不好”）

GUI 的 battle 模式会给所有蛇加载同一个模型，容易出现“看起来不聪明/互相同归于尽/僵持”的错觉。
建议先用离线评估脚本看 win rate vs random：

```bash
python eval_battle.py --algo dqn --model agent/checkpoints/dqn_battle.final.pth --episodes 200 --opponent random
```

## 🛠️ 深度技术规格 (V7.0 Balance)

### 观测空间 (25-dim Vector + CNN Grid)
- **食物定位 (4)**: 上下左右食物距离感应。
- **危险雷达 (9)**: 距离 1 & 2 的障碍物检测 + 线性全向雷达。
- **运动状态 (4)**: 当前蛇头朝向（One-hot）。
- **对抗雷达 (4)**: 最近敌蛇相对方位感应。
- **生存特征 (4)**: 尾部相对位置 (2) + 长度占比 (1) + **冲刺冷却 (1)**。

### 奖励矩阵 (Balanced V7.0)
| 事件 | 奖励 | 说明 |
| :--- | :--- | :--- |
| **Eat Food** | `+25.0` | 核心成长动力 |
| **Survival** | `+0.01` | **V7.0 新增**，鼓励活着就有收益 |
| **Kill Enemy** | `+30.0` | 激进对抗激励 |
| **Death** | `-15.0` | **V7.0 调优**，减轻惩罚以防避战 |
| **Navigation** | `+0.05` | 持续的寻路引导补偿 |

---
*Created for Machine Learning Course Project. Optimized for High-Performance Deep RL.*
