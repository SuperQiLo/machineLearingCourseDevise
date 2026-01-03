# Multi-Snake Battle AI（课程设计工程）

本项目实现了一个多蛇对战的强化学习训练与可视化框架，支持：

- **两阶段课程学习**：Phase1（单蛇导航）→ Phase2（多蛇对战 + 自博弈）
- **算法**：DQN / DDQN / PER / Dueling（DQN 家族统一训练器）与 PPO
- **规则与训练分离**：游戏“得分（score）/MVP”与训练奖励（reward）解耦，便于稳定训练与正确评估

> 说明：本 README 以 Windows（PowerShell）为主；如果你在 Linux/macOS，同样命令基本可直接使用（把 PowerShell 环境变量写法换成 bash 即可）。

---

## 目录

- [项目结构](#项目结构)
- [环境准备与安装](#环境准备与安装)
- [快速运行（可视化对战）](#快速运行可视化对战)
- [训练：DQN 变体（课程学习）](#训练dqn-变体课程学习)
- [训练：PPO（课程学习）](#训练ppo课程学习)
- [日志指标与评估口径（非常重要）](#日志指标与评估口径非常重要)
- [模型保存规则：best vs final](#模型保存规则best-vs-final)
- [环境规则要点（MVP/掉落/奖励解耦）](#环境规则要点mvp掉落奖励解耦)
- [常见问题排查（Windows 常见）](#常见问题排查windows-常见)
- [自检：运行逻辑测试](#自检运行逻辑测试)

---

## 项目结构

```text
machineLearningCourseDevise/
  ├── agent/                     # 各算法网络与推理封装（dqn/ddqn/per/dueling/ppo）
  │   └── checkpoints/            # 训练输出与自带示例模型（*.pth / *.final.pth）
  ├── env/                        # 对战环境与 gymnasium wrapper（奖励整形/信息字典）
  ├── net/                        # 网络对战（client/server）
  ├── utils/                      # 自博弈池、渲染等
  ├── gui_game.py                 # 本地可视化（single/battle）
  ├── train_dqn_curriculum.py     # DQN 变体：Phase1→Phase2 课程学习调度器
  ├── train_dqn_variants.py       # DQN 变体统一训练器（带自博弈与 KPI 日志）
  ├── train_ppo_curriculum.py     # PPO：Phase1→Phase2 课程学习调度器
  ├── train_ppo.py                # PPO 训练器（带 KPI 日志）
  ├── test_env_logic.py           # 环境逻辑冒烟测试（非 pytest）
  ├── requirements.txt
  └── logs/                       # 已训练日志（用于对比 Win%/S0）
```

---

## 环境准备与安装

### 1）Python 版本

建议 Python 3.10+（你当前环境能跑 `python gui_game.py` 就可用）。

### 2）安装依赖

在项目根目录执行：

```powershell
pip install -r requirements.txt
```

### 3）确认 PyTorch 可用

训练需要 `torch`。如果你遇到 `ModuleNotFoundError: No module named 'torch'`：

- 说明当前 python 环境没装 torch（常见于你用 `conda run` 跑到了另一个环境）
- 解决方式：在“你平时能跑 `python gui_game.py` 的那个 python”里安装 torch

（不同 CUDA/CPU 安装方式不同，这里不硬写版本，避免误导；按你机器对应的官方安装命令安装即可）

---

## 快速运行（可视化对战）

`gui_game.py` 支持两种模式：

- `--mode single`：单蛇
- `--mode battle`：多蛇对战

支持算法：`dqn / ddqn / per / dueling / ppo`。

### 1）运行一个已训练模型（推荐用 best `.pth`）

```powershell
python gui_game.py --mode battle --algo dqn --model agent/checkpoints/dqn_battle.pth
python gui_game.py --mode battle --algo per --model agent/checkpoints/per_battle.pth
python gui_game.py --mode battle --algo ddqn --model agent/checkpoints/ddqn_battle.pth
python gui_game.py --mode battle --algo dueling --model agent/checkpoints/dueling_battle.pth
python gui_game.py --mode battle --algo ppo --model agent/checkpoints/ppo_battle_best.pth
```

### 2）如果你只有 final

`.final.pth` 是“训练结束时刻”的快照，不一定是 best：

```powershell
python gui_game.py --mode battle --algo dueling --model agent/checkpoints/dueling_battle.final.pth
```

---

## 训练：DQN 变体（课程学习）

入口：`train_dqn_curriculum.py`

### 1）最简单的课程训练

```powershell
python train_dqn_curriculum.py --variant dqn
python train_dqn_curriculum.py --variant ddqn
python train_dqn_curriculum.py --variant per
python train_dqn_curriculum.py --variant dueling
```

默认会跑：

- Phase1（单蛇）→ `agent/checkpoints/<variant>_pretrain.pth` 与 `.final.pth`
- Phase2（对战）→ `agent/checkpoints/<variant>_battle.pth` 与 `.final.pth`

### 2）常用可调参数（只列代码里真实支持的）

```powershell
# 自定义步数
python train_dqn_curriculum.py --variant dqn --steps1 5000000 --steps2 3000000

# 只调 Phase2 的探索与自博弈
python train_dqn_curriculum.py --variant dqn --eps-start2 0.6 --eps-min2 0.05 --sp-prob-start 0.7 --sp-prob-end 0.4 --sp-prob-frac 0.30

# 指定输出路径（避免覆盖）
python train_dqn_curriculum.py --variant per --save2 agent/checkpoints/per_battle_exp1.pth

# Phase2 微调学习率倍率（Phase2 会用 --load，训练器会把 LR 乘上这个倍率）
python train_dqn_curriculum.py --variant ddqn --finetune-lr-mult2 0.35
```

### 3）性能/稳定性环境变量（可选）

这些在 `train_dqn_variants.py` 中读取：

```powershell
# 限制 torch CPU 线程，减少 AsyncVectorEnv worker 互抢
$env:DQN_CPU_THREADS = "1"

# 对手刷新间隔（步数），降低模型加载抖动
$env:RIVAL_UPDATE_INTERVAL = "100000"
```

---

## 训练：PPO（课程学习）

入口：`train_ppo_curriculum.py`（它内部调用 `train_ppo.py`）

### 1）最简单的课程训练

```powershell
python train_ppo_curriculum.py
```

### 2）常用可调参数（与代码一致）

```powershell
# 改步数
python train_ppo_curriculum.py --steps1 10000000 --steps2 10000000

# Phase1/Phase2 并行环境数
python train_ppo_curriculum.py --envs1 64 --envs2 64

# Phase2 自博弈概率（0 表示关闭自博弈，对战对手全随机，会影响上限）
python train_ppo_curriculum.py --self-play-prob2 0.30

# Phase2 fine-tune 学习率（推荐直接用 --finetune-lr2 显式指定“有效 LR”）
python train_ppo_curriculum.py --finetune-lr2 5e-5
```

PPO 训练完成后，脚本会打印：

- `Best Model: agent/checkpoints/ppo_battle_best.pth`
- `Final Snapshot: agent/checkpoints/ppo_battle_best.final.pth`

---

## 日志指标与评估口径（非常重要）

在 Phase2（battle）里，不要只看训练 reward（因为 reward 是“训练用整形信号”的混合），推荐以日志 KPI 为准：

- `Win%`：学习者（通常是 snake0）成为 MVP 的比例
- `S0` / `Score0`：学习者的**游戏得分**（不是训练 reward）
- `EPS`：epsilon-greedy 的探索率（DQN 家族）

为什么这样评估：

- **MVP = 得分最高**（与训练 reward 解耦），因此 `Win%` + `S0` 才反映真正对战强度
- 很多“看似 reward 高”的策略在对战里可能并不强（例如过度 dash、只求存活不吃分）

建议对比时选“低探索尾段”（例如 `EPS≈0.05` 附近）的 `Win%/S0`。

---

## 模型保存规则：best vs final

本项目对模型保存采用两类文件：

- `*.pth`：**best 模型**（训练中会覆盖保存）
- `*.final.pth`：**最终快照**（训练结束时保存一次，不保证最优）

### DQN 变体（`train_dqn_variants.py`）的 best 判定

- Phase2（battle）：优先比较 `S0`，其次比较 `Win%`；满足阈值提升会写入 `--save` 指定的 `.pth`
- Phase1（single）：按平均回报（`Rew`）提升保存 best

因此：

1. 训练/测试优先用 `.pth`
2. 需要“可复现某次训练结束状态”再用 `.final.pth`

---

## 环境规则要点（MVP/掉落/奖励解耦）

以下规则与你看到的 UI/评估一致：

- **得分（score）与训练 reward 分离**：游戏胜负/MVP 只看 score；训练 reward 可以是 score-delta + 额外整形
- **MVP 定义**：分数最高者为 MVP，环境信息里的 `winner_idx` 表示 MVP
- **死亡掉落**：蛇死亡时，**整条身体**会转化为食物
- **dash 整形**：wrapper 会对 dash 做“短窗口内得分增益奖励/否则惩罚”，避免无脑 dash

---

## 常见问题排查（Windows 常见）

### 1）`ModuleNotFoundError: No module named 'torch'`

原因：你用的 python 环境没装 torch（常见于 `conda run` 跑到了另一个环境）。

解决：用你当前能跑 `python gui_game.py` 的那个 python，执行：

```powershell
python -c "import torch; print(torch.__version__)"
```

如果还不行，就在该环境里安装 torch（按你的 CPU/CUDA 版本选择对应安装命令）。

### 2）训练很慢 / FPS 很低

- battle 默认 `num_envs` 会占 CPU，多进程争抢会降速
- 可先限制线程：

```powershell
$env:DQN_CPU_THREADS = "1"
```

### 3）为什么 `.final.pth` 打不过 `.pth`？

正常现象：`.final.pth` 是“最后一步”的快照，不保证最优；`.pth` 才是训练过程中保存的 best。

---

## 自检：运行逻辑测试

该工程带了一个环境逻辑自检脚本（非 pytest）：

```powershell
python test_env_logic.py
```

输出 `--- 逻辑测试通过！ ---` 即表示关键规则（dash、全灭判定、死亡掉落、MVP 语义等）通过。
