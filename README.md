# Multi-Snake Battle AI（课程设计工程）

本项目实现了一个“多蛇对战（Battle Snake）”的强化学习训练 + 可视化框架，核心特点：

- **两阶段课程学习（Curriculum）**：Phase 1 单蛇（single）→ Phase 2 多蛇对战（battle）
- **算法**：DQN / DDQN / PER / Dueling（统一训练器 `train_dqn_variants.py`）与 PPO（训练器 `train_ppo.py`）
- **规则与训练解耦**：
  - 游戏胜负与 MVP 依据 `scores`（得分）
  - 训练使用 `env.step()` 返回的 shaping reward（可选叠加 score-delta）

说明：本文档以 Windows + PowerShell 为主；Linux/macOS 仅需把环境变量写法换成 bash。

---

## 目录

- [快速开始（建议按这个顺序）](#快速开始建议按这个顺序)
- [项目结构](#项目结构)
- [环境准备与安装（Windows PowerShell）](#环境准备与安装windows-powershell)
- [可视化运行：`gui_game.py`](#可视化运行gui_gamepy)
- [训练：DQN 变体课程学习：`train_dqn_curriculum.py`](#训练dqn-变体课程学习train_dqn_curriculumpy)
- [训练：PPO 课程学习：`train_ppo_curriculum.py`](#训练ppo-课程学习train_ppo_curriculumpy)
- [直接运行训练器（进阶）](#直接运行训练器进阶)
- [模型文件命名与 best/final 规则](#模型文件命名与-bestfinal-规则)
- [日志指标与评估口径（非常重要）](#日志指标与评估口径非常重要)
- [环境规则要点（MVP/掉落/奖励解耦）](#环境规则要点mvp掉落奖励解耦)
- [常见问题排查（Windows 常见）](#常见问题排查windows-常见)
- [自检：运行逻辑测试](#自检运行逻辑测试)

---

## 快速开始（建议按这个顺序）

在项目根目录（本仓库目录）依次执行：

1）安装依赖

```powershell
pip install -r requirements.txt
```

2）跑环境逻辑自检（推荐先跑，能快速确认规则/依赖没问题）

```powershell
python test_env_logic.py
```

3）打开 GUI 跑一局对战（不加载模型也能运行，用于确认 GUI 正常）

```powershell
python gui_game.py --mode battle
```

4）如果你已经训练出了模型（或你手上有 `.pth/.final.pth`），再用 `--model` 加载

```powershell
python gui_game.py --mode battle --algo dueling --model agent/checkpoints/dueling_battle.final.pth
```

注意：如果 `--model` 指向的文件不存在/无法加载，GUI 仍会启动，但会在控制台打印加载错误，并让该蛇走默认动作（直走）。

---

## 项目结构

```text
machineLearningCourseDevise/
  ├── agent/                     # 各算法网络与推理封装（dqn/ddqn/per/dueling/ppo）
  │   ├── checkpoints/            # 训练输出模型（*.pth / *.final.pth）
  │   └── pool/                   # 自博弈历史池（*.pth，会自动维护）
  ├── env/                        # 环境 + gymnasium wrapper（奖励整形/信息字典）
  ├── net/                        # 网络对战（client/server）
  ├── utils/                      # 自博弈池、渲染等
  ├── gui_game.py                 # 本地可视化（single/battle）
  ├── train_dqn_curriculum.py     # DQN 课程学习调度器（Phase1→Phase2）
  ├── train_dqn_variants.py       # DQN 变体统一训练器（单蛇/对战，自博弈，KPI 日志）
  ├── train_ppo_curriculum.py     # PPO 课程学习调度器（Phase1→Phase2）
  ├── train_ppo.py                # PPO 训练器（单蛇/对战，自博弈，KPI 日志）
  ├── test_env_logic.py           # 环境逻辑自检脚本（非 pytest）
  ├── requirements.txt
  └── scripts/                    # bash 脚本（Linux/macOS 更方便，Windows 可忽略）
```

---

## 环境准备与安装（Windows PowerShell）

### 1）Python 版本

建议 Python 3.10+。

### 2）可选：创建虚拟环境（强烈建议）

如果你不想污染全局 Python：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3）依赖说明（来自 `requirements.txt`）

- `numpy`
- `torch`
- `pygame`
- `PyQt6`
- `gymnasium`

建议用下面命令快速确认关键依赖是否能 import：

```powershell
python -c "import numpy, torch, pygame, PyQt6, gymnasium; print('ok'); print('torch=', torch.__version__)"
```

如果 `torch` 安装失败或与你的 CUDA/CPU 不匹配：以你机器对应的 PyTorch 官方安装命令为准（本文档不硬写版本，避免误导）。

---

## 可视化运行：`gui_game.py`

### 1）参数一览（与代码完全一致）

`gui_game.py` 的参数（每个参数都可单独使用，互不强制依赖）：

- `--mode {single,battle}`（默认 `battle`）
  - 作用：决定环境蛇数。`single`=1 条蛇；`battle`=4 条蛇。
  - 影响：蛇数不同会影响观测、dash 频率、以及“是否存在对手”。

- `--algo {dqn,ppo,ddqn,per,dueling}`（默认 `dqn`）
  - 作用：选择加载/推理的智能体类型（对应 `agent/get_agent()` 的注册表）。
  - 影响：决定网络结构与动作选择逻辑；必须与模型文件的算法类型匹配。

- `--model <path>`（默认不填）
  - 作用：指定权重文件路径（`.pth` 或 `.final.pth`）。
  - 影响：
    - 填了：非人类控制的蛇会尝试加载该模型。
    - 不填：蛇使用默认动作（直走/无策略），仅用于验证 GUI/环境。
  - 注意：路径不存在或加载失败时，GUI 仍会启动，但控制台会打印错误，该蛇会退化为默认动作。

- `--fps <int>`（默认 `10`）
  - 作用：GUI 刷新/步进速度。
  - 影响：仅影响可视化速度，不影响训练。

- `--grid <int>`（默认 `20`，即 20x20）
  - 作用：设置地图边长。
  - 影响：地图越大越难；训练与模型一般默认按 20x20 设计（尤其卷积输入）。

- `--human`（默认关闭）
  - 作用：让 P0（第一条蛇）由键盘控制。
  - 操作：方向键转向；空格键 Dash。
  - 影响：
    - 开启后：只有 P0 是人控，其它蛇才会按 `--model` 加载模型（如果提供）。

- `--food <int>`（默认不填）
  - 作用：覆盖环境最小食物数量。
  - 默认行为：不填时会自动设置 `food_count = max(2, num_snakes)`。
  - 影响：食物越多越“富”；对策略学习与对战节奏影响很大。

### 2）最常用运行方式

启动 battle（4 条蛇），不加载模型：

```powershell
python gui_game.py --mode battle
```

启动 battle，并加载一个模型给所有非人类控制的蛇（`--human` 时只有 P0 是人，P1~P3 会加载模型）：

```powershell
python gui_game.py --mode battle --algo dueling --model agent/checkpoints/dueling_battle.pth
```

启动 single（1 条蛇）并手动控制：

```powershell
python gui_game.py --mode single --human
```

### 3）如何确认模型文件是否存在

PowerShell 下可以用：

```powershell
Test-Path agent\checkpoints\dueling_battle.pth
Test-Path agent\checkpoints\dueling_battle.final.pth
```

如果返回 `False`，说明你还没训练出该文件（或路径写错）。

---

## 训练：DQN 变体课程学习：`train_dqn_curriculum.py`

这个脚本会自动跑两段：

- Phase 1：单蛇（`--single`）预训练 → 产物 `agent/checkpoints/<variant>_pretrain.pth` 及其 `.final.pth`
- Phase 2：对战（battle）微调（使用 `--load <pretrain.final.pth>`）→ 产物 `agent/checkpoints/<variant>_battle.pth` 及其 `.final.pth`

这里的 `<variant>` 指的是你在命令行传入的 `--variant` 原始字符串（脚本会用它拼接文件名）。例如：

- `--variant dueling` → `dueling_battle.pth`
- `--variant ddqn_per_dueling` → `ddqn_per_dueling_battle.pth`

注意：训练器内部会把 `ddqn_per_dueling` 归一化为 `dueling` 算法变体，但保存文件名前缀仍保留你传入的原始值。

### 1）最简单的一键训练（推荐）

```powershell
python train_dqn_curriculum.py --variant dqn
python train_dqn_curriculum.py --variant ddqn
python train_dqn_curriculum.py --variant per
python train_dqn_curriculum.py --variant dueling
```

### 2）参数一览（与代码一致）

`train_dqn_curriculum.py` 支持的参数（它会把参数转发给 `train_dqn_variants.py`）：

- `--variant {dqn,ddqn,per,dueling,ddqn_per,ddqn_per_dueling}`（默认 `dqn`）
  - 作用：选择 DQN 家族变体。
  - 影响：网络结构/损失计算/是否 PER/是否 Dueling。
  - 注意：训练器内部会把 `ddqn_per` 归一化为 `per`、把 `ddqn_per_dueling` 归一化为 `dueling`；
    但保存文件名前缀仍使用你传入的原始字符串（见上文说明）。

- `--steps1 <int>`（默认 `5000000`）
  - 作用：Phase1（单蛇）总交互帧数。
  - 影响：越大越充分，但耗时更长。

- `--steps2 <int>`（默认 `3000000`）
  - 作用：Phase2（对战）总交互帧数。
  - 影响：越大越能适应对战动态与自博弈对手。

- `--force`
  - 作用：强制从 Phase1 重新跑。
  - 默认行为：如果检测到 `agent/checkpoints/<variant>_pretrain.final.pth` 已存在，会跳过 Phase1。

- `--save1 <path>` / `--save2 <path>`（默认不填）
  - 作用：覆盖默认保存路径，避免覆盖已有实验。
  - 建议：做对比实验时一定要改 `--save2`。

- `--num-envs1 <int>`（默认 `128`）
  - 作用：Phase1 并行环境数（越大吞吐越高）。
  - 影响：CPU/内存占用显著上升；过大可能导致系统卡顿或速度反而下降。

- `--num-envs2 <int>`（默认 `32`）
  - 作用：Phase2 并行环境数。
  - 影响：对战环境更重，建议比 Phase1 小。

- `--eps-start1 <float>` / `--eps-min1 <float>`（默认不填）
  - 作用：覆盖 Phase1 的 epsilon-greedy 探索起始值/最小值。
  - 影响：探索更高更随机；探索太低可能早收敛到差策略。

- `--eps-start2 <float>` / `--eps-min2 <float>`（默认不填）
  - 作用：覆盖 Phase2 的探索起始值/最小值。
  - 影响：对战阶段建议保留一定探索以适应对手变化。

- `--sp-prob-start <float>`（默认 `0.7`）
  - 作用：Phase2 前期“使用历史模型当对手”的概率。
  - 影响：越高越偏向自博弈；越低越偏向随机对手（速度快但上限可能低）。

- `--sp-prob-end <float>`（默认 `0.4`）
  - 作用：Phase2 后期自博弈概率。
  - 影响：降低后期自博弈可减少过拟合到池中对手。

- `--sp-prob-frac <float>`（默认 `0.30`）
  - 作用：切换点占总训练步数比例。
  - 例：`0.30` 表示训练到 30% 后，从 `sp-prob-start` 切到 `sp-prob-end`。

- `--finetune-lr-mult2 <float>`（默认 `0.35`）
  - 作用：Phase2 使用 `--load` 微调时，训练器会把学习率乘以该倍率。
  - 影响：倍率更小更稳（但适应更慢）；倍率更大更快但更可能不稳定。

### 3）常用“可复制就跑”的例子

缩短训练（用于验证流程，不追求最终性能）：

```powershell
python train_dqn_curriculum.py --variant dueling --steps1 200000 --steps2 200000
```

给 Phase2 另存为新文件，避免覆盖历史 best：

```powershell
python train_dqn_curriculum.py --variant dueling --save2 agent/checkpoints/dueling_battle_exp1.pth
```

训练完成后立刻用 GUI 验证：

```powershell
python gui_game.py --mode battle --algo dueling --model agent/checkpoints/dueling_battle.pth
```

### 4）训练/自博弈相关环境变量（可选，但很实用）

这些变量由代码读取（不用改代码即可生效）：

- `DQN_CPU_THREADS`：限制 torch CPU 线程数，减少并行环境 worker 互抢（`train_dqn_variants.py` 读取）
- `RIVAL_UPDATE_INTERVAL`：对战阶段对手模型刷新间隔（步数），减少频繁加载模型造成的抖动（`train_dqn_variants.py` 读取，默认 50000）
- `SELF_PLAY_POOL_SIZE`：自博弈历史池最大容量（`utils/self_play.py` 读取，默认 10）

PowerShell 示例（设置后启动训练才会生效）：

```powershell
$env:DQN_CPU_THREADS = "1"
$env:RIVAL_UPDATE_INTERVAL = "100000"
$env:SELF_PLAY_POOL_SIZE = "50"

python train_dqn_curriculum.py --variant dueling
```

---

## 训练：PPO 课程学习：`train_ppo_curriculum.py`

这个脚本同样分两段：

- Phase 1：单蛇 → best 保存到 `agent/checkpoints/ppo_best.pth`（final 为 `ppo_best.final.pth`）
- Phase 2：对战（`--load agent/checkpoints/ppo_best.final.pth`）→ best 保存到 `agent/checkpoints/ppo_battle_best.pth`（final 为 `ppo_battle_best.final.pth`）

### 1）最简单的一键训练

```powershell
python train_ppo_curriculum.py
```

### 2）参数一览（与代码一致）

`train_ppo_curriculum.py` 支持的参数（它会把参数拼接并转发给 `train_ppo.py`）：

- `--steps1 <int>` / `--steps2 <int>`（默认各 `10000000`）
  - 作用：两阶段总 timesteps。
  - 影响：越大越充分，但耗时更长。

- `--envs1 <int>` / `--envs2 <int>`（默认各 `64`）
  - 作用：并行环境数。
  - 影响：越大吞吐越高，但 CPU/内存压力越大；Windows 上过大可能出现明显卡顿。

- `--rollout-steps1 <int>` / `--rollout-steps2 <int>`（默认 `256/128`）
  - 作用：每个环境每次 rollout 的步数。
  - 影响：越大单次更新更“长视野”，但显存/内存更大、更新更慢。

- `--update-epochs1 <int>` / `--update-epochs2 <int>`（默认各 `2`）
  - 作用：每次 rollout 之后 PPO 反复更新的轮数。
  - 影响：越大样本复用越多但更慢；过大可能导致策略更新过头。

- `--minibatch-size1 <int>` / `--minibatch-size2 <int>`（默认各 `4096`）
  - 作用：PPO 更新时的 minibatch 大小。
  - 影响：越大越稳但更吃显存；过小可能噪声大。

- `--lr1 <float>` / `--lr2 <float>`（默认 `2e-4 / 1.5e-4`）
  - 作用：基础学习率。
  - 影响：学习率过大容易不稳定；过小学习很慢。

- `--target-kl1 <float>` / `--target-kl2 <float>`（默认各 `0.015`）
  - 作用：PPO 更新的 KL 早停阈值。
  - 影响：阈值越小越保守（更稳）；越大更新更激进（更快但风险更高）。

- `--finetune-lr2 <float>`（默认 `5e-5`）
  - 作用：Phase2 微调的“有效学习率”（最终会直接传给 `train_ppo.py --finetune-lr`）。
  - 影响：比 `--lr2` 更重要，因为 Phase2 使用 `--load` 时会进入 fine-tune 模式。

- `--finetune-lr-mult2 <float>`（默认不填）
  - 作用：如果你不想显式给 `finetune-lr2`，可用倍率让 fine-tune LR = `lr2 * finetune-lr-mult2`。
  - 注意：只要设置了 `--finetune-lr2`，该倍率会被忽略。

- `--finetune-target-kl2 <float>`（默认 `0.030`）
  - 作用：Phase2 fine-tune 的 KL 早停阈值。
  - 影响：对战阶段通常需要更“能适应”，所以默认比 Phase1 更宽松。

- `--self-play-prob2 <float>`（默认 `0.30`）
  - 作用：Phase2 中对手使用历史池模型的概率（其余情况用随机对手）。
  - 影响：
    - 越大：更偏自博弈，强度更高但可能更慢。
    - 设为 `0`：关闭自博弈，速度更快但上限可能更低。

- `--force`
  - 作用：强制重跑 Phase1。

### 3）常用例子

关闭 Phase2 自博弈（速度更快，但上限可能更低）：

```powershell
python train_ppo_curriculum.py --self-play-prob2 0
```

训练完成后用 GUI 验证：

```powershell
python gui_game.py --mode battle --algo ppo --model agent/checkpoints/ppo_battle_best.pth
```

---

## 直接运行训练器（进阶）

当你只想跑某一段、或希望更细粒度控制时，用训练器脚本：

### 1）DQN 统一训练器：`train_dqn_variants.py`

参数（与代码一致，脚本内 argparse 定义）：

- `--variant {dqn,ddqn,per,dueling,ddqn_per,ddqn_per_dueling}`（默认 `dqn`）
  - 作用：选择 DQN 家族变体。
  - 注意：脚本内部会把 `ddqn_per` 归一化为 `per`、`ddqn_per_dueling` 归一化为 `dueling`。

- `--steps <int>`（默认 `1000000`）
  - 作用：总交互帧数（训练长度）。

- `--single`
  - 作用：单蛇模式（Phase1）。
  - 不传：默认 battle（多蛇对战，Phase2）。

- `--load <path>`（默认不填）
  - 作用：从权重文件继续训练（常用于 Phase2 微调，或断点续训）。
  - 建议：如果你是 curriculum 的 Phase2，优先 load `*_pretrain.final.pth`（对应 Phase1 结束快照）。

- `--save <path>`（默认 `agent/checkpoints/dqn_best.pth`）
  - 作用：best 模型保存路径（训练过程中会覆盖更新）。
  - 旁路文件：训练结束时还会写入 `<save>.final.pth` 作为 final 快照。

- `--num-envs <int>`（默认不填，脚本会按阶段给默认值）
  - 作用：覆盖并行环境数。
  - 影响：越大吞吐越高但更吃 CPU/内存。

- `--eps-start <float>` / `--eps-min <float>`（默认不填）
  - 作用：覆盖 epsilon-greedy 探索率起始/下限。
  - 影响：探索越高越随机；过低可能早收敛到差策略。

- `--sp-prob <float>`（默认 `0.6`）
  - 作用：对战模式下，对手使用历史池模型的概率（其余情况用随机对手）。
  - 说明：如果同时使用 schedule 参数（见下），训练器会在不同阶段动态调整自博弈概率。

- `--sp-prob-start <float>`（默认 `0.7`）
- `--sp-prob-end <float>`（默认 `0.4`）
- `--sp-prob-frac <float>`（默认 `0.30`）
  - 作用：自博弈概率分段调度。
  - 例：训练前 30% 用 `sp-prob-start`，之后用 `sp-prob-end`。

- `--finetune-lr-mult <float>`（默认 `0.5`）
  - 作用：当使用 `--load` 时，训练器会把学习率乘以该倍率（fine-tune 更稳）。
  - 建议：Phase2 微调一般用较小倍率（更稳），但太小会适应很慢。

例子：只跑单蛇（Phase1 的等价形式）：

```powershell
python train_dqn_variants.py --variant dueling --single --steps 500000 --save agent/checkpoints/dueling_pretrain.pth
```

例子：对战微调（Phase2 的等价形式，注意 `--load` 推荐指向 `.final.pth`）：

```powershell
python train_dqn_variants.py --variant dueling --load agent/checkpoints/dueling_pretrain.final.pth --steps 500000 --save agent/checkpoints/dueling_battle.pth
```

### 2）PPO 训练器：`train_ppo.py`

参数（与代码一致，脚本内 argparse 定义）：

- `--single`
  - 作用：单蛇训练（Phase1）。
  - 不传：默认 battle（4 蛇对战，Phase2）。

- `--load <path>`（默认不填）
  - 作用：从权重文件继续训练。
  - 影响：只要设置了 `--load`，训练器就会进入 fine-tune 模式（会调整有效 LR/熵系数/KL 阈值等）。

- `--steps <int>`（默认 `10000000`）
  - 作用：总 timesteps。

- `--envs <int>`（默认 single=64 / battle=64）
  - 作用：并行环境数。
  - 影响：越大吞吐越高但 CPU/内存压力越大。

- `--rollout-steps <int>`（默认 single=256 / battle=512；也可由 curriculum 覆盖）
  - 作用：每个环境每次 rollout 的步数。
  - 影响：越大单次更新更“长视野”，但显存/内存开销更大。

- `--update-epochs <int>`（默认 2）
  - 作用：每次 rollout 后 PPO 更新轮数。

- `--minibatch-size <int>`（默认 4096，且不会超过 batch_size）
  - 作用：PPO 更新时的 minibatch 大小。

- `--lr <float>`（默认 single=2e-4 / battle=1.5e-4）
  - 作用：基础学习率（fine-tune 时会进一步缩放或被 `--finetune-lr` 覆盖）。

- `--target-kl <float>`（默认 0.015）
  - 作用：PPO 更新的 KL 早停阈值。

- `--finetune-lr <float>`（默认不填）
  - 作用：当 `--load` 时，直接指定“有效 fine-tune 学习率”。
  - 建议：Phase2 通常优先显式给这个值（更直观）。

- `--finetune-lr-mult <float>`（默认 0.25）
  - 作用：当 `--load` 且未设置 `--finetune-lr` 时，fine-tune LR = `lr * finetune_lr_mult`。

- `--finetune-target-kl <float>`（默认 0.030）
  - 作用：fine-tune 模式的 KL 早停阈值。
  - 说明：默认比 `target-kl` 更宽松，使策略能更快适应对战动态。

- `--self-play-prob <float>`（默认 battle=0.3，single=0.0）
  - 作用：battle 下对手使用历史池模型的概率（0 表示禁用自博弈，全部随机对手）。

---

## 模型文件命名与 best/final 规则

### 1）两类保存文件

- `*.pth`：训练过程中保存的 **best**（会覆盖更新）
- `*.final.pth`：训练结束时写一次的 **final 快照**（不保证最优）

### 2）DQN 变体：best 判定口径（来自 `train_dqn_variants.py` 的设计）

- battle：更偏向真实对战强度，因此会重点看学习者（snake0）的游戏 KPI（例如 `S0/Win%`）而不是仅看 shaping reward
- single：更偏向回报/生存与吃食，按单蛇训练口径保存 best

因此：

1）想“对战更强”优先用 `.pth`

2）想“复现训练结束那一刻状态”再用 `.final.pth`

---

## 日志指标与评估口径（非常重要）

在 battle（Phase2）阶段，不要只看训练 reward（reward 是整形混合信号），建议以环境提供的游戏 KPI 为准：

- `scores`：每条蛇的游戏得分
- `winner_idx` / `mvp_idx`：MVP（得分最高者；wrapper 里会把它归一到 `winner_idx`）
- `score0` / `delta_score0`：学习者（snake0）的分数与分数增量（在 gymnasium wrapper 的 `info` 里）

推荐对比方式：

- DQN 家族：在探索率较低时段（例如 EPS 接近 `--eps-min`）对比 `Win%`（成为 MVP 的比例）与 `S0/score0`
- PPO：对比 `score0` 与 `winner_idx` 统计（自博弈打开/关闭会影响强度与速度）

---

## 环境规则要点（MVP/掉落/奖励解耦）

这些规则与 `env/battle_snake_env.py` / `env/gymnasium_wrapper.py` 的实现一致：

- **得分（score）与训练 reward 分离**：胜负/MVP 只看 `scores`；训练 reward 可以叠加 score-delta（`use_score_delta_reward`）
- **MVP 定义**：得分最高者，信息字段为 `winner_idx`（wrapper 会同步 `mvp_idx`）
- **死亡掉落**：蛇死亡时整条身体会转化为食物（自检脚本里有断言）
- **Dash**：动作 `3` 为 Dash；环境会消耗长度并进入持续冲刺窗口；wrapper 可对 Dash 有额外 shaping（成功/失败窗口）

---

## 常见问题排查（Windows 常见）

### 1）`ModuleNotFoundError: No module named 'torch'`

原因通常是“你运行脚本的 Python 环境”和“你安装 torch 的环境”不是同一个。

用下面命令确认当前解释器的 torch：

```powershell
python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.__version__)"
```

### 2）训练很慢 / CPU 占用过高

这是并行环境 + 多进程常见现象。建议从降低并行度开始：

- DQN：降低 `--num-envs` 或 curriculum 的 `--num-envs1/--num-envs2`
- PPO：降低 `--envs` 或 curriculum 的 `--envs1/--envs2`

此外，DQN 训练器支持限制 torch 线程，减少 worker 互抢：

```powershell
$env:DQN_CPU_THREADS = "1"
```

### 3）为什么 `.final.pth` 不如 `.pth`？

正常：`.final.pth` 是最后一步快照，不保证最优；`.pth` 才是训练过程中的 best。

---

## 自检：运行逻辑测试

本工程带了一个环境逻辑自检脚本（非 pytest）：

```powershell
python test_env_logic.py
```

看到 `--- 逻辑测试通过！ ---` 说明关键规则（dash、全灭判定、死亡掉落、MVP 语义、wrapper full_obs 输出等）通过。
