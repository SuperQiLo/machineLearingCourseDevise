# 多蛇博弈生存战（Multi-Snake Battle）

一个用于机器学习课程作业的多智能体贪吃蛇系统，集成强化学习训练与局域网对战功能。

## 功能概览

- **环境（`env/multi_snake_env.py`）**：
  - 默认 30x30 网格，可自定义任意蛇数量，随机生成食物。
  - 动作空间：相对转向（直行/左转/右转），避免后退自杀。
  - 状态空间：3 通道网格（自身、敌人、食物），方便 CNN 处理。
  - 奖励：吃食物 +10、存活 -0.01、死亡 -10、击杀 +20、靠近食物 +0.1。

- **算法（`agent/`）**：
  - `model.py`：卷积 Actor-Critic 网络。
  - `ppo.py`：参数共享 PPO，实现动作采样、更新与推理。
  - `train.py`：完整训练循环，自动保存最新/最终模型到 `agent/checkpoints/`。

 - **联机（`network/`）**：
  - `server.py`：Server-Authoritative 设计，可通过 `--tick-rate`/`--countdown` 等命令行参数调整节奏（默认 0.12s/步）并在每局开始前广播倒计时；
  - `client.py`：带“模型训练 / 本地对练 / 联机大厅”三个选项卡的 GUI，大厅渲染、AI 推理和输入驱动，并在倒计时阶段提示玩家所控蛇位；
  - `local_arena.py`：独立的本地对练渲染循环，内置 3s 倒计时、30 FPS 渲染与速度预设下拉，可在 HUD 中明确提示人类操控的蛇；
  - `constants.py`、`utils.py`：集中颜色、方向转换等共用逻辑，避免重复代码。

- **辅助**：`test_env.py` 提供文本模式调试；`main.py` 保留为旧版 CLI（可选，不再推荐）。

## 快速开始

```cmd
pip install -r requirements.txt
python network\client.py
```

客户端脚本会直接弹出“多蛇客户端控制台”（Tk 窗口），并以选项卡方式区分三个场景：

1. **模型训练**：填写网格大小、蛇数量、训练轮数，点击“开始训练”后即刻运行 PPO 训练，日志打印在当前终端；
2. **本地对练**：设置地图、蛇数、是否人工操控、速度/流畅度预设以及模型路径，点击“启动本地对练”后才会打开 Pygame 窗口，方向键操控页面中提示的蛇颜色，回合结束需按 Enter 决定是否继续；
3. **联机大厅**：填写服务器 IP/端口、昵称与可选模型路径，点击按钮即可进入 Pygame 大厅进行局域网对战。

> 如需保留旧的命令行菜单，可运行 `python main.py`，但新的 GUI 已满足全部客户端场景。

建议先通过“本地训练”生成 `agent/checkpoints/ppo_snake_latest.pth`，再开启服务器获得更强的 AI 对手。

### GPU 训练指南

1. **安装 CUDA 版 PyTorch**：根据显卡与驱动版本，到 [PyTorch 官网](https://pytorch.org/) 选择合适的 CUDA wheel，例如：

  ```cmd
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

2. **确认 GPU 可见**：

  ```cmd
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
  ```

3. **选择训练设备**：
   - GUI 的“模型训练”页新增 “训练设备 (auto/cpu/cuda:0)” 输入框；填入 `cuda` 或 `cuda:0` 即可强制使用对应 GPU，`auto` 则在检测到 GPU 时自动启用；
   - 直接运行 `agent/train.py` 时，同样会提示“训练设备”，可输入 `auto`/`cpu`/`cuda:0`；
   - 在代码中使用 `TrainConfig(device="cuda")` 传入配置也支持脚本化控制。

> 如果显式填写了 `cuda` 但本机没有可用 GPU，训练脚本会抛出错误提醒，请改为 `auto` 或 `cpu`。

### 联机大厅操作

1. **服务器端**：
  - 运行 `python network\server.py`（或旧 CLI）可自定义主机地址、端口、tick 速度、倒计时等；默认 0.12s/步、30x30。
  - 例如：

    ```cmd
    python network\server.py --tick-rate 0.1 --countdown 2.0 --grid-size 32
    ```

  - 服务器终端可随时 Ctrl+C 关闭。

2. **客户端**：
  - 运行 `python network\client.py`，在 “联机大厅” 选项卡填写服务器 IP、端口、昵称与可选模型路径，然后点击按钮即可启动大厅。
  - 大厅规则：
     - **单房间、首位房主**：第一个进入的玩家自动成为房主；房主离开时自动移交。
  - **开局条件**：至少 2 名参赛者，并且全员准备就绪后，只有房主点击“开始游戏”按钮才能开局；游戏内蛇条数与参赛人数完全一致。
  - 大厅操作：
    - 右侧提供“模式切换 / 准备就绪 / 开始游戏”按钮，可完全通过图形化界面操作；
    - 仍可使用快捷键：`F1`/`TAB`（人工）、`F2`/`A`（AI）、`F3`/`O`（观察）、`SPACE`（准备/取消）、`ESC`（退出客户端）。
   - 进入游戏后：
     - 开局前会显示 3 秒倒计时，并在屏幕底部提示“你控制的蛇/颜色”；
     - 方向键 `↑↓←→` 直接控制绝对方向（系统自动转换为相对动作并禁止调头），界面左下角会显示“我的蛇颜色”；
     - AI 或观察模式不会发送键盘动作，可专注围观或查看提示条。
  - 回合结束：屏幕中央会弹出按钮提示，玩家可选择“继续作战”（自动准备下一局）、“休息一下”（停留在大厅）或直接退出客户端，实现“由玩家决定是否继续”的体验。

3. **本地训练/对练**：
  - 在 “模型训练” 选项卡填写参数后开始训练，模型会保存到 `agent/checkpoints/`；
  - 在 “本地对练” 选项卡填写参数并点击启动，即可在 Pygame 窗口用方向键控制指定蛇，可在启动前选择“慢速/标准/快速”预设、指定 “AI 推理设备 (auto/cpu/cuda:0)” 以决定 PPO 推理运行在 CPU 还是 GPU，开局前会显示 3 秒倒计时与身份提示，局末弹出提示让玩家决定是否继续下一局。

> 脚本化使用同样支持 `start_local_arena(device="cuda")`，若显卡可用会将 PPO 模型加载到 GPU 以加速对练推理。

## 目录结构

```
agent/
  model.py        # 卷积 Actor-Critic
  ppo.py          # PPO 算法与 Memory
'train.py'       # 训练脚本
env/
  multi_snake_env.py  # 多蛇环境逻辑
network/
  server.py       # 局域网服务器
  client.py       # Pygame 客户端
requirements.txt
README.md
```

## 改进建议
- 增加 Lobby/角色选择，支持观战或 2v2。
- 引入插值/缓冲机制，缓解网络延迟。
- 加入更多奖励 shaping（例如压制对手、控制区域）。

欢迎扩展并在课堂演示中展示 1 人对 3 AI 或 2 人对 2 AI 的精彩对战！
