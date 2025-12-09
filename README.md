# 多蛇云端生存战（Multi-Snake Cloud Arena）

基于 `MultiSnakeEnv` 的云端多人贪吃蛇实验平台。服务器集中托管环境、房间与状态步进；客户端（真人或 AI）只需连入房间发送动作即可训练、对战或渲染。项目面向教学演示、算法实验与远程联机活动。

## 亮点速览

- **Authoritative Server**：`network/server.py` 使用 asyncio 管理所有房间、成员与动作收集，统一判定胜负与奖励，杜绝状态漂移。
- **单一 Online 模式**：移除历史多模式分支，房主只需配置网格、蛇数、食物、步数与 tick 间隔即可开局，真人与 AI 客户端共用协议。
- **PyQt 客户端**：`network/client.py` 集成房间列表、参数表单、成员管理、日志与渲染入口；CLI 模式（`--mode cli`）适合无桌面环境。
- **Rainbow 基线**：`agent/` 目录提供 Rainbow DQN（NoisyNet、分布式回报、n-step、优先经验回放）及训练脚本，便于快速获得强力 AI。
- **Pygame 渲染器**：`network/renderer.py` 展示房间号、用户-角色-得分与步数，支持真人按键或纯观战，退出时自动处理角色/槽位。
- **中文注释 & 教学友好**：主要模块均附中文说明，方便课堂讲解与二次开发。

## 目录速查

```
agent/
   base_agent.py       # 智能体抽象基类
   config.py           # RainbowConfig 与共享超参
   rainbow_agent.py    # Rainbow 推理/训练逻辑
   rainbow_model.py    # CNN + NoisyLinear 建模
   replay_buffer.py    # 优先经验回放
   trainer.py          # 训练循环封装
   utils.py            # 设备/随机数工具
env/
   config.py           # 环境配置数据结构
   multi_snake_env.py  # 多蛇对战环境
network/
   server.py           # 云端服务器
   client.py           # PyQt/CLI 客户端
   renderer.py         # Pygame 渲染窗口
   utils.py            # 方向与端口辅助
rainbow_train.py      # 本地训练入口
rainbow_eval.py       # 本地评估/演示
requirements.txt
README.md
```

## 快速开始

1. **安装依赖**：`pip install -r requirements.txt`
2. **启动服务器**：`python network/server.py --host 0.0.0.0 --port 5555`
3. **启动客户端**：`python network/client.py`（Windows/Mac/Linux 桌面）或 `python network/client.py --mode cli`
4. **创建房间**：在 GUI 的“房间参数”面板设置网格、蛇数、食物、步数与 tick（毫秒）后点击“创建房间”。
5. **加入/准备**：其他客户端选择房间 -> 点击“加入” -> 在“玩家”面板切换 `human`/`ai`/`spectator`。
6. **开局**：非观战成员全部准备后，房主点击“开始游戏”，倒计时结束即进入对局。

> 服务器只负责环境演化；若要训练 AI，请在本地运行 `rainbow_train.py` 或自定义脚本，作为房间中的 `ai` 成员上传动作。

## 房间与角色

- **角色规则**：`human` 和 `ai` 占用槽位并参与战斗；`spectator` 释放槽位。AI 角色在未加载模型前无法准备。
- **自动判负**：对局进行中退出渲染窗口或切换至 `spectator` 会立即判定蛇死亡，避免僵尸蛇；对局结束后自动恢复退出前的角色。
- **广播内容**：服务器会推送成员列表、蛇状态、得分、颜色与步数，客户端据此渲染 UI/HUD。

## 渲染与输入

- `network/renderer.py` 使用方向键控制（禁止立即调头），同时显示房间号、步数以及“用户名-角色：得分”。
- ESC 退出渲染窗口：如果仍在对局，会自动切换到观战；对局结束则保持原角色，方便下一局。

## Rainbow 训练/评估

- **训练**：
  ```bash
  python rainbow_train.py --grid-size 24 --num-snakes 4 --total-frames 2_000_000
  ```
  输出 `agent/checkpoints/` 目录下的 latest/best/final 模型，可在客户端加载。
- **评估/演示**：
  ```bash
  python rainbow_eval.py --model agent/checkpoints/rainbow_snake_latest.pth \
      --grid-size 24 --num-snakes 2 --human-slot 0 --ai-slot 1
  ```
  支持多人 AI、纯观战或人工演示，复用与服务器一致的渲染体验。

## 常见问题

- **是否支持离线单机？** 可运行 `rainbow_eval.py` 进行本地演示；标准流程仍推荐“服务器 + 客户端”以保持一致的时序与协议。
- **如何扩展参数/玩法？** 直接修改 `RoomConfig` 与前端表单即可，例如增加额外道具、随机事件或自定义奖励。
- **渲染器需要哪些依赖？** Windows 默认可用；Linux/macOS 需安装 SDL 运行库及可用的显示环境。
- **AI 掉线后会怎样？** 服务器检测断开后即释放槽位并广播最新房间状态；如果该蛇在对局中，会被判定阵亡。

## 未来方向

- WebSocket/Web UI 版房间管理和观战
- 训练/评估指标上报（Prometheus、InfluxDB 等）
- Self-Play、Curriculum 等强化学习策略扩展
- WebGL/Three.js 渲染器，支持零客户端观看

欢迎在课程、讲座、社团活动中展示“多蛇云端生存战”，也欢迎 Issue/PR 共同完善！
