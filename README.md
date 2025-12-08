# 多蛇云端生存战（Multi-Snake Cloud Arena）

这是一个将多智能体贪吃蛇环境、Rainbow 强化学习算法与“云游戏”玩法融合的教学/演示项目。服务器专职托管 `MultiSnakeEnv`，负责房间创建、参数化环境搭建、状态步进与广播；客户端（人类或 AI）仅需发送动作并消费返回的状态/奖励即可训练、预测或渲染，非常适合课堂展示、社团比赛或远程联机实验。

## 核心特性

- **云游戏架构**：`network/server.py` 基于 asyncio 托管环境，集中处理客户端参数、创建房间、广播状态与收集动作，确保判定与同步一致。
- **多房间模式**：服务器可创建训练房、人机演练房与联机房三种模式，全部按房主传入的参数（网格、蛇数、tick 等）即时搭建环境。
- **Rainbow DQN 默认策略**：`agent/rainbow_agent.py` + `agent/trainer.py` 集成 NoisyNet、分布式数值、n-step + 优先经验回放，提供高质量基线模型。
- **多形态客户端**：`network/client.py` 默认启动 PyQt GUI（含房间管理 + 日志 + 渲染入口），也保留 CLI（`--mode cli`）以应对无桌面环境。
- **远程训练与推理**：训练房用于纯 AI 训练循环，人机演练房允许 AI/玩家混合对战，联机房面向纯真人或真人 + AI 同屏展示，所有动作都由客户端上传、服务端执行。
- **本地训练脚本**：`rainbow_train.py` 直接调用 `RainbowTrainer`，可在任意客户端独立训练模型并上传至服务器。
- **全中文注释**：关键类与函数均附中文说明，易于在课堂中讲解或二次改造。

## 目录结构

```
agent/
   base_agent.py       # 智能体抽象基类
   config.py           # RainbowConfig 与共享超参
   rainbow_agent.py    # Rainbow 推理与训练步骤
   rainbow_model.py    # CNN Backbone + NoisyLinear
   replay_buffer.py    # 优先经验回放实现
   trainer.py          # 训练循环封装
   utils.py            # 设备/通用工具
env/
   config.py           # MultiSnakeEnv 配置
   multi_snake_env.py  # 多蛇对战环境
network/
   server.py           # 云端服务器（房间/推理）
   client.py           # GUI 客户端 + 渲染入口
   renderer.py         # Pygame 渲染器
   constants.py        # 渲染配色
   utils.py            # 方向转换/端口工具
rainbow_train.py      # 本地训练入口
rainbow_eval.py       # 本地评估 / 人机对练
requirements.txt
README.md
```
## 房间模式

| 房间类型 | 参数说明 | 典型用途 |
| --- | --- | --- |
| `training` | `sparring_bots` 决定陪练蛇数量，系统会额外保留 1 条“自训练 AI”槽位；房主默认观战，负责监控日志/渲染。 | 远程训练、算法调试、教学演示 |
| `ai_duel`（人机演练） | 默认 2 条蛇，创建房间时需填写 `agent_label` 与可选 `model_path`；房主固定为人工控制，另一个槽位交由指定 AI 执行。 | 人机或机机对战，适合展示 |
| `online` | 多个客户端各占一个槽位，所有动作通过 TCP 发送。 | 真人混战或团队赛 |

> 提示：服务器不会托管模型或训练任务；创建训练/对练房后，需要额外的客户端（可运行在同一机器）持续发送动作才能驱动环境。可参考 `rainbow_train.py` 或自定义控制脚本作为“AI 客户端”。

PyQt 客户端在“创建房间”区域新增了 `Mode` 下拉框与专用参数面板：

- `online`：可自由设置蛇数量；
- `training`：启用 `陪练蛇` 选项，服务器会生成对应数量的启发式 BOT，并预留 1 个训练槽位供 AI 客户端占用；
- `ai_duel`：需填写 `AI 标签` 与可选的 `模型路径`，方便记录演练对手来源。

服务端始终以 authoritative 模式运行——状态、奖励、结束判定都只在服务器发生，客户端只渲染最新 `state` 并发送按键，天然避免作弊与状态漂移。

### 环境参数与数据流

- **训练房**：房主默认为观众角色，可设置 `grid_size`、`sparring_bots`（陪练数量）与 `tick_rate`。总蛇数会被自动设置为“陪练 + 1”，预留出的 1 条蛇用于房主自己的训练 AI（固定 agent_count=1），其余陪练会按简单策略陪跑，便于专注于模型迭代。
- **人机演练房**：仍维持 2 条蛇，创建时必须填写 `agent_label`，并可提供 `model_path` 方便记录 AI 模型来源；房主固定为人工控制，另一个槽位交由指定 AI 或远程客户端占据，适合进行“玩家 + AI”展示。
- **联机房**：允许多个真人或真人 + AI 共享同一棋盘。房主负责输入环境参数，其他客户端加入后即可渲染，并把按键或 AI 推理结果发送回服务器。服务器会把所有蛇的状态、分数与回合进度广播给每个成员，保证全员同步。

## Rainbow 训练

`rainbow_train.py` 直接构建 `RainbowConfig` 并交给 `RainbowTrainer`，推荐流程如下：

1. 打开脚本并根据需要修改 `RainbowConfig(...)` 中的字段（如 `grid_size`、`num_snakes`、`total_frames`、`save_interval`、`device` 等）。所有可调超参及默认值都定义在 `agent/config.py`。
2. 运行 `python rainbow_train.py`。脚本会自动创建多蛇环境、经验回放池与 Rainbow 智能体并开始训练循环。
3. 控制台每 `log_interval` 帧输出一次回合统计（帧数、平均得分、当前 epsilon、存活数），训练结束后给出耗时与最终模型路径。
4. 训练过程中会在 `agent/checkpoints/` 生成三类权重文件：
   - `rainbow_snake_latest.pth`：按 `save_interval` 滚动覆盖的最新权重；
   - `rainbow_snake_best.pth`：平均得分刷新时的最佳权重；
   - `rainbow_snake_final.pth`：达到 `total_frames` 后导出的终稿。

上述脚本完全在本地运行多蛇环境，适合离线实验或课堂演示。若想把训练逻辑接入服务器 `training` 房，只需在自定义客户端里复用 `RainbowAgent` 与回放/优化循环，并将动作通过 `network/client.py` 的通信协议发送至房间。训练好的模型可以用 `rainbow_eval.py` 人机/机机对练验证：修改其中的 `model_path`、`grid_size` 等参数后运行脚本即可加载渲染窗口。

## 本地评估 / 演示

`rainbow_eval.py` 现已提供 CLI 化入口，可一键拉起本地环境 + Pygame 渲染窗口：

```cmd
python rainbow_eval.py --model agent/checkpoints/rainbow_snake_latest.pth --grid-size 24 --num-snakes 2 --human-slot 0 --ai-slot 1
```

- `--model`：要加载的 `.pth` 模型路径，未找到时会使用随机权重；
- `--grid-size`、`--num-snakes`、`--num-food`、`--tick`：控制棋盘大小、蛇数量与步进速度；
- `--human-slot`：指定手动控制的槽位，配合 `--spectator` 可改为纯观察；
- `--ai-slot`（可多次填写）：声明由 Rainbow 模型托管的槽位；
- `--epsilon`、`--device`：设置评估用的探索率与推理设备；
- `--no-ai`：仅渲染/手动演示，不加载模型。

评估脚本内部通过 `LocalGameAdapter` 模拟服务器客户端协议，因此可以直接复用 `BattleRenderer` 的 UI/按键体验；若指定多个 `--ai-slot` 会让同一个模型同时托管多条蛇，便于快速机机对战展示。

## 客户端渲染 / 输入

- 加入房间后可随时打开或关闭渲染窗口，不影响房间占位；
- 方向键 `↑↓←→` 会自动转换成“直行/左转/右转”三值动作，并禁止立即调头；
- `ESC` 关闭渲染窗口，回到客户端界面（GUI 或 CLI）；
- 如果玩家槽位为空，客户端会以观察者身份渲染整个棋盘。

## 常见问题

- **服务器会帮忙训练吗？** 不会。服务器只负责环境演化与状态广播；训练/推理逻辑需由客户端运行（例如 `rainbow_train.py` 连接训练房并发送动作）。若要与他人共享模型，可自行同步 `.pth` 文件到约定目录或对象存储。
- **还能用 `network/local_arena.py` 吗？** 该单机脚本已移除；若要离线展示，请在本机启动服务器、创建房间并用 GUI/CLI 客户端加入，以便复用与线上一致的时序/回放逻辑。
- **PPO 还在吗？** `agent/ppo.py` 只保留了兼容提示，调用会直接抛出异常并指导使用 Rainbow 模块。
- **如何自定义 tick rate / grid？** GUI 的“创建房间”标签页提供所有参数输入，若偏好命令行，可用 `--mode cli` 进入交互式菜单，或直接在 `network/server.py` 中扩展配置策略。
- **断线后槽位怎么办？** 服务器检测到 TCP 关闭会自动释放槽位并广播最新房间列表，无需人工清理。
- **GUI 模式需要什么？** 依赖 `PyQt6` 与可用的图形环境（Windows/macOS/Linux 桌面均可安装）。若在纯终端服务器上运行，可加 `--mode cli` 使用命令行菜单。

## 进阶方向

- 添加 WebSocket/HTTP 控制面板，浏览器即可创建房间与观看；
- 把训练进度写入数据库/Prometheus，制作可视化大屏；
- 引入 Self-Play 或联邦训练，探索更多多智能体协作策略；
- 将渲染窗口替换为 WebGL/Three.js，实现零客户端部署。

欢迎在课堂、讲座或社团活动中展示“多蛇云端生存战”，也期待你基于该架构继续拓展更复杂的强化学习玩法！
