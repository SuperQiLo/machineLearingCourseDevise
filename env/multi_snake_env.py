"""多蛇对战环境定义。

目标：提供“可训练、可复现、接口稳定”的多智能体蛇环境。

关键设计点：
- 环境直接返回 reward（不要在 trainer 里二次拼接），避免不一致与隐性 bug
- reward 默认采用 step penalty + 食物奖励 + 死亡惩罚 + 击杀奖励 + 距离塑形
    其中 step penalty 用于消除“原地转圈拿存活奖励”的局部最优
- 观测采用 (3, 84, 84) RGB 图像，便于 CNN 学习
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Set

import numpy as np
import pygame

from env.config import EnvConfig


@dataclass
class StepEvent:
    """记录单条蛇在一步内的关键事件，便于统一计算奖励与日志。"""

    alive: bool = True
    died: bool = False
    ate_food: bool = False
    kills: int = 0
    killed_by: Optional[int] = None


class Action(Enum):
    """相对动作枚举。"""

    STRAIGHT = 0
    LEFT = 1
    RIGHT = 2


class Direction(Enum):
    """绝对方向枚举。"""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class MultiSnakeEnv:
    """多蛇博弈生存战的核心环境类。"""

    def __init__(
        self,
        width: int = 30,
        height: int = 30,
        num_snakes: int = 4,
        num_food: int = 6,
        max_steps: int = 1500,
        *,
        config: Optional[EnvConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        """配置地图尺寸、蛇/食物数量以及最大步数等基本参数，可通过 `EnvConfig` 覆盖。"""

        cfg = config or EnvConfig(
            width=width,
            height=height,
            num_snakes=num_snakes,
            num_food=num_food,
            max_steps=max_steps,
        )

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.config = cfg
        self.width = cfg.width
        self.height = cfg.height
        self.num_snakes = cfg.num_snakes
        self.num_food = cfg.num_food
        self.max_steps = cfg.max_steps

        # 统一的配色与单元尺寸，供渲染和协议复用
        self.colors: List[Tuple[int, int, int]] = [
            (0, 255, 159),
            (255, 0, 85),
            (0, 243, 255),
            (255, 204, 0),
            (189, 0, 255),
            (255, 153, 0),
        ]
        self.cell_size = 20

        self.snakes: List[Dict] = []
        self.food: List[Tuple[int, int]] = []
        self.steps = 0
        self._surface = None
        # 记录每条蛇的上一次动作，用于计算重复动作惩罚
        self._last_actions: List[int] = [0] * num_snakes

    # ------------------------------------------------------------------
    # 环境生命周期
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None) -> List[np.ndarray]:
        """重置环境并返回所有蛇的初始观测。"""

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.snakes = []
        self.steps = 0

        spawn_layouts = self._generate_spawn_layouts()
        for i, layout in enumerate(spawn_layouts):
            self.snakes.append(
                {
                    "id": i,
                    "body": layout["body"],
                    "direction": layout["direction"],
                    "alive": True,
                    "score": 0,
                    "steps_alive": 0,
                }
            )

        self.food = []
        while len(self.food) < self.num_food:
            self._spawn_food()

        # 重置动作记录
        self._last_actions = [0] * self.num_snakes

        return self._get_observations()

    # ------------------------------------------------------------------
    # 核心交互
    # ------------------------------------------------------------------
    def step(self, actions: Sequence[int]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        """推进一步模拟，输出 (obs, rewards, dones, info)。"""

        self._ensure_action_length(actions)

        self.steps += 1

        prev_heads = [snake["body"][0] if snake.get("alive", True) and snake.get("body") else None for snake in self.snakes]
        prev_food = list(self.food)

        events = self._init_step_events()
        next_heads, will_grow = self._plan_next_heads(actions)
        snakes_to_die, kill_credits = self._detect_collisions(next_heads, will_grow, events)
        for owner, victims in kill_credits.items():
            events[owner].kills += len(victims)
        dones = self._apply_movements(next_heads, will_grow, snakes_to_die, events)

        curr_heads = [snake["body"][0] if snake.get("alive", True) and snake.get("body") else None for snake in self.snakes]
        rewards = self._compute_rewards(events, prev_heads, curr_heads, prev_food, actions)

        alive_count = sum(1 for snake in self.snakes if snake["alive"])
        game_over = alive_count <= 1 or self.steps >= self.max_steps
        if game_over:
            for i, snake in enumerate(self.snakes):
                if snake["alive"]:
                    dones[i] = True

        info = self._build_info(events, alive_count, game_over)
        return self._get_observations(), rewards, dones, info

    def _compute_rewards(
        self,
        events: Sequence[StepEvent],
        prev_heads: Sequence[Optional[Tuple[int, int]]],
        curr_heads: Sequence[Optional[Tuple[int, int]]],
        prev_food: Sequence[Tuple[int, int]],
        actions: Sequence[int],
    ) -> List[float]:
        """计算每条蛇的即时奖励。
        
        奖励设计原则：
        1. 每步都有微小惩罚 (step_penalty)，防止无意义移动
        2. 吃到食物有大奖励 (food_reward)，引导蛇主动寻找食物
        3. 死亡有惩罚 (death_penalty)，避免冒险行为
        4. 击杀其他蛇有奖励 (kill_reward)，鼓励竞争
        5. 距离塑形 (distance_shaping)，靠近食物有微小正奖励
        6. 重复动作惩罚 (repetition_penalty)，防止原地转圈
        """
        cfg = self.config
        rewards = [0.0 for _ in range(self.num_snakes)]

        for i, event in enumerate(events):
            # 已死亡且本帧未死的蛇不给奖励
            if not event.alive and not event.died:
                rewards[i] = 0.0
                continue

            # 基础每步惩罚
            r = float(cfg.step_penalty)

            # 吃到食物奖励
            if event.ate_food:
                r += float(cfg.food_reward)
            # 击杀奖励
            if event.kills:
                r += float(cfg.kill_reward) * float(event.kills)
            # 死亡惩罚
            if event.died:
                r += float(cfg.death_penalty)

            # 距离塑形：让“靠近最近食物”有微弱增益，减少稀疏性
            if cfg.distance_shaping_scale != 0.0 and prev_food:
                ph = prev_heads[i] if i < len(prev_heads) else None
                ch = curr_heads[i] if i < len(curr_heads) else None
                if ph is not None and ch is not None and event.alive:
                    prev_d = min(abs(ph[0] - fx) + abs(ph[1] - fy) for fx, fy in prev_food)
                    curr_d = min(abs(ch[0] - fx) + abs(ch[1] - fy) for fx, fy in self.food) if self.food else prev_d
                    r += float(cfg.distance_shaping_scale) * float(prev_d - curr_d)

            # 重复动作惩罚：连续执行相同动作时给予额外惩罚，打破转圈圈局部最优
            if hasattr(cfg, 'repetition_penalty') and cfg.repetition_penalty != 0.0:
                if i < len(actions) and i < len(self._last_actions):
                    if actions[i] == self._last_actions[i]:
                        r += float(cfg.repetition_penalty)

            rewards[i] = float(r)

        # 更新上一次动作记录
        for i, action in enumerate(actions):
            if i < len(self._last_actions):
                self._last_actions[i] = action

        return rewards

    # ------------------------------------------------------------------
    # Step 子流程
    # ------------------------------------------------------------------
    def _ensure_action_length(self, actions: Sequence[int]) -> None:
        """校验动作数组长度，与蛇数量不符时直接抛错。"""

        if len(actions) != self.num_snakes:
            raise ValueError(f"actions 长度为 {len(actions)}，必须与蛇数量 {self.num_snakes} 一致。")

    def _init_step_events(self) -> List[StepEvent]:
        """为每条蛇初始化事件记录，默认继承当前存活状态。"""

        return [StepEvent(alive=snake.get("alive", True)) for snake in self.snakes]

    def _plan_next_heads(self, actions: Sequence[int]) -> Tuple[List[Optional[Tuple[int, int]]], List[bool]]:
        """根据相对动作计算下一帧蛇头位置，并判断是否会成长。"""

        next_heads: List[Optional[Tuple[int, int]]] = [None] * self.num_snakes
        will_grow: List[bool] = [False] * self.num_snakes
        for idx, snake in enumerate(self.snakes):
            if not snake["alive"]:
                continue

            new_dir = self._turn(snake["direction"], actions[idx])
            snake["direction"] = new_dir

            dx, dy = self._dir_delta(new_dir)
            head_x, head_y = snake["body"][0]
            next_head = (head_x + dx, head_y + dy)
            next_heads[idx] = next_head
            will_grow[idx] = next_head in self.food

        return next_heads, will_grow

    def _detect_collisions(
        self,
        next_heads: Sequence[Optional[Tuple[int, int]]],
        will_grow: Sequence[bool],
        events: List[StepEvent],
    ) -> Tuple[Set[int], Dict[int, Set[int]]]:
        """检测墙面、头对头与身体碰撞，返回死亡集合与击杀归属。"""

        snakes_to_die: Set[int] = set()
        kill_credits: Dict[int, Set[int]] = {}

        for idx, head in enumerate(next_heads):
            if head is None or not self.snakes[idx]["alive"]:
                continue
            x, y = head
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                snakes_to_die.add(idx)

        head_counts: Dict[Tuple[int, int], List[int]] = {}
        for idx, head in enumerate(next_heads):
            if head is None or not self.snakes[idx]["alive"]:
                continue
            head_counts.setdefault(head, []).append(idx)

        for indices in head_counts.values():
            if len(indices) > 1:
                snakes_to_die.update(indices)

        occupied_future: Dict[Tuple[int, int], int] = {}
        for idx, snake in enumerate(self.snakes):
            if not snake["alive"]:
                continue
            body = snake["body"]
            body_space = body if will_grow[idx] else body[:-1]
            for cell in body_space:
                occupied_future[cell] = idx

        for idx, head in enumerate(next_heads):
            if head is None or not self.snakes[idx]["alive"]:
                continue
            owner = occupied_future.get(head)
            if owner is None:
                continue
            snakes_to_die.add(idx)
            if owner != idx:
                killers = kill_credits.setdefault(owner, set())
                killers.add(idx)
                events[idx].killed_by = owner

        return snakes_to_die, kill_credits

    def _apply_movements(
        self,
        next_heads: Sequence[Optional[Tuple[int, int]]],
        will_grow: Sequence[bool],
        snakes_to_die: Set[int],
        events: List[StepEvent],
    ) -> List[bool]:
        """执行位移、生长与死亡判定，返回 dones 列表。"""

        dones = [False for _ in range(self.num_snakes)]
        for idx, snake in enumerate(self.snakes):
            if not snake["alive"]:
                continue

            if idx in snakes_to_die:
                snake["alive"] = False
                dones[idx] = True
                events[idx].alive = False
                events[idx].died = True
                continue

            new_head = next_heads[idx]
            if new_head is None:
                snake["alive"] = False
                dones[idx] = True
                events[idx].alive = False
                events[idx].died = True
                continue

            snake["body"].insert(0, new_head)
            if will_grow[idx]:
                if new_head in self.food:
                    self.food.remove(new_head)
                self._spawn_food()
                snake["score"] += 1
                events[idx].ate_food = True
            else:
                snake["body"].pop()

            snake["steps_alive"] += 1

        return dones

    def _build_info(self, events: List[StepEvent], alive_count: int, game_over: bool) -> Dict:
        """组装 info 字典，保持与旧版服务器兼容。"""

        return {
            "steps": self.steps,
            "alive_count": alive_count,
            "scores": [snake["score"] for snake in self.snakes],
            "game_over": game_over,
            "events": [asdict(event) for event in events],
        }

    # ------------------------------------------------------------------
    # 观测与渲染
    # ------------------------------------------------------------------
    def _get_observations(self) -> List[np.ndarray]:
        """返回所有蛇的图像观测 (3, 84, 84)。"""
        
        # 渲染全局图像
        full_surface = self._render_full_surface()
        
        obs_list = []
        for i in range(self.num_snakes):
            obs = self._get_agent_observation(full_surface, i)
            obs_list.append(obs)
        return obs_list

    def _render_full_surface(self) -> pygame.Surface:
        """渲染整个游戏画面（精致霓虹风格，用于 RL 观测或静态导出）。"""

        if not pygame.get_init():
            pygame.init()

        # 配色方案 (与 Renderer 保持一致)
        bg_color = (13, 15, 26)
        grid_color = (25, 30, 45)
        food_color = (255, 0, 85)
        snake_colors = self.colors

        width_px = self.width * self.cell_size
        height_px = self.height * self.cell_size

        if not self._surface or self._surface.get_size() != (width_px, height_px):
            self._surface = pygame.Surface((width_px, height_px), flags=pygame.SRCALPHA)

        surf = self._surface
        surf.fill(bg_color)

        # 1. 绘制网格
        for x in range(self.width + 1):
            pygame.draw.line(surf, grid_color, (x * self.cell_size, 0), (x * self.cell_size, height_px))
        for y in range(self.height + 1):
            pygame.draw.line(surf, grid_color, (0, y * self.cell_size), (width_px, y * self.cell_size))

        # 2. 绘制食物
        for fx, fy in self.food:
            center = ((fx + 0.5) * self.cell_size, (fy + 0.5) * self.cell_size)
            radius = self.cell_size * 0.35
            
            # 辉光 (静态环境使用固定发光)
            glow_radius = radius * 2.0
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*food_color, 60), (glow_radius, glow_radius), glow_radius)
            surf.blit(glow_surf, (center[0] - glow_radius, center[1] - glow_radius))
            
            pygame.draw.circle(surf, (255, 255, 255), center, radius)
            pygame.draw.circle(surf, food_color, center, radius, width=2)

        # 3. 绘制蛇
        for i, snake in enumerate(self.snakes):
            if not snake.get("alive", True):
                continue

            base_color = snake_colors[i % len(snake_colors)]
            body = snake.get("body", [])
            if not body: continue

            # 绘制连接的身体
            for idx in range(len(body) - 1):
                p1 = body[idx]
                p2 = body[idx+1]
                
                start_px = ((p1[0] + 0.5) * self.cell_size, (p1[1] + 0.5) * self.cell_size)
                end_px = ((p2[0] + 0.5) * self.cell_size, (p2[1] + 0.5) * self.cell_size)
                
                shade = max(0.4, 1.0 - (idx / len(body)) * 0.6)
                color = tuple(int(c * shade) for c in base_color)
                
                width = self.cell_size * 0.8
                pygame.draw.line(surf, color, start_px, end_px, int(width))
                pygame.draw.circle(surf, color, start_px, width // 2)
                pygame.draw.circle(surf, color, end_px, width // 2)

            # 绘制头部
            head = body[0]
            head_px = ((head[0] + 0.5) * self.cell_size, (head[1] + 0.5) * self.cell_size)
            
            # 头部发光
            hg_radius = self.cell_size * 1.0
            hg_surf = pygame.Surface((hg_radius*2, hg_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(hg_surf, (*base_color, 120), (hg_radius, hg_radius), hg_radius)
            surf.blit(hg_surf, (head_px[0] - hg_radius, head_px[1] - hg_radius))
            
            pygame.draw.circle(surf, (255, 255, 255), head_px, self.cell_size * 0.45)
            pygame.draw.circle(surf, base_color, head_px, self.cell_size * 0.45, width=2)

        return surf

    def _get_agent_observation(self, surface: pygame.Surface, agent_idx: int) -> np.ndarray:
        """获取特定 Agent 的观测图像 (3, 84, 84)。
        
        目前实现为 Global View，即所有 Agent 看到的一样。
        如果需要 Egocentric，需要在这里做裁剪或旋转。
        鉴于这是 Snake 游戏，全局视野通常较好。
        """
        if agent_idx >= len(self.snakes) or not self.snakes[agent_idx]["alive"]:
            # 死亡则返回全黑或特定画面
            return np.zeros((3, 84, 84), dtype=np.float32)

        # 缩放到 84x84
        target_size = (84, 84)
        scaled = pygame.transform.scale(surface, target_size)
        
        # 转为 numpy (W, H, 3)
        # pygame.surfarray.pixels3d 返回的是 (W, H, 3) 且 referencing the surface pixels
        # 复制一份 channel order (H, W, 3) -> Transpose to (3, H, W) for PyTorch
        
        # pixels3d: (width, height, rgb)
        pixels = pygame.surfarray.array3d(scaled)
        # PyTorch expect (C, H, W). pygame is (W, H, C) usually?
        # Let's verify: pygame x is width, y is height.
        # Transpose to (C, H, W) -> (2, 1, 0)
        
        # Normalize to 0-1
        obs = pixels.transpose(2, 1, 0).astype(np.float32) / 255.0
        
        return obs

    def render_text(self) -> None:
        """简易字符渲染，方便调试。"""

        grid = [["." for _ in range(self.height)] for _ in range(self.width)]

        for fx, fy in self.food:
            grid[fx][fy] = "F"

        for i, snake in enumerate(self.snakes):
            if not snake["alive"]:
                continue
            for idx, (bx, by) in enumerate(snake["body"]):
                char = str(i)
                if idx == 0:
                    char = char.upper()
                grid[bx][by] = char

        border = "-" * (self.width + 2)
        print(border)
        for y in range(self.height):
            row = "|"
            for x in range(self.width):
                row += grid[x][y]
            row += "|"
            print(row)
        print(border)

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------
    @staticmethod
    def _turn(direction: Direction, action: int) -> Direction:
        """根据相对动作更新方向。"""

        if action == Action.LEFT.value:
            return Direction((direction.value - 1) % 4)
        if action == Action.RIGHT.value:
            return Direction((direction.value + 1) % 4)
        return direction

    @staticmethod
    def _dir_delta(direction: Direction) -> Tuple[int, int]:
        """方向转位移增量。"""

        if direction == Direction.UP:
            return 0, -1
        if direction == Direction.DOWN:
            return 0, 1
        if direction == Direction.LEFT:
            return -1, 0
        return 1, 0

    def _spawn_food(self) -> None:
        """生成新的食物，避免与蛇身体冲突。"""

        while True:
            pos = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))

            occupied = pos in self.food
            if not occupied:
                for snake in self.snakes:
                    if snake["alive"] and pos in snake["body"]:
                        occupied = True
                        break

            if not occupied:
                self.food.append(pos)
                break


    # ------------------------------------------------------------------
    # 启动/布置辅助
    # ------------------------------------------------------------------
    def _generate_spawn_layouts(self) -> List[Dict]:
        """根据地图尺寸随机生成互不重叠的初始身体。"""

        cells = [
            (x, y)
            for x in range(2, max(3, self.width - 2))
            for y in range(2, max(3, self.height - 2))
        ]
        if not cells:
            raise ValueError("地图尺寸过小，无法放置蛇。")

        random.shuffle(cells)
        layouts: List[Dict] = []
        occupied: set[Tuple[int, int]] = set()

        for head in cells:
            for direction in Direction:
                body = []
                dx, dy = self._dir_delta(direction)
                valid = True
                for seg in range(3):
                    cx = head[0] - dx * seg
                    cy = head[1] - dy * seg
                    if not (0 <= cx < self.width and 0 <= cy < self.height):
                        valid = False
                        break
                    if (cx, cy) in occupied:
                        valid = False
                        break
                    body.append((cx, cy))
                if not valid:
                    continue
                layouts.append({"body": body, "direction": direction})
                occupied.update(body)
                break
            if len(layouts) >= self.num_snakes:
                break

        if len(layouts) < self.num_snakes:
            raise RuntimeError("无法初始化所有蛇，请增大地图尺寸或减少蛇数量。")
        return layouts
