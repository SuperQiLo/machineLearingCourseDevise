"""多蛇对战环境定义。

目标：提供“可训练、可复现、接口稳定”的多智能体蛇环境。

关键设计点：
- 环境直接返回 reward（不要在 trainer 里二次拼接），避免不一致与隐性 bug
- reward 默认采用 step penalty + 食物奖励 + 死亡惩罚 + 击杀奖励 + 距离塑形
    其中 step penalty 用于消除“原地转圈拿存活奖励”的局部最优
- 观测采用轻量 6 通道网格，便于卷积网络学习，且每步构造开销小
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Set

import numpy as np

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

        self.snakes: List[Dict] = []
        self.food: List[Tuple[int, int]] = []
        self.steps = 0
        self.grid_shape = (self.width, self.height)

        # 渲染辅助颜色
        self.colors = [
            "green",
            "blue",
            "red",
            "yellow",
            "purple",
            "orange",
            "teal",
            "pink",
            "lime",
            "cyan",
            "magenta",
            "silver",
        ]

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
        rewards = self._compute_rewards(events, prev_heads, curr_heads, prev_food)

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
    ) -> List[float]:
        cfg = self.config
        rewards = [0.0 for _ in range(self.num_snakes)]

        for i, event in enumerate(events):
            if not event.alive and not event.died:
                rewards[i] = 0.0
                continue

            r = float(cfg.step_penalty)

            if event.ate_food:
                r += float(cfg.food_reward)
            if event.kills:
                r += float(cfg.kill_reward) * float(event.kills)
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

            rewards[i] = float(r)

        return rewards

    # ------------------------------------------------------------------
    # Step 子流程
    # ------------------------------------------------------------------
    def _ensure_action_length(self, actions: Sequence[int]) -> None:
        """校验动作数组长度，与蛇数量不符时直接抛错。"""

        if len(actions) != self.num_snakes:
            raise ValueError("actions 数组长度必须与蛇数量一致。")

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
        """返回所有蛇的 6 通道网格观测。"""

        return [
            build_observation_from_snapshot(
                width=self.width,
                height=self.height,
                snakes=self.snakes,
                food=self.food,
                slot=i,
            )
            for i in range(self.num_snakes)
        ]

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


def build_observation_from_snapshot(
    *,
    width: int,
    height: int,
    snakes: Sequence[Dict],
    food: Sequence[Tuple[int, int]],
    slot: int,
) -> np.ndarray:
    """根据状态快照构造单条蛇的 10 通道观测（轻量，可训练）。

    通道说明:
    - 0: 自己头
    - 1: 自己身体(不含头)
    - 2: 敌方头
    - 3: 敌方身体(不含头)
    - 4: 食物
    - 5: 墙（边界格子为 1）
    - 6~9: 自身朝向 one-hot（UP/RIGHT/DOWN/LEFT），在“蛇头格子”置 1
    """

    num_channels = 10

    obs = np.zeros((num_channels, width, height), dtype=np.float32)
    if slot >= len(snakes) or slot < 0:
        return obs

    snake = snakes[slot]
    if not snake.get("alive", True):
        return obs

    body = snake.get("body", [])
    if body:
        hx, hy = body[0]
        if 0 <= hx < width and 0 <= hy < height:
            obs[0, hx, hy] = 1.0

            # 朝向编码（相对动作必须知道当前朝向，否则会学成固定转圈）
            direction = snake.get("direction", Direction.UP)
            if isinstance(direction, Direction):
                dir_index = int(direction.value) % 4
            else:
                try:
                    dir_index = int(direction) % 4
                except Exception:
                    dir_index = 0
            obs[6 + dir_index, hx, hy] = 1.0
        for bx, by in body[1:]:
            if 0 <= bx < width and 0 <= by < height:
                obs[1, bx, by] = 1.0

    for idx, other in enumerate(snakes):
        if idx == slot or not other.get("alive", True):
            continue
        other_body = other.get("body", [])
        if not other_body:
            continue
        ohx, ohy = other_body[0]
        if 0 <= ohx < width and 0 <= ohy < height:
            obs[2, ohx, ohy] = 1.0
        for bx, by in other_body[1:]:
            if 0 <= bx < width and 0 <= by < height:
                obs[3, bx, by] = 1.0

    for fx, fy in food:
        if 0 <= fx < width and 0 <= fy < height:
            obs[4, fx, fy] = 1.0

    obs[5, 0, :] = 1.0
    obs[5, width - 1, :] = 1.0
    obs[5, :, 0] = 1.0
    obs[5, :, height - 1] = 1.0

    return obs
