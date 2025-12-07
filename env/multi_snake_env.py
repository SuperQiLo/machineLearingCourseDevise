"""多蛇对战环境定义。

该模块实现了一个支持 *任意数量* 蛇同场竞技的网格世界环境，
提供了基于网格的全局观测、相对动作空间以及奖励塑形策略，
可用于强化学习训练、局域网服务器或本地检验。
"""

from __future__ import annotations

import random
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


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
    ) -> None:
        """配置地图尺寸、蛇/食物数量以及最大步数等基本参数。"""
        self.width = width
        self.height = height
        self.num_snakes = num_snakes
        self.num_food = num_food
        self.max_steps = max_steps

        self.snakes: List[Dict] = []
        self.food: List[Tuple[int, int]] = []
        self.steps = 0
        self.grid_shape = (width, height)

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
    def reset(self) -> List[np.ndarray]:
        """重置环境并返回所有蛇的初始观测。"""

        self.snakes = []
        self.steps = 0

        spawn_layouts = self._generate_spawn_layouts()
        for i, layout in enumerate(spawn_layouts):
            body = layout["body"]
            self.snakes.append(
                {
                    "id": i,
                    "body": body,
                    "direction": layout["direction"],
                    "alive": True,
                    "score": 0,
                    "steps_alive": 0,
                    "last_food_distance": None,
                }
            )

        self.food = []
        while len(self.food) < self.num_food:
            self._spawn_food()

        # 初始化每条蛇的最近食物距离，方便引导奖励
        for snake in self.snakes:
            snake["last_food_distance"] = self._nearest_food_distance(snake["body"][0])

        return self._get_observations()

    # ------------------------------------------------------------------
    # 核心交互
    # ------------------------------------------------------------------
    def step(self, actions: Sequence[int]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        """推进一步模拟。

        Args:
            actions: 每条蛇的相对动作列表 (0=直行,1=左转,2=右转)。
        Returns:
            obs, rewards, dones, info
        """

        if len(actions) != self.num_snakes:
            raise ValueError("actions 数组长度必须与蛇数量一致。")

        self.steps += 1
        rewards = [0.0 for _ in range(self.num_snakes)]
        dones = [False for _ in range(self.num_snakes)]
        info: Dict = {}

        next_heads: List[Optional[Tuple[int, int]]] = [None] * self.num_snakes

        # 1) 根据动作更新方向并计算下一帧的蛇头坐标
        for i, snake in enumerate(self.snakes):
            if not snake["alive"]:
                continue

            action = actions[i]
            new_dir = self._turn(snake["direction"], action)
            snake["direction"] = new_dir

            dx, dy = self._dir_delta(new_dir)
            head_x, head_y = snake["body"][0]
            next_heads[i] = (head_x + dx, head_y + dy)

        # 2) 检测墙面、头撞头、头撞身体等致死事件
        snakes_to_die = set()

        # 2.1 撞墙
        for i, head in enumerate(next_heads):
            if head is None or not self.snakes[i]["alive"]:
                continue
            x, y = head
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                snakes_to_die.add(i)
                rewards[i] -= 10

        # 2.2 头对头
        head_counts: Dict[Tuple[int, int], List[int]] = {}
        for i, head in enumerate(next_heads):
            if head is None or not self.snakes[i]["alive"] or i in snakes_to_die:
                continue
            head_counts.setdefault(head, []).append(i)

        for indices in head_counts.values():
            if len(indices) > 1:
                for idx in indices:
                    snakes_to_die.add(idx)
                    rewards[idx] -= 10

        # 2.3 头撞身体（考虑尾巴是否移动）
        future_bodies: List[set] = []
        for i, snake in enumerate(self.snakes):
            if not snake["alive"]:
                future_bodies.append(set())
                continue

            will_grow = next_heads[i] in self.food if next_heads[i] else False
            body = snake["body"]
            # 如果不生长，尾巴会移除，最后一个格可以被占用
            occupied = body if will_grow else body[:-1]
            future_bodies.append(set(occupied))

        for i, head in enumerate(next_heads):
            if head is None or not self.snakes[i]["alive"] or i in snakes_to_die:
                continue

            for j, body in enumerate(future_bodies):
                if head in body:
                    snakes_to_die.add(i)
                    rewards[i] -= 10
                    if i != j:
                        rewards[j] += 20  # 击杀奖励
                    break

        # 3) 真正执行移动与奖励塑形
        for i, snake in enumerate(self.snakes):
            if not snake["alive"]:
                dones[i] = True
                continue

            if i in snakes_to_die:
                snake["alive"] = False
                dones[i] = True
                continue

            new_head = next_heads[i]
            snake["body"].insert(0, new_head)
            ate_food = new_head in self.food

            if ate_food:
                self.food.remove(new_head)
                self._spawn_food()
                rewards[i] += 10
                snake["score"] += 1
            else:
                snake["body"].pop()
                rewards[i] -= 0.01

            # 引导奖励：接近或远离食物
            prev_dist = snake["last_food_distance"]
            new_dist = self._nearest_food_distance(new_head)
            if prev_dist is not None and new_dist is not None:
                if new_dist < prev_dist:
                    rewards[i] += 0.1
                elif new_dist > prev_dist:
                    rewards[i] -= 0.1
            snake["last_food_distance"] = new_dist

            snake["steps_alive"] += 1

        # 4) 最大步数截断
        if self.steps >= self.max_steps:
            for i in range(self.num_snakes):
                dones[i] = True

        alive_count = sum(1 for snake in self.snakes if snake["alive"])
        info = {
            "steps": self.steps,
            "alive_count": alive_count,
            "scores": [snake["score"] for snake in self.snakes],
            "game_over": alive_count <= 1 or self.steps >= self.max_steps,
        }

        return self._get_observations(), rewards, dones, info

    # ------------------------------------------------------------------
    # 观测与渲染
    # ------------------------------------------------------------------
    def _get_observations(self) -> List[np.ndarray]:
        """返回所有蛇的三通道网格观测。"""

        obs_list: List[np.ndarray] = []
        food_grid = np.zeros(self.grid_shape, dtype=np.float32)
        for fx, fy in self.food:
            food_grid[fx, fy] = 1.0

        for i in range(self.num_snakes):
            obs_list.append(
                build_observation_from_snapshot(
                    width=self.width,
                    height=self.height,
                    snakes=self.snakes,
                    food=self.food,
                    slot=i,
                )
            )

        return obs_list

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

    def _nearest_food_distance(self, head: Tuple[int, int]) -> Optional[int]:
        """计算蛇头到最近食物的曼哈顿距离。"""

        if not self.food:
            return None

        hx, hy = head
        return min(abs(hx - fx) + abs(hy - fy) for fx, fy in self.food)

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
    """根据状态快照构造单条蛇的 3 通道观测。"""

    if slot >= len(snakes) or slot < 0:
        return np.zeros((3, width, height), dtype=np.float32)

    food_grid = np.zeros((width, height), dtype=np.float32)
    for fx, fy in food:
        if 0 <= fx < width and 0 <= fy < height:
            food_grid[fx, fy] = 1.0

    snake = snakes[slot]
    if not snake.get("alive", True):
        return np.zeros((3, width, height), dtype=np.float32)

    self_grid = np.zeros((width, height), dtype=np.float32)
    enemies_grid = np.zeros((width, height), dtype=np.float32)

    for idx, (bx, by) in enumerate(snake.get("body", [])):
        if 0 <= bx < width and 0 <= by < height:
            self_grid[bx, by] = 2.0 if idx == 0 else 1.0

    for idx, other in enumerate(snakes):
        if idx == slot or not other.get("alive", True):
            continue
        for seg, (bx, by) in enumerate(other.get("body", [])):
            if 0 <= bx < width and 0 <= by < height:
                enemies_grid[bx, by] = 2.0 if seg == 0 else 1.0

    return np.stack([self_grid, enemies_grid, food_grid], axis=0)
