"""
Multi-Agent Battle Snake Environment V5.0.
Supports Dash with Cooldown, Hybrid Obs (CNN+MLP), and Death Drops.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, TypedDict
import numpy as np

class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class Action(IntEnum):
    STRAIGHT = 0
    LEFT = 1
    RIGHT = 2
    DASH = 3

class ObservationDict(TypedDict):
    """V3+ Hybrid Observation Format"""
    grid: np.ndarray    # (3, 7, 7) Local view
    vector: np.ndarray  # (25,) Global features

@dataclass
class BattleSnakeConfig:
    width: int = 20
    height: int = 20
    num_snakes: int = 2
    min_food: int = 2
    max_steps: int = 1000
    
    # Mechanics
    dash_cooldown_steps: int = 30 # New in V5.0
    
    # Rewards
    food_reward: float = 20.0
    death_penalty: float = -15.0 
    kill_reward: float = 15.0
    closer_reward: float = 0.3
    farther_penalty: float = -0.2
    step_penalty: float = -0.05

class BattleSnakeEnv:
    """Multi-Agent Snake Environment with Multiple Foods."""
    
    DIR_DELTA = {
        Direction.UP: (0, -1),
        Direction.DOWN: (0, 1),
        Direction.LEFT: (-1, 0),
        Direction.RIGHT: (1, 0),
    }

    def __init__(self, config: Optional[BattleSnakeConfig] = None, seed: Optional[int] = None):
        self.config = config or BattleSnakeConfig()
        self.width = self.config.width
        self.height = self.config.height
        
        if seed is not None:
            self.seed(seed)
            
        # State
        self.snakes: List[List[Tuple[int, int]]] = []
        self.directions: List[Direction] = []
        self.dead: List[bool] = []
        self.scores: List[int] = []
        self.dash_cooldowns: List[int] = [] # New: Track cooldown
        self.foods: List[Tuple[int, int]] = []
        self.steps = 0
        
    def seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        
    @property
    def obs_dim(self) -> int:
        return 25 # V5.0: Added cooldown (24 -> 25)

    @property
    def action_dim(self) -> int:
        return 4 # Straight, Left, Right, Dash

    def reset(self) -> List[ObservationDict]:
        self.snakes = []
        self.directions = []
        self.dead = []
        self.scores = []
        self.dash_cooldowns = []
        self.steps = 0
        self.foods = []
        
        safe_margin = 3
        for _ in range(self.config.num_snakes):
            while True:
                x = random.randint(safe_margin, self.width - 1 - safe_margin)
                y = random.randint(safe_margin, self.height - 1 - safe_margin)
                conflict = False
                for s in self.snakes:
                    if abs(s[0][0] - x) < 4 and abs(s[0][1] - y) < 4:
                        conflict = True; break
                if not conflict: break
            
            d = Direction(random.randint(0, 3))
            self.directions.append(d)
            dx, dy = self.DIR_DELTA[d]
            self.snakes.append([(x, y), (x - dx, y - dy), (x - 2*dx, y - 2*dy)])
            self.dead.append(False)
            self.scores.append(0)
            self.dash_cooldowns.append(0)
            
        self._spawn_food()
        return self._get_observations()

    def step(self, actions: List[int]) -> Tuple[List[ObservationDict], List[float], List[bool], Dict]:
        rewards = [self.config.step_penalty] * self.config.num_snakes
        dones = [False] * self.config.num_snakes
        self.steps += 1
        
        # 0. Process Dash & Cooldown
        move_repeats = [1] * self.config.num_snakes
        for i in range(self.config.num_snakes):
            if self.dead[i]:
                rewards[i] = 0.0
                continue
            
            # Update Cooldown
            if self.dash_cooldowns[i] > 0:
                self.dash_cooldowns[i] -= 1
                
            if actions[i] == Action.DASH:
                if self.dash_cooldowns[i] == 0 and len(self.snakes[i]) > 3:
                    move_repeats[i] = 2
                    self.snakes[i].pop() # Tactical cost
                    self.dash_cooldowns[i] = self.config.dash_cooldown_steps # Trigger cooldown
                    rewards[i] -= 0.1
                else:
                    # Cooldown active or length too short -> FALLBACK
                    actions[i] = Action.STRAIGHT

        # Run moves
        for sub_step in range(2):
            next_heads = []
            for i in range(self.config.num_snakes):
                if self.dead[i] or move_repeats[i] <= sub_step:
                    next_heads.append(None); continue
                
                if sub_step == 0:
                    self.directions[i] = self._turn(self.directions[i], actions[i])
                
                head = self.snakes[i][0]
                dx, dy = self.DIR_DELTA[self.directions[i]]
                new_head = (head[0] + dx, head[1] + dy)
                next_heads.append(new_head)
                
                # Distance shaping
                if self.foods:
                    old_min = min(abs(head[0]-fx)+abs(head[1]-fy) for fx, fy in self.foods)
                    new_min = min(abs(new_head[0]-fx)+abs(new_head[1]-fy) for fx, fy in self.foods)
                    if new_min < old_min: rewards[i] += self.config.closer_reward
                    elif new_min > old_min: rewards[i] += self.config.farther_penalty

            # Collisions
            alive_indices = [i for i, d in enumerate(self.dead) if not d and next_heads[i] is not None]
            dying_now = set()
            for i in alive_indices:
                nh = next_heads[i]
                if not (0 <= nh[0] < self.width and 0 <= nh[1] < self.height):
                    dying_now.add(i); rewards[i] += self.config.death_penalty
                for j in range(self.config.num_snakes):
                    if not self.dead[j] and nh in self.snakes[j][:-1]:
                        dying_now.add(i); rewards[i] += self.config.death_penalty
                        if i != j: rewards[j] += self.config.kill_reward
                for j in alive_indices:
                    if i != j and nh == next_heads[j]:
                        if len(self.snakes[i]) <= len(self.snakes[j]):
                            dying_now.add(i); rewards[i] += self.config.death_penalty
                        if len(self.snakes[j]) <= len(self.snakes[i]):
                            dying_now.add(j); rewards[j] += self.config.death_penalty

            # Update State
            for i in alive_indices:
                if i in dying_now:
                    self._handle_death(i); dones[i] = True
                else:
                    nh = next_heads[i]
                    self.snakes[i].insert(0, nh)
                    if nh in self.foods:
                        rewards[i] += self.config.food_reward; self.scores[i] += 1
                        self.foods.remove(nh)
                    else:
                        self.snakes[i].pop()

        if len(self.foods) < self.config.min_food: self._spawn_food()
        if self.steps >= self.config.max_steps:
            for i in range(self.config.num_snakes): dones[i] = True
        
        return self._get_observations(), rewards, dones, {"scores": self.scores}

    def _handle_death(self, idx: int):
        self.dead[idx] = True
        for segment in self.snakes[idx][1:]:
            if random.random() < 0.5 and segment not in self.foods:
                self.foods.append(segment)

    def _get_observations(self) -> List[ObservationDict]:
        return [self._get_agent_obs(i) for i in range(self.config.num_snakes)]

    def _get_agent_obs(self, agent_idx: int) -> ObservationDict:
        if self.dead[agent_idx]:
             return {
                 "vector": np.zeros(self.obs_dim, dtype=np.float32),
                 "grid": np.zeros((3, 7, 7), dtype=np.float32)
             }
             
        head = self.snakes[agent_idx][0]
        direction = self.directions[agent_idx]
        
        # 1. Vector Features
        food_up, food_down, food_left, food_right = 0.0, 0.0, 0.0, 0.0
        if self.foods:
            closest_food = min(self.foods, key=lambda f: abs(head[0]-f[0]) + abs(head[1]-f[1]))
            dy = head[1] - closest_food[1]
            if dy > 0: food_up = max(0, 1.0 - dy / self.height)
            elif dy < 0: food_down = max(0, 1.0 - abs(dy) / self.height)
            dx = head[0] - closest_food[0]
            if dx > 0: food_left = max(0, 1.0 - dx / self.width)
            elif dx < 0: food_right = max(0, 1.0 - abs(dx) / self.width)
        
        dirs = [direction, self._turn(direction, Action.LEFT), self._turn(direction, Action.RIGHT)]
        danger_1, danger_2, radar = [], [], []
        for d in dirs:
            p1 = self._get_next_pos(head, d)
            danger_1.append(float(self._is_danger(agent_idx, p1)))
            p2 = self._get_next_pos(p1, d)
            danger_2.append(float(self._is_danger(agent_idx, p2)))
            dist, cur = 1, p1
            while 0 <= cur[0] < self.width and 0 <= cur[1] < self.height:
                if self._is_danger(agent_idx, cur): break
                dist += 1; cur = self._get_next_pos(cur, d)
            radar.append(1.0 / dist)
            
        dir_vec = [float(direction == d) for d in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]]
        enemy_vec = [0.0, 0.0, 0.0, 0.0]
        closest_dist, found = float('inf'), False
        for j in range(self.config.num_snakes):
            if agent_idx == j or self.dead[j]: continue
            e_head = self.snakes[j][0]
            d = abs(head[0]-e_head[0]) + abs(head[1]-e_head[1])
            if d < closest_dist:
                closest_dist = d; found = True; ex, ey = e_head
        if found:
            enemy_vec = [float(ey < head[1]), float(ey > head[1]), float(ex < head[0]), float(ex > head[0])]
            
        tail = self.snakes[agent_idx][-1]
        tail_rel = [(tail[0] - head[0]) / self.width, (tail[1] - head[1]) / self.height]
        len_pct = [len(self.snakes[agent_idx]) / (self.width * self.height)]
        
        # New in V5.0: Cooldown Obs (1-dim)
        cd_val = [self.dash_cooldowns[agent_idx] / self.config.dash_cooldown_steps]
            
        vector = np.concatenate([
            [food_up, food_down, food_left, food_right],
            danger_1, danger_2, radar,
            dir_vec,
            enemy_vec,
            tail_rel,
            len_pct,
            cd_val
        ]).astype(np.float32)

        # 2. Grid (Local 7x7)
        grid = np.zeros((3, 7, 7), dtype=np.float32)
        view_r = 3
        for dy in range(-view_r, view_r + 1):
            for dx in range(-view_r, view_r + 1):
                x, y, gx, gy = head[0] + dx, head[1] + dy, dx + view_r, dy + view_r
                if not (0 <= x < self.width and 0 <= y < self.height):
                    grid[0, gy, gx] = 1.0
                else:
                    for j in range(self.config.num_snakes):
                        if not self.dead[j] and (x, y) in self.snakes[j]:
                            grid[0, gy, gx] = 1.0
                            if (x, y) == self.snakes[j][0] and j != agent_idx: grid[2, gy, gx] = 1.0
                    if (x, y) in self.foods: grid[1, gy, gx] = 1.0
        return {"vector": vector, "grid": grid}

    def _is_danger(self, agent_idx: int, pos: Tuple[int, int]) -> bool:
        x, y = pos
        if not (0 <= x < self.width and 0 <= y < self.height): return True
        for i in range(self.config.num_snakes):
            if not self.dead[i] and pos in self.snakes[i]: return True
        return False

    def _get_next_pos(self, head: Tuple[int, int], direction: Direction) -> Tuple[int, int]:
        dx, dy = self.DIR_DELTA[direction]
        return (head[0] + dx, head[1] + dy)
        
    @staticmethod
    def _turn(direction: Direction, action: int) -> Direction:
        if action == Action.LEFT: return Direction((direction - 1) % 4)
        elif action == Action.RIGHT: return Direction((direction + 1) % 4)
        return direction

    def _spawn_food(self):
        occupied = set()
        for i in range(self.config.num_snakes):
            if not self.dead[i]: occupied.update(self.snakes[i])
        occupied.update(self.foods)
        while len(self.foods) < self.config.min_food:
            found = False
            for _ in range(100):
                x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
                if (x, y) not in occupied:
                    self.foods.append((x, y)); occupied.add((x, y)); found = True; break
            if not found: break

