"""
Multi-Agent Battle Snake Environment.
Supports multiple snakes, collision logic, and kill rewards.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Set

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

@dataclass
class BattleSnakeConfig:
    width: int = 20
    height: int = 20
    num_snakes: int = 2
    min_food: int = 2          # New: Target number of food items on board
    max_steps: int = 1000        # Increased: Allow longer episodes
    
    # Rewards
    food_reward: float = 10.0
    death_penalty: float = -10.0
    kill_reward: float = 20.0
    closer_reward: float = 0.2
    farther_penalty: float = -0.3

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
        self.foods: List[Tuple[int, int]] = [] # Changed: List of foods
        self.steps = 0
        
    def seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        
    @property
    def obs_dim(self) -> int:
        return 24 # Corrected: 4(food)+3(d1)+3(d2)+3(radar)+4(dir)+4(enemy)+2(tail)+1(len) = 24

    @property
    def action_dim(self) -> int:
        return 3

    def reset(self) -> List[np.ndarray]:
        # ... (rest stays same)
        self.snakes = []
        self.directions = []
        self.dead = []
        self.scores = []
        self.steps = 0
        self.foods = []
        
        # Initialize snakes
        safe_margin = 3
        for _ in range(self.config.num_snakes):
            while True:
                x = random.randint(safe_margin, self.width - 1 - safe_margin)
                y = random.randint(safe_margin, self.height - 1 - safe_margin)
                conflict = False
                for s in self.snakes:
                    if abs(s[0][0] - x) < 4 and abs(s[0][1] - y) < 4:
                        conflict = True
                        break
                if not conflict: break
            
            d = Direction(random.randint(0, 3))
            self.directions.append(d)
            dx, dy = self.DIR_DELTA[d]
            head = (x, y)
            self.snakes.append([head, (x - dx, y - dy), (x - 2*dx, y - 2*dy)])
            self.dead.append(False)
            self.scores.append(0)
            
        self._spawn_food()
        return self._get_observations()

    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        rewards = [0.0] * self.config.num_snakes
        dones = [False] * self.config.num_snakes
        self.steps += 1
        
        # 0. Pre-process Dash
        move_repeats = [1] * self.config.num_snakes
        for i in range(self.config.num_snakes):
            if not self.dead[i] and actions[i] == Action.DASH: # DASH
                if len(self.snakes[i]) > 3: # Minimum length to dash
                    move_repeats[i] = 2
                    # Penalty for dashing (cost of length)
                    self.snakes[i].pop() 
                    rewards[i] -= 0.1 # Small tactical cost
                else:
                    # Not enough length to dash, fallback to straight
                    actions[i] = Action.STRAIGHT

        # Run moves (1 or 2 steps)
        for sub_step in range(2):
            next_heads = []
            for i in range(self.config.num_snakes):
                if self.dead[i] or move_repeats[i] <= sub_step:
                    next_heads.append(None)
                    continue
                
                # Turn logic (only on first step of dash or single move)
                if sub_step == 0:
                    self.directions[i] = self._turn(self.directions[i], actions[i])
                
                head = self.snakes[i][0]
                dx, dy = self.DIR_DELTA[self.directions[i]]
                new_head = (head[0] + dx, head[1] + dy)
                next_heads.append(new_head)
                
                # Distance shaping (relative to NEAREST food)
                if self.foods:
                    old_min_dist = min(abs(head[0] - fx) + abs(head[1] - fy) for fx, fy in self.foods)
                    new_min_dist = min(abs(new_head[0] - fx) + abs(new_head[1] - fy) for fx, fy in self.foods)
                    if new_min_dist < old_min_dist:
                        rewards[i] += self.config.closer_reward
                    elif new_min_dist > old_min_dist:
                        rewards[i] += self.config.farther_penalty

            # Check collisions for this sub-step
            still_alive_indices = [i for i, d in enumerate(self.dead) if not d and next_heads[i] is not None]
            dying_now = set()
            
            for i in still_alive_indices:
                nh = next_heads[i]
                # Wall
                if not (0 <= nh[0] < self.width and 0 <= nh[1] < self.height):
                    dying_now.add(i)
                    rewards[i] += self.config.death_penalty
                # Self/Enemy Body
                for j in range(self.config.num_snakes):
                    if not self.dead[j] and nh in self.snakes[j][:-1]:
                        dying_now.add(i)
                        rewards[i] += self.config.death_penalty
                        if i != j: rewards[j] += self.config.kill_reward
                # Head-to-Head
                for j in still_alive_indices:
                    if i != j and nh == next_heads[j]:
                        # Only the shorter snake dies in a head-to-head, or both if equal length
                        if len(self.snakes[i]) <= len(self.snakes[j]):
                            dying_now.add(i)
                            rewards[i] += self.config.death_penalty
                        if len(self.snakes[j]) <= len(self.snakes[i]):
                            dying_now.add(j)
                            rewards[j] += self.config.death_penalty

            # Apply updates for this sub-step
            for i in still_alive_indices:
                if i in dying_now:
                    self._handle_death(i)
                    dones[i] = True
                else:
                    nh = next_heads[i]
                    self.snakes[i].insert(0, nh)
                    if nh in self.foods:
                        rewards[i] += self.config.food_reward
                        self.scores[i] += 1
                        self.foods.remove(nh)
                    else:
                        self.snakes[i].pop()

        if len(self.foods) < self.config.min_food:
            self._spawn_food()
            
        if self.steps >= self.config.max_steps:
            for i in range(self.config.num_snakes):
                dones[i] = True
        
        return self._get_observations(), rewards, dones, {"scores": self.scores}

    def _handle_death(self, idx):
        """Convert snake body to food on death (V3)"""
        self.dead[idx] = True
        # Probability to drop food for each segment
        for segment in self.snakes[idx][1:]: # Exclude head, as it might be on another snake's head
            if random.random() < 0.5: # 50% drop rate
                if segment not in self.foods:
                    self.foods.append(segment)

    def _get_observations(self) -> List[Dict]:
        """In V3, observation is a Dict: {'vector': ..., 'grid': ...}"""
        return [self._get_agent_obs(i) for i in range(self.config.num_snakes)]

    def _get_agent_obs(self, agent_idx: int) -> Dict:
        if self.dead[agent_idx]:
             return {
                 "vector": np.zeros(24, dtype=np.float32),
                 "grid": np.zeros((3, 7, 7), dtype=np.float32)
             }
             
        head = self.snakes[agent_idx][0]
        direction = self.directions[agent_idx]
        
        # 1. Vector Obs (Same as V2, 24-dim)
        food_up, food_down, food_left, food_right = 0.0, 0.0, 0.0, 0.0
        if self.foods:
            closest_food = min(self.foods, key=lambda f: abs(head[0]-f[0]) + abs(head[1]-f[1]))
            food_up = float(closest_food[1] < head[1])
            food_down = float(closest_food[1] > head[1])
            food_left = float(closest_food[0] < head[0])
            food_right = float(closest_food[0] > head[0])
        
        # 2. Multi-level Danger & Radar (3 directions: Straight, Left, Right)
        # Directions to check
        dirs = [
            direction, 
            self._turn(direction, Action.LEFT),
            self._turn(direction, Action.RIGHT)
        ]
        
        danger_1 = [] # 1-step (3)
        danger_2 = [] # 2-step (3)
        radar = []    # 1/dist (3)
        
        for d in dirs:
            # Danger level 1
            p1 = self._get_next_pos(head, d)
            d1 = float(self._is_danger(agent_idx, p1))
            danger_1.append(d1)
            
            # Danger level 2
            p2 = self._get_next_pos(p1, d)
            danger_2.append(float(self._is_danger(agent_idx, p2)))
            
            # Radar distance
            dist = 1
            cur = p1
            while 0 <= cur[0] < self.width and 0 <= cur[1] < self.height:
                if self._is_danger(agent_idx, cur): break
                dist += 1
                cur = self._get_next_pos(cur, d)
            radar.append(1.0 / dist)
            
        # 3. Current Direction (4)
        dir_vec = [float(direction == d) for d in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]]
        
        # 4. Nearest Enemy Head (4)
        enemy_vec = [0.0, 0.0, 0.0, 0.0]
        closest_enemy_dist = float('inf')
        found_enemy = False
        for j in range(self.config.num_snakes):
            if agent_idx == j or self.dead[j]: continue
            e_head = self.snakes[j][0]
            dist = abs(head[0]-e_head[0]) + abs(head[1]-e_head[1])
            if dist < closest_enemy_dist:
                closest_enemy_dist = dist
                found_enemy = True; ex, ey = e_head
        if found_enemy:
            enemy_vec = [float(ey < head[1]), float(ey > head[1]), float(ex < head[0]), float(ex > head[0])]
            
        # 5. Tail Relative Position (2) - Very important for long snakes
        tail = self.snakes[agent_idx][-1]
        tail_rel = [(tail[0] - head[0]) / self.width, (tail[1] - head[1]) / self.height]
        
        # 6. Length Percentage (1)
        len_pct = [len(self.snakes[agent_idx]) / (self.width * self.height)]
            
        vector = np.concatenate([
            [food_up, food_down, food_left, food_right],
            danger_1, danger_2, radar,
            dir_vec,
            enemy_vec,
            tail_rel,
            len_pct
        ]).astype(np.float32)

        # 2. Grid Obs (7x7 Local View)
        # Channels: 0: Obstacle, 1: Food, 2: Enemy Head
        grid = np.zeros((3, 7, 7), dtype=np.float32)
        view_r = 3 # 7x7
        
        for dy in range(-view_r, view_r + 1):
            for dx in range(-view_r, view_r + 1):
                x, y = head[0] + dx, head[1] + dy
                gx, gy = dx + view_r, dy + view_r
                
                # Check Obstacle (Wall/Body)
                if not (0 <= x < self.width and 0 <= y < self.height):
                    grid[0, gy, gx] = 1.0
                else:
                    for j in range(self.config.num_snakes):
                        if not self.dead[j] and (x, y) in self.snakes[j]:
                            grid[0, gy, gx] = 1.0
                            if (x, y) == self.snakes[j][0] and j != agent_idx:
                                grid[2, gy, gx] = 1.0 # Enemy head
                    # Food
                    if (x, y) in self.foods:
                        grid[1, gy, gx] = 1.0
                        
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
                    self.foods.append((x, y))
                    occupied.add((x, y))
                    found = True
                    break
            if not found: break # Grid full?

