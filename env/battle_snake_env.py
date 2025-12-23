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
        return 15 # 11 (Standard with Nearest Food) + 4 (Nearest Enemy)

    @property
    def action_dim(self) -> int:
        return 3

    def reset(self) -> List[np.ndarray]:
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
        
        # 1. Update directions and potential heads
        next_heads = []
        for i in range(self.config.num_snakes):
            if self.dead[i]:
                next_heads.append(None)
                continue
                
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

        # 2. Check collisions
        still_alive_indices = [i for i, d in enumerate(self.dead) if not d]
        dying_indices = set()
        
        for i in still_alive_indices:
            nh = next_heads[i]
            # Wall
            if not (0 <= nh[0] < self.width and 0 <= nh[1] < self.height):
                dying_indices.add(i)
                rewards[i] += self.config.death_penalty
            # Self/Enemy Body
            for j in range(self.config.num_snakes):
                if not self.dead[j] and nh in self.snakes[j][:-1]:
                    dying_indices.add(i)
                    rewards[i] += self.config.death_penalty
                    if i != j: rewards[j] += self.config.kill_reward
            # Head-to-Head
            for j in still_alive_indices:
                if i != j and nh == next_heads[j]:
                    dying_indices.add(i)
                    dying_indices.add(j)
                    rewards[i] += self.config.death_penalty
                    rewards[j] += self.config.death_penalty

        # 3. Resolve Updates
        foods_eaten = []
        for i in still_alive_indices:
            if i in dying_indices:
                self.dead[i] = True
                dones[i] = True
            else:
                nh = next_heads[i]
                self.snakes[i].insert(0, nh)
                if nh in self.foods:
                    rewards[i] += self.config.food_reward
                    self.scores[i] += 1
                    foods_eaten.append(nh)
                    # If multiple snakes eat the same food, the first index gets the reward 
                    # and the food is removed (fairly rare due to move sync)
                    if nh in self.foods: self.foods.remove(nh)
                else:
                    self.snakes[i].pop()

        if foods_eaten:
            self._spawn_food()
            
        if self.steps >= self.config.max_steps:
            for i in range(self.config.num_snakes):
                dones[i] = True
        
        return self._get_observations(), rewards, dones, {"scores": self.scores, "steps": self.steps}

    def _get_observations(self) -> List[np.ndarray]:
        obs_list = []
        for i in range(self.config.num_snakes):
            obs_list.append(self._get_agent_obs(i))
        return obs_list

    def _get_agent_obs(self, agent_idx: int) -> np.ndarray:
        if self.dead[agent_idx]:
            return np.zeros(self.obs_dim, dtype=np.float32)
             
        head = self.snakes[agent_idx][0]
        direction = self.directions[agent_idx]
        
        # 1. Nearest Food Relative (4)
        food_up, food_down, food_left, food_right = 0.0, 0.0, 0.0, 0.0
        if self.foods:
            closest_food = min(self.foods, key=lambda f: abs(head[0]-f[0]) + abs(head[1]-f[1]))
            food_up = float(closest_food[1] < head[1])
            food_down = float(closest_food[1] > head[1])
            food_left = float(closest_food[0] < head[0])
            food_right = float(closest_food[0] > head[0])
        
        # 2. Danger (3)
        danger_straight = float(self._is_danger(agent_idx, self._get_next_pos(head, direction)))
        danger_left = float(self._is_danger(agent_idx, self._get_next_pos(head, self._turn(direction, Action.LEFT))))
        danger_right = float(self._is_danger(agent_idx, self._get_next_pos(head, self._turn(direction, Action.RIGHT))))
        
        # 3. Current Direction (4)
        dir_up = float(direction == Direction.UP)
        dir_down = float(direction == Direction.DOWN)
        dir_left = float(direction == Direction.LEFT)
        dir_right = float(direction == Direction.RIGHT)
        
        # 4. Nearest Enemy Head (4)
        enemy_up, enemy_down, enemy_left, enemy_right = 0.0, 0.0, 0.0, 0.0
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
            enemy_up = float(ey < head[1])
            enemy_down = float(ey > head[1])
            enemy_left = float(ex < head[0])
            enemy_right = float(ex > head[0])
            
        return np.array([
            food_up, food_down, food_left, food_right,
            danger_straight, danger_left, danger_right,
            dir_up, dir_down, dir_left, dir_right,
            enemy_up, enemy_down, enemy_left, enemy_right
        ], dtype=np.float32)

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

