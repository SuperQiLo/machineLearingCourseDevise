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
    grid: np.ndarray    # (5, H, W) Full view (uint8 0/1)
    vector: np.ndarray  # (28,) Global features

@dataclass
class BattleSnakeConfig:
    width: int = 20
    height: int = 20
    num_snakes: int = 2
    min_food: int = 2
    max_steps: int = 1000
    
    # Mechanics
    # 冲刺持续时间（步数）
    dash_duration_steps: int = 5
    
    # Rewards
    # NOTE: These are TRAINING rewards returned by env.step().
    # Game scoring/MVP uses `scores` and the *_score_mult fields below.
    food_reward: float = 1.2
    death_penalty: float = -3.0
    kill_reward: float = 2.0
    closer_reward: float = 0.05
    farther_penalty: float = -0.04
    step_penalty: float = -0.01
    self_collision_penalty: float = -4.0
    win_reward: float = 5.0
    loss_penalty: float = -2.0

    # Game scoring (NOT training reward)
    food_score_mult: int = 10
    kill_score_mult: int = 50
    survive_score_mult: int = 100

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
        self.dash_durations: List[int] = [] # 新增：冲刺持续步数
        self.pre_terminal_lengths: List[int] = []  # 记录“游戏结束前一时刻”的长度快照（用于全员死亡时判赢家）
        self.foods: List[Tuple[int, int]] = []
        self.steps = 0
        
    def seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        
    @property
    def obs_dim(self) -> int:
        return 28 # V11.0: +Time awareness (steps_left)

    @property
    def action_dim(self) -> int:
        return 4 # Straight, Left, Right, Dash

    def reset(self) -> List[ObservationDict]:
        self.snakes = []
        self.directions = []
        self.dead = []
        self.scores = []
        self.dash_durations = []
        self.pre_terminal_lengths = []
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
            self.dash_durations.append(0)
            self.pre_terminal_lengths.append(len(self.snakes[-1]))
            
        self._spawn_food()
        return self._get_observations()

    def step(self, actions: List[int]) -> Tuple[List[ObservationDict], List[float], List[bool], Dict]:
        # Snapshot lengths at the moment right before this step executes.
        # Used for winner selection when the game ends with everyone dead.
        # IMPORTANT: only count snakes that are alive at this moment.
        self.pre_terminal_lengths = [
            (len(self.snakes[i]) if (i < len(self.snakes) and self.snakes[i]) else 0)
            if (i < len(self.dead) and not self.dead[i])
            else 0
            for i in range(self.config.num_snakes)
        ]

        rewards = [self.config.step_penalty] * self.config.num_snakes
        dones = [False] * self.config.num_snakes
        self.steps += 1
        
        # 0. Process Dash & Cooldown
        move_repeats = [1] * self.config.num_snakes
        for i in range(self.config.num_snakes):
            if self.dead[i]:
                rewards[i] = 0.0
                continue
            
            # Update Duration
            if self.dash_durations[i] > 0:
                self.dash_durations[i] -= 1
                move_repeats[i] = 2 # 处于冲刺状态，移动两次
                
            if actions[i] == Action.DASH:
                # 仅在非冲刺状态且长度大于 3 (初始长度) 时触发
                if self.dash_durations[i] == 0 and len(self.snakes[i]) > 3:
                    move_repeats[i] = 2
                    self.snakes[i].pop() # 消耗一个长度
                    self.dash_durations[i] = self.config.dash_duration_steps - 1
                    rewards[i] -= 0.1
                else:
                    # 无法触发 DASH 时，强制转为 STRAIGHT
                    actions[i] = Action.STRAIGHT

        # Run moves
        for sub_step in range(2):
            # Compute planned next heads once.
            next_heads: List[Optional[Tuple[int, int]]] = [None] * self.config.num_snakes

            for i in range(self.config.num_snakes):
                if self.dead[i] or move_repeats[i] <= sub_step:
                    continue

                if sub_step == 0:
                    self.directions[i] = self._turn(self.directions[i], actions[i])

                head = self.snakes[i][0]
                dx, dy = self.DIR_DELTA[self.directions[i]]
                new_head = (head[0] + dx, head[1] + dy)
                next_heads[i] = new_head

                # Distance shaping (V6.0: Smooth Potential-based Reward)
                if self.foods:
                    old_min = min(abs(head[0] - fx) + abs(head[1] - fy) for fx, fy in self.foods)
                    new_min = min(abs(new_head[0] - fx) + abs(new_head[1] - fy) for fx, fy in self.foods)
                    dist_diff = old_min - new_min
                    if dist_diff > 0:
                        rewards[i] += self.config.closer_reward
                    elif dist_diff < 0:
                        rewards[i] += self.config.farther_penalty

            alive_indices = [i for i in range(self.config.num_snakes) if (not self.dead[i] and next_heads[i] is not None)]
            if not alive_indices:
                continue

            # Build a compact occupancy table (H*W is tiny: 20*20).
            # Stores owner snake idx for body cells (excluding tail), else -1.
            occ = np.full((self.height, self.width), -1, dtype=np.int16)
            for j in range(self.config.num_snakes):
                if self.dead[j] or not self.snakes[j]:
                    continue
                for x, y in self.snakes[j][:-1]:
                    if 0 <= x < self.width and 0 <= y < self.height:
                        occ[y, x] = j

            dying_now = set()

            # 1) Wall + body collisions
            for i in alive_indices:
                nh = next_heads[i]
                if nh is None:
                    continue
                if not (0 <= nh[0] < self.width and 0 <= nh[1] < self.height):
                    dying_now.add(i)
                    rewards[i] += self.config.death_penalty
                    continue

                owner = int(occ[nh[1], nh[0]])
                if owner != -1 and not self.dead[owner]:
                    dying_now.add(i)
                    if owner == i:
                        rewards[i] += self.config.self_collision_penalty
                    else:
                        rewards[i] += self.config.death_penalty
                        rewards[owner] += self.config.kill_reward
                        # Game scoring: killer gets points based on current length
                        if self.config.num_snakes > 1:
                            killer_len = len(self.snakes[owner]) if (owner < len(self.snakes) and self.snakes[owner]) else 0
                            self.scores[owner] += int(killer_len) * int(self.config.kill_score_mult)

            # 2) Head-on collisions: group by next head position
            head_groups: Dict[Tuple[int, int], List[int]] = {}
            for i in alive_indices:
                nh = next_heads[i]
                if nh is not None:
                    head_groups.setdefault(nh, []).append(i)

            for pos, idxs in head_groups.items():
                if len(idxs) <= 1:
                    continue
                lengths = [len(self.snakes[i]) for i in idxs]
                max_len = max(lengths)
                # Shorter (or equal) snakes die; if multiple tie for max, all survive (matches previous pairwise rule)
                if lengths.count(max_len) == 1:
                    winner = idxs[lengths.index(max_len)]
                    for i in idxs:
                        if i != winner:
                            dying_now.add(i)
                            rewards[i] += self.config.death_penalty
                            # Game scoring: winner gets a kill
                            if self.config.num_snakes > 1:
                                winner_len = len(self.snakes[winner]) if (winner < len(self.snakes) and self.snakes[winner]) else 0
                                self.scores[winner] += int(winner_len) * int(self.config.kill_score_mult)
                else:
                    # all tied -> all die (pairwise rule with <= would kill both)
                    for i in idxs:
                        dying_now.add(i)
                        rewards[i] += self.config.death_penalty

            # 3) Apply state updates
            for i in alive_indices:
                if i in dying_now:
                    self._handle_death(i)
                    dones[i] = True
                    continue

                nh = next_heads[i]
                if nh is None:
                    continue
                pre_len = len(self.snakes[i]) if (i < len(self.snakes) and self.snakes[i]) else 0
                self.snakes[i].insert(0, nh)
                if nh in self.foods:
                    rewards[i] += self.config.food_reward
                    # Game scoring: single-player only scores via food; battle also scores via food
                    self.scores[i] += int(pre_len) * int(self.config.food_score_mult)
                    self.foods.remove(nh)
                else:
                    self.snakes[i].pop()

        if len(self.foods) < self.config.min_food: self._spawn_food()

        # 统一游戏结束规则：
        # - 单人：蛇死亡或达到最大步数结束
        # - 多人：只剩最后一条蛇存活（含全员死亡）或达到最大步数结束
        def _is_alive(i: int) -> bool:
            if i >= self.config.num_snakes:
                return False
            if i >= len(self.dead) or self.dead[i]:
                return False
            return i < len(self.snakes) and bool(self.snakes[i])

        alive_idxs = [i for i in range(self.config.num_snakes) if _is_alive(i)]
        alive_count = len(alive_idxs)

        if self.config.num_snakes <= 1:
            game_over = (not _is_alive(0)) or self.steps >= self.config.max_steps
        else:
            game_over = alive_count <= 1 or self.steps >= self.config.max_steps

        # Winner/MVP selection:
        # - MVP is the highest score.
        # - Tie-break: alive at end > longer (alive uses current len, else uses pre_terminal_lengths) > lower idx.
        mvp_idx: Optional[int] = None
        winner_len: int = 0
        winner_is_all_dead: bool = False

        if game_over:
            if self.config.num_snakes > 1:
                # Survival score only in battle mode
                for i in range(self.config.num_snakes):
                    if _is_alive(i):
                        cur_len = len(self.snakes[i]) if (i < len(self.snakes) and self.snakes[i]) else 0
                        self.scores[i] += int(cur_len) * int(self.config.survive_score_mult)

            def _mvp_key(i: int):
                score = int(self.scores[i]) if i < len(self.scores) else 0
                alive_flag = 1 if _is_alive(i) else 0
                length_like = (len(self.snakes[i]) if (i < len(self.snakes) and self.snakes[i]) else 0) if alive_flag else (
                    int(self.pre_terminal_lengths[i]) if i < len(self.pre_terminal_lengths) else 0
                )
                return (score, alive_flag, int(length_like), -i)

            mvp_idx = max(range(self.config.num_snakes), key=_mvp_key) if self.config.num_snakes > 0 else None
            if self.config.num_snakes > 1:
                winner_is_all_dead = (len(alive_idxs) == 0)
            if mvp_idx is not None:
                winner_len = len(self.snakes[mvp_idx]) if (mvp_idx < len(self.snakes) and self.snakes[mvp_idx]) else 0

        if game_over:
            for i in range(self.config.num_snakes): dones[i] = True
            
            # 终局奖励：多人模式给赢家 win_reward，其余给 loss_penalty
            if self.config.num_snakes > 1 and mvp_idx is not None:
                for i in range(self.config.num_snakes):
                    rewards[i] += self.config.win_reward if i == mvp_idx else self.config.loss_penalty
        
        info = {
            "scores": self.scores,
            "game_over": game_over,
            "alive_count": alive_count,
            # Backward-compat: winner == MVP (highest score)
            "winner_idx": mvp_idx,
            "mvp_idx": mvp_idx,
            "winner_len": winner_len,
            "winner_all_dead": winner_is_all_dead,
            "pre_terminal_lengths": self.pre_terminal_lengths,
            # Backward-compat alias (was briefly used as "max_lengths")
            "max_lengths": self.pre_terminal_lengths,
        }

        return self._get_observations(), rewards, dones, info

    def _handle_death(self, idx: int):
        self.dead[idx] = True
        # Game rule: convert the whole body into food (deterministic)
        if idx < len(self.snakes) and self.snakes[idx]:
            for segment in self.snakes[idx]:
                if segment not in self.foods:
                    self.foods.append(segment)

    def _get_observations(self) -> List[ObservationDict]:
        cache = self._build_obs_cache()
        return [self._get_agent_obs_cached(i, cache) for i in range(self.config.num_snakes)]

    def _build_obs_cache(self):
        """Build reusable O(1) lookup tables for observation construction."""
        occupied_all = set()
        head_map: Dict[Tuple[int, int], int] = {}
        for i in range(self.config.num_snakes):
            if self.dead[i] or not self.snakes[i]:
                continue
            occupied_all.update(self.snakes[i])
            head_map[self.snakes[i][0]] = i
        food_set = set(self.foods)
        return occupied_all, head_map, food_set

    def _get_agent_obs(self, agent_idx: int) -> ObservationDict:
        # Keep API stable (used by net/game_client.py). This path is slower than
        # `_get_observations()` but fine for occasional calls.
        cache = self._build_obs_cache()
        return self._get_agent_obs_cached(agent_idx, cache)

    def _get_agent_obs_cached(self, agent_idx: int, cache) -> ObservationDict:
        occupied_all, head_map, food_set = cache
        if self.dead[agent_idx]:
             return {
                 "vector": np.zeros(self.obs_dim, dtype=np.float32),
                 "grid": np.zeros((5, self.height, self.width), dtype=np.uint8)
             }
             
        head = self.snakes[agent_idx][0]
        direction = self.directions[agent_idx]
        
        # 1. Vector Features
        food_up, food_down, food_left, food_right = 0.0, 0.0, 0.0, 0.0
        if self.foods:
            # Deterministic tie-break: when two foods are equidistant, always pick the same one
            # to avoid observation jitter that can cause limit-cycles.
            closest_food = min(
                self.foods,
                key=lambda f: (
                    abs(head[0] - f[0]) + abs(head[1] - f[1]),
                    f[0],
                    f[1],
                ),
            )
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
            danger_1.append(float(self._is_danger_cached(p1, occupied_all, self.width, self.height)))
            p2 = self._get_next_pos(p1, d)
            danger_2.append(float(self._is_danger_cached(p2, occupied_all, self.width, self.height)))
            dist, cur = 1, p1
            while 0 <= cur[0] < self.width and 0 <= cur[1] < self.height:
                if self._is_danger_cached(cur, occupied_all, self.width, self.height):
                    break
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
        
        # Updated Vector features: Remaining Dash steps vs Can Dash
        can_dash_val = [1.0 if (self.dash_durations[agent_idx] == 0 and len(self.snakes[agent_idx]) > 3) else 0.0]
        
        # V10.0: Relative Standing Features (Crucial for rule awareness)
        # NOTE: With score-based MVP rules, standing should be based on game score (not length).
        is_leader = 0.0
        rel_score = 1.0
        if self.config.num_snakes > 1:
            scores = list(self.scores) if isinstance(self.scores, list) else []
            # Defensive: keep shape stable even if scores are missing/misaligned.
            if len(scores) < self.config.num_snakes:
                scores = scores + [0] * (self.config.num_snakes - len(scores))

            my_score = float(scores[agent_idx])
            max_other = max([float(scores[j]) for j in range(self.config.num_snakes) if j != agent_idx] + [0.0])

            # Avoid the trivial all-zero start state marking everyone as leader.
            if (my_score > 0.0) or (max_other > 0.0):
                is_leader = 1.0 if my_score >= max_other else 0.0

            if max_other > 0.0:
                rel_score = my_score / max_other
            else:
                rel_score = 2.0 if my_score > 0.0 else 1.0
        
        # V11.0: Time Management Feature (Crucial for tournament strategy)
        steps_left = [(self.config.max_steps - self.steps) / self.config.max_steps]
            
        vector = np.concatenate([
            [food_up, food_down, food_left, food_right],
            danger_1, danger_2, radar,
            dir_vec,
            enemy_vec,
            tail_rel,
            len_pct,
            can_dash_val,
            [is_leader, rel_score],
            steps_left
        ]).astype(np.float32)
        # V11.0: Maintain full 28-dimensional vector
        if len(vector) > 28: vector = vector[:28]
        elif len(vector) < 28: vector = np.pad(vector, (0, 28 - len(vector)))

        # 2. Grid (Full Map Encoding - 5 channels)
        # Channel 0: Food
        # Channel 1: Self Body
        # Channel 2: Enemy Heads
        # Channel 3: Enemy Bodies
        # Channel 4: Obstacles (Walls are implicit by grid boundaries, here we mark them as 1 if outside)
        grid = np.zeros((5, self.height, self.width), dtype=np.uint8)

        # Food (vectorized)
        if self.foods:
            fx, fy = zip(*self.foods)
            grid[0, np.asarray(fy, dtype=np.intp), np.asarray(fx, dtype=np.intp)] = 1

        # Snakes (vectorized per snake)
        for i in range(self.config.num_snakes):
            if self.dead[i] or not self.snakes[i]:
                continue

            if i == agent_idx:
                bx, by = zip(*self.snakes[i])
                grid[1, np.asarray(by, dtype=np.intp), np.asarray(bx, dtype=np.intp)] = 1
            else:
                hx, hy = self.snakes[i][0]
                grid[2, hy, hx] = 1
                if len(self.snakes[i]) > 1:
                    bx, by = zip(*self.snakes[i][1:])
                    grid[3, np.asarray(by, dtype=np.intp), np.asarray(bx, dtype=np.intp)] = 1
        
        # Channel 4 could be used for static obstacles if any, or specialized features. 
        # Here we leave it or fill with boundaries if needed (though CNN handles coords fine).
        
        return {"vector": vector, "grid": grid}

    @staticmethod
    def _is_danger_cached(pos: Tuple[int, int], occupied_all: set, width: int, height: int) -> bool:
        x, y = pos
        if not (0 <= x < width and 0 <= y < height):
            return True
        return pos in occupied_all

    def _is_danger(self, agent_idx: int, pos: Tuple[int, int]) -> bool:
        x, y = pos
        if not (0 <= x < self.width and 0 <= y < self.height): return True
        occupied_all, _, _ = self._build_obs_cache()
        return pos in occupied_all

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

