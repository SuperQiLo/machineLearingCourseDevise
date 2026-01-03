"""gymnasium_wrapper.py

Gymnasium 兼容包装器：把多智能体 `BattleSnakeEnv` 适配到单智能体 Gymnasium 接口。

核心约定
- 学习者固定为 snake0；其它蛇作为对手。
- `reset()` / `step()` 返回的 obs/reward/terminated 针对 snake0。
- 但在 `info` 中会额外提供 `full_obs_*` / `scores` / `winner_idx` 等，方便训练器做自博弈与 KPI 统计。

奖励（非常重要）
- `env_reward0`：来自原始环境 `BattleSnakeEnv.step()` 的 reward（dense shaping + 终局奖惩）
- `delta_score0`：本步 snake0 的游戏得分增量（score delta）
- 最终返回给 RL 的 `reward`：
    `reward = env_reward_coef * env_reward0 + (use_score_delta_reward ? score_reward_coef * delta_score0 : 0)`

这样做的目的是：在不改变游戏胜负判定（score/MVP）的前提下，可控地把“得分信号”融入训练。
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from typing import Optional, Tuple, Dict, Any, List, Union

from env.battle_snake_env import BattleSnakeEnv, BattleSnakeConfig


class BattleSnakeGymnasiumEnv(gym.Env):
    """Gymnasium-compatible wrapper for BattleSnakeEnv.
    
    This wrapper adapts the multi-agent BattleSnakeEnv to the single-agent
    Gymnasium interface, exposing only the first snake (index 0) as the
    learning agent.
    """
    
    metadata = {"render_modes": ["rgb_array"]}
    
    def __init__(
        self,
        config: Optional[BattleSnakeConfig] = None,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        return_full_obs: bool = True,
        *,
        use_score_delta_reward: bool = False,
        score_reward_coef: float = 0.001,
        env_reward_coef: float = 1.0,
        dash_effect_window: int = 0,
        dash_success_bonus: float = 0.2,
        dash_fail_penalty: float = -0.2,
        unsafe_move_penalty: float = 0.0,
        unsafe_move2_penalty: float = 0.0,
        unsafe_dash_penalty: float = 0.0,
        invalid_dash_penalty: float = 0.0,
        revisit_penalty: float = 0.0,
        revisit_window: int = 24,
    ):
        super().__init__()
        self.config = config or BattleSnakeConfig()
        self._env = BattleSnakeEnv(self.config, seed=seed)
        self.render_mode = render_mode
        self.return_full_obs = bool(return_full_obs)

        # 训练奖励选项（与游戏 score/MVP 解耦）
        self.use_score_delta_reward = bool(use_score_delta_reward)
        self.score_reward_coef = float(score_reward_coef)
        self.env_reward_coef = float(env_reward_coef)

        # Dash 效果整形（仅学习者）：在 dash_effect_window 窗口内若 score 增加则奖励，否则惩罚
        self.dash_effect_window = int(dash_effect_window)
        self.dash_success_bonus = float(dash_success_bonus)
        self.dash_fail_penalty = float(dash_fail_penalty)
        self._dash_pending_steps = 0

        # 安全整形（仅学习者）：利用观测 vector 里的危险特征惩罚明显不安全动作
        # 目的：减少“撞墙/撞自己”的探索成本（不改变观测 shape）。
        self.unsafe_move_penalty = float(unsafe_move_penalty)
        self.unsafe_move2_penalty = float(unsafe_move2_penalty)
        self.unsafe_dash_penalty = float(unsafe_dash_penalty)
        self.invalid_dash_penalty = float(invalid_dash_penalty)

        # 反循环整形（仅学习者）：惩罚短周期回到最近头位置（防止原地打转）
        self.revisit_penalty = float(revisit_penalty)
        self.revisit_window = int(max(0, revisit_window))
        self._recent_pos0 = deque(maxlen=self.revisit_window) if self.revisit_window > 0 else None

        # Score tracker for delta-score reward
        self._last_score0 = 0

        # Cache last learner obs for action-based shaping
        self._last_obs0: Optional[Dict[str, np.ndarray]] = None
        
        # Define observation space (Dict space for grid + vector)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(5, self.config.height, self.config.width),
                dtype=np.uint8,
            ),
            "vector": spaces.Box(
                low=-10.0,
                high=10.0,
                shape=(28,),
                dtype=np.float32,
            ),
        })
        
        # Action space: 0=Straight, 1=Left, 2=Right, 3=Dash
        self.action_space = spaces.Discrete(4)
        
        # Store opponent actions (for multi-snake mode)
        self._opponent_actions: Optional[list] = None
        
    @property
    def num_snakes(self) -> int:
        return self.config.num_snakes
    
    def set_opponent_actions(self, actions: list) -> None:
        """Set actions for opponent snakes (indices 1+)."""
        self._opponent_actions = actions
        
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if seed is not None:
            self._env.seed(seed)
            
        obs_list = self._env.reset()
        self._last_score0 = int(self._env.scores[0]) if (self._env.scores and len(self._env.scores) > 0) else 0
        self._dash_pending_steps = 0
        if self.return_full_obs:
            # Pre-process ALL observations into arrays for fast global indexing.
            # NOTE: Even in single-snake mode, downstream trainers expect these keys.
            all_grids = np.asarray([o["grid"] for o in obs_list], dtype=np.uint8)
            all_vecs = np.asarray([o["vector"] for o in obs_list])
            # full_obs_*：把所有蛇的观测打包（训练器可用于给对手模型推理动作）
            info = {"full_obs_grids": all_grids, "full_obs_vecs": all_vecs, "raw_obs": obs_list}
        else:
            info = {}

        self._last_obs0 = obs_list[0]

        # Reset anti-loop history (learner head positions)
        if self._recent_pos0 is not None:
            self._recent_pos0.clear()
            try:
                if self._env.snakes and len(self._env.snakes) > 0 and self._env.snakes[0]:
                    self._recent_pos0.append(tuple(self._env.snakes[0][0]))
            except Exception:
                pass

        return obs_list[0], info
    
    def step(
        self, action: Union[int, List[int], np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        # Action-based shaping needs the pre-step learner observation.
        pre_vec = None
        if self._last_obs0 is not None:
            try:
                pre_vec = self._last_obs0.get("vector", None)
            except Exception:
                pre_vec = None

        # 构造所有蛇的动作列表：
        # - 如果传入的是 list/ndarray：认为已经包含全部蛇动作
        # - 否则：仅 learner 动作，其它蛇动作来自 set_opponent_actions 或随机
        if isinstance(action, (list, np.ndarray)):
            # If a list/array of actions is provided, use them directly
            actions = list(action)
            # Ensure we have enough actions for all snakes
            while len(actions) < self.config.num_snakes:
                actions.append(0) # Default to STRAIGHT
        else:
            # Only learner action provided
            actions = [int(action)]
            
            if self.config.num_snakes > 1:
                if self._opponent_actions is not None:
                    actions.extend(self._opponent_actions)
                else:
                    # Default: random actions for opponents
                    import random
                    for _ in range(1, self.config.num_snakes):
                        actions.append(random.randint(0, 3))
        
        obs_list, rewards, dones, info = self._env.step(actions)
        
        # 只返回 snake0 的 (obs, reward, terminated)，但会在 info 中保留全局信息
        obs = obs_list[0]
        env_reward = float(rewards[0])

        # 游戏得分增量（learner only）：用于可选的 score-delta 奖励
        scores = info.get("scores", [])
        score0 = int(scores[0]) if scores else 0
        delta_score0 = score0 - int(self._last_score0)
        self._last_score0 = score0

        # 最终训练 reward：环境 reward * 系数 + 可选 score-delta
        reward = (self.env_reward_coef * env_reward)
        if self.use_score_delta_reward:
            reward += self.score_reward_coef * float(delta_score0)

        # 安全整形（learner only）：根据 pre_vec 的危险特征，对明显危险动作加惩罚
        # Vector layout (28 dims):
        # [0:4]=food, [4:7]=danger_1(s,l,r), [7:10]=danger_2(s,l,r), ... , [24]=can_dash
        unsafe_move0 = 0
        unsafe_move20 = 0
        unsafe_dash0 = 0
        invalid_dash0 = 0
        if pre_vec is not None and pre_vec.shape[0] >= 25:
            a0 = int(actions[0]) if actions else 0
            d1_s, d1_l, d1_r = float(pre_vec[4]), float(pre_vec[5]), float(pre_vec[6])
            d2_s, d2_l, d2_r = float(pre_vec[7]), float(pre_vec[8]), float(pre_vec[9])
            can_dash = float(pre_vec[24]) > 0.5

            if a0 in (0, 1, 2) and self.unsafe_move_penalty != 0.0:
                d1 = d1_s if a0 == 0 else (d1_l if a0 == 1 else d1_r)
                if d1 >= 0.5:
                    reward += self.unsafe_move_penalty
                    unsafe_move0 = 1

            # 2-step lookahead penalty for normal moves (helps reduce "corridor" crashes)
            if a0 in (0, 1, 2) and self.unsafe_move2_penalty != 0.0:
                d2 = d2_s if a0 == 0 else (d2_l if a0 == 1 else d2_r)
                if d2 >= 0.5:
                    reward += self.unsafe_move2_penalty
                    unsafe_move20 = 1
            elif a0 == 3:
                if (not can_dash) and self.invalid_dash_penalty != 0.0:
                    reward += self.invalid_dash_penalty
                    invalid_dash0 = 1
                if self.unsafe_dash_penalty != 0.0:
                    # Dash is essentially a fast forward move; if the 2-step lookahead is dangerous,
                    # penalize to reduce wall/self crashes during/after dash.
                    if d2_s >= 0.5 or d1_s >= 0.5:
                        reward += self.unsafe_dash_penalty
                        unsafe_dash0 = 1

        # 反循环整形：惩罚回到最近 head 位置
        revisit0 = 0
        if self._recent_pos0 is not None and self.revisit_penalty != 0.0:
            try:
                if self._env.snakes and len(self._env.snakes) > 0 and self._env.snakes[0] and (not self._env.dead[0]):
                    head0 = tuple(self._env.snakes[0][0])
                    if head0 in self._recent_pos0:
                        reward += self.revisit_penalty
                        revisit0 = 1
                    self._recent_pos0.append(head0)
            except Exception:
                pass

        # Dash 效果整形：若 Dash 后短窗口内 score 有提升则奖励，否则惩罚
        learner_action = int(actions[0]) if actions else 0
        if self.dash_effect_window > 0:
            # Start a new dash window
            if learner_action == 3 and self._dash_pending_steps <= 0:
                self._dash_pending_steps = self.dash_effect_window

            if self._dash_pending_steps > 0:
                if delta_score0 > 0:
                    reward += self.dash_success_bonus
                    self._dash_pending_steps = 0
                else:
                    self._dash_pending_steps -= 1
                    if self._dash_pending_steps <= 0:
                        reward += self.dash_fail_penalty
        terminated = bool(info.get("game_over", all(dones)))
        truncated = False  # Gymnasium convention
        
        # 输出给训练器的 info（尽量保持“易于堆叠/统计”的标量字段）
        # - scores / winner_idx：用于对战 KPI
        # - score0 / delta_score0 / env_reward0：用于区分“训练 reward”与“游戏得分”
        winner_idx = info.get("winner_idx", info.get("mvp_idx", None))
        if winner_idx is None:
            winner_idx = -1
        else:
            try:
                winner_idx = int(winner_idx)
            except Exception:
                winner_idx = -1

        info_out: Dict[str, Any] = {
            "full_rewards": rewards,
            "full_dones": dones,
            "scores": scores,
            "score0": score0,
            "delta_score0": delta_score0,
            "env_reward0": env_reward,
            "alive_count": int(info.get("alive_count", -1)),
            "game_over": bool(info.get("game_over", terminated)),
            "winner_idx": winner_idx,
            "mvp_idx": winner_idx,
            "unsafe_move0": int(unsafe_move0),
            "unsafe_move20": int(unsafe_move20),
            "unsafe_dash0": int(unsafe_dash0),
            "invalid_dash0": int(invalid_dash0),
            "revisit0": int(revisit0),
        }

        if self.return_full_obs:
            info_out["full_obs_grids"] = np.asarray([o["grid"] for o in obs_list], dtype=np.uint8)
            info_out["full_obs_vecs"] = np.asarray([o["vector"] for o in obs_list])

        # Update cache for next step
        self._last_obs0 = obs_list[0]
        
        return obs, reward, terminated, truncated, info_out
    
    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            # Simple grid visualization (can be extended)
            return None
        return None
    
    def close(self) -> None:
        pass


def make_gymnasium_env(
    num_snakes: int = 1,
    grid_size: int = 20,
    seed: Optional[int] = None,
    return_full_obs: bool = True,
    use_score_delta_reward: bool = False,
    score_reward_coef: float = 0.001,
    env_reward_coef: float = 1.0,
    dash_effect_window: int = 0,
    dash_success_bonus: float = 0.2,
    dash_fail_penalty: float = -0.2,
    unsafe_move_penalty: float = 0.0,
    unsafe_move2_penalty: float = 0.0,
    unsafe_dash_penalty: float = 0.0,
    invalid_dash_penalty: float = 0.0,
    revisit_penalty: float = 0.0,
    revisit_window: int = 24,
    **reward_kwargs,
) -> BattleSnakeGymnasiumEnv:
    """Factory function for creating Gymnasium-compatible BattleSnake environments.
    
    Args:
        num_snakes: Number of snakes in the environment
        grid_size: Width and height of the grid
        seed: Random seed
        **reward_kwargs: Reward configuration (food_reward, death_penalty, etc.)
    
    Returns:
        BattleSnakeGymnasiumEnv instance
    """
    config = BattleSnakeConfig(
        width=grid_size,
        height=grid_size,
        num_snakes=num_snakes,
    )
    
    # Apply reward kwargs
    for key, value in reward_kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return BattleSnakeGymnasiumEnv(
        config,
        seed=seed,
        return_full_obs=return_full_obs,
        use_score_delta_reward=use_score_delta_reward,
        score_reward_coef=score_reward_coef,
        env_reward_coef=env_reward_coef,
        dash_effect_window=dash_effect_window,
        dash_success_bonus=dash_success_bonus,
        dash_fail_penalty=dash_fail_penalty,
        unsafe_move_penalty=unsafe_move_penalty,
        unsafe_move2_penalty=unsafe_move2_penalty,
        unsafe_dash_penalty=unsafe_dash_penalty,
        invalid_dash_penalty=invalid_dash_penalty,
        revisit_penalty=revisit_penalty,
        revisit_window=revisit_window,
    )
