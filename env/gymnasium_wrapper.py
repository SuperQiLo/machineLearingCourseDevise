"""
Gymnasium-compatible wrapper for BattleSnakeEnv.
Enables AsyncVectorEnv for parallel environment execution.
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any

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
    ):
        super().__init__()
        self.config = config or BattleSnakeConfig()
        self._env = BattleSnakeEnv(self.config, seed=seed)
        self.render_mode = render_mode
        
        # Define observation space (Dict space for grid + vector)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(5, self.config.height, self.config.width),
                dtype=np.float32,
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
        # Return only the first snake's observation
        return obs_list[0], {"full_obs": obs_list}
    
    def step(
        self, action: Union[int, List[int], np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        # Build actions list for all snakes
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
        
        # Return only the first snake's data
        obs = obs_list[0]
        reward = rewards[0]
        terminated = dones[0]
        truncated = False  # Gymnasium convention
        
        info_out = {
            "full_obs": obs_list,
            "full_rewards": rewards,
            "full_dones": dones,
            "scores": info.get("scores", []),
        }
        
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
    
    return BattleSnakeGymnasiumEnv(config, seed=seed)
