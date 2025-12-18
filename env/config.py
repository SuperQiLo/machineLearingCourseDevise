"""多蛇环境配置。

训练效果差时，最常见的问题并不是网络或算法本身，而是：
- 奖励信号过于稀疏/方向错误（例如正的“存活奖励”会鼓励原地转圈）
- 环境接口不稳定（reward 在 trainer 拼接，导致不一致/不可复现）

这里把“环境动力学 + reward 计算”固化在 env 层，A2C/Rainbow 共用同一套定义。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass
class EnvConfig:
    width: int = 30
    height: int = 30
    num_snakes: int = 4
    num_food: int = 6
    max_steps: int = 1500

    # Reward 设计（默认可训练：避免“转圈”局部最优）
    step_penalty: float = -0.01
    food_reward: float = 1.0
    death_penalty: float = -1.0
    kill_reward: float = 0.5

    # 形势塑形：基于“到最近食物距离”的差分 shaping（越靠近越好）
    distance_shaping_scale: float = 0.01

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], defaults: Optional["EnvConfig"] = None) -> "EnvConfig":
        base = defaults or cls()

        def _int(key: str, fallback: int, minimum: int = 1) -> int:
            try:
                value = int(raw.get(key, fallback))
            except Exception:
                value = fallback
            return max(minimum, value)

        def _float(key: str, fallback: float) -> float:
            try:
                return float(raw.get(key, fallback))
            except Exception:
                return fallback

        grid = raw.get("grid_size")
        width = _int("width", base.width, minimum=4)
        height = _int("height", base.height, minimum=4)
        if grid is not None:
            try:
                size = max(4, int(grid))
                width = size
                height = size
            except Exception:
                pass

        return cls(
            width=width,
            height=height,
            num_snakes=_int("num_snakes", base.num_snakes, minimum=1),
            num_food=_int("num_food", base.num_food, minimum=1),
            max_steps=_int("max_steps", base.max_steps, minimum=1),
            step_penalty=_float("step_penalty", base.step_penalty),
            food_reward=_float("food_reward", base.food_reward),
            death_penalty=_float("death_penalty", base.death_penalty),
            kill_reward=_float("kill_reward", base.kill_reward),
            distance_shaping_scale=_float("distance_shaping_scale", base.distance_shaping_scale),
        )

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "num_snakes": self.num_snakes,
            "num_food": self.num_food,
            "max_steps": self.max_steps,
            "step_penalty": self.step_penalty,
            "food_reward": self.food_reward,
            "death_penalty": self.death_penalty,
            "kill_reward": self.kill_reward,
            "distance_shaping_scale": self.distance_shaping_scale,
        }
