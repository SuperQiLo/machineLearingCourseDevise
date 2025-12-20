"""多蛇环境配置。

训练效果差时，最常见的问题并不是网络或算法本身，而是：
- 奖励信号过于稀疏/方向错误（例如正的"存活奖励"会鼓励原地转圈）
- 环境接口不稳定（reward 在 trainer 拼接，导致不一致/不可复现）

这里把"环境动力学 + reward 计算"固化在 env 层，A2C/Rainbow 共用同一套定义。
OBS: 返回 (3, 84, 84) RGB 图像。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass
class EnvConfig:
    """多蛇环境的核心配置类。

    Attributes:
        width: 地图宽度（格子数）
        height: 地图高度（格子数）
        num_snakes: 蛇的数量
        num_food: 食物的数量
        max_steps: 每局游戏的最大步数
    """

    # ==================== 环境基础参数 ====================
    width: int = 30                    # 地图宽度
    height: int = 30                   # 地图高度
    num_snakes: int = 4                # 同场蛇数量
    num_food: int = 6                  # 食物数量
    max_steps: int = 1500              # 最大步数（防止无限游戏）

    # ==================== 奖励函数设计 ====================
    # 核心思想：每一步都有代价，吃到食物有奖励，死亡有惩罚
    # 这样可以鼓励蛇主动寻找食物，而非原地转圈"存活"

    step_penalty: float = -0.01        # 每步惩罚：防止无意义移动/转圈
    food_reward: float = 1.0           # 吃到食物的奖励
    death_penalty: float = -1.0        # 死亡惩罚
    kill_reward: float = 0.5           # 击杀其他蛇的奖励

    # ==================== 奖励塑形 (Reward Shaping) ====================
    # 距离塑形：基于"到最近食物距离"的差分 shaping
    # 如果蛇靠近食物，给予微小正奖励；远离则给予微小负奖励
    # 这有助于缓解稀疏奖励问题，加速学习
    distance_shaping_scale: float = 0.01

    # 重复动作惩罚：连续执行相同动作时的额外惩罚
    # 有助于打破"转圈圈"的局部最优
    repetition_penalty: float = -0.05

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], defaults: Optional["EnvConfig"] = None) -> "EnvConfig":
        """从字典创建配置对象，支持提供默认值。

        Args:
            raw: 包含配置的字典
            defaults: 可选的默认配置对象

        Returns:
            新的 EnvConfig 实例
        """
        base = defaults or cls()

        def _int(key: str, fallback: int, minimum: int = 1) -> int:
            """安全解析整数，带最小值限制。"""
            try:
                value = int(raw.get(key, fallback))
            except Exception:
                value = fallback
            return max(minimum, value)

        def _float(key: str, fallback: float) -> float:
            """安全解析浮点数。"""
            try:
                return float(raw.get(key, fallback))
            except Exception:
                return fallback

        # 支持 grid_size 作为 width 和 height 的简写
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
            repetition_penalty=_float("repetition_penalty", base.repetition_penalty),
        )

    def to_dict(self) -> dict:
        """将配置转换为字典格式。"""
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
            "repetition_penalty": self.repetition_penalty,
        }
