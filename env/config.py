"""Shared configuration helpers for the multi-snake environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass
class EnvConfig:
    """环境基础配置，统一定义网格尺寸、蛇数量等核心参数。"""
    width: int = 30
    height: int = 30
    num_snakes: int = 4
    num_food: int = 6
    max_steps: int = 1500

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], defaults: Optional["EnvConfig"] = None) -> "EnvConfig":
        """根据外部字典生成配置，自动处理缺省值与合法范围。"""
        base = defaults or cls()

        def _int(key: str, fallback: int, minimum: int = 1) -> int:
            try:
                value = int(raw.get(key, fallback))
            except Exception:
                value = fallback
            return max(minimum, value)

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
        )

    def to_dict(self) -> dict:
        """导出当前配置为字典，方便序列化与日志输出。"""
        return {
            "width": self.width,
            "height": self.height,
            "num_snakes": self.num_snakes,
            "num_food": self.num_food,
            "max_steps": self.max_steps,
        }
