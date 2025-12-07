"""Utility helpers shared across network modules."""

from __future__ import annotations

from typing import Optional

from env.multi_snake_env import Direction


def direction_to_relative(current: Direction, desired: Direction) -> int:
    """将绝对方向转换为相对动作 (0直行/1左转/2右转)。"""
    if current == desired:
        return 0
    if desired == Direction((current.value - 1) % 4):
        return 1
    if desired == Direction((current.value + 1) % 4):
        return 2
    return 0


def normalize_port(value: Optional[str], default: int = 5555) -> int:
    """将端口字符串安全转换为整数，无法解析时返回默认值。"""
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default
