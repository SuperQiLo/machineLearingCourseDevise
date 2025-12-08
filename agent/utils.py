"""训练/推理阶段复用的公共工具函数。"""

from __future__ import annotations

from typing import Optional

import torch


def resolve_device(preference: Optional[str] = None) -> torch.device:
    """根据用户偏好自动推断可用的 torch.device。

    Args:
        preference: "auto"/"cpu"/"cuda"/"cuda:0" 等字符串，默认为 "auto"。

    Returns:
        torch.device: 可以直接用于模型与张量的设备对象。

    Raises:
        ValueError: 当输入字符串无法解析为合法设备时抛出。
        RuntimeError: 当用户显式请求 CUDA 但当前环境不可用时抛出。
    """

    pref = (preference or "auto").strip()
    if pref.lower() == "auto":
        target = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        target = pref

    try:
        device = torch.device(target)
    except RuntimeError as exc:  # noqa: BLE001
        raise ValueError(f"无法识别的设备：{target}") from exc

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("已请求 CUDA 但当前环境未检测到可用的 GPU。")

    return device
