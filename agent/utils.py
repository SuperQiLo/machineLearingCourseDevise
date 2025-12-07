"""Shared helpers for training/inference utilities."""

from __future__ import annotations

from typing import Optional

import torch


def resolve_device(preference: Optional[str] = None) -> torch.device:
    """Resolve a human-readable device string into a torch.device instance.

    Args:
        preference: "auto", "cpu", "cuda", or "cuda:0" style string. Defaults to "auto".

    Returns:
        torch.device: A validated torch device.

    Raises:
        ValueError: If the device string cannot be parsed.
        RuntimeError: If CUDA was explicitly requested but unavailable.
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
