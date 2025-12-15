"""A2C 算法使用的卷积 Actor-Critic 网络。"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class A2CBackbone(nn.Module):
    """面向网格观测的卷积骨干，输出 (policy_logits, value)。"""

    def __init__(
        self,
        input_channels: int = 3,
        grid_size: int = 30,
        action_dim: int = 3,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim

        dummy = torch.zeros(1, input_channels, grid_size, grid_size)

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            encoded = self.encoder(dummy)
            flat_dim = int(encoded.numel())

        self.policy_head = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, action_dim),
        )

        self.value_head = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value
