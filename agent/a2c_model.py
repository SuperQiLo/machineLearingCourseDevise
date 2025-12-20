"""A2C 算法使用的卷积 Actor-Critic 网络 (增强版)。"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """残差块，加速深层网络训练并提升特征学习能力。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class SpatialAttention(nn.Module):
    """空间注意力模块，帮助模型关注重要区域(如蛇头、食物)。"""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = torch.sigmoid(self.conv(x))
        return x * attention


class A2CBackbone(nn.Module):
    """增强版卷积骨干，包含残差连接、注意力机制和更深的网络结构。"""

    def __init__(
        self,
        input_channels: int = 3,
        obs_size: int = 84,
        action_dim: int = 3,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.obs_size = obs_size

        dummy = torch.zeros(1, input_channels, obs_size, obs_size)

        # 初始特征提取
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 残差块堆叠 - 增强特征学习
        self.res_blocks1 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
        )

        # 下采样 + 通道增加
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.res_blocks2 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
        )

        # 空间注意力
        self.attention = SpatialAttention(128)

        # 第二次下采样
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.res_blocks3 = nn.Sequential(
            ResidualBlock(256),
        )

        # 全局平均池化 + 特征融合
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 空间特征池化：固定输出尺寸，避免 grid_size 变化导致展平维度变化
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))

        # 计算展平维度
        with torch.no_grad():
            x = self.stem(dummy)
            x = self.res_blocks1(x)
            x = self.down1(x)
            x = self.res_blocks2(x)
            x = self.attention(x)
            x = self.down2(x)
            x = self.res_blocks3(x)
            spatial_features = self.spatial_pool(x).view(x.size(0), -1)
            spatial_dim = spatial_features.shape[1]
            
            global_features = self.global_pool(x).view(x.size(0), -1)
            global_dim = global_features.shape[1]

        # 策略头 - 更宽更深
        self.policy_head = nn.Sequential(
            nn.Linear(spatial_dim + global_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, action_dim),
        )

        # 价值头 - 独立的深层网络
        self.value_head = nn.Sequential(
            nn.Linear(spatial_dim + global_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self) -> None:
        """使用正交初始化，有助于策略梯度训练。"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 特征提取
        x = self.stem(x)
        x = self.res_blocks1(x)
        x = self.down1(x)
        x = self.res_blocks2(x)
        x = self.attention(x)
        x = self.down2(x)
        x = self.res_blocks3(x)

        # 融合空间特征和全局特征
        spatial_features = self.spatial_pool(x).view(x.size(0), -1)
        global_features = self.global_pool(x).view(x.size(0), -1)
        features = torch.cat([spatial_features, global_features], dim=-1)

        # 输出
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value
