"""Rainbow 算法使用的卷积骨干与 NoisyLinear 模块。"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class NoisyLinear(nn.Module):
    """Factorized NoisyLinear，用于取代传统全连接层以实现探索。"""

    def __init__(self, in_features: int, out_features: int, sigma_zero: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 可学习的均值与方差参数
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # 运行期生成的噪声缓存
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))

        self.sigma_zero = sigma_zero
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        """按照论文建议初始化均值/方差参数。"""

        bound = 1 / self.in_features**0.5
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, self.sigma_zero / self.in_features**0.5)
        nn.init.constant_(self.bias_sigma, self.sigma_zero / self.out_features**0.5)

    def reset_noise(self) -> None:
        """为权重与偏置重新采样独立噪声。"""

        device = self.weight_mu.device
        eps_out = self._scale_noise(self.out_features, device)
        eps_in = self._scale_noise(self.in_features, device)
        self.weight_epsilon.copy_(eps_out.unsqueeze(1) * eps_in.unsqueeze(0))
        self.bias_epsilon.copy_(eps_out)

    @staticmethod
    def _scale_noise(size: int, device: torch.device) -> torch.Tensor:
        """根据 NoisyNet 论文构造 factorized 噪声。"""

        noise = torch.randn(size, device=device)
        return noise.sign() * noise.abs().sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)


class RainbowBackbone(nn.Module):
    """面向网格观测的卷积骨干网络，输出 dueling 结构的 (A, V) 表示。"""

    def __init__(
        self,
        input_channels: int = 3,
        grid_size: int = 30,
        action_dim: int = 3,
        atom_size: int = 51,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.atom_size = atom_size

        # 采样网格大小以自动推导展平维度，避免手算
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

        self.advantage_head = nn.Sequential(
            NoisyLinear(flat_dim, 512),
            nn.ReLU(inplace=True),
            NoisyLinear(512, action_dim * atom_size),
        )

        self.value_head = nn.Sequential(
            NoisyLinear(flat_dim, 512),
            nn.ReLU(inplace=True),
            NoisyLinear(512, atom_size),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回优势项与价值项的原始 logits，供上层组装分布。"""

        features = self.encoder(x)
        features = features.view(features.size(0), -1)

        advantage = self.advantage_head(features)
        advantage = advantage.view(-1, self.action_dim, self.atom_size)

        value = self.value_head(features)
        value = value.view(-1, 1, self.atom_size)

        return advantage, value

    def reset_noise(self) -> None:
        """训练阶段每个 step 调用以刷新 NoisyLinear 的噪声。"""

        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
