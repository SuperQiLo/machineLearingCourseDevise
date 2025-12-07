"""PPO 中使用的卷积 Actor-Critic 网络。"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    """使用共享卷积分支的策略-价值网络。"""

    def __init__(self, input_channels: int = 3, grid_size: int = 20, action_dim: int = 3) -> None:
        """构建卷积特征提取与 actor/critic 头。"""
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.flatten_dim = 64 * grid_size * grid_size

        self.actor_fc = nn.Linear(self.flatten_dim, 256)
        self.actor_out = nn.Linear(256, action_dim)

        self.critic_fc = nn.Linear(self.flatten_dim, 256)
        self.critic_out = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向推理，返回动作概率与状态价值。"""

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        actor_feat = F.relu(self.actor_fc(x))
        logits = self.actor_out(actor_feat)
        probs = F.softmax(logits, dim=-1)

        critic_feat = F.relu(self.critic_fc(x))
        value = self.critic_out(critic_feat)
        return probs, value

    def act(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """与环境交互时采样动作，返回动作与对数概率。"""

        if state.dim() == 3:
            state = state.unsqueeze(0)
        probs, _ = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """训练阶段计算动作对数概率、价值以及熵。"""

        probs, state_values = self.forward(states)
        dist = torch.distributions.Categorical(probs)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        return action_logprobs, torch.squeeze(state_values, -1), dist_entropy
