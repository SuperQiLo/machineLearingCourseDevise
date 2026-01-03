"""agent/ppo.py

【中文说明】
PPO 推理侧实现（Actor-Critic 网络 + 推理封装）。

- `ActorCritic`：共享特征提取（CNN+拼接 vector），actor 输出 logits，critic 输出 value。
- `PPOAgent`：加载权重并输出 greedy 动作（评估/对战时常用）。

历史说明：本文件早期包含英文版本号注释，已统一为中文说明。
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict

class ActorCritic(nn.Module):
    """PPO 的 Actor-Critic 网络。

    中文：本实现的 actor 输出 logits（不做 softmax），采样/贪心选择在分布层完成。
    """
    def __init__(self, vector_dim: int = 28, grid_shape: tuple = (5, 20, 20), action_dim: int = 4):
        super().__init__()
        
        # 1) 特征提取：CNN 处理 grid
        self.conv = nn.Sequential(
            nn.Conv2d(grid_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 20x20 -> 10x10
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 10x10 -> 5x5
            nn.Conv2d(64, 128, kernel_size=2, stride=1), # 5x5 -> 4x4
            nn.ReLU(),
            nn.Flatten()
        )
        # 128 * 4 * 4 = 2048
        cnn_out_dim = 2048
        
        # 2) Actor 头：输出 logits
        self.actor = nn.Sequential(
            nn.Linear(vector_dim + cnn_out_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim)
            # 中文：这里不做 softmax，交给 Categorical 分布层处理
        )
        
        # 3) Critic 头：输出 value
        self.critic = nn.Sequential(
            nn.Linear(vector_dim + cnn_out_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, grid, vector):
        shared = self.forward_shared(grid, vector)
        return self.actor(shared), self.critic(shared)

    def forward_shared(self, grid, vector):
        cnn_feat = self.conv(grid)
        combined = torch.cat([cnn_feat, vector], dim=1)
        return combined

    def get_value(self, grid, vector):
        shared = self.forward_shared(grid, vector)
        return self.critic(shared)

    def get_action_and_value(self, grid, vector, action=None):
        shared = self.forward_shared(grid, vector)
        logits = self.actor(shared)
        dist = Categorical(logits=logits) # Stability: Use Logits
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(shared)

class PPOAgent:
    """PPO 推理封装（加载权重 + greedy 动作）。"""
    def __init__(self, input_dim=28, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ActorCritic(vector_dim=input_dim).to(self.device)
        self.net.eval()
        
        if model_path:
            self.load(model_path)
            
    def load(self, path: str):
        """加载模型权重（`.pth` 的 state_dict）。"""
        path = Path(path)
        if path.exists():
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            self.net.load_state_dict(state_dict)
        else:
            print(f"Warning: PPO model not found at {path}")

    def act(self, obs: Dict[str, np.ndarray]) -> int:
        """根据观测选择动作（贪心 argmax logits）。"""
        with torch.no_grad():
            t_grid = torch.as_tensor(obs['grid'], dtype=torch.float32, device=self.device).unsqueeze(0)
            t_vec = torch.as_tensor(obs['vector'], dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.net.actor(self.net.forward_shared(t_grid, t_vec))
            return int(logits.argmax(dim=1).item())
