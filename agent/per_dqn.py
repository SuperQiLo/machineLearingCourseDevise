"""agent/per_dqn.py

【中文说明】
DDQN + PER（Prioritized Experience Replay）推理侧实现。

注意：PER 的重点在训练侧（按 TD-error 抽样与重要性采样权重修正）。
本文件主要用于：加载训练好的权重并在环境中做动作推断。

历史说明：本文件早期包含英文模块介绍，已统一为中文说明。
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple

class SumTree:
    """PER 的 SumTree 结构（本项目训练脚本当前未必使用该实现）。"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.ptr = 0
        self.size = 0

    def add(self, p: float, data: object):
        idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        self.update(idx, p)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx: int, p: float):
        change = p - self.tree[idx]
        self.tree[idx] = p
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get_leaf(self, v: float) -> Tuple[int, float, object]:
        parent = 0
        while True:
            left = 2 * parent + 1
            right = left + 1
            if left >= len(self.tree): break
            if v <= self.tree[left]:
                parent = left
            else:
                v -= self.tree[left]
                parent = right
        data_idx = parent - self.capacity + 1
        return parent, self.tree[parent], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]

class PERDQNNet(nn.Module):
    """Hybrid CNN-MLP 网络（PER 兼容）。"""
    def __init__(self, vector_dim: int = 28, grid_shape: tuple = (5, 20, 20), action_dim: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(grid_shape[0], 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 20x20 -> 10x10
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 10x10 -> 5x5
            nn.Conv2d(32, 128, kernel_size=2, stride=1), # 5x5 -> 4x4
            nn.ReLU(),
            nn.Flatten()
        )
        # 128 * 4 * 4 = 2048
        cnn_out_dim = 2048
        self.fc = nn.Sequential(
            nn.Linear(vector_dim + cnn_out_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None: nn.init.zeros_(m.bias)
    
    def forward(self, grid: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.conv(grid)
        combined = torch.cat([cnn_feat, vector], dim=1)
        return self.fc(combined)

class PERDQNAgent:
    """PER-DQN 推理封装（加载权重 + greedy 动作）。"""
    def __init__(self, input_dim: int = 28, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = PERDQNNet(vector_dim=input_dim).to(self.device)
        self.net.eval()
        if model_path: self.load(model_path)
            
    def load(self, path: str):
        """加载模型权重（`.pth` 的 state_dict）。"""
        path = Path(path)
        if path.exists():
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            self.net.load_state_dict(state_dict)
        else:
            print(f"Warning: PER-DQN model not found at {path}")

    def act(self, obs: Dict[str, np.ndarray]) -> int:
        """根据观测选择动作（贪心 argmax）。"""
        with torch.no_grad():
            t_grid = torch.as_tensor(obs['grid'], dtype=torch.float32, device=self.device).unsqueeze(0)
            t_vec = torch.as_tensor(obs['vector'], dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.net(t_grid, t_vec)
            return int(q_values.argmax(dim=1).item())
