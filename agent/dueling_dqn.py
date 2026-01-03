"""agent/dueling_dqn.py

【中文说明】
Dueling DDQN + PER 推理侧实现。

中文要点：Dueling 结构把 $Q(s,a)$ 拆成 $V(s)$ 与 $A(s,a)$，在复杂状态下更稳定；
本文件提供网络与推理封装，便于加载模型进行对战/演示。

历史说明：本文件早期包含英文模块介绍，已统一为中文说明。
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Dict

class DuelingDQNNet(nn.Module):
    """Hybrid CNN-MLP + Dueling Heads。

    中文：输出采用 $Q(s,a)=V(s)+A(s,a)-\\mathrm{mean}_a A(s,a)$，避免不可辨识性。
    """
    def __init__(self, vector_dim: int = 28, grid_shape: tuple = (5, 20, 20), action_dim: int = 4):
        super().__init__()
        
        # 特征提取：CNN 处理 grid
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
        
        # 2) 共享全连接层
        self.shared_fc = nn.Sequential(
            nn.Linear(vector_dim + cnn_out_dim, 1024),
            nn.ReLU()
        )
        
        # 3) 两个分支：Value / Advantage
        self.value_stream = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.advantage_stream = nn.Sequential(
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
        features = self.shared_fc(combined)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

class DuelingDQNAgent:
    """Dueling-DQN 推理封装（加载权重 + greedy 动作）。"""
    def __init__(self, input_dim: int = 28, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = DuelingDQNNet(vector_dim=input_dim).to(self.device)
        self.net.eval()
        if model_path: self.load(model_path)
            
    def load(self, path: str):
        """加载模型权重（`.pth` 的 state_dict）。"""
        path = Path(path)
        if path.exists():
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            self.net.load_state_dict(state_dict)
        else:
            print(f"Warning: Dueling-DQN model not found at {path}")

    def act(self, obs: Dict[str, np.ndarray]) -> int:
        """根据观测选择动作（贪心 argmax）。"""
        with torch.no_grad():
            t_grid = torch.as_tensor(obs['grid'], dtype=torch.float32, device=self.device).unsqueeze(0)
            t_vec = torch.as_tensor(obs['vector'], dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.net(t_grid, t_vec)
            return int(q_values.argmax(dim=1).item())
