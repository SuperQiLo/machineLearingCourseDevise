"""agent/dqn.py

【中文说明】
最基础的 DQN 推理侧实现（提供网络 `DQNNet` 与推理封装 `DQNAgent`）。

- 输入观测是字典：`{'grid': (5,20,20), 'vector': (28,)}`。
- `act()` 返回离散动作 id（与环境方向映射保持一致）。

历史说明：本文件早期包含英文模块介绍，已统一为中文说明。
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Dict

class DQNNet(nn.Module):
    """Hybrid CNN-MLP Architecture for Snake AI.

    中文：卷积网络提取 grid 特征，MLP 融合 vector 全局特征，输出各动作的 Q 值。
    """
    def __init__(self, vector_dim: int = 28, grid_shape: tuple = (5, 20, 20), action_dim: int = 4):
        super().__init__()
        
        # 1) CNN：处理 20x20x5 的 grid
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
        
        # CNN 输出维度：128 * 4 * 4 = 2048
        cnn_out_dim = 2048
        
        # 2) MLP：拼接 grid 特征与 vector 特征
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
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, grid: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.conv(grid)
        combined = torch.cat([cnn_feat, vector], dim=1)
        return self.fc(combined)

class DQNAgent:
    """DQN 推理封装。

    中文：负责加载权重、把 numpy 观测转为 torch 张量，并输出 greedy 动作。
    """
    def __init__(self, input_dim: int = 28, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = DQNNet(vector_dim=input_dim).to(self.device)
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
            print(f"Warning: DQN model not found at {path}")

    def act(self, obs: Dict[str, np.ndarray]) -> int:
        """根据观测选择动作（贪心 argmax）。

        参数：
        - `obs['grid']`：形状 (5, 20, 20)
        - `obs['vector']`：形状 (28,)
        """
        with torch.no_grad():
            t_grid = torch.as_tensor(obs['grid'], dtype=torch.float32, device=self.device).unsqueeze(0)
            t_vec = torch.as_tensor(obs['vector'], dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.net(t_grid, t_vec)
            return int(q_values.argmax(dim=1).item())
