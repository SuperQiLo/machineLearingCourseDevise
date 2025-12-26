"""
DQN Agent V3 Implementation.
Hybrid Architecture: CNN (Local Grid) + MLP (Global Vector).
V5.0: Updated to 25D vector (Cooldown).
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Dict

class DQNNet(nn.Module):
    """Hybrid CNN-MLP Architecture for Snake AI"""
    def __init__(self, vector_dim: int = 25, grid_shape: tuple = (3, 7, 7), action_dim: int = 4):
        super().__init__()
        
        # 1. CNN for Local Grid (7x7x3)
        self.conv = nn.Sequential(
            nn.Conv2d(grid_shape[0], 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate CNN output size (32 * 6 * 6 = 1152)
        cnn_out_dim = 32 * 6 * 6
        
        # 2. MLP for Global Features
        self.fc = nn.Sequential(
            nn.Linear(vector_dim + cnn_out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
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
    """Helper class for DQN inference in V3"""
    def __init__(self, input_dim: int = 25, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = DQNNet(vector_dim=input_dim).to(self.device)
        self.net.eval()
        
        if model_path:
            self.load(model_path)
            
    def load(self, path: str):
        path = Path(path)
        if path.exists():
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            self.net.load_state_dict(state_dict)
        else:
            print(f"Warning: DQN model not found at {path}")

    def act(self, obs: Dict[str, np.ndarray]) -> int:
        """Process dict observation: {'vector': ..., 'grid': ...}"""
        with torch.no_grad():
            t_grid = torch.as_tensor(obs['grid'], dtype=torch.float32, device=self.device).unsqueeze(0)
            t_vec = torch.as_tensor(obs['vector'], dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.net(t_grid, t_vec)
            return int(q_values.argmax(dim=1).item())
