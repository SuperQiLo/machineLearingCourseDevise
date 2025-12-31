"""
Dueling DDQN + PER Agent V5.
Focus: State value and advantage decomposition for better decision making in complex states.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Dict

class DuelingDQNNet(nn.Module):
    """Hybrid CNN-MLP Architecture with Dueling Heads."""
    def __init__(self, vector_dim: int = 28, grid_shape: tuple = (5, 20, 20), action_dim: int = 4):
        super().__init__()
        
        # Feature Extraction (CNN for Full Grid)
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
        
        # 2. Shared FC (V15.0: Widened to 1024)
        self.shared_fc = nn.Sequential(
            nn.Linear(vector_dim + cnn_out_dim, 1024),
            nn.ReLU()
        )
        
        # 3. Streams
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
    """Helper class for Dueling DDQN + PER inference"""
    def __init__(self, input_dim: int = 28, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = DuelingDQNNet(vector_dim=input_dim).to(self.device)
        self.net.eval()
        if model_path: self.load(model_path)
            
    def load(self, path: str):
        path = Path(path)
        if path.exists():
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            self.net.load_state_dict(state_dict)
        else:
            print(f"Warning: Dueling-DQN model not found at {path}")

    def act(self, obs: Dict[str, np.ndarray]) -> int:
        with torch.no_grad():
            t_grid = torch.as_tensor(obs['grid'], dtype=torch.float32, device=self.device).unsqueeze(0)
            t_vec = torch.as_tensor(obs['vector'], dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.net(t_grid, t_vec)
            return int(q_values.argmax(dim=1).item())
