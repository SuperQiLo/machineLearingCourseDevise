"""
DQN Agent Implementation.
Encapsulates Network Definition, Loading, and Inference.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional

class DQNNet(nn.Module):
    """Deep Q-Network Architecture"""
    def __init__(self, input_dim: int, hidden_dim: int = 256, action_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DQNAgent:
    """Helper class for DQN inference"""
    def __init__(self, input_dim: int, model_path: Optional[str] = None, device: str = "auto"):
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else "cpu")
        self.net = DQNNet(input_dim=input_dim).to(self.device)
        self.net.eval()
        
        if model_path:
            self.load(model_path)
            
    def load(self, path: str):
        path = Path(path)
        if path.exists():
            # Use weights_only=True for security
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            self.net.load_state_dict(state_dict)
            print(f"DQN Agent loaded from {path}")
        else:
            print(f"Warning: DQN model not found at {path}")

    def act(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            t_obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.net(t_obs)
            return int(q_values.argmax(dim=1).item())
