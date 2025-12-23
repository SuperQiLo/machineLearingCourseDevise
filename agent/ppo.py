"""
PPO Agent V3 Implementation.
Hybrid Actor-Critic: CNN (Local Grid) + MLP (Global Vector).
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict

class ActorCritic(nn.Module):
    """Hybrid CNN-MLP Architecture for PPO"""
    def __init__(self, vector_dim: int = 24, grid_shape: tuple = (3, 7, 7), action_dim: int = 4):
        super().__init__()
        
        # 1. Feature Extractor (CNN for Grid)
        self.conv = nn.Sequential(
            nn.Conv2d(grid_shape[0], 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # 32 * 6 * 6 = 1152
        cnn_out_dim = 32 * 6 * 6
        
        # 2. Actor Head
        self.actor = nn.Sequential(
            nn.Linear(vector_dim + cnn_out_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # 3. Critic Head
        self.critic = nn.Sequential(
            nn.Linear(vector_dim + cnn_out_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_shared(self, grid, vector):
        cnn_feat = self.conv(grid)
        combined = torch.cat([cnn_feat, vector], dim=1)
        return combined

    def get_value(self, grid, vector):
        shared = self.forward_shared(grid, vector)
        return self.critic(shared)

    def get_action_and_value(self, grid, vector, action=None):
        shared = self.forward_shared(grid, vector)
        probs = self.actor(shared)
        dist = Categorical(probs)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(shared)

class PPOAgent:
    """Helper class for PPO inference in V3"""
    def __init__(self, input_dim=24, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ActorCritic(vector_dim=input_dim).to(self.device)
        self.net.eval()
        
        if model_path:
            self.load(model_path)
            
    def load(self, path: str):
        path = Path(path)
        if path.exists():
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            self.net.load_state_dict(state_dict)
            print(f"PPO V3 Agent loaded from {path}")
        else:
            print(f"Warning: PPO model not found at {path}")

    def act(self, obs: Dict[str, np.ndarray]) -> int:
        with torch.no_grad():
            t_grid = torch.as_tensor(obs['grid'], dtype=torch.float32, device=self.device).unsqueeze(0)
            t_vec = torch.as_tensor(obs['vector'], dtype=torch.float32, device=self.device).unsqueeze(0)
            probs = self.net.actor(self.net.forward_shared(t_grid, t_vec))
            return int(probs.argmax(dim=1).item())
