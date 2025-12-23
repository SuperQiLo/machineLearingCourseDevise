"""
PPO Agent Implementation.
Encapsulates Actor-Critic Network and Inference.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

class ActorCritic(nn.Module):
    """PPO Actor-Critic Network"""
    def __init__(self, input_dim: int, hidden_dim: int = 256, action_dim: int = 3):
        super().__init__()
        
        # Shared Feature Extractor (Optional, here we separate)
        
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        probs = self.actor(x)
        dist = Categorical(probs)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(x)

class PPOAgent:
    """Helper class for PPO inference"""
    def __init__(self, input_dim: int, model_path: Optional[str] = None, device: str = "auto"):
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else "cpu")
        self.net = ActorCritic(input_dim=input_dim).to(self.device)
        self.net.eval()
        
        if model_path:
            self.load(model_path)
            
    def load(self, path: str):
        path = Path(path)
        if path.exists():
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            self.net.load_state_dict(state_dict)
            print(f"PPO Agent loaded from {path}")
        else:
            print(f"Warning: PPO model not found at {path}")

    def act(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            t_obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            probs = self.net.actor(t_obs)
            return int(probs.argmax(dim=1).item()) # Deterministic for eval
