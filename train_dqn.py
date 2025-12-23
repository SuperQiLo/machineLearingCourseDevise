"""
Unified Multi-Agent DQN Training.
Trains a shared policy for N snakes using BattleSnakeEnv.
"""

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

# Import Refactored Modules
from env.battle_snake_env import BattleSnakeEnv, BattleSnakeConfig
from agent.dqn import DQNNet # Use shared Arch

# Re-use ReplayBuffer (Generic)
from collections import deque, namedtuple
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int, batch_size: int):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
        )
    
    def __len__(self): return len(self.buffer)

@dataclass
class TrainConfig:
    total_frames: int = 1_000_000
    batch_size: int = 128
    lr: float = 1e-4
    gamma: float = 0.99
    
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: int = 200_000
    
    target_update: int = 1000
    checkpoint_path: str = "agent/checkpoints/dqn_battle_best.pth"
    load_path: Optional[str] = None # New: Path to pretrained model
    num_snakes: int = 4
    grid_size: int = 20

class UnifiedDQNTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.env = BattleSnakeEnv(BattleSnakeConfig(
            width=cfg.grid_size, height=cfg.grid_size, num_snakes=cfg.num_snakes,min_food=cfg.num_snakes,max_steps=500
        ))
        
        # Init Net from Agent Abstraction
        self.policy_net = DQNNet(input_dim=15, hidden_dim=256).to(self.device)
        self.target_net = DQNNet(input_dim=15, hidden_dim=256).to(self.device)
        
        # Load Pretrained if provided
        if cfg.load_path and Path(cfg.load_path).exists():
            print(f"Loading pretrained model from {cfg.load_path}...")
            state_dict = torch.load(cfg.load_path, map_location=self.device, weights_only=True)
            self.policy_net.load_state_dict(state_dict)
            
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.memory = ReplayBuffer(50_000, cfg.batch_size)
        self.steps = 0
        
    def select_action(self, state):
        eps = self.cfg.eps_end + (self.cfg.eps_start - self.cfg.eps_end) * \
              math.exp(-1. * self.steps / self.cfg.eps_decay)
        
        if random.random() < eps:
            return random.randint(0, 2)
        
        with torch.no_grad():
            t_state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.policy_net(t_state).argmax(dim=1).item()

    def update(self):
        if len(self.memory) < 5000: return
        
        states, actions, rewards, next_states, dones = self.memory.sample()
        
        # To Tensor
        s = torch.tensor(states, device=self.device)
        a = torch.tensor(actions, device=self.device)
        r = torch.tensor(rewards, device=self.device)
        ns = torch.tensor(next_states, device=self.device)
        d = torch.tensor(dones, device=self.device)
        
        q_curr = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            q_next = self.target_net(ns).max(1)[0]
            target = r + self.cfg.gamma * q_next * (~d)
            
        loss = nn.SmoothL1Loss()(q_curr, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()

    def train(self):
        print(f"Starting Unified DQN Training ({self.cfg.num_snakes} snakes)...")
        obs_list = self.env.reset()
        episode_rewards = [0] * self.cfg.num_snakes
        
        ep_count = 0 
        
        while self.steps < self.cfg.total_frames:
            self.steps += 1
            actions = []
            
            # 1. Select Actions for Alive Snakes
            for i in range(self.cfg.num_snakes):
                if self.env.dead[i]:
                    actions.append(0)
                else:
                    actions.append(self.select_action(obs_list[i]))
            
            # 2. Step
            next_obs_list, rewards, dones, info = self.env.step(actions)
            
            # 3. Store Transitions
            for i in range(self.cfg.num_snakes):
                if not self.env.dead[i] or dones[i]: # Store if was alive or just died
                    # If just died/done, done flag is True
                    # We need to distinguish: "Just died this step" vs "Already dead"
                    # But `BattleSnakeEnv` logic: if in `dead` list (updated inside step), it returns zero obs.
                    # rewards[i] != 0 usually if just died.
                    
                    # Store only if it was a meaningful step
                    # Simplified: Store everything where prev state wasn't "Zero/Dead"
                    # Obs check: if np.sum(obs_list[i]) == 0 and not starting? (Unreliable)
                    
                    # Better: Check if we took an action for it.
                    if not (self.env.dead[i] and rewards[i] == 0): # Rough check
                         self.memory.push(obs_list[i], actions[i], rewards[i], next_obs_list[i], dones[i])
                         episode_rewards[i] += rewards[i]

            obs_list = next_obs_list
            
            # 4. Update
            loss = self.update()
            
            if self.steps % self.cfg.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                
            if all(dones):
                ep_count += 1
                if ep_count % 10 == 0:
                    avg = sum(episode_rewards)/len(episode_rewards)
                    print(f"Step {self.steps} | Ep {ep_count} | Avg Reward: {avg:.2f}")
                
                obs_list = self.env.reset()
                episode_rewards = [0] * self.cfg.num_snakes
                
            if self.steps % 50_000 == 0:
                Path(self.cfg.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.policy_net.state_dict(), self.cfg.checkpoint_path)
                print(f"Saved to {self.cfg.checkpoint_path}")

        torch.save(self.policy_net.state_dict(), self.cfg.checkpoint_path)
        print("Training Done.")

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--single", action="store_true", help="Train single snake mode")
    parser.add_argument("--load", type=str, default=None, help="Path to pretrained model to load")
    args = parser.parse_args()
    
    num = 1 if args.single else 4
    path = "agent/checkpoints/dqn_best.pth" if args.single else "agent/checkpoints/dqn_battle_best.pth"
    
    trainer = UnifiedDQNTrainer(TrainConfig(
        num_snakes=num, 
        checkpoint_path=path,
        load_path=args.load
    ))
    trainer.train()
