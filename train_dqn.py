"""
Unified Multi-Agent DQN Training V3 with Self-Play.
Hybrid CNN+MLP for competitive play.
"""

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from env.battle_snake_env import BattleSnakeEnv, BattleSnakeConfig
from agent.dqn import DQNNet, DQNAgent
from utils.self_play import SelfPlayManager

class ReplayBuffer:
    def __init__(self, capacity: int, batch_size: int):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        v_s = np.array([s['vector'] for s in states], dtype=np.float32)
        g_s = np.array([s['grid'] for s in states], dtype=np.float32)
        v_ns = np.array([s['vector'] for s in next_states], dtype=np.float32)
        g_ns = np.array([s['grid'] for s in next_states], dtype=np.float32)
        
        return (
            {"vector": v_s, "grid": g_s},
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            {"vector": v_ns, "grid": g_ns},
            np.array(dones, dtype=np.bool_),
        )
    
    def __len__(self): return len(self.buffer)

@dataclass
class TrainConfig:
    total_frames: int = 2_000_000
    batch_size: int = 128
    lr: float = 2e-4
    gamma: float = 0.99
    
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: int = 400_000
    
    target_update: int = 2000
    checkpoint_path: str = "agent/checkpoints/dqn_battle_best.pth"
    load_path: Optional[str] = None
    num_snakes: int = 4
    grid_size: int = 20
    
    # Self-Play
    self_play_prob: float = 0.3 # 30% episodes involve history agents
    pool_dir: str = "agent/pool/dqn"

class UnifiedDQNTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.env = BattleSnakeEnv(BattleSnakeConfig(
            width=cfg.grid_size, height=cfg.grid_size, 
            num_snakes=cfg.num_snakes,
            min_food=max(2, cfg.num_snakes // 2),
            max_steps=500
        ))
        
        self.policy_net = DQNNet(vector_dim=24).to(self.device)
        self.target_net = DQNNet(vector_dim=24).to(self.device)
        
        if cfg.load_path and Path(cfg.load_path).exists():
            print(f"Loading pretrained model from {cfg.load_path}...")
            state_dict = torch.load(cfg.load_path, map_location=self.device, weights_only=True)
            self.policy_net.load_state_dict(state_dict)
            
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.memory = ReplayBuffer(100_000, cfg.batch_size)
        self.steps = 0
        
        self.sp_manager = SelfPlayManager(cfg.pool_dir)
        self.opponents: List[Optional[DQNAgent]] = [None] * cfg.num_snakes

    def select_action(self, state, opponent_idx=None):
        # If controlled by independent opponent (Self-Play)
        if opponent_idx is not None and self.opponents[opponent_idx]:
            return self.opponents[opponent_idx].act(state)

        eps = self.cfg.eps_end + (self.cfg.eps_start - self.cfg.eps_end) * \
              math.exp(-1. * self.steps / self.cfg.eps_decay)
        
        if random.random() < eps:
            return random.randint(0, 3)
        
        with torch.no_grad():
            t_grid = torch.tensor(state['grid'], dtype=torch.float32, device=self.device).unsqueeze(0)
            t_vec = torch.tensor(state['vector'], dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.policy_net(t_grid, t_vec).argmax(dim=1).item()

    def update(self):
        if len(self.memory) < 10000: return 0.0
        
        states, actions, rewards, next_states, dones = self.memory.sample()
        
        s_g = torch.tensor(states['grid'], device=self.device)
        s_v = torch.tensor(states['vector'], device=self.device)
        a = torch.tensor(actions, device=self.device)
        r = torch.tensor(rewards, device=self.device)
        ns_g = torch.tensor(next_states['grid'], device=self.device)
        ns_v = torch.tensor(next_states['vector'], device=self.device)
        d = torch.tensor(dones, device=self.device)
        
        q_curr = self.policy_net(s_g, s_v).gather(1, a.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            q_next = self.target_net(ns_g, ns_v).max(1)[0]
            target = r + self.cfg.gamma * q_next * (~d)
            
        loss = nn.SmoothL1Loss()(q_curr, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()

    def _prepare_opponents(self):
        """Randomly pick history models for some opponents (Self-Play)"""
        self.opponents = [None] * self.cfg.num_snakes
        if self.cfg.num_snakes > 1 and random.random() < self.cfg.self_play_prob:
            # Pick 1-2 opponents to be historical versions
            count = random.randint(1, max(1, self.cfg.num_snakes // 2))
            indices = random.sample(range(1, self.cfg.num_snakes), count) # Skip P0
            
            for idx in indices:
                model_path = self.sp_manager.sample_model()
                if model_path:
                    # In V3, DQNAgent expects vector_dim=24
                    self.opponents[idx] = DQNAgent(vector_dim=24, model_path=str(model_path))

    def train(self):
        print(f"Starting V3 DQN Training ({self.cfg.num_snakes} snakes) with Self-Play...")
        obs_list = self.env.reset()
        episode_rewards = [0.0] * self.cfg.num_snakes
        ep_count = 0 
        
        total_loss = 0
        best_avg_rew = -float('inf')

        while self.steps < self.cfg.total_frames:
            self.steps += 1
            actions = []
            
            # Select actions
            for i in range(self.cfg.num_snakes):
                if self.env.dead[i]:
                    actions.append(0)
                else:
                    # Only train with experiences from non-Self-Play snakes (or all?)
                    # Let's train using ALL experience but recognize that some are fixed.
                    actions.append(self.select_action(obs_list[i], opponent_idx=i))
            
            next_obs_list, rewards, dones, info = self.env.step(actions)
            
            # Store Transitions
            for i in range(self.cfg.num_snakes):
                # If snake i is controlled by a history agent, we DON'T train on its actions?
                # Actually, training on diverse data is good. But the action selection was different.
                # Standard practice: push to buffer if it's the learning agent (P0).
                # To maximize data, we push for all snakes that AREN'T history agents.
                if self.opponents[i] is None:
                    if not self.env.dead[i] or dones[i]:
                        if not (self.env.dead[i] and rewards[i] == 0):
                             self.memory.push(obs_list[i], actions[i], rewards[i], next_obs_list[i], dones[i])
                             episode_rewards[i] += rewards[i]

            obs_list = next_obs_list
            loss = self.update()
            if loss: total_loss += loss
            
            if self.steps % self.cfg.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                
            if all(dones):
                ep_count += 1
                if ep_count % 10 == 0:
                    avg = sum(episode_rewards)/max(1, sum(1 for x in self.opponents if x is None))
                    print(f"Step {self.steps} | Ep {ep_count} | Avg Training Reward: {avg:.2f}")
                
                # New Episode: Maybe new self-play opponents
                self._prepare_opponents()
                obs_list = self.env.reset()
                episode_rewards = [0.0] * self.cfg.num_snakes
                
            # Periodically Save and Expand Pool
            if self.steps % 100_000 == 0:
                Path(self.cfg.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.policy_net.state_dict(), self.cfg.checkpoint_path)
                # Add to pool for self-play diversity
                timestamp = int(time.time())
                self.sp_manager.add_model(self.policy_net.state_dict(), f"dqn_step_{self.steps}_{timestamp}")
                print(f"Periodic Save: {self.cfg.checkpoint_path} | Pool count: {len(self.sp_manager.history_models)}")

        print("Training Done.")

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--load", type=str, default=None)
    args = parser.parse_args()
    
    num = 1 if args.single else 4
    path = "agent/checkpoints/dqn_best.pth" if args.single else "agent/checkpoints/dqn_battle_best.pth"
    
    trainer = UnifiedDQNTrainer(TrainConfig(
        num_snakes=num, 
        checkpoint_path=path,
        load_path=args.load
    ))
    trainer.train()
