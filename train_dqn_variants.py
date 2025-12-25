"""
Unified Trainer for DQN Variants (V6.1 FIX).
Supports: DQN, DDQN, PER, Dueling-PER.
Features: 30-step Cooldown Awareness, 25D Observation.
"""

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys

from env.battle_snake_env import BattleSnakeEnv, BattleSnakeConfig
from agent.dqn import DQNNet, DQNAgent
from agent.ddqn import DDQNNet, DDQNAgent
from agent.per_dqn import PERDQNNet, PERDQNAgent, SumTree
from agent.dueling_dqn import DuelingDQNNet, DuelingDQNAgent
from utils.self_play import SelfPlayManager

def log(msg):
    print(msg, flush=True)

# --- Buffer Implementations ---

class FastReplayBuffer:
    def __init__(self, capacity, grid_shape, vector_dim, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.grids = np.zeros((capacity, *grid_shape), dtype=np.float32)
        self.vectors = np.zeros((capacity, vector_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_grids = np.zeros((capacity, *grid_shape), dtype=np.float32)
        self.next_vectors = np.zeros((capacity, vector_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.ptr = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        self.grids[self.ptr] = state['grid']
        self.vectors[self.ptr] = state['vector']
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_grids[self.ptr] = next_state['grid']
        self.next_vectors[self.ptr] = next_state['vector']
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self):
        idxs = np.random.randint(0, self.size, size=self.batch_size)
        return (
            {"grid": torch.from_numpy(self.grids[idxs]).to(self.device), "vector": torch.from_numpy(self.vectors[idxs]).to(self.device)},
            torch.from_numpy(self.actions[idxs]).to(self.device),
            torch.from_numpy(self.rewards[idxs]).to(self.device),
            {"grid": torch.from_numpy(self.next_grids[idxs]).to(self.device), "vector": torch.from_numpy(self.next_vectors[idxs]).to(self.device)},
            torch.from_numpy(self.dones[idxs]).to(self.device),
            None, # weights
            None  # indices
        )

class PrioritizedReplayBuffer:
    def __init__(self, capacity, grid_shape, vector_dim, batch_size, device, alpha=0.6, beta=0.4):
        self.tree = SumTree(capacity)
        self.batch_size = batch_size
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-6
        self.max_priority = 1.0

    @property
    def size(self):
        """Fixed AttributeError by exposing size property"""
        return self.tree.size

    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.tree.add(self.max_priority ** self.alpha, data)

    def sample(self):
        idxs, weights, batch = [], [], []
        segment = self.tree.total_priority / self.batch_size
        
        for i in range(self.batch_size):
            a, b = segment * i, segment * (i + 1)
            v = random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            idxs.append(idx)
            weights.append(p / self.tree.total_priority)
            batch.append(data)
            
        # Importance Sampling Weights
        weights = np.array(weights)
        # Fixed AttributeError: 'list' object has no attribute 'size'
        weights = (len(batch) * weights) ** (-self.beta) 
        weights /= (weights.max() + 1e-8)
        
        # Unpack
        o_grids = np.array([x[0]['grid'] for x in batch])
        o_vecs = np.array([x[0]['vector'] for x in batch])
        acts = np.array([x[1] for x in batch])
        rews = np.array([x[2] for x in batch])
        n_grids = np.array([x[3]['grid'] for x in batch])
        n_vecs = np.array([x[3]['vector'] for x in batch])
        dones = np.array([x[4] for x in batch])
        
        return (
            {"grid": torch.from_numpy(o_grids).to(self.device), "vector": torch.from_numpy(o_vecs).to(self.device)},
            torch.from_numpy(acts).to(self.device),
            torch.from_numpy(rews).to(self.device),
            {"grid": torch.from_numpy(n_grids).to(self.device), "vector": torch.from_numpy(n_vecs).to(self.device)},
            torch.from_numpy(dones).to(self.device),
            torch.from_numpy(weights.astype(np.float32)).to(self.device),
            idxs
        )

    def update_priorities(self, idxs, td_errors):
        for idx, err in zip(idxs, td_errors):
            p = (abs(err) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, abs(err) + self.epsilon)

# --- Main Trainer ---

@dataclass
class TrainConfig:
    variant: str = "dqn" # dqn, ddqn, per, dueling
    total_frames: int = 1_000_000
    num_envs: int = 8
    batch_size: int = 256
    lr: float = 2e-4
    eps_decay: int = 500_000
    target_update: int = 2000
    num_snakes: int = 4
    pool_dir: str = "agent/pool/dqn_variants"
    load_path: Optional[str] = None
    save_path: str = "agent/checkpoints/dqn_best.pth"
    single_snake: bool = False

class DQNVariantTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Override for single snake mode
        if cfg.single_snake:
            cfg.num_snakes = 1
            
        log(f">>> [V6.1 FIX] Variant: {cfg.variant.upper()} | Device: {self.device}")
        
        env_cfg = BattleSnakeConfig(num_snakes=cfg.num_snakes, dash_cooldown_steps=30)
        if cfg.num_snakes == 1:
            env_cfg.closer_reward, env_cfg.food_reward, env_cfg.death_penalty = 0.35, 12.0, -10.0
            log(">>> Running in SINGLE SNAKE mode.")
        
        self.envs = [BattleSnakeEnv(env_cfg) for _ in range(cfg.num_envs)]
        
        # Select Architecture
        if cfg.variant == "dqn": self.net_cls = DQNNet
        elif cfg.variant == "ddqn": self.net_cls = DDQNNet
        elif cfg.variant == "per": self.net_cls = PERDQNNet
        elif cfg.variant == "dueling": self.net_cls = DuelingDQNNet
        else: raise ValueError(f"Unknown variant {cfg.variant}")
        
        # Explicit 25D Vector Dim
        self.policy_net = self.net_cls(vector_dim=25).to(self.device)
        self.target_net = self.net_cls(vector_dim=25).to(self.device)
        
        # Load weights
        if cfg.load_path and Path(cfg.load_path).exists():
            log(f">>> Loading weights from {cfg.load_path}...")
            state_dict = torch.load(cfg.load_path, map_location=self.device, weights_only=True)
            self.policy_net.load_state_dict(state_dict)
            
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        
        # Select Buffer
        if "per" in cfg.variant or "dueling" in cfg.variant:
            self.memory = PrioritizedReplayBuffer(100_000, (3, 7, 7), 25, cfg.batch_size, self.device)
        else:
            self.memory = FastReplayBuffer(150_000, (3, 7, 7), 25, cfg.batch_size, self.device)
            
        self.steps = 0
        self.sp_manager = SelfPlayManager(cfg.pool_dir)
        self.best_reward = -float('inf')

    def save_model(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), p)
        # log(f">>> Model saved to {path}")

    def train(self):
        log(f">>> Starting {self.cfg.variant.upper()} Training...")
        obs_batch = [env.reset() for env in self.envs]
        ep_rewards = [0.0] * self.cfg.num_envs
        recent_rewards = []
        
        last_log_time = time.time()
        
        while self.steps < self.cfg.total_frames:
            self.steps += self.cfg.num_envs
            eps = 0.05 + (0.95) * math.exp(-1. * self.steps / self.cfg.eps_decay)
            
            # Action selection
            all_actions = []
            for e_idx in range(self.cfg.num_envs):
                env_acts = []
                for s_idx in range(self.cfg.num_snakes):
                    if s_idx == 0: # Learning Agent
                        if random.random() < eps: env_acts.append(random.randint(0, 3))
                        else:
                            obs = obs_batch[e_idx][0]
                            # Debug: verify obs shape once
                            if self.steps == self.cfg.num_envs and e_idx == 0:
                                log(f">>> Obs Vector Shape: {obs['vector'].shape}")
                            
                            t_g = torch.tensor(obs['grid'], dtype=torch.float32, device=self.device).unsqueeze(0)
                            t_v = torch.tensor(obs['vector'], dtype=torch.float32, device=self.device).unsqueeze(0)
                            with torch.no_grad():
                                env_acts.append(self.policy_net(t_g, t_v).argmax().item())
                    else: env_acts.append(random.randint(0, 3)) 
                all_actions.append(env_acts)
            
            # Env Step
            next_obs_batch = []
            for e_idx in range(self.cfg.num_envs):
                n_obs, rews, dones, _ = self.envs[e_idx].step(all_actions[e_idx])
                
                # Push transition (Agent 0)
                self.memory.push(obs_batch[e_idx][0], all_actions[e_idx][0], rews[0], n_obs[0], dones[0])
                ep_rewards[e_idx] += rews[0]
                
                if dones[0]:
                    recent_rewards.append(ep_rewards[e_idx])
                    if len(recent_rewards) > 100: recent_rewards.pop(0)
                    
                    ep_rewards[e_idx] = 0.0
                    next_obs_batch.append(self.envs[e_idx].reset())
                else:
                    next_obs_batch.append(n_obs)
            
            obs_batch = next_obs_batch
            
            # Update
            if self.memory.size > 2000:
                self.update()
            
            if self.steps % self.cfg.target_update < self.cfg.num_envs:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                
            if self.steps % 10000 < self.cfg.num_envs:
                fps = 10000 / (time.time() - last_log_time)
                avg_r = np.mean(recent_rewards) if recent_rewards else 0
                log(f"Step: {self.steps} | EPS: {eps:.2f} | Rew: {avg_r:.2f} | FPS: {fps:.1f} | Var: {self.cfg.variant}")
                last_log_time = time.time()
                
                # Save best
                if avg_r > self.best_reward and len(recent_rewards) >= 20:
                    self.best_reward = avg_r
                    self.save_model(self.cfg.save_path)
        
        self.save_model(self.cfg.save_path) # Final save

    def update(self):
        states, actions, rewards, next_states, dones, weights, idxs = self.memory.sample()
        
        # Current Q
        q_curr = self.policy_net(states['grid'], states['vector']).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q
        with torch.no_grad():
            if self.cfg.variant == "dqn":
                q_next = self.target_net(next_states['grid'], next_states['vector']).max(1)[0]
            else: # DDQN Logic
                best_actions = self.policy_net(next_states['grid'], next_states['vector']).argmax(1)
                q_next = self.target_net(next_states['grid'], next_states['vector']).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target = rewards + 0.99 * q_next * (~dones)
            
        td_errors = q_curr - target
        
        if weights is not None: # PER logic
            loss = (weights * (td_errors ** 2)).mean()
            self.memory.update_priorities(idxs, td_errors.detach().cpu().numpy())
        else:
            loss = nn.SmoothL1Loss()(q_curr, target)
            
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

if __name__ == "__main__":
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument("--variant", type=str, default="dqn", choices=["dqn", "ddqn", "per", "dueling"])
    p.add_argument("--steps", type=int, default=1000000)
    p.add_argument("--single", action="store_true", help="Single snake mode")
    p.add_argument("--load", type=str, default=None, help="Path to pre-trained model")
    p.add_argument("--save", type=str, default="agent/checkpoints/dqn_best.pth", help="Path to save model")
    args = p.parse_args()
    
    cfg = TrainConfig(
        variant=args.variant, 
        total_frames=args.steps,
        single_snake=args.single,
        load_path=args.load,
        save_path=args.save
    )
    DQNVariantTrainer(cfg).train()
