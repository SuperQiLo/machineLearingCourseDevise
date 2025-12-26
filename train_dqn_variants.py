"""
Unified Trainer for DQN Variants (V6.7 - Turbo Battle Performance).
Supports: DQN, DDQN, PER, Dueling-PER.
Features: Omni-Batch Inference (Massive FPS boost), Soft Updates, Algorithm-Specific Hyperparameters.
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
    def __init__(self, capacity, grid_shape, vector_dim, batch_size, device, alpha=0.5, beta=0.4):
        self.tree = SumTree(capacity)
        self.batch_size = batch_size
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-6
        self.max_priority = 1.0

    @property
    def size(self):
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
            
        weights = np.array(weights)
        weights = (len(batch) * weights) ** (-self.beta) 
        weights /= (weights.max() + 1e-8)
        
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
    lr: float = 1e-4
    eps_decay: int = 200_000
    tau: float = 0.005 # V7.9: Restored to 0.005 for faster alignment/convergence
    num_snakes: int = 4
    pool_dir: str = "agent/pool/dqn"
    load_path: Optional[str] = None
    save_path: str = "agent/checkpoints/dqn_best.pth"
    single_snake: bool = False
    self_play_prob: float = 0.5

class DQNVariantTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if cfg.single_snake:
            cfg.num_snakes = 1

        # Algorithm-Specific Hyperparameters (V6.9 Optim-Matrix)
        if cfg.variant == "dqn":
            self.lr = 1.0e-4 
            self.tau = 0.005 # V8.0: Aggressive for vanilla
            self.closer_reward = 0.15
            self.buffer_size = 300_000
            self.grad_clip = 1.0
        elif cfg.variant == "ddqn":
            self.lr = 1.0e-4 
            self.tau = 0.005 # V8.0: Aggressive for vanilla
            self.closer_reward = 0.08
            self.buffer_size = 400_000
            self.grad_clip = 1.0
        elif cfg.variant == "per":
            self.lr = 5.0e-5 
            self.tau = 0.002 # V8.0: Reverted to smooth for PER stability
            self.closer_reward = 0.05
            self.buffer_size = 300_000
            self.grad_clip = 0.5 
        elif cfg.variant == "dueling":
            # V8.0: Re-tuned Dueling for Ph1 (1.5e-4 -> 1.2e-4)
            self.lr = 1.2e-4 if cfg.single_snake else 2.5e-4 
            self.tau = 0.002 # V8.0: Reverted to smooth for Dueling stability
            self.closer_reward = 0.05
            self.buffer_size = 400_000
            self.grad_clip = 0.8
        else:
            self.lr = 2.0e-4
            self.tau = 0.005
            self.closer_reward = 0.01
            self.buffer_size = 250_000
            self.grad_clip = 1.0

        log(f">>> [V8.0 Asymmetric-Tuning] Variant: {cfg.variant.upper()} | LR: {self.lr} | Tau: {self.tau} | GradClip: {self.grad_clip}")
        
        env_cfg = BattleSnakeConfig(num_snakes=cfg.num_snakes, dash_cooldown_steps=15)
        if cfg.num_snakes == 1:
            # Phase 1: High focus on navigation
            env_cfg.closer_reward = self.closer_reward
            env_cfg.food_reward = 25.0
            env_cfg.death_penalty = -15.0 # V7.5: More forgiving Ph1 (-20 -> -15)
            log(f">>> PHASE 1 (Single) | Closer: {env_cfg.closer_reward} | Death: {env_cfg.death_penalty}")
        else:
            # Phase 2: Survival & Combat Balance (V6.9)
            env_cfg.closer_reward = 0.05   # Keep some guiding in battle
            env_cfg.step_reward = 0.01     # Survival incentive
            env_cfg.death_penalty = -15.0  # Reduced penalty to avoid cowardice
            env_cfg.kill_reward = 30.0
            env_cfg.food_reward = 20.0
            log(f">>> PHASE 2 (Battle) | Closer: {env_cfg.closer_reward} | Death: {env_cfg.death_penalty} | SurvivalBonus: 0.01")
        
        self.envs = [BattleSnakeEnv(env_cfg) for _ in range(cfg.num_envs)]
        
        # Select Architecture
        if cfg.variant == "dqn": self.net_cls = DQNNet
        elif cfg.variant == "ddqn": self.net_cls = DDQNNet
        elif cfg.variant == "per": self.net_cls = PERDQNNet
        elif cfg.variant == "dueling": self.net_cls = DuelingDQNNet
        else: raise ValueError(f"Unknown variant {cfg.variant}")
        
        self.policy_net = self.net_cls(vector_dim=25).to(self.device)
        self.target_net = self.net_cls(vector_dim=25).to(self.device)
        
        if cfg.load_path and Path(cfg.load_path).exists():
            log(f">>> Loading weights from {cfg.load_path}...")
            state_dict = torch.load(cfg.load_path, map_location=self.device, weights_only=True)
            self.policy_net.load_state_dict(state_dict)
            
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        if "per" in cfg.variant or "dueling" in cfg.variant:
            self.memory = PrioritizedReplayBuffer(self.buffer_size, (3, 7, 7), 25, cfg.batch_size, self.device)
        else:
            self.memory = FastReplayBuffer(self.buffer_size, (3, 7, 7), 25, cfg.batch_size, self.device)
            
        self.steps = 0
        self.sp_manager = SelfPlayManager(cfg.pool_dir)
        
        # Self-Play Manager (V6.7 Model Cache)
        # Store model paths for opponents. None means random.
        self.opp_model_paths = [[None]*cfg.num_snakes for _ in range(cfg.num_envs)]
        self.loaded_opp_models: Dict[str, nn.Module] = {}
        
        self.best_reward = -float('inf')

    def save_model(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), p)

    def _get_opp_model(self, path: str) -> nn.Module:
        """Get model from cache or load from disk"""
        if path not in self.loaded_opp_models:
            model = self.net_cls(vector_dim=25).to(self.device)
            try:
                state_dict = torch.load(path, map_location=self.device, weights_only=True)
                model.load_state_dict(state_dict)
                model.eval()
                self.loaded_opp_models[path] = model
                # Limit cache size to prevent memory leak
                if len(self.loaded_opp_models) > 5:
                    key_to_del = next(iter(self.loaded_opp_models))
                    del self.loaded_opp_models[key_to_del]
            except Exception:
                return None
        return self.loaded_opp_models.get(path)

    def train(self):
        log(f">>> Starting {self.cfg.variant.upper()} Training (V6.7 Omni-Batch)...")
        obs_batch = [env.reset() for env in self.envs]
        ep_rewards = [0.0] * self.cfg.num_envs
        recent_rewards = []
        last_log_time = time.time()
        
        while self.steps < self.cfg.total_frames:
            # 1. Group all snakes by model for Omni-Batch Inference
            # { model_ptr: [(env_idx, snake_idx, obs)] }
            groups: Dict[Optional[nn.Module], List[Tuple[int, int, Dict]]] = {None: []}
            
            self.steps += self.cfg.num_envs
            # V7.5: Protect PER Ph1 exploration (Min 0.1 if single_snake)
            eps_min = 0.1 if (self.cfg.variant == "per" and self.cfg.single_snake) else 0.05
            eps = max(eps_min, 1.0 - self.steps / self.cfg.eps_decay)
            
            all_actions = [ [None]*self.cfg.num_snakes for _ in range(self.cfg.num_envs) ]
            
            for e_idx in range(self.cfg.num_envs):
                # Learner Agent (0)
                if random.random() < eps:
                    all_actions[e_idx][0] = random.randint(0, 3)
                else:
                    groups[self.policy_net] = groups.get(self.policy_net, [])
                    groups[self.policy_net].append((e_idx, 0, obs_batch[e_idx][0]))
                
                # Opponent Agents (1+)
                for s_idx in range(1, self.cfg.num_snakes):
                    m_path = self.opp_model_paths[e_idx][s_idx]
                    if m_path:
                        model = self._get_opp_model(m_path)
                        if model:
                            groups[model] = groups.get(model, [])
                            groups[model].append((e_idx, s_idx, obs_batch[e_idx][s_idx]))
                        else:
                            all_actions[e_idx][s_idx] = random.randint(0, 3)
                    else:
                        all_actions[e_idx][s_idx] = random.randint(0, 3)

            # 2. Execute Omni-Batch Inference
            for model, samples in groups.items():
                if not samples: continue
                if model is None: continue # Handled by random
                
                with torch.no_grad():
                    grids = np.array([s[2]['grid'] for s in samples])
                    vecs = np.array([s[2]['vector'] for s in samples])
                    t_g = torch.as_tensor(grids, dtype=torch.float32, device=self.device)
                    t_v = torch.as_tensor(vecs, dtype=torch.float32, device=self.device)
                    q_vals = model(t_g, t_v)
                    acts = q_vals.argmax(dim=1).cpu().numpy()
                    for i, (env_idx, snake_idx, _) in enumerate(samples):
                        all_actions[env_idx][snake_idx] = int(acts[i])
            
            # 3. Env Step
            next_obs_batch = []
            for e_idx in range(self.cfg.num_envs):
                n_obs, rews, dones, _ = self.envs[e_idx].step(all_actions[e_idx])
                
                self.memory.push(obs_batch[e_idx][0], all_actions[e_idx][0], rews[0], n_obs[0], dones[0])
                ep_rewards[e_idx] += rews[0]
                
                if dones[0]:
                    recent_rewards.append(ep_rewards[e_idx])
                    if len(recent_rewards) > 100: recent_rewards.pop(0)
                    ep_rewards[e_idx] = 0.0
                    next_obs_batch.append(self.envs[e_idx].reset())
                    
                    # Self-Play Shuffle
                    if self.cfg.num_snakes > 1 and random.random() < self.cfg.self_play_prob:
                        opp_idx = random.randint(1, self.cfg.num_snakes-1)
                        m_p = self.sp_manager.sample_model()
                        if m_p:
                            self.opp_model_paths[e_idx][opp_idx] = str(m_p)
                else:
                    next_obs_batch.append(n_obs)
            
            obs_batch = next_obs_batch
            
            if self.memory.size > 2000:
                self.update()
            
            # 4. Multi-Stage Learning Rate Decay (V7.0)
            # Final 20% of frames: Decay LR to 10% for precision
            if self.steps > self.cfg.total_frames * 0.8:
                current_lr = self.lr * 0.1
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr

            # 5. Soft Update
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
                
            # 5. Heartbeat Logging
            log_interval = 2000 if self.steps < 20000 else 10000
            if self.steps % log_interval < self.cfg.num_envs:
                fps = log_interval / (time.time() - last_log_time)
                avg_r = np.mean(recent_rewards) if recent_rewards else 0
                log(f"Step: {self.steps} | EPS: {eps:.2f} | Rew: {avg_r:.2f} | FPS: {fps:.1f} | Var: {self.cfg.variant}")
                last_log_time = time.time()
                
                if avg_r > self.best_reward and len(recent_rewards) >= 20:
                    self.best_reward = avg_r
                    self.save_model(self.cfg.save_path)
            
            if self.steps % 150000 < self.cfg.num_envs:
                self.sp_manager.add_model(self.policy_net.state_dict(), f"{self.cfg.variant}_step_{self.steps}")
        
        self.save_model(self.cfg.save_path)

    def update(self):
        states, actions, rewards, next_states, dones, weights, idxs = self.memory.sample()
        q_curr = self.policy_net(states['grid'], states['vector']).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            if self.cfg.variant == "dqn":
                q_next = self.target_net(next_states['grid'], next_states['vector']).max(1)[0]
            else:
                best_actions = self.policy_net(next_states['grid'], next_states['vector']).argmax(1)
                q_next = self.target_net(next_states['grid'], next_states['vector']).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target = rewards + 0.99 * q_next * (~dones)
            
        td_errors = q_curr - target
        if weights is not None:
            loss = (weights * (td_errors ** 2)).mean()
            self.memory.update_priorities(idxs, td_errors.detach().cpu().numpy())
        else:
            loss = nn.SmoothL1Loss()(q_curr, target)
            
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

if __name__ == "__main__":
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument("--variant", type=str, default="dqn", choices=["dqn", "ddqn", "per", "dueling"])
    p.add_argument("--steps", type=int, default=1000000)
    p.add_argument("--single", action="store_true")
    p.add_argument("--load", type=str, default=None)
    p.add_argument("--save", type=str, default="agent/checkpoints/dqn_best.pth")
    p.add_argument("--sp-prob", type=float, default=0.5)
    args = p.parse_args()
    
    p_dir = f"agent/pool/{args.variant}"
    
    cfg = TrainConfig(
        variant=args.variant, 
        total_frames=args.steps,
        single_snake=args.single,
        load_path=args.load,
        save_path=args.save,
        self_play_prob=args.sp_prob,
        pool_dir=p_dir
    )
    DQNVariantTrainer(cfg).train()
