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
import os

from env.battle_snake_env import BattleSnakeEnv, BattleSnakeConfig
from env.gymnasium_wrapper import make_gymnasium_env
import gymnasium as gym

from agent.dqn import DQNNet, DQNAgent
from agent.ddqn import DDQNNet, DDQNAgent
from agent.per_dqn import PERDQNNet, PERDQNAgent, SumTree
from agent.dueling_dqn import DuelingDQNNet, DuelingDQNAgent
from utils.self_play import SelfPlayManager

# V17.4: AMP for faster GPU training (Updated API for PyTorch 2.x)
from torch.amp import autocast, GradScaler

def log(msg):
    print(msg, flush=True)

# --- Buffer Implementations ---

class FastReplayBuffer:
    def __init__(self, capacity, grid_shape, vector_dim, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        # V17.0: Use uint8 for grids to save 75% memory (45GB -> 11GB for 3M capacity)
        self.grids = np.zeros((capacity, *grid_shape), dtype=np.uint8)
        self.vectors = np.zeros((capacity, vector_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_grids = np.zeros((capacity, *grid_shape), dtype=np.uint8)
        self.next_vectors = np.zeros((capacity, vector_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.ptr = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        # V17.2: Direct assignment - numpy handles the type conversion automatically
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
        
        # Pinned memory transfer for faster CPU->GPU throughput
        def to_device(numpy_array, dtype=None):
            t = torch.as_tensor(numpy_array, dtype=dtype)
            if self.device.type == 'cuda':
                return t.pin_memory().to(self.device, non_blocking=True)
            return t.to(self.device)

        return (
            {
                "grid": to_device(self.grids[idxs], torch.float32), 
                "vector": to_device(self.vectors[idxs])
            },
            to_device(self.actions[idxs]),
            to_device(self.rewards[idxs]),
            {
                "grid": to_device(self.next_grids[idxs], torch.float32), 
                "vector": to_device(self.next_vectors[idxs])
            },
            to_device(self.dones[idxs]),
            None, # weights
            None  # indices
        )



class PrioritizedReplayBuffer:
    def __init__(self, capacity, grid_shape, vector_dim, batch_size, device, alpha=0.5, beta=0.4):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-6
        self.max_priority = 1.0

        # V17.0: Use uint8 for grids (20x20x5)
        self.grids = np.zeros((capacity, *grid_shape), dtype=np.uint8)
        self.vectors = np.zeros((capacity, vector_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_grids = np.zeros((capacity, *grid_shape), dtype=np.uint8)
        self.next_vectors = np.zeros((capacity, vector_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    @property
    def size(self):
        return self.tree.size

    def push(self, state, action, reward, next_state, done):
        data_idx = self.tree.ptr
        # V17.2: Direct assignment - numpy handles the type conversion
        self.grids[data_idx] = state['grid']
        self.vectors[data_idx] = state['vector']
        self.actions[data_idx] = action
        self.rewards[data_idx] = reward
        self.next_grids[data_idx] = next_state['grid']
        self.next_vectors[data_idx] = next_state['vector']
        self.dones[data_idx] = done
        self.tree.add(self.max_priority ** self.alpha, int(data_idx))

    def sample(self, beta: float):
        segment = self.tree.total_priority / self.batch_size
        v = np.random.uniform(segment * np.arange(self.batch_size), segment * np.arange(1, self.batch_size + 1))
        
        idxs = []
        priorities = []
        batch_data_idxs = []
        
        for val in v:
            idx, p, data_idx = self.tree.get_leaf(val)
            idxs.append(idx)
            priorities.append(p)
            batch_data_idxs.append(int(data_idx))
            
        idxs = np.array(idxs)
        batch = np.array(batch_data_idxs)
        
        probs = np.array(priorities) / (self.tree.total_priority + 1e-8)
        weights = (self.size * probs) ** (-beta)
        weights /= (weights.max() + 1e-8)
        
        def to_device(numpy_array, dtype=None):
            t = torch.as_tensor(numpy_array, dtype=dtype)
            if self.device.type == 'cuda':
                return t.pin_memory().to(self.device, non_blocking=True)
            return t.to(self.device)

        return (
            {
                "grid": to_device(self.grids[batch], torch.float32), 
                "vector": to_device(self.vectors[batch])
            },
            to_device(self.actions[batch]),
            to_device(self.rewards[batch]),
            {
                "grid": to_device(self.next_grids[batch], torch.float32), 
                "vector": to_device(self.next_vectors[batch])
            },
            to_device(self.dones[batch]),
            to_device(weights.astype(np.float32)),
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
    total_frames: int = 5_000_000 
    num_envs: int = 64 # V18.0: Increased to 64 with AsyncVectorEnv
    batch_size: int = 512 
    lr: float = 1e-4
    eps_decay: int = 0  
    tau: float = 0.005 
    num_snakes: int = 4
    pool_dir: str = "agent/pool/dqn"
    save_path: str = "agent/checkpoints/dqn_best.pth"
    load_path: Optional[str] = None
    self_play_prob: float = 0.5
    buffer_size: int = 200_000 # V17.4: Reduced to 200k for much faster sampling
    single_snake: bool = False
    eps_start: float = 1.0  

class DQNVariantTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg  # FIXED: Restored self.cfg assignment
        # Variant-aware exploration schedule.
        # Use percentage-based decay to adapt to any total_frames count (1M, 5M, etc.)
        total = max(1, int(cfg.total_frames))
        
        # Decide Decay Duration: 85% of total steps for single, 90% for battle (need more exploration)
        if cfg.single_snake:
            decay_ratio = 0.80 # V46.3: Compressed for 300W run
        else:
            decay_ratio = 0.90
            
        self.decay_steps = int(total * decay_ratio)
        self.cfg.eps_decay = self.decay_steps # Update config for logging
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if cfg.single_snake:
            cfg.num_snakes = 1

        # Decide Hyperparameters based on Variant and Phase (V9.0 Specialized Tuning)
        v = cfg.variant
        if cfg.single_snake:
            # Phase 1: High Exploration & Robust Learning
            self.eps_start, self.eps_min = 1.0, 0.1
            if v == "dqn":
                self.lr, self.tau, self.batch_size = 1.5e-4, 0.002, 512 # V17.3: Fixed batch
            else:
                self.lr, self.tau, self.batch_size = 2.0e-4, 0.005, 512 # V17.3: Fixed batch
        else:
            # Phase 2: High Stability & Combat Precision
            self.eps_start, self.eps_min = 0.3, 0.05
            if v == "dqn":
                self.lr, self.tau, self.batch_size = 1.0e-4, 0.002, 512 # V17.3: Fixed batch
            else:
                self.lr, self.tau, self.batch_size = 8.0e-5, 0.005, 512 # V17.3: Fixed batch
            
        self.grad_clip = 0.5 
        
        # Override algorithm-specific if needed
        if cfg.variant == "per" or cfg.variant == "dueling":
            # V11.0: Lower Alpha (0.4) for Phase 2 to handle high-noise battle environments
            self.per_alpha = 0.6 if cfg.single_snake else 0.4
            self.per_beta_start = 0.4
        
        # V12.0: Buffer size managed via TrainConfig for A6000 visibility
        self.buffer_size = cfg.buffer_size

        # V45.0: DDQN Fix - Remove LR penalty that caused 26% performance loss
        # Analysis: V44.6 halved LR (8e-5->4e-5), causing DDQN Ph1 reward=101 vs DQN=136
        # Fix: Restore full LR 8e-5 to match DQN convergence speed
        # Removed: if cfg.variant == "ddqn": self.lr = self.lr * 0.5

        log(f">>> [V8.0 Asymmetric-Tuning] Variant: {cfg.variant.upper()} | LR: {self.lr} | Tau: {self.tau} | GradClip: {self.grad_clip} | Buf: {self.buffer_size}")
        
        env_cfg = BattleSnakeConfig(width=20, height=20, num_snakes=cfg.num_snakes)
        if cfg.num_snakes == 1:
            # Phase 1: High focus on navigation (V46.0: Aggressive Reward Tuning)
            env_cfg.closer_reward = 0.25      # Stronger pull to food
            env_cfg.farther_penalty = -0.20   # Stronger penalty for going away
            env_cfg.food_reward = 120.0
            env_cfg.death_penalty = -30.0     # V16.0: Encouraging exploration on 20x20
            env_cfg.step_penalty = -0.02      
            env_cfg.self_collision_penalty = -50.0  # V16.0: Early Survival Awareness
            log(f">>> [V16.0 Turbo] PHASE 1 (Single) | Batch: {self.batch_size} | Food: {env_cfg.food_reward} | Self-Penalty: {env_cfg.self_collision_penalty}")
        else:
            # Phase 2: V9.0 Combat (High Aggression)
            env_cfg.closer_reward = 0.15 
            env_cfg.farther_penalty = -0.10  
            env_cfg.step_penalty = -0.05      
            env_cfg.death_penalty = -100.0    # V15.0: Penalize environment death heavily in Battle
            env_cfg.kill_reward = 150.0      # Aligned with PPO V7.0
            env_cfg.food_reward = 80.0       # V10.0: Re-aligned with survival (Old 40 caused starvation)
            env_cfg.self_collision_penalty = -150.0 # V15.0: Crucial fix for long-snake self-collision
            log(f">>> [V9.0 Battle] PHASE 2 | Kill: {env_cfg.kill_reward} | Food: {env_cfg.food_reward} | Stable LR/Tau")
        
        log(f">>> [V18.0 Turbo] Initializing {cfg.num_envs} Parallel Battle Environments...")
        def env_fn():
             return make_gymnasium_env(num_snakes=cfg.num_snakes, grid_size=20)
        self.envs = gym.vector.AsyncVectorEnv([env_fn for _ in range(cfg.num_envs)])
        log(">>> [V18.0 Turbo] Environments Ready.")
        
        # Select Architecture
        if cfg.variant == "dqn": self.net_cls = DQNNet
        elif cfg.variant == "ddqn": self.net_cls = DDQNNet
        elif cfg.variant == "per": 
            from agent.per_dqn import PERDQNNet
            self.net_cls = PERDQNNet 
        elif cfg.variant == "dueling": self.net_cls = DuelingDQNNet
        else: raise ValueError(f"Unknown variant {cfg.variant}")
        
        self.policy_net = self.net_cls(vector_dim=28).to(self.device)
        self.target_net = self.net_cls(vector_dim=28).to(self.device)
        
        # V18.0: torch.compile for massive throughput boost
        if hasattr(torch, 'compile') and os.name != 'nt':
            try:
                self.policy_net = torch.compile(self.policy_net, mode='reduce-overhead')
                self.target_net = torch.compile(self.target_net, mode='reduce-overhead')
                log(">>> [DQN-Turbo] torch.compile() enabled.")
            except Exception as e:
                log(f">>> [DQN-Turbo] torch.compile() failed: {e}")
        
        if cfg.load_path and Path(cfg.load_path).exists():
            log(f">>> Loading weights from {cfg.load_path}...")
            state_dict = torch.load(cfg.load_path, map_location=self.device, weights_only=True)
            self.policy_net.load_state_dict(state_dict)
            
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        if "per" in cfg.variant or "dueling" in cfg.variant:
            # V8.1: Support phase-aware alpha
            alpha = getattr(self, 'per_alpha', 0.6)
            self.memory = PrioritizedReplayBuffer(self.buffer_size, (5, 20, 20), 28, self.batch_size, self.device, alpha=alpha)
        else:
            self.memory = FastReplayBuffer(self.buffer_size, (5, 20, 20), 28, self.batch_size, self.device)
            
        self.steps = 0
        log(f">>> [V13.3 Heartbeat] Setup Finished. Buffer: {self.buffer_size} | Device: {self.device}")
        self.sp_manager = SelfPlayManager(cfg.pool_dir)
        
        # Self-Play Manager (V6.7 Model Cache)
        # Store model paths for opponents. None means random.
        self.opp_model_paths = [[None]*cfg.num_snakes for _ in range(cfg.num_envs)]
        self.loaded_opp_models: Dict[str, nn.Module] = {}
        
        self.best_reward = -float('inf')
        
        # V39.0: Unified Gamma (Dueling was 0.995 -> unstable)
        # All variants now use 0.99 for stable value estimation
        self.gamma = 0.99
        # V17.2: Reduced Update Frequency to 1
        self.updates_per_step = 1
        
        # V17.4: AMP Scaler for mixed precision training
        self.scaler = GradScaler('cuda')
        self.use_amp = True

    def save_model(self, path):
        # V44.4: Atomic Save to prevent file corruption/locking during self-play loading
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = p.with_suffix(p.suffix + ".tmp")
        torch.save(self.policy_net.state_dict(), tmp_path)
        if os.path.exists(p):
            try:
                os.replace(tmp_path, p)
            except OSError:
                # Windows atomic replace retry fallback
                os.remove(p)
                os.rename(tmp_path, p)
        else:
            os.rename(tmp_path, p)

    def _get_opp_model(self, path: str) -> nn.Module:
        """Get model from cache or load from disk"""
        if path not in self.loaded_opp_models:
            model = self.net_cls(vector_dim=28).to(self.device)
            try:
                state_dict = torch.load(path, map_location=self.device, weights_only=True)
                model.load_state_dict(state_dict)
                model.eval()
                self.loaded_opp_models[path] = model
                # Limit cache size to prevent memory leak
                # V44.3: Increase Cache to 50 (was 5). 8 envs * 3 opps = 24 models needed.
                # 5 was causing severe IO thrashing -> 3 FPS.
                if len(self.loaded_opp_models) > 50:
                    key_to_del = next(iter(self.loaded_opp_models))
                    del self.loaded_opp_models[key_to_del]
            except Exception:
                return None
        return self.loaded_opp_models.get(path)

    def train(self):
        log(f">>> Starting {self.cfg.variant.upper()} Training (V18.1 Turbo Battle)...")
        obs_dict, info = self.envs.reset()
        ep_rewards = [0.0] * self.cfg.num_envs
        recent_rewards = []
        last_log_time = time.time()
        
        total_frames = max(1, int(self.cfg.total_frames))
        
        while self.steps < total_frames:
            # 1. Group all snakes by model for Omni-Batch Inference
            groups: Dict[Optional[nn.Module], List[Tuple[int, int, Dict]]] = {None: []}
            
            self.steps += self.cfg.num_envs
            completion = min(1.0, self.steps / self.decay_steps)
            eps = max(0.10, self.eps_start - completion * (self.eps_start - self.eps_min))
            
            all_actions = np.zeros((self.cfg.num_envs, self.cfg.num_snakes), dtype=np.int32)
            all_full_obs = info["full_obs"]
            
            for e_idx in range(self.cfg.num_envs):
                # Learner Agent (0)
                if random.random() < eps:
                    all_actions[e_idx, 0] = random.randint(0, 3)
                else:
                    groups[self.policy_net] = groups.get(self.policy_net, [])
                    # obs_dict['grid'][e_idx] is the obs for snake 0
                    groups[self.policy_net].append((e_idx, 0, {"grid": obs_dict["grid"][e_idx], "vector": obs_dict["vector"][e_idx]}))
                
                # Opponent Agents (1+)
                for s_idx in range(1, self.cfg.num_snakes):
                    m_path = self.opp_model_paths[e_idx][s_idx]
                    if m_path:
                        model = self._get_opp_model(m_path)
                        if model:
                            groups[model] = groups.get(model, [])
                            groups[model].append((e_idx, s_idx, all_full_obs[e_idx][s_idx]))
                        else:
                            all_actions[e_idx, s_idx] = random.randint(0, 3)
                    else:
                        all_actions[e_idx, s_idx] = random.randint(0, 3)

            # 2. Execute Omni-Batch Inference
            for model, samples in groups.items():
                if not samples or model is None: continue
                with torch.inference_mode():
                    grids = np.asarray([s[2]['grid'] for s in samples])
                    vecs = np.asarray([s[2]['vector'] for s in samples])
                    t_g = torch.as_tensor(grids, dtype=torch.float32, device=self.device)
                    t_v = torch.as_tensor(vecs, dtype=torch.float32, device=self.device)
                    q_vals = model(t_g, t_v)
                    acts = q_vals.argmax(dim=1).cpu().numpy()
                    for i, (env_idx, snake_idx, _) in enumerate(samples):
                        all_actions[env_idx, snake_idx] = int(acts[i])
            
            # 3. Env Step (Batched)
            next_obs_dict, rews, terminated, truncated, next_info = self.envs.step(all_actions)
            
            for e_idx in range(self.cfg.num_envs):
                # Only push snake 0 (learner) to memory
                # Note: We need the single-agent observation for memory
                state = {"grid": obs_dict["grid"][e_idx], "vector": obs_dict["vector"][e_idx]}
                n_state = {"grid": next_obs_dict["grid"][e_idx], "vector": next_obs_dict["vector"][e_idx]}
                
                self.memory.push(state, all_actions[e_idx, 0], rews[e_idx], n_state, terminated[e_idx])
                ep_rewards[e_idx] += rews[e_idx]
                
                if terminated[e_idx] or truncated[e_idx]:
                    recent_rewards.append(ep_rewards[e_idx])
                    if len(recent_rewards) > 100: recent_rewards.pop(0)
                    ep_rewards[e_idx] = 0.0
                    
                    # Self-Play Shuffle
                    if self.cfg.num_snakes > 1 and random.random() < self.cfg.self_play_prob:
                        opp_idx = random.randint(1, self.cfg.num_snakes-1)
                        m_p = self.sp_manager.sample_model()
                        if m_p:
                            self.opp_model_paths[e_idx][opp_idx] = str(m_p)
            
            obs_dict, info = next_obs_dict, next_info


            # V16.0: 10% LR annealing floor
            progress = self.steps / total_frames
            frac = max(0.0, 1.0 - progress)
            current_lr = self.lr * (0.10 + 0.90 * frac) 
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr

            # V9.0 Hyper-Precision Schedule
            warmup1, warmup2 = 50000, 150000
            late_half = total_frames * 0.5
            late_40 = total_frames * 0.4
            
            # V13.0 CRITICAL FIX: Actually train the network!
            # V24.0: Double Update Frequency for Phase 1 (Ratio 0.25)
            if self.memory.size >= self.cfg.batch_size:
                # Warm-up: avoid overfitting noisy early experience for PER/Dueling.
                if self.cfg.variant in ("per", "dueling"):
                    if self.steps < warmup1:
                        updates = 1
                    elif self.steps < warmup2:
                        updates = min(2, self.updates_per_step)
                    else:
                        updates = self.updates_per_step
                elif self.cfg.variant == "ddqn":
                    # DDQN often peaks mid-training then regresses; reduce update pressure late.
                    late_cut = late_half if self.cfg.single_snake else late_40
                    updates = self.updates_per_step if self.steps < late_cut else 1
                elif self.cfg.variant == "per":
                    # PER can become unstable after it starts exploiting; reduce update pressure late.
                    updates = self.updates_per_step if self.steps < late_half else 1
                else:
                    updates = self.updates_per_step

                for _ in range(updates):
                    self.update()

            # 5. Soft Update
            tau_eff = self.tau
            if self.cfg.variant == "ddqn":
                late_cut = late_half if self.cfg.single_snake else late_40
                if self.steps >= late_cut:
                    tau_eff = self.tau * 0.5
            if self.cfg.variant == "dqn" and (not self.cfg.single_snake) and self.steps >= late_half:
                tau_eff = self.tau * 0.5
            if self.cfg.variant == "per" and self.steps >= late_half:
                tau_eff = self.tau * 0.5
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(tau_eff * policy_param.data + (1.0 - tau_eff) * target_param.data)
                
            # 5. Heartbeat Logging (V13.3 Enhanced for A6000 mode)
            log_interval = 400 if self.steps < 10000 else 2000
            if self.steps % log_interval < self.cfg.num_envs:
                fps = log_interval / (time.time() - last_log_time)
                avg_r = np.mean(recent_rewards) if recent_rewards else 0
                log(f"Step: {self.steps} | EPS: {eps:.2f} | Rew: {avg_r:.2f} | FPS: {fps:.1f} | Var: {self.cfg.variant}")
                last_log_time = time.time()
                
                # Simplified Saving (User requested ONLY final model or periodic snapshot)
                # Keep periodic pool snapshots for self-play diversity
                pool_interval = max(150_000, int(total_frames * 0.03))
                if self.steps % pool_interval < self.cfg.num_envs:
                    self.sp_manager.add_model(self.policy_net.state_dict(), f"{self.cfg.variant}_step_{self.steps}")
            
        # Final Save (Only at the end of total_steps)
        self.save_model(self.cfg.save_path)
        final_path = str(Path(self.cfg.save_path).with_suffix(".final.pth"))
        self.save_model(final_path)

    def update(self):
        # V6.3: Calculate dynamic Beta for PER
        frac = self.steps / self.cfg.total_frames
        beta_start = getattr(self, "per_beta_start", 0.4)
        current_beta = min(1.0, beta_start + frac * (1.0 - beta_start))
        
        if "per" in self.cfg.variant or "dueling" in self.cfg.variant:
            states, actions, rewards, next_states, dones, weights, idxs = self.memory.sample(current_beta)
        else:
            states, actions, rewards, next_states, dones, weights, idxs = self.memory.sample()
        
        # V17.4: AMP Mixed Precision Training
        with autocast('cuda', enabled=self.use_amp):
            q_curr = self.policy_net(states['grid'], states['vector']).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            with torch.no_grad():
                if self.cfg.variant == "dqn":
                    q_next = self.target_net(next_states['grid'], next_states['vector']).max(1)[0]
                else:
                    best_actions = self.policy_net(next_states['grid'], next_states['vector']).argmax(1)
                    q_next = self.target_net(next_states['grid'], next_states['vector']).gather(1, best_actions.unsqueeze(1)).squeeze(1)
                target = rewards + self.gamma * q_next * (1.0 - dones.float())
                
            td_errors = q_curr - target
            if weights is not None:
                per_sample = nn.SmoothL1Loss(reduction="none")(q_curr, target)
                loss = (weights * per_sample).mean()
            else:
                loss = nn.SmoothL1Loss()(q_curr, target)
        
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        if weights is not None:
            self.memory.update_priorities(idxs, td_errors.detach().abs().cpu().numpy())

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
