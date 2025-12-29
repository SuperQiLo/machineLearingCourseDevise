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
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-6
        self.max_priority = 1.0

        # Store transitions in contiguous numpy arrays for speed.
        # SumTree stores only indices into these arrays.
        self.grids = np.zeros((capacity, *grid_shape), dtype=np.float32)
        self.vectors = np.zeros((capacity, vector_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_grids = np.zeros((capacity, *grid_shape), dtype=np.float32)
        self.next_vectors = np.zeros((capacity, vector_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    @property
    def size(self):
        return self.tree.size

    def push(self, state, action, reward, next_state, done):
        data_idx = self.tree.ptr
        self.grids[data_idx] = state['grid']
        self.vectors[data_idx] = state['vector']
        self.actions[data_idx] = action
        self.rewards[data_idx] = reward
        self.next_grids[data_idx] = next_state['grid']
        self.next_vectors[data_idx] = next_state['vector']
        self.dones[data_idx] = done
        self.tree.add(self.max_priority ** self.alpha, int(data_idx))

    def sample(self, beta: float):
        idxs, weights, batch = [], [], []
        segment = self.tree.total_priority / self.batch_size
        
        for i in range(self.batch_size):
            a, b = segment * i, segment * (i + 1)
            v = random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            idxs.append(idx)
            weights.append(p / self.tree.total_priority)
            batch.append(int(data))
            
        weights = np.array(weights)
        weights = (len(batch) * weights) ** (-beta) 
        weights /= (weights.max() + 1e-8)
        
        b_idx = np.asarray(batch, dtype=np.int64)
        o_grids = self.grids[b_idx]
        o_vecs = self.vectors[b_idx]
        acts = self.actions[b_idx]
        rews = self.rewards[b_idx]
        n_grids = self.next_grids[b_idx]
        n_vecs = self.next_vectors[b_idx]
        dones = self.dones[b_idx]
        
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
    eps_decay: int = 0  # 0 = Use Percentage-based Decay (80% of total)
    tau: float = 0.005 
    num_snakes: int = 4
    pool_dir: str = "agent/pool/dqn"
    load_path: Optional[str] = None
    save_path: str = "agent/checkpoints/dqn_best.pth"
    single_snake: bool = False
    self_play_prob: float = 0.5

class DQNVariantTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg  # FIXED: Restored self.cfg assignment
        # Variant-aware exploration schedule.
        # Use percentage-based decay to adapt to any total_frames count (1M, 5M, etc.)
        total = max(1, int(cfg.total_frames))
        
        # Decide Decay Duration: 80% of total steps for single, 90% for battle (need more exploration)
        if cfg.single_snake:
            decay_ratio = 0.80
        else:
            decay_ratio = 0.90
            
        self.decay_steps = int(total * decay_ratio)
        self.cfg.eps_decay = self.decay_steps # Update config for logging
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if cfg.single_snake:
            cfg.num_snakes = 1

        # Algorithm-Specific Hyperparameters (V6.9 Optim-Matrix)
        # V8.1: Dual-Phase Hyperparameter Matrix
        if cfg.variant == "dqn":
            if cfg.single_snake:
                # V38.0: DQN Ph1 - Restore Winner Config (LR 8e-5, Tau 0.005)
                # User Log 18:14 proved 8e-5 works (Rew 268) vs 5e-5
                self.lr, self.tau = 8.0e-5, 0.005 
                self.closer_reward = 0.15
            else:
                # V33.0: DQN Ph2 - Stability Tuning (LR 4e-5, Tau 0.001)
                self.lr, self.tau = 4.0e-5, 0.001 
                self.closer_reward = 0.05
            self.buffer_size = 600_000 
            self.grad_clip = 0.5 
        elif cfg.variant == "ddqn":
            if cfg.single_snake:
                # V38.0: DDQN Ph1 - Strict Align with DQN (Winner)
                # GradClip 1.0 -> 0.5 was the likely culprit for failure.
                self.lr, self.tau = 8.0e-5, 0.005 # V38.0: Match DQN
                self.closer_reward = 0.15
            else:
                # DDQN battle is sensitive to self-play non-stationarity; use slightly lower LR.
                self.lr, self.tau = 4.0e-5, 0.001
                self.closer_reward = 0.05
            self.buffer_size = 600_000 
            self.grad_clip = 0.5 # V38.0: Critical Fix (1.0 -> 0.5) to match DQN stability
        elif cfg.variant == "per":
            if cfg.single_snake:
                # V39.0: PER Ph1 Fix - Combat late-stage collapse (peak@210k then crash)
                # Root cause: tau=0.002 caused target lag + alpha=0.6 over-prioritized outliers
                # Fix: tau 0.005 (faster sync), lr 6e-5 (slower), alpha 0.5 (balanced sampling)
                # NOTE: PER now uses DDQNNet; restore LR to avoid under-training.
                # Stability is handled via weighted Huber + milder prioritization.
                self.lr, self.tau = 8.0e-5, 0.005
                self.per_alpha = 0.5
                self.per_beta_start = 0.4
                self.closer_reward = 0.15
            else:
                self.lr, self.tau = 6.0e-5, 0.002
                self.per_alpha = 0.5
                self.per_beta_start = 0.6
                self.closer_reward = 0.05 
            self.buffer_size = 600_000 
            self.grad_clip = 0.5 
        elif cfg.variant == "dueling":
            if cfg.single_snake:
                # V42.0: Dueling Ph1 - Rebalance for V41.0 self_collision_penalty
                # V40.0+V41.0 result: peak@90k (98) then crash (self_collision too harsh)
                # Fix: lr 6e-5 (gentler), per_alpha 0.5 (balanced sampling)
                self.lr, self.tau = 6.0e-5, 0.005
                self.per_alpha = 0.5
                self.per_beta_start = 0.4
                self.closer_reward = 0.15
            else:
                self.lr, self.tau = 4.0e-5, 0.001 
                self.per_alpha = 0.5
                self.per_beta_start = 0.6
                self.closer_reward = 0.05 
            self.buffer_size = 600_000 
            self.buffer_size = 600_000 
            self.grad_clip = 0.5 
        
        if cfg.total_frames <= 1_200_000 and cfg.single_snake:
            # Phase 1 Short Run: 
            # V44.6: Increase Buffer to 1M (Whole History) to prevent Catastrophic Forgetting.
            # 200k was too small, causing agent to forget how to handle early/mid game states 
            # once it reached late game (long snake) states.
            self.buffer_size = 1_000_000
        elif cfg.total_frames <= 2_000_000:
            self.buffer_size = min(self.buffer_size, 400_000)

        # V44.6: DDQN specific tuning - Lower LR to stabilize Phase 1
        if cfg.variant == "ddqn":
            self.lr = self.lr * 0.5 # 8e-5 -> 4e-5

        log(f">>> [V8.0 Asymmetric-Tuning] Variant: {cfg.variant.upper()} | LR: {self.lr} | Tau: {self.tau} | GradClip: {self.grad_clip} | Buf: {self.buffer_size}")
        
        env_cfg = BattleSnakeConfig(num_snakes=cfg.num_snakes, dash_cooldown_steps=15)
        if cfg.num_snakes == 1:
            # Phase 1: High focus on navigation (V6.2 Fixed)
            env_cfg.closer_reward = self.closer_reward
            env_cfg.farther_penalty = -0.10
            env_cfg.food_reward = 50.0 
            env_cfg.death_penalty = -20.0 
            env_cfg.step_penalty = -0.01
            env_cfg.self_collision_penalty = -15.0  # V43.0: Reduced from -22 (Prevent timidity)
            log(f">>> PHASE 1 (Single) | Closer: {env_cfg.closer_reward} | Penalty: -0.01")
        else:
            # Phase 2: Aggressive Combat & Survival (V6.2 Fixed)
            env_cfg.closer_reward = self.closer_reward
            env_cfg.farther_penalty = -0.10  
            env_cfg.step_penalty = -0.02      
            env_cfg.death_penalty = -30.0    
            env_cfg.kill_reward = 30.0       
            env_cfg.food_reward = 50.0
            env_cfg.self_collision_penalty = -20.0  # V43.0: Reduced from -35
            log(f">>> PHASE 2 (Battle) | Closer: {env_cfg.closer_reward} | Penalty: -0.02")
        
        self.envs = [BattleSnakeEnv(env_cfg) for _ in range(cfg.num_envs)]
        
        # Select Architecture
        if cfg.variant == "dqn": self.net_cls = DQNNet
        elif cfg.variant == "ddqn": self.net_cls = DDQNNet
        elif cfg.variant == "per": self.net_cls = DDQNNet 
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
            # V8.1: Support phase-aware alpha
            alpha = getattr(self, 'per_alpha', 0.6)
            self.memory = PrioritizedReplayBuffer(self.buffer_size, (3, 7, 7), 25, cfg.batch_size, self.device, alpha=alpha)
        else:
            self.memory = FastReplayBuffer(self.buffer_size, (3, 7, 7), 25, cfg.batch_size, self.device)
            
        self.steps = 0
        self.sp_manager = SelfPlayManager(cfg.pool_dir)
        
        # Self-Play Manager (V6.7 Model Cache)
        # Store model paths for opponents. None means random.
        self.opp_model_paths = [[None]*cfg.num_snakes for _ in range(cfg.num_envs)]
        self.loaded_opp_models: Dict[str, nn.Module] = {}
        
        self.best_reward = -float('inf')
        
        # V39.0: Unified Gamma (Dueling was 0.995 -> unstable)
        # All variants now use 0.99 for stable value estimation
        self.gamma = 0.99
        # V44.2: Global Stability Fix - Force 1 update/step for ALL variants (including PER).
        # Previous values (PER=3, DDQN=2) caused collapse with high LR.
        # Stability > Speed.
        self.updates_per_step = 1

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
            model = self.net_cls(vector_dim=25).to(self.device)
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
        log(f">>> Starting {self.cfg.variant.upper()} Training (V6.7 Omni-Batch)...")
        obs_batch = [env.reset() for env in self.envs]
        ep_rewards = [0.0] * self.cfg.num_envs
        recent_rewards = []
        last_log_time = time.time()
        saved_best = False
        
        total_frames = max(1, int(self.cfg.total_frames))
        # Scale key schedule points with training length.
        # For 20M runs, linear warmup(10%/30%) is too long. Use sqrt scaling + clamps.
        scale = math.sqrt(total_frames / 1_000_000)
        warmup1 = int(max(50_000, min(int(total_frames * 0.10), 100_000 * scale)))
        warmup2 = int(max(150_000, min(int(total_frames * 0.30), 300_000 * scale)))
        # Late-stage throttling (reduce update pressure / tau / self-play churn).
        late_half = int(total_frames * 0.50)
        late_40 = int(total_frames * 0.40)

        while self.steps < total_frames:
            # 1. Group all snakes by model for Omni-Batch Inference
            # { model_ptr: [(env_idx, snake_idx, obs)] }
            groups: Dict[Optional[nn.Module], List[Tuple[int, int, Dict]]] = {None: []}
            
            self.steps += self.cfg.num_envs
            # V43.0: Dynamic Epsilon Decay (Scaled to total_frames)
            if self.cfg.single_snake:
                eps_min = 0.05 # V44.0: Increased from 0.02 to prevent overfitting/collapse
            else:
                eps_min = 0.05 # Keep exploring in battle
            
            # Linear decay over defined duration
            completion = min(1.0, self.steps / self.decay_steps)
            eps = 1.0 - completion * (1.0 - eps_min)
            eps = max(eps_min, eps)
            
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
                
                with torch.inference_mode():
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
                    sp_prob = self.cfg.self_play_prob
                    # Gradual reduction of shuffle rate? No, keep it dynamic for Battle.
                    if self.cfg.num_snakes > 1 and random.random() < sp_prob:
                        opp_idx = random.randint(1, self.cfg.num_snakes-1)
                        m_p = self.sp_manager.sample_model()
                        if m_p:
                            self.opp_model_paths[e_idx][opp_idx] = str(m_p)
                else:
                    next_obs_batch.append(n_obs)
            
            # V14.0 CRITICAL FIX: Update obs_batch for next iteration!
            # This was MISSING - old observations were reused indefinitely!
            obs_batch = next_obs_batch

            # 4. V43.0: Standard Linear LR Decay
            # Remove complex variant-specific floor logic. Just decay to 1% (V44.0).
            progress = self.steps / total_frames
            frac = max(0.0, 1.0 - progress)
            current_lr = self.lr * (0.01 + 0.99 * frac)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr

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
                    saved_best = True
            
            pool_interval = max(150_000, int(total_frames * 0.03))
            if self.steps % pool_interval < self.cfg.num_envs:
                self.sp_manager.add_model(self.policy_net.state_dict(), f"{self.cfg.variant}_step_{self.steps}")
        
        # IMPORTANT: many variants can peak and then regress late.
        # Keep `save_path` as the best checkpoint; avoid overwriting it with a worse final model.
        if not saved_best:
            self.save_model(self.cfg.save_path)
        else:
            final_path = str(Path(self.cfg.save_path).with_suffix(".final.pth"))
            self.save_model(final_path)

    def update(self):
        # V6.3: Calculate dynamic Beta for PER (Annealing from 0.4 to 1.0)
        frac = self.steps / self.cfg.total_frames
        beta_start = getattr(self, "per_beta_start", 0.4)
        current_beta = min(1.0, beta_start + frac * (1.0 - beta_start))
        
        if "per" in self.cfg.variant or "dueling" in self.cfg.variant:
            states, actions, rewards, next_states, dones, weights, idxs = self.memory.sample(current_beta)
        else:
            states, actions, rewards, next_states, dones, weights, idxs = self.memory.sample()
            
        q_curr = self.policy_net(states['grid'], states['vector']).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            if self.cfg.variant == "dqn":
                q_next = self.target_net(next_states['grid'], next_states['vector']).max(1)[0]
            else:
                best_actions = self.policy_net(next_states['grid'], next_states['vector']).argmax(1)
                q_next = self.target_net(next_states['grid'], next_states['vector']).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            # V14.0 FIX: Use explicit float conversion instead of bitwise NOT
            # V6.0: Use self.gamma (with N-step extension)
            target = rewards + self.gamma * q_next * (1.0 - dones.float())
            
        td_errors = q_curr - target
        if weights is not None:
            # Weighted Huber loss is significantly more stable than weighted MSE under PER.
            per_sample = nn.SmoothL1Loss(reduction="none")(q_curr, target)
            loss = (weights * per_sample).mean()
            self.memory.update_priorities(idxs, td_errors.detach().abs().cpu().numpy())
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
