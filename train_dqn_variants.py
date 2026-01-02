"""
Unified Trainer for DQN Variants (V6.7 - Turbo Battle Performance).
Supports: DQN, DDQN, PER, Dueling-PER.
Features: Omni-Batch Inference (Massive FPS boost), Soft Updates, Algorithm-Specific Hyperparameters.
"""

import math
import random
import time
import threading
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# cuDNN autotune for fixed-shape conv nets (20x20 grid)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Optional CPU thread caps to reduce AsyncVectorEnv worker contention.
# Recommended: 1 (or 2) on typical multi-process rollouts.
_cpu_threads_env = os.getenv("DQN_CPU_THREADS")
if _cpu_threads_env:
    try:
        _n = max(1, int(_cpu_threads_env))
        torch.set_num_threads(_n)
        # Inter-op threads can be smaller to reduce oversubscription
        try:
            torch.set_num_interop_threads(min(4, _n))
        except Exception:
            pass
        # Best-effort for libraries used in workers; user can still override in shell.
        os.environ.setdefault("OMP_NUM_THREADS", str(_n))
        os.environ.setdefault("MKL_NUM_THREADS", str(_n))
    except Exception:
        pass

# V31.1: Align with PPO - Skip CUDAGraphs for dynamic opponent batches
if hasattr(torch, '_inductor'):
    import torch._inductor.config as inductor_config
    inductor_config.triton.cudagraph_skip_dynamic_graphs = True

# V18.4: Enable TF32 for Tensor Core acceleration (Ampere+)
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

from env.battle_snake_env import BattleSnakeEnv, BattleSnakeConfig
from env.gymnasium_wrapper import make_gymnasium_env
import gymnasium as gym

from agent.dqn import DQNNet, DQNAgent
from agent.ddqn import DDQNNet, DDQNAgent
from agent.per_dqn import PERDQNNet, PERDQNAgent
from agent.dueling_dqn import DuelingDQNNet, DuelingDQNAgent
from utils.self_play import SelfPlayManager


def _normalize_variant(v: str) -> str:
    """Normalize user-facing variant names to internal canonical names.

    Canonical variants:
      - dqn
      - ddqn
      - per        (DDQN + PER)
      - dueling    (DDQN + PER + Dueling)
    """
    v0 = (v or "").strip().lower()
    v1 = v0.replace("+", "_").replace("-", "_")
    if v1 in {"dqn"}:
        return "dqn"
    if v1 in {"ddqn"}:
        return "ddqn"
    if v1 in {"per", "ddqn_per", "ddqn_per_nodueling", "ddqnper"}:
        return "per"
    if v1 in {"dueling", "ddqn_per_dueling", "per_dueling", "ddqnperdueling"}:
        return "dueling"
    raise ValueError(
        f"Unknown variant '{v}'. Use one of: dqn, ddqn, per(=ddqn+per), dueling(=ddqn+per+dueling)."
    )

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
        
        # V28.0: Full GPU Zero-Sync Buffer for Standard DQN
        self.grids = torch.zeros((capacity, *grid_shape), dtype=torch.uint8, device=device)
        self.vectors = torch.zeros((capacity, vector_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_grids = torch.zeros((capacity, *grid_shape), dtype=torch.uint8, device=device)
        self.next_vectors = torch.zeros((capacity, vector_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
        
        # GPU state trackers
        self.register_buffer("_ptr_val", torch.zeros(1, dtype=torch.long, device=device))
        self.register_buffer("_size_val", torch.zeros(1, dtype=torch.long, device=device))
        
        # CPU Shadow variables for zero-sync indexing
        self._ptr = 0
        self._size = 0

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    @property
    def ptr(self): return self._ptr

    @property
    def size(self): return self._size

    def push_batch(self, grids, vecs, actions, rewards, next_grids, next_vecs, dones):
        num = len(grids)
        idx_range = (torch.arange(self._ptr, self._ptr + num, device=self.device) % self.capacity)
        
        self.grids[idx_range] = torch.as_tensor(grids, dtype=torch.uint8, device=self.device)
        self.vectors[idx_range] = torch.as_tensor(vecs, dtype=torch.float32, device=self.device)
        self.actions[idx_range] = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        self.rewards[idx_range] = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        self.next_grids[idx_range] = torch.as_tensor(next_grids, dtype=torch.uint8, device=self.device)
        self.next_vectors[idx_range] = torch.as_tensor(next_vecs, dtype=torch.float32, device=self.device)
        self.dones[idx_range] = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
        
        # Update trackers (Shadowed)
        self._ptr = (self._ptr + num) % self.capacity
        self._size = min(self._size + num, self.capacity)
        self._ptr_val[0] = self._ptr
        self._size_val[0] = self._size

    def sample(self):
        # V28.0: Pure indexing on GPU with Zero-Sync
        batch = torch.randint(0, self._size, (self.batch_size,), device=self.device)
        
        return (
            { "grid": self.grids[batch].float(), "vector": self.vectors[batch] },
            self.actions[batch], self.rewards[batch],
            { "grid": self.next_grids[batch].float(), "vector": self.next_vectors[batch] },
            self.dones[batch],
            None, # weights
            None  # indices
        )

    def update_priorities(self, idxs, td_errors):
        # Optional: Standard DQN ignores this, but we keep it for API compatibility
        pass



# V25.0 DEPRECATED: TorchSumTree removed for O(1) linear GPU priorities

class PrioritizedReplayBuffer:
    def __init__(self, capacity, grid_shape, vector_dim, batch_size, device, alpha=0.5, beta=0.4):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-6
        
        # Buffers on GPU
        self.grids = torch.zeros((capacity, *grid_shape), dtype=torch.uint8, device=device)
        self.vectors = torch.zeros((capacity, vector_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_grids = torch.zeros((capacity, *grid_shape), dtype=torch.uint8, device=device)
        self.next_vectors = torch.zeros((capacity, vector_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.priorities = torch.zeros(capacity, dtype=torch.float32, device=device)

        # Maintain running sum of priorities on GPU to avoid O(capacity) reductions during sampling.
        self.register_buffer("_total_priority", torch.zeros(1, dtype=torch.float32, device=device))
        
        # GPU state trackers
        self.register_buffer("_ptr_val", torch.zeros(1, dtype=torch.long, device=device))
        self.register_buffer("_size_val", torch.zeros(1, dtype=torch.long, device=device))
        self.register_buffer("_max_pri_tensor", torch.ones(1, dtype=torch.float32, device=device))
        
        # V25.2: CPU Shadow variables for zero-sync indexing
        self._ptr = 0
        self._size = 0

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    @property
    def ptr(self):
        return self._ptr

    @property
    def size(self):
        return self._size

    def push_batch(self, grids, vecs, actions, rewards, next_grids, next_vecs, dones):
        num = len(grids)
        idx_range = (torch.arange(self._ptr, self._ptr + num, device=self.device) % self.capacity)
        
        self.grids[idx_range] = torch.as_tensor(grids, dtype=torch.uint8, device=self.device)
        self.vectors[idx_range] = torch.as_tensor(vecs, dtype=torch.float32, device=self.device)
        self.actions[idx_range] = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        self.rewards[idx_range] = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        self.next_grids[idx_range] = torch.as_tensor(next_grids, dtype=torch.uint8, device=self.device)
        self.next_vectors[idx_range] = torch.as_tensor(next_vecs, dtype=torch.float32, device=self.device)
        self.dones[idx_range] = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
        
        # O(1) GPU Priority Update (No host sync)
        old_sum = self.priorities[idx_range].sum()
        new_p = self._max_pri_tensor[0].pow(self.alpha)
        self.priorities[idx_range] = new_p
        self._total_priority[0] += (new_p * num) - old_sum
        
        # Update trackers (Both GPU and CPU shadow)
        self._ptr = (self._ptr + num) % self.capacity
        self._size = min(self._size + num, self.capacity)
        self._ptr_val[0] = self._ptr
        self._size_val[0] = self._size

    def update_priorities(self, idxs, td_errors):
        ps = (td_errors.detach().abs() + self.epsilon).pow(self.alpha)

        # idxs may contain duplicates (sampling with replacement). Aggregate by max priority per index.
        if hasattr(ps, "scatter_reduce_"):
            uniq, inv = torch.unique(idxs, return_inverse=True)
            agg = torch.zeros((uniq.shape[0],), device=self.device, dtype=ps.dtype)
            agg.scatter_reduce_(0, inv, ps, reduce="amax", include_self=False)
            old = self.priorities[uniq]
            self.priorities[uniq] = agg
            self._total_priority[0] += (agg - old).sum()
            new_max = agg.max()
        else:
            # Fallback: best-effort without exact duplicate handling
            old = self.priorities[idxs]
            self.priorities[idxs] = ps
            self._total_priority[0] += (ps - old).sum()
            new_max = ps.max()

        # Update max_priority on GPU
        self._max_pri_tensor[0] = torch.max(self._max_pri_tensor[0], new_max)

    def sample(self, beta: float):
        # V25.1: Zero-Sync GPU Multinomial Sampling
        curr_size = self.size
        # Slicing is okay, but we use the fixed capacity if we want to avoid host-syncing 'size'
        # pr = self.priorities[:curr_size]
        # To truly avoid host-sync, one could pad priorities with 0 for unused indices.
        # But for multinomial, we need a 1D tensor of weights.
        
        pr = self.priorities[:curr_size]
        idxs = torch.multinomial(pr, self.batch_size, replacement=True)
        
        # Importance weights calculation
        total_p = self._total_priority[0]
        probs = pr[idxs] / (total_p + 1e-8)
        weights = (curr_size * probs).pow(-beta)
        weights = weights / (weights.max() + 1e-8)
        
        return (
            { "grid": self.grids[idxs].float(), "vector": self.vectors[idxs] },
            self.actions[idxs], self.rewards[idxs],
            { "grid": self.next_grids[idxs].float(), "vector": self.next_vectors[idxs] },
            self.dones[idxs], weights, idxs
        )

# --- Main Trainer ---

@dataclass
class TrainConfig:
    variant: str = "dqn" # dqn, ddqn, per(=ddqn+per), dueling(=ddqn+per+dueling)
    total_frames: int = 5_000_000 
    num_envs: int = 64 # V26.0: Optimal IPC for A6000 Phase 2
    num_envs_override: Optional[int] = None
    batch_size: int = 512 
    lr: float = 1e-4
    eps_decay: int = 0  
    tau: float = 0.005 
    num_snakes: int = 4
    pool_dir: str = "agent/pool/dqn"
    save_path: str = "agent/checkpoints/dqn_best.pth"
    load_path: Optional[str] = None
    # Recommended defaults for best performance in battle/self-play
    self_play_prob: float = 0.6
    buffer_size: int = 200_000 # V17.4: Reduced to 200k for much faster sampling
    single_snake: bool = False
    # Exploration (can override via CLI)
    eps_start: Optional[float] = None
    eps_min: Optional[float] = None

    # Self-play probability schedule (front-high then low)
    # Defaults tuned for stronger generalization: more self-play early, more randomness later.
    self_play_prob_start: Optional[float] = 0.7
    self_play_prob_end: Optional[float] = 0.4
    self_play_prob_frac: float = 0.30

    # Fine-tune learning rate multiplier when --load is used.
    finetune_lr_mult: float = 0.5

class DQNVariantTrainer:
    def __init__(self, cfg: TrainConfig):
        # Normalize variants early (support DDQN+PER naming requested by user)
        cfg.variant = _normalize_variant(cfg.variant)
        self.cfg = cfg  # FIXED: Restored self.cfg assignment

        # Curriculum correctness: when running Phase 1 ("single"), force a true single-snake env.
        # This keeps the observation/action spaces consistent while matching intended training.
        if cfg.single_snake:
            self.cfg.num_snakes = 1
        # Variant-aware exploration schedule.
        # Use percentage-based decay to adapt to any total_frames count (1M, 5M, etc.)
        total = max(1, int(cfg.total_frames))
        
        # Decide Decay Duration: 85% of total steps for single, 90% for battle (need more exploration)
        # V35.0: Universal Compile Config - Skip dynamic graphs to stay stable with groups
        if hasattr(torch, '_inductor'):
            torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Restore Decay Steps (V35.1 Fix)
        if cfg.single_snake:
            decay_ratio = 0.80 
        else:
            decay_ratio = 0.90
        self.decay_steps = int(total * decay_ratio)
        self.cfg.eps_decay = self.decay_steps

        # V35.0 Aggressive Optimization: Reduce overhead
        # Respect explicit overrides for A6000 benchmarking.
        if cfg.num_envs_override is not None:
            self.cfg.num_envs = int(cfg.num_envs_override)
        else:
            if not cfg.single_snake:
                self.cfg.num_envs = 32 # Reduced from 64 to save CPU/Memory
            else:
                self.cfg.num_envs = 128

        if not cfg.single_snake:
            self.active_rivals = [] # Global Active Rival models (paths)
            # Reduce FPS spikes from frequent model refresh/load.
            # Can override via env var for experiments.
            self.rival_update_interval = int(os.getenv("RIVAL_UPDATE_INTERVAL", "50000"))
            self.next_rival_update = 0
        
        # Opponent Grouping Cache
        self.cached_opp_groups = {}
        self.opp_model_paths = [] 
        
        v = cfg.variant
        num_snakes = cfg.num_snakes
        if cfg.single_snake:
            # Phase 1: High Exploration & Robust Learning
            default_eps_start, default_eps_min = 1.0, 0.05
            self.batch_size = 1024 # V21.1: Increased for GPU saturation
            if v == "dqn":
                self.lr, self.tau = 1.5e-4, 0.002
            else:
                self.lr, self.tau = 2.0e-4, 0.005
        else:
            # Phase 2: High Stability & Combat Precision
            default_eps_start, default_eps_min = 0.5, 0.10
            self.batch_size = 1024 # V21.1: Increased
            if v == "dqn":
                self.lr, self.tau = 1.0e-4, 0.002
            else:
                self.lr, self.tau = 8.0e-5, 0.005

        # Allow overriding eps schedule from config/CLI
        self.eps_start = float(cfg.eps_start) if cfg.eps_start is not None else float(default_eps_start)
        self.eps_min = float(cfg.eps_min) if cfg.eps_min is not None else float(default_eps_min)

        # Self-play probability schedule (front-high then low)
        self.sp_prob_start = float(cfg.self_play_prob_start) if cfg.self_play_prob_start is not None else float(cfg.self_play_prob)
        self.sp_prob_end = float(cfg.self_play_prob_end) if cfg.self_play_prob_end is not None else float(cfg.self_play_prob)
        self.sp_prob_frac = float(cfg.self_play_prob_frac) if cfg.self_play_prob_frac is not None else 0.30

        # If self-play is fully disabled, avoid requesting full observations and avoid opponent model inference.
        self.use_model_opps = (
            (not cfg.single_snake)
            and (cfg.num_snakes > 1)
            and (max(self.sp_prob_start, self.sp_prob_end, float(cfg.self_play_prob)) > 0.0)
        )
            
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
        log(f">>> [Explore] eps_start={self.eps_start:.2f} eps_min={self.eps_min:.2f} | [SelfPlay] prob={self.sp_prob_start:.2f}->{self.sp_prob_end:.2f} @ {self.sp_prob_frac:.2f}")
        
        env_cfg = BattleSnakeConfig(width=20, height=20, num_snakes=cfg.num_snakes)
        if cfg.num_snakes == 1:
            # Phase 1: High focus on navigation (V18.3 Turbo Sync)
            env_cfg.closer_reward = 0.15
            env_cfg.farther_penalty = -0.12
            env_cfg.food_reward = 50.0 # Match PPO V18.3
            env_cfg.death_penalty = -50.0
            env_cfg.step_penalty = -0.05
            env_cfg.self_collision_penalty = -60.0
            env_cfg.min_food = 5 # Add min_food to env_cfg
            log(f">>> [V18.3 Turbo] PHASE 1 (Single) | FoodDensity: {env_cfg.min_food} | FoodRew: {env_cfg.food_reward}")
        else:
            # Phase 2: V9.0 Combat (High Aggression)
            env_cfg.closer_reward = 0.15 
            env_cfg.farther_penalty = -0.10  
            env_cfg.step_penalty = -0.05      
            env_cfg.death_penalty = -100.0    
            env_cfg.kill_reward = 150.0      
            env_cfg.food_reward = 80.0       
            env_cfg.self_collision_penalty = -150.0 
            env_cfg.min_food = 2
            log(f">>> [V9.0 Battle] PHASE 2 | Kill: {env_cfg.kill_reward} | FoodDensity: {env_cfg.min_food}")
        
        log(f">>> [V18.3 Turbo] Initializing {cfg.num_envs} Parallel Environments...")
        def env_fn():
             # Convert BattleSnakeConfig to gymnasium wrapper keywords
             return make_gymnasium_env(
                 num_snakes=cfg.num_snakes, 
                 grid_size=20,
                 min_food=env_cfg.min_food,
                 closer_reward=env_cfg.closer_reward,
                 farther_penalty=env_cfg.farther_penalty,
                 food_reward=env_cfg.food_reward,
                 death_penalty=env_cfg.death_penalty,
                 step_penalty=env_cfg.step_penalty,
                 self_collision_penalty=env_cfg.self_collision_penalty,
                 # Only passed if present in BattleSnakeConfig
                 kill_reward=getattr(env_cfg, 'kill_reward', 150.0),
                 win_reward=getattr(env_cfg, 'win_reward', 800.0),
                 loss_penalty=getattr(env_cfg, 'loss_penalty', -200.0),
                 return_full_obs=self.use_model_opps,
             )
        self.envs = gym.vector.AsyncVectorEnv([env_fn for _ in range(cfg.num_envs)])
        log(">>> [V18.3 Turbo] Environments Ready.")
        
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

        # Memory format optimization for Conv2d on CUDA
        if self.device.type == "cuda":
            self.policy_net = self.policy_net.to(memory_format=torch.channels_last)
            self.target_net = self.target_net.to(memory_format=torch.channels_last)
        
        # 1. Load weights (V37.0: Dynamic stripping of _orig_mod prefixes)
        if cfg.load_path and Path(cfg.load_path).exists():
            log(f">>> Loading weights from {cfg.load_path}...")
            sd = torch.load(cfg.load_path, map_location=self.device, weights_only=True)
            # Clean Prefix: Handle models saved with torch.compile enabled
            sd = { k.replace("_orig_mod.", ""): v for k, v in sd.items() }
            self.policy_net.load_state_dict(sd)
            self.lr = self.lr * float(getattr(cfg, "finetune_lr_mult", 0.25))
            log(f">>> [V22.0] Fine-tuning mode: Lowered LR to {self.lr:.2e}")
            
        # 2. Sync target net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 3. Initialize Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # 4. Final Compilation (Only on Linux, after loading/syncing)
        if os.name != 'nt' and hasattr(torch, 'compile'):
            try:
                self.policy_net = torch.compile(self.policy_net, mode='reduce-overhead')
                log(">>> [DQN-Turbo] torch.compile() enabled.")
            except Exception as e:
                log(f">>> [DQN-Turbo] torch.compile() failed: {e}")
        else:
            log(">>> [DQN-Turbo] torch.compile() skipped (Windows/Unsupported).")
        
        # V28.0: Revert DQN to vanilla sampling while keeping PER for variants
        # V28.1: dqn and ddqn use uniform sampling; per and dueling use prioritized
        if cfg.variant in ["dqn", "ddqn"]:
            self.memory = FastReplayBuffer(self.buffer_size, (5, 20, 20), 28, self.batch_size, self.device)
        else:
            alpha = getattr(self, 'per_alpha', 0.6)
            self.memory = PrioritizedReplayBuffer(self.buffer_size, (5, 20, 20), 28, self.batch_size, self.device, alpha=alpha)
            
        self.steps = 0
        log(f">>> [V13.3 Heartbeat] Setup Finished. Buffer: {self.buffer_size} | Device: {self.device}")
        self.sp_manager = SelfPlayManager(cfg.pool_dir)
        
        # Self-Play Manager (V6.7 Model Cache)
        # Store model paths for opponents. None means random.
        self.opp_model_paths = [[None]*cfg.num_snakes for _ in range(cfg.num_envs)]
        self.loaded_opp_models: Dict[str, nn.Module] = {}

        # Async prefetch for opponent weights (CPU) to avoid I/O spikes in battle mode.
        # Main thread will materialize GPU modules lazily when weights are ready.
        self._prefetch_queue: "queue.SimpleQueue[str]" = queue.SimpleQueue()
        self._prefetch_lock = threading.Lock()
        self._prefetch_pending: set[str] = set()
        self._prefetched_sd: Dict[str, Tuple[float, Dict[str, torch.Tensor]]] = {}
        self._prefetch_stop = threading.Event()
        self._prefetch_max_items = int(os.getenv("PREFETCH_MAX_ITEMS", "64"))
        self._prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._prefetch_thread.start()
        
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
        # V37.0: Always save the clean state_dict (stripping torch.compile wrappers)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = p.with_suffix(p.suffix + ".tmp")
        
        # Access the raw module if it's compiled
        raw_net = self.policy_net._orig_mod if hasattr(self.policy_net, "_orig_mod") else self.policy_net
        # Save CPU tensors for portability and to avoid serializing CUDA tensors.
        sd_cpu = {k: v.detach().cpu() for k, v in raw_net.state_dict().items()}
        torch.save(sd_cpu, tmp_path)
        
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
        if path in self.loaded_opp_models:
            return self.loaded_opp_models.get(path)

        # If weights were prefetched on CPU, materialize GPU module fast.
        try:
            mtime = os.stat(path).st_mtime
        except OSError:
            return None

        cached = self._prefetched_sd.get(path)
        if cached is not None and cached[0] == mtime:
            try:
                model = self.net_cls(vector_dim=28).to(self.device).eval()
                if self.device.type == "cuda":
                    model = model.to(memory_format=torch.channels_last)
                sd = cached[1]
                model.load_state_dict(sd)
                for p in model.parameters():
                    p.requires_grad = False
                self.loaded_opp_models[path] = model
                if len(self.loaded_opp_models) > 500:
                    key_to_del = next(iter(self.loaded_opp_models))
                    del self.loaded_opp_models[key_to_del]
                return model
            except Exception:
                return None

        # Not ready yet: enqueue async prefetch and skip (caller will fallback to random action)
        self._enqueue_prefetch(path)
        return None

    def _enqueue_prefetch(self, path: Optional[str]) -> None:
        if not path:
            return
        with self._prefetch_lock:
            if path in self._prefetch_pending:
                return
            self._prefetch_pending.add(path)
        self._prefetch_queue.put(path)

    def _prefetch_worker(self) -> None:
        while not self._prefetch_stop.is_set():
            try:
                path = self._prefetch_queue.get(timeout=0.5)
            except Exception:
                continue

            with self._prefetch_lock:
                self._prefetch_pending.discard(path)

            try:
                mtime = os.stat(path).st_mtime
            except OSError:
                continue

            cached = self._prefetched_sd.get(path)
            if cached is not None and cached[0] == mtime:
                continue

            try:
                sd = torch.load(path, map_location="cpu", weights_only=True)
                sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
                self._prefetched_sd[path] = (mtime, sd)

                # simple LRU-ish eviction
                if len(self._prefetched_sd) > self._prefetch_max_items:
                    self._prefetched_sd.pop(next(iter(self._prefetched_sd)))
            except Exception:
                continue

    def train(self):
        log(f">>> Starting {self.cfg.variant.upper()} Training (V21.0 Flash-Batch)...")
        obs_dict, info = self.envs.reset()
        ep_rewards = [0.0] * self.cfg.num_envs
        recent_rewards = []
        last_log_time = time.time()
        
        total_frames = max(1, int(self.cfg.total_frames))
        num_snakes = self.cfg.num_snakes
        total_snakes = self.cfg.num_envs * num_snakes

        # Precompute indices and pinned CPU buffer for fast env stepping
        learner_map = torch.arange(0, total_snakes, num_snakes, device=self.device)
        actions_cpu = torch.empty((total_snakes,), dtype=torch.int32, device="cpu", pin_memory=True)

        # Reusable tensors to avoid per-step allocations
        current_actions = torch.empty(total_snakes, dtype=torch.int32, device=self.device)
        
        # V26.0 Initialization
        if self.use_model_opps:
            self.opp_model_paths = [([None] * num_snakes) for _ in range(self.cfg.num_envs)]
            self._rebuild_opp_groups()
        
        while self.steps < total_frames:
            self.steps += self.cfg.num_envs
            completion = min(1.0, self.steps / self.decay_steps)
            eps = max(self.eps_min, self.eps_start - completion * (self.eps_start - self.eps_min))
            
            # 1-2. Action selection
            if self.use_model_opps:
                # Faster batch creation (V34.0)
                # Transfer grids as uint8 to reduce PCIe bandwidth, then cast on GPU.
                full_grids = info.get("full_obs_grids") if isinstance(info, dict) else None
                full_vecs = info.get("full_obs_vecs") if isinstance(info, dict) else None
                if full_grids is None or full_vecs is None:
                    raise KeyError(
                        "Missing 'full_obs_grids'/'full_obs_vecs' in env info for multi-snake mode. "
                        "Ensure the Gymnasium wrapper returns full observations (return_full_obs=True)."
                    )

                all_g_u8 = torch.from_numpy(full_grids).view(total_snakes, 5, 20, 20)
                all_v_f32 = torch.from_numpy(full_vecs).view(total_snakes, -1)

                t_all_g_u8 = all_g_u8.to(self.device, non_blocking=True)
                t_all_v = all_v_f32.to(self.device, non_blocking=True)

                with torch.inference_mode(), autocast('cuda', enabled=self.use_amp):
                    grid_dtype = torch.float16 if (self.device.type == "cuda") else torch.float32
                    t_all_g = t_all_g_u8.to(dtype=grid_dtype)
                    if self.device.type == "cuda":
                        t_all_g = t_all_g.contiguous(memory_format=torch.channels_last)

                    # A. Learner
                    q_vals = self.policy_net(t_all_g[learner_map], t_all_v[learner_map])
                    la = q_vals.argmax(dim=1).to(torch.int32)
                    if eps > 0:
                        rv = torch.rand(self.cfg.num_envs, device=self.device)
                        la = torch.where(
                            rv < eps,
                            torch.randint(0, 4, (self.cfg.num_envs,), device=self.device, dtype=torch.int32),
                            la,
                        )
                    current_actions[learner_map] = la

                    # B. Opponents
                    for m_path, idxs in self.cached_opp_groups.items():
                        if m_path is None:
                            current_actions[idxs] = torch.randint(
                                0, 4, (len(idxs),), device=self.device, dtype=torch.int32
                            )
                        else:
                            m = self._get_opp_model(m_path)
                            if m:
                                current_actions[idxs] = m(t_all_g[idxs], t_all_v[idxs]).argmax(dim=1).to(torch.int32)

                    actions_cpu.copy_(current_actions, non_blocking=True)
                    all_actions_flat_np = actions_cpu.numpy()

                all_actions = all_actions_flat_np.reshape(self.cfg.num_envs, num_snakes)
            else:
                # No self-play models: learner uses its own obs; all opponents random.
                t_g = torch.from_numpy(obs_dict["grid"]).to(self.device, non_blocking=True)
                t_v = torch.from_numpy(obs_dict["vector"]).to(self.device, non_blocking=True)
                with torch.inference_mode(), autocast('cuda', enabled=self.use_amp):
                    grid_dtype = torch.float16 if (self.device.type == "cuda") else torch.float32
                    t_g = t_g.to(dtype=grid_dtype)
                    if self.device.type == "cuda":
                        t_g = t_g.contiguous(memory_format=torch.channels_last)
                    q_vals = self.policy_net(t_g, t_v)
                    la = q_vals.argmax(dim=1).to(torch.int32)
                    if eps > 0:
                        rv = torch.rand(self.cfg.num_envs, device=self.device)
                        la = torch.where(
                            rv < eps,
                            torch.randint(0, 4, (self.cfg.num_envs,), device=self.device, dtype=torch.int32),
                            la,
                        )
                if num_snakes == 1:
                    all_actions = la.view(self.cfg.num_envs, 1).cpu().numpy()
                else:
                    opp = torch.randint(0, 4, (self.cfg.num_envs, num_snakes - 1), device="cpu", dtype=torch.int32)
                    all_actions = torch.cat([la.cpu().view(self.cfg.num_envs, 1), opp], dim=1).numpy()

            # 3. Step Environment
            next_obs_dict, rews, terminated, truncated, next_info = self.envs.step(all_actions)
            
            # 4. Global Experience Store
            self.memory.push_batch(
                obs_dict["grid"], obs_dict["vector"],
                all_actions[:, 0], rews,
                next_obs_dict["grid"], next_obs_dict["vector"],
                terminated
            )
            
            # 5. Monitor & Self-Play (V35.1: Balanced Update)
            should_rebuild = False
            for e_idx in range(self.cfg.num_envs):
                ep_rewards[e_idx] += rews[e_idx]
                if terminated[e_idx] or truncated[e_idx]:
                    recent_rewards.append(ep_rewards[e_idx])
                    if len(recent_rewards) > 100: recent_rewards.pop(0)
                    ep_rewards[e_idx] = 0.0

                    # Two-stage self-play probability schedule: early higher, later lower.
                    if not self.cfg.single_snake and num_snakes > 1:
                        prog = self.steps / float(total_frames)
                        sp_prob = self.sp_prob_start if prog < self.sp_prob_frac else self.sp_prob_end
                        for s in range(1, num_snakes):
                            old = self.opp_model_paths[e_idx][s]
                            if self.active_rivals and (random.random() < sp_prob):
                                new = random.choice(self.active_rivals)
                            else:
                                new = None
                            if new != old:
                                self.opp_model_paths[e_idx][s] = new
                                should_rebuild = True

            # Global Rival Update (V35.0)
            if self.use_model_opps and self.steps >= self.next_rival_update:
                self.next_rival_update = self.steps + self.rival_update_interval
                new_rivals = []
                for _ in range(4):
                    m = self.sp_manager.sample_model()
                    if m: new_rivals.append(str(m))
                if new_rivals:
                    self.active_rivals = new_rivals
                    for p in self.active_rivals:
                        self._enqueue_prefetch(p)
                
            if self.use_model_opps and should_rebuild:
                self._rebuild_opp_groups()
            
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
            # V22.0: Multi-update to utilize GPU throughput
            # With num_envs=128, one loop adds 128 samples.
            # V23.0: 8 updates per 128 transitions (1:16 samples-to-steps ratio)
            # This is much more balanced than 32 updates.
            # V25.0: Consistent 1-update ratio to maximize FPS
            if self.memory.size >= self.batch_size:
                updates = 1
                for _ in range(updates):
                    self.update()

            # 5. Soft Update
            # 5. Soft Update (V31.1: Vectorized update - 100x faster than loops)
            tau_eff = self.tau
            if self.cfg.variant == "ddqn":
                late_cut = late_half if self.cfg.single_snake else late_40
                if self.steps >= late_cut:
                    tau_eff = self.tau * 0.5
            if self.cfg.variant == "dqn" and (not self.cfg.single_snake) and self.steps >= late_half:
                tau_eff = self.tau * 0.5
            if self.cfg.variant == "per" and self.steps >= late_half:
                tau_eff = self.tau * 0.5
                
            with torch.no_grad():
                for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                    target_param.mul_(1.0 - tau_eff).add_(policy_param, alpha=tau_eff)
                
            # 5. Heartbeat Logging (V25.0: 1024 interval for better visibility)
            if self.steps % 1024 < self.cfg.num_envs:
                elapsed = time.time() - last_log_time
                fps = (1024) / (elapsed + 1e-6)
                avg_r = np.mean(recent_rewards) if recent_rewards else 0
                log(f"Step: {self.steps} | EPS: {eps:.2f} | Rew: {avg_r:.2f} | FPS: {fps:.1f} | Var: {self.cfg.variant}")
                last_log_time = time.time()
                
                # Simplified Saving (User requested ONLY final model or periodic snapshot)
                # Keep periodic pool snapshots for self-play diversity
                pool_interval = max(150_000, int(total_frames * 0.03))
                if self.steps % pool_interval < self.cfg.num_envs:
                    raw_net = self.policy_net._orig_mod if hasattr(self.policy_net, "_orig_mod") else self.policy_net
                    sd_cpu = {k: v.detach().cpu() for k, v in raw_net.state_dict().items()}
                    self.sp_manager.add_model(sd_cpu, f"{self.cfg.variant}_step_{self.steps}")
            
        # Final Save (Only at the end of total_steps)
        self.save_model(self.cfg.save_path)
        final_path = str(Path(self.cfg.save_path).with_suffix(".final.pth"))
        self.save_model(final_path)

    def _rebuild_opp_groups(self):
        # V34.0: Pre-compute Tensors to avoid CPU-GPU sync during推断
        temp_groups = {}
        for e in range(self.cfg.num_envs):
            for s in range(1, self.cfg.num_snakes):
                m_path = self.opp_model_paths[e][s]
                idx = e * self.cfg.num_snakes + s
                temp_groups.setdefault(m_path, []).append(idx)

        # Async prefetch any non-random opponent models
        for m in temp_groups.keys():
            if m is not None:
                self._enqueue_prefetch(m)
        
        # Convert to cached tensors for constant-time GPU usage
        self.cached_opp_groups = {
            m: torch.tensor(idxs, device=self.device) for m, idxs in temp_groups.items()
        }

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
            # V24.0: Async priority update
            self.memory.update_priorities(idxs, td_errors.detach())

if __name__ == "__main__":
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument(
        "--variant",
        type=str,
        default="dqn",
        choices=["dqn", "ddqn", "per", "dueling", "ddqn_per", "ddqn_per_dueling"],
        help="Variants: dqn, ddqn, per(=ddqn+per), dueling(=ddqn+per+dueling)",
    )
    p.add_argument("--steps", type=int, default=1000000)
    p.add_argument("--single", action="store_true")
    p.add_argument("--load", type=str, default=None)
    p.add_argument("--save", type=str, default="agent/checkpoints/dqn_best.pth")
    p.add_argument("--num-envs", type=int, default=None, help="Override number of parallel envs")
    p.add_argument("--sp-prob", type=float, default=0.6)
    p.add_argument("--finetune-lr-mult", type=float, default=0.5, help="LR multiplier when --load is used (default 0.5)")

    # Optional knobs (battle/single)
    p.add_argument("--eps-start", type=float, default=None, help="Override epsilon start (default depends on phase)")
    p.add_argument("--eps-min", type=float, default=None, help="Override epsilon min (default depends on phase)")

    # Self-play probability schedule: early high then low
    p.add_argument("--sp-prob-start", type=float, default=0.7, help="Self-play prob in early phase (default 0.7)")
    p.add_argument("--sp-prob-end", type=float, default=0.4, help="Self-play prob after switch (default 0.4)")
    p.add_argument("--sp-prob-frac", type=float, default=0.30, help="Switch point as fraction of total steps (default 0.30)")
    args = p.parse_args()

    v_norm = _normalize_variant(args.variant)
    p_dir = f"agent/pool/{v_norm}"
    
    cfg = TrainConfig(
        variant=v_norm,
        total_frames=args.steps,
        single_snake=args.single,
        load_path=args.load,
        save_path=args.save,
        num_envs_override=args.num_envs,
        self_play_prob=args.sp_prob,
        eps_start=args.eps_start,
        eps_min=args.eps_min,
        self_play_prob_start=args.sp_prob_start,
        self_play_prob_end=args.sp_prob_end,
        self_play_prob_frac=args.sp_prob_frac,
        finetune_lr_mult=args.finetune_lr_mult,
        pool_dir=p_dir
    )
    DQNVariantTrainer(cfg).train()
