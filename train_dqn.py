"""
Fast DQN Training V4.2 - Vectorized & Optimized.
Hybrid CNN+MLP for competitive play with Self-Play.
V4.2: Fixed stdout buffering & Added faster heartbeat logs.
"""

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys

from env.battle_snake_env import BattleSnakeEnv, BattleSnakeConfig
from agent.dqn import DQNNet, DQNAgent
from utils.self_play import SelfPlayManager

def log(msg):
    print(msg, flush=True)

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
            torch.from_numpy(self.dones[idxs]).to(self.device)
        )

class VectorizedEnvForDQN:
    def __init__(self, num_envs, config):
        self.envs = [BattleSnakeEnv(config) for _ in range(num_envs)]
        self.num_envs = num_envs

    def reset(self):
        return [env.reset() for env in self.envs]

    def step(self, actions_list):
        results = [env.step(a) for env, a in zip(self.envs, actions_list)]
        obs_n, rew_n, done_n, info_n = zip(*results)
        return list(obs_n), list(rew_n), list(done_n), list(info_n)

@dataclass
class TrainConfig:
    total_frames: int = 2_000_000
    num_envs: int = 8 
    batch_size: int = 256 
    lr: float = 2e-4
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: int = 500_000
    target_update: int = 2000
    update_freq: int = 4 
    update_iters: int = 1 
    checkpoint_path: str = "agent/checkpoints/dqn_battle_best.pth"
    load_path: Optional[str] = None
    num_snakes: int = 4
    grid_size: int = 20
    self_play_prob: float = 0.3
    pool_dir: str = "agent/pool/dqn"

class FastDQNTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        log(">>> [Heartbeat] Initializing CUDA Device...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f">>> [Heartbeat] Using device: {self.device}")
        
        env_cfg = BattleSnakeConfig(
            width=cfg.grid_size, height=cfg.grid_size, 
            num_snakes=cfg.num_snakes,
            min_food=max(2, cfg.num_snakes // 2), 
            max_steps=500
        )
        if cfg.num_snakes == 1:
            env_cfg.closer_reward, env_cfg.food_reward, env_cfg.death_penalty = 0.35, 12.0, -10.0
        else:
            env_cfg.closer_reward, env_cfg.kill_reward, env_cfg.death_penalty = 0.1, 30.0, -25.0
            
        log(">>> [Heartbeat] Spawning Vectorized Environments...")
        self.envs = VectorizedEnvForDQN(cfg.num_envs, env_cfg)
        self.policy_net = DQNNet(vector_dim=24).to(self.device)
        self.target_net = DQNNet(vector_dim=24).to(self.device)
        
        if cfg.load_path and Path(cfg.load_path).exists():
            log(f">>> Loading pretrained model from {cfg.load_path}...")
            self.policy_net.load_state_dict(torch.load(cfg.load_path, map_location=self.device, weights_only=True))
            
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.memory = FastReplayBuffer(150_000, (3, 7, 7), 24, cfg.batch_size, self.device)
        self.total_env_steps = 0
        self.sp_manager = SelfPlayManager(cfg.pool_dir)
        self.history_agents = [[None]*cfg.num_snakes for _ in range(cfg.num_envs)]

    def train(self):
        log(f">>> Starting V4.2 Fast DQN Training: {self.cfg.num_envs} Envs | {self.cfg.num_snakes} Snakes.")
        obs_batch = self.envs.reset()
        ep_rewards = [0.0] * self.cfg.num_envs
        total_episodes = 0
        last_log_time = time.time()
        last_log_steps = 0
        update_step = 0

        while self.total_env_steps < self.cfg.total_frames:
            update_step += 1
            self.total_env_steps += self.cfg.num_envs
            
            # Action Selection logic (Simplified for readability here)
            all_actions = []
            flat_grids, flat_vecs = [], []
            for e_idx in range(self.cfg.num_envs):
                for s_idx in range(self.cfg.num_snakes):
                    flat_grids.append(obs_batch[e_idx][s_idx]['grid'])
                    flat_vecs.append(obs_batch[e_idx][s_idx]['vector'])
            
            t_grids = torch.tensor(np.array(flat_grids), dtype=torch.float32, device=self.device)
            t_vecs = torch.tensor(np.array(flat_vecs), dtype=torch.float32, device=self.device)
            
            eps = self.cfg.eps_end + (self.cfg.eps_start - self.cfg.eps_end) * math.exp(-1. * self.total_env_steps / self.cfg.eps_decay)
            with torch.no_grad():
                q_values = self.policy_net(t_grids, t_vecs)
                greedy_actions = q_values.argmax(dim=1).cpu().numpy()
            
            idx = 0
            for e_idx in range(self.cfg.num_envs):
                env_acts = []
                for s_idx in range(self.cfg.num_snakes):
                    if self.history_agents[e_idx][s_idx]:
                        env_acts.append(self.history_agents[e_idx][s_idx].act(obs_batch[e_idx][s_idx]))
                    elif random.random() < eps:
                        env_acts.append(random.randint(0, 3))
                    else:
                        env_acts.append(greedy_actions[idx])
                    idx += 1
                all_actions.append(env_acts)
            
            # Step Env
            next_obs_batch, rewards_batch, dones_batch, _ = self.envs.step(all_actions)
            
            # Store & Reset
            for e_idx in range(self.cfg.num_envs):
                for s_idx in range(self.cfg.num_snakes):
                    if self.history_agents[e_idx][s_idx] is None:
                        # Fixed dones_batch handling for both single and multi mode
                        env_done = dones_batch[e_idx]
                        is_snake_done = env_done[s_idx] if isinstance(env_done, (list, np.ndarray)) else env_done
                        
                        if not self.envs.envs[e_idx].dead[s_idx] or rewards_batch[e_idx][s_idx] != 0:
                             self.memory.push(obs_batch[e_idx][s_idx], all_actions[e_idx][s_idx], 
                                             rewards_batch[e_idx][s_idx], next_obs_batch[e_idx][s_idx], 
                                             is_snake_done)
                        ep_rewards[e_idx] += rewards_batch[e_idx][s_idx]
                
                # Check if entire environment reset is needed
                env_done = dones_batch[e_idx]
                should_reset = all(env_done) if isinstance(env_done, (list, np.ndarray)) else env_done
                if should_reset:
                    total_episodes += 1
                    if total_episodes % 20 == 0:
                        log(f"Env {e_idx} Step {self.total_env_steps} | Ep {total_episodes} | Avg Reward: {ep_rewards[e_idx]:.2f}")
                    ep_rewards[e_idx] = 0.0
                    next_obs_batch[e_idx] = self.envs.envs[e_idx].reset()
                    # Self-Play Shuffle (Simplified)
                    self.history_agents[e_idx] = [None] * self.cfg.num_snakes
                    if self.cfg.num_snakes > 1 and random.random() < self.cfg.self_play_prob:
                         for idx_ in random.sample(range(1, self.cfg.num_snakes), random.randint(1,1)):
                             m_p = self.sp_manager.sample_model()
                             if m_p: self.history_agents[e_idx][idx_] = DQNAgent(input_dim=24, model_path=str(m_p))

            obs_batch = next_obs_batch
            
            # Update (V4.2 High Freq Initial Logs)
            # Replay threshold
            if self.memory.size >= 1000 and update_step % self.cfg.update_freq == 0:
                self.update()
                
            if update_step % self.cfg.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                
            # Periodic FPS/Milestone Log (Every 10000 steps for visibility)
            if self.total_env_steps % 10000 < self.cfg.num_envs:
                now = time.time()
                fps = (self.total_env_steps - last_log_steps) / (now - last_log_time)
                log(f">>> [Status] Step: {self.total_env_steps} | FPS: {fps:.1f} | Buffer: {self.memory.size}")
                last_log_time, last_log_steps = now, self.total_env_steps

                if self.total_env_steps % 200000 < self.cfg.num_envs:
                    torch.save(self.policy_net.state_dict(), self.cfg.checkpoint_path)
                    self.sp_manager.add_model(self.policy_net.state_dict(), f"dqn_v4_step_{self.total_env_steps}")

    def update(self):
        states, actions, rewards, next_states, dones = self.memory.sample()
        q_curr = self.policy_net(states['grid'], states['vector']).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target_net(next_states['grid'], next_states['vector']).max(1)[0]
            target = rewards + self.cfg.gamma * q_next * (~dones)
        loss = nn.SmoothL1Loss()(q_curr, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

if __name__ == "__main__":
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument("--single", action="store_true")
    p.add_argument("--load", type=str, default=None)
    p.add_argument("--steps", type=int, default=2000000)
    args = p.parse_args()
    num = 1 if args.single else 4
    ckpt = "agent/checkpoints/dqn_best.pth" if args.single else "agent/checkpoints/dqn_battle_best.pth"
    FastDQNTrainer(TrainConfig(num_snakes=num, total_frames=args.steps, checkpoint_path=ckpt, load_path=args.load)).train()
