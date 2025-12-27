"""
Unified Multi-Agent PPO Training V4.3 (V6.1 FIX).
V4.3: Robust 25D Enforcement & Checkpoint Mismatch Diagnostic.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import time
import random
import sys

from env.battle_snake_env import BattleSnakeEnv, BattleSnakeConfig
from agent.ppo import ActorCritic, PPOAgent
from utils.self_play import SelfPlayManager

def log(msg):
    print(msg, flush=True)

def make_env(num_snakes, grid_size, seed=None):
    def thunk():
        env_cfg = BattleSnakeConfig(
            width=grid_size, height=grid_size, 
            num_snakes=num_snakes,
            min_food=max(2, num_snakes//2), 
            max_steps=500,
            dash_cooldown_steps=15 # ENSURE V5.0 MECHANICS
        )
        if num_snakes == 1:
            # Phase 1: Navigation emphasis (V9.2 Aligned with DQN)
            env_cfg.closer_reward = 0.15
            env_cfg.farther_penalty = -0.10  # V12.0: Explicit alignment with DQN
            env_cfg.food_reward = 25.0
            env_cfg.death_penalty = -15.0
        else:
            # Phase 2: Aggressive Combat & Survival (V9.0)
            env_cfg.closer_reward = 0.05
            env_cfg.step_reward = 0.05   # V9.0: (0.01 -> 0.05)
            env_cfg.kill_reward = 50.0   # V9.0: (30 -> 50)
            env_cfg.death_penalty = -15.0
            env_cfg.food_reward = 30.0   # V9.0: (20 -> 30)
        return BattleSnakeEnv(env_cfg, seed=seed)
    return thunk

class VectorizedEnv:
    def __init__(self, env_fns, device):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.device = device
        # V6.8: Store model paths instead of agent objects
        self.opp_model_paths: List[List[Optional[str]]] = [[None]*env.config.num_snakes for env in self.envs]
        self.loaded_models: Dict[str, nn.Module] = {}

    def _get_model(self, path: str) -> Optional[nn.Module]:
        if path not in self.loaded_models:
            # V11.2 I/O Shield: Prevent deadlock during disk write collision
            try:
                model = ActorCritic(vector_dim=25).to(self.device).eval()
                state_dict = torch.load(path, map_location=self.device, weights_only=True)
                model.load_state_dict(state_dict)
                for p in model.parameters(): p.requires_grad = False
                self.loaded_models[path] = model
                if len(self.loaded_models) > 5:
                    del self.loaded_models[next(iter(self.loaded_models))]
            except Exception as e:
                # Log but don't hang/crash
                log(f"--- [I/O CACHE] Load failed for {Path(path).name}: {e}. Using Random.")
                return None
        return self.loaded_models.get(path)

    def reset(self): return [env.reset() for env in self.envs]

    def step(self, learning_actions, current_obs):
        # learning_actions: [num_envs] (int actions for learner)
        all_actions = [[None]*env.config.num_snakes for env in self.envs]
        
        # 1. Group by model
        groups: Dict[Optional[nn.Module], List[Tuple[int, int, dict]]] = {None: []}
        for i in range(self.num_envs):
            all_actions[i][0] = int(learning_actions[i]) # Always greedy/sampled learner action
            for j in range(1, self.envs[0].config.num_snakes):
                m_path = self.opp_model_paths[i][j]
                if m_path:
                    model = self._get_model(m_path)
                    if model:
                        groups[model] = groups.get(model, [])
                        groups[model].append((i, j, current_obs[i][j]))
                    else: all_actions[i][j] = random.randint(0, 3)
                else: all_actions[i][j] = random.randint(0, 3)

        # 2. Omni-Batch Inference for opponents
        for model, samples in groups.items():
            if not samples or model is None: continue
            with torch.inference_mode(): # V9.0 Speedup
                grids = np.array([s[2]['grid'] for s in samples])
                vecs = np.array([s[2]['vector'] for s in samples])
                t_g = torch.as_tensor(grids, dtype=torch.float32, device=self.device)
                t_v = torch.as_tensor(vecs, dtype=torch.float32, device=self.device)
                logits, _ = model(t_g, t_v) # Standard AC call
                acts = logits.argmax(dim=1).cpu().numpy()
                for idx, (e_idx, s_idx, _) in enumerate(samples):
                    all_actions[e_idx][s_idx] = int(acts[idx])

        # 3. Environment Step
        res = [self.envs[i].step(all_actions[i]) for i in range(self.num_envs)]
        obs_n, rew_n, done_n, info_n = zip(*res)
        return list(obs_n), list(rew_n), list(done_n), list(info_n)

def train_ppo(num_envs=8, num_snakes=4, total_timesteps=2_000_000, 
              load_path=None, checkpoint_path="agent/checkpoints/ppo_battle_best.pth",
              self_play_prob=0.3, lr=2.5e-4):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # V8.1: Precision smoothing for battle phase
    if num_snakes > 1:
        lr = lr * 0.6
    log(f">>> [V8.1 FIX] PPO Device: {device} | Snakes: {num_snakes} | BaseLR: {lr:.2e}")
    Path("agent/checkpoints").mkdir(parents=True, exist_ok=True)
    sp_manager = SelfPlayManager("agent/pool/ppo")
    
    # 强制 25D
    agent = ActorCritic(vector_dim=25).to(device)
    
    if load_path and Path(load_path).exists():
        log(f">>> Loading PPO weights from {load_path}...")
        try:
            state_dict = torch.load(load_path, map_location=device, weights_only=True)
            # Diagnostic for size mismatch
            for k, v in state_dict.items():
                if "actor.0.weight" in k:
                    log(f">>> Checkpoint '{k}' shape: {v.shape}")
            agent.load_state_dict(state_dict)
        except Exception as e:
            log(f">>> ERROR loading {load_path}: {e}")
            log(">>> TIP: Phase 1 checkpoint might be OLD (24D). Please re-run Phase 1.")
            sys.exit(1)
        
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    # V10.1 A6000 Turbo: Scale up for high-VRAM throughput
    num_steps, batch_size, mini_batch = 128, num_envs * 128, 512
    
    envs = VectorizedEnv([make_env(num_snakes, 20, i) for i in range(num_envs)], device)
    log(f">>> [A6000 TURBO] Envs: {num_envs} | Batch: {batch_size} | Mini: {mini_batch}")
    obs_list = envs.reset() 
    
    # Verify observation dimension at start
    log(f">>> First obsession dimension: {obs_list[0][0]['vector'].shape[0]}")
    
    global_step, ep_rewards = 0, []
    current_rewards = np.zeros(num_envs)

    while global_step < total_timesteps:
        # V10.6 Dynamic Cooling: Linearly decay entropy to Lock-in policy
        # Heartbeat added to track stagnation at 60% (possible Race Condition)
        frac = max(0.0, 1.0 - (global_step / total_timesteps))
        min_factor = 0.1 # V11.1: Restored to 10% floor for late stability
        current_lr = lr * max(min_factor, frac)
        ent_coef = max(0.01, 0.05 * frac) 

        for g in optimizer.param_groups: 
            g['lr'] = current_lr

        if global_step % 20480 < num_envs:
            log(f">>> [HEARTBEAT] Loop Start | Progress: {(global_step/total_timesteps)*100:.1f}%")
        
        # V10.8: Restore buffer initialization (Fixed NameError)
        b_grid, b_vector, b_acts, b_logprobs, b_rewards, b_dones, b_values = [], [], [], [], [], [], []
        
        # --- STAGE 1: Collection ---
        for _ in range(num_steps):
            global_step += num_envs
            # Sub-Loop Heartbeat for V11.2 (Settle 60% Freeze)
            if global_step % 2048 == 0:
                pass # Minimal overhead
            
            flat_grid, flat_vec = [o[0]['grid'] for o in obs_list], [o[0]['vector'] for o in obs_list]
            t_grid = torch.tensor(np.array(flat_grid), dtype=torch.float32).to(device)
            t_vec = torch.tensor(np.array(flat_vec), dtype=torch.float32).to(device)
            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(t_grid, t_vec)
                
            next_obs, rewards, dones, _ = envs.step(action.cpu().numpy(), obs_list)
            
            for i in range(num_envs):
                current_rewards[i] += rewards[i][0]
                if all(dones[i]):
                    ep_rewards.append(current_rewards[i])
                    current_rewards[i] = 0
                    next_obs[i] = envs.envs[i].reset()
                    # Self-Play Shuffle (V6.8)
                    if num_snakes > 1 and random.random() < self_play_prob:
                         for s_idx in range(1, num_snakes):
                             try:
                                 m_p = sp_manager.sample_model(chaos_prob=0.1) 
                                 if m_p: envs.opp_model_paths[i][s_idx] = str(m_p)
                             except Exception as e:
                                 log(f"!!! [SP ERROR] Sample failed: {e}. Skipping to avoid deadlock.")
            
            b_grid.append(t_grid); b_vector.append(t_vec); b_acts.append(action)
            b_logprobs.append(logprob); b_values.append(value.flatten())
            b_rewards.append(torch.tensor([r[0] for r in rewards], device=device))
            b_dones.append(torch.tensor([d[0] for d in dones], device=device).float())
            obs_list = next_obs
        
        # --- STAGE 2: GAE ---
        if global_step % 20480 < num_envs:
            log(f">>> [HEARTBEAT] Advantage Calculation...")

        # Advantage (GAE)
        with torch.no_grad():
            f_g, f_v = [o[0]['grid'] for o in obs_list], [o[0]['vector'] for o in obs_list]
            next_v = agent.get_value(torch.tensor(np.array(f_g), dtype=torch.float32).to(device), torch.tensor(np.array(f_v), dtype=torch.float32).to(device)).reshape(1, -1)
            
        bg, bv, ba, blp, bval = torch.stack(b_grid).view(-1,3,7,7), torch.stack(b_vector).view(-1,25), torch.stack(b_acts).view(-1), torch.stack(b_logprobs).view(-1), torch.stack(b_values)
        br, bd = torch.stack(b_rewards), torch.stack(b_dones)
        
        advantages = torch.zeros_like(br).to(device)
        lastgaelam, gamma, gae_lambda = 0, 0.99, 0.95
        for t in reversed(range(num_steps)):
            nextnonterminal = 1.0 - bd[t] if t == num_steps - 1 else 1.0 - bd[t+1]
            nextvalues = next_v.view(-1) if t == num_steps - 1 else bval[t+1]
            delta = br[t] + gamma * nextvalues * nextnonterminal - bval[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        returns, b_adv = (advantages + bval).reshape(-1), advantages.reshape(-1)
        
        # --- STAGE 3: Optimization ---
        if global_step % 20480 < num_envs:
            log(f">>> [HEARTBEAT] Optimization Phase...")
        inds = np.arange(batch_size)
        for _ in range(4):
            np.random.shuffle(inds)
            for s in range(0, batch_size, mini_batch):
                mb = inds[s:s+mini_batch]
                _, nlp, ent, nv = agent.get_action_and_value(bg[mb], bv[mb], ba[mb])
                ratio = (nlp - blp[mb]).exp()
                mb_a = (b_adv[mb] - b_adv[mb].mean())/(b_adv[mb].std() + 1e-8)
                # V11.0 Smoothing: Tighter clip_range (0.15) to reduce oscillations
                pg_loss = torch.max(-mb_a * ratio, -mb_a * torch.clamp(ratio, 0.85, 1.15)).mean()
                v_loss = 0.5 * ((nv.view(-1) - returns[mb])**2).mean()
                # V10.6: Dynamic Entropy to consolidated policy
                optimizer.zero_grad(); (pg_loss - ent_coef * ent.mean() + 0.5 * v_loss).backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.3); optimizer.step()
                
        # Logging
        if global_step % 10240 < num_envs:
            avg_rew = np.mean(ep_rewards[-50:]) if ep_rewards else 0
            perc = (global_step/total_timesteps)*100
            log(f">>> [Progress {perc:.1f}%] Step: {global_step} | Ep_Rew: {avg_rew:.2f} | Info: LR={current_lr:.2e}")
            if global_step % 204800 < num_envs:
                torch.save(agent.state_dict(), checkpoint_path)
                sp_manager.add_model(agent.state_dict(), f"ppo_v4_step_{global_step}")

    torch.save(agent.state_dict(), checkpoint_path)
    log(f"PPO V4.3 Training Finished. Model: {checkpoint_path}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--single", action="store_true"); p.add_argument("--load", type=str); p.add_argument("--steps", type=int, default=1000000)
    args = p.parse_args(); num = 1 if args.single else 4
    # V11.2: num_envs defined by hardware (A6000 scale)
    # Reduced from 32 to 16 to avoid I/O Deadlock during model loading
    n_envs = 8 if args.single else 16
    # V11.2 saturation training: 4.0M steps for Battle Phase
    steps = args.steps if args.single else 4000000
    ckpt = "agent/checkpoints/ppo_best.pth" if args.single else "agent/checkpoints/ppo_battle_best.pth"
    
    train_ppo(num_envs=n_envs, num_snakes=num, total_timesteps=steps, load_path=args.load, checkpoint_path=ckpt)
