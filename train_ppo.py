"""
Unified Multi-Agent PPO Training.
Refactored to support Curriculum Learning (Single -> Battle).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time
from typing import List, Tuple, Dict, Optional

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent))

from env.battle_snake_env import BattleSnakeEnv, BattleSnakeConfig
from agent.ppo import ActorCritic

def make_env(num_snakes, grid_size, seed=None):
    def thunk():
        # Using competitive settings or training settings based on phase
        env = BattleSnakeEnv(BattleSnakeConfig(
            width=grid_size, height=grid_size, 
            num_snakes=num_snakes,
            min_food=num_snakes, # Abundant food for faster learning
            max_steps=500
        ), seed=seed)
        return env
    return thunk

class VectorizedEnv:
    """Simple wrapper for parallel environments"""
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
    
    def reset(self):
        return [env.reset() for env in self.envs]
    
    def step(self, actions_list):
        # actions_list: [ [act_env0_s0, ...], [act_env1_s0, ...] ]
        results = [env.step(a) for env, a in zip(self.envs, actions_list)]
        obs_n, rew_n, done_n, info_n = zip(*results)
        return list(obs_n), list(rew_n), list(done_n), list(info_n)

def train_ppo(num_envs=8, num_snakes=4, total_timesteps=1_000_000, 
              load_path=None, checkpoint_path="agent/checkpoints/ppo_battle_best.pth"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path("agent/checkpoints").mkdir(parents=True, exist_ok=True)
    
    # Init Envs
    envs = VectorizedEnv([make_env(num_snakes, 20, i) for i in range(num_envs)])
    
    # Init Agent
    agent = ActorCritic(input_dim=15, hidden_dim=256, action_dim=3).to(device)
    
    if load_path and Path(load_path).exists():
        print(f">>> Loading PPO weights from {load_path}...")
        agent.load_state_dict(torch.load(load_path, map_location=device, weights_only=True))
        
    optimizer = optim.Adam(agent.parameters(), lr=2.5e-4, eps=1e-5)
    
    # Hyperparams
    num_steps = 128 
    batch_size = num_envs * num_snakes * num_steps
    
    print(f"Starting PPO Training: {num_envs} Envs x {num_snakes} Snakes.")
    
    obs_list = envs.reset()
    next_obs = torch.tensor(np.array(obs_list), dtype=torch.float32).to(device)
    
    global_step = 0
    best_rew = -float('inf')
    
    while global_step < total_timesteps:
        b_obs, b_acts, b_logprobs, b_rewards, b_dones, b_values = [], [], [], [], [], []
        
        # 1. Collection
        for _ in range(num_steps):
            global_step += num_envs * num_snakes
            flat_obs = next_obs.view(-1, 15)
            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(flat_obs)
                
            b_obs.append(flat_obs)
            b_acts.append(action)
            b_logprobs.append(logprob)
            b_values.append(value.flatten())
            
            actions_reshaped = action.view(num_envs, num_snakes).cpu().numpy().tolist()
            next_obs_list, rewards, dones, _ = envs.step(actions_reshaped)
            
            b_rewards.append(torch.tensor(rewards, device=device).view(-1))
            b_dones.append(torch.tensor(dones, device=device).view(-1))
            next_obs = torch.tensor(np.array(next_obs_list), dtype=torch.float32).to(device)

        # 2. Advantage Estimation (GAE)
        with torch.no_grad():
            next_value = agent.get_value(next_obs.view(-1, 15)).reshape(1, -1)
            
        b_obs, b_acts, b_logprobs = torch.stack(b_obs), torch.stack(b_acts), torch.stack(b_logprobs)
        b_rewards, b_dones, b_values = torch.stack(b_rewards), torch.stack(b_dones), torch.stack(b_values)
        
        advantages = torch.zeros_like(b_rewards).to(device)
        lastgaelam = 0
        gamma, gae_lambda = 0.99, 0.95
        
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - b_dones[t]
                nextvalues = next_value.view(-1)
            else:
                nextnonterminal = 1.0 - b_dones[t+1]
                nextvalues = b_values[t+1]
                
            delta = b_rewards[t] + gamma * nextvalues * nextnonterminal - b_values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            
        returns = advantages + b_values
        
        # 3. Optimization
        b_obs, b_acts, b_logprobs = b_obs.reshape(-1, 15), b_acts.reshape(-1), b_logprobs.reshape(-1)
        b_advantages, b_returns = advantages.reshape(-1), returns.reshape(-1)
        
        inds = np.arange(batch_size)
        for _ in range(4): # epochs
            np.random.shuffle(inds)
            for start in range(0, batch_size, 32):
                end = start + 32
                mb_inds = inds[start:end]
                
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_acts[mb_inds])
                ratio = (newlogprob - b_logprobs[mb_inds]).exp()
                
                mb_adv = b_advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                
                pg_loss = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 0.8, 1.2)).mean()
                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds])**2).mean()
                
                loss = pg_loss - 0.01 * entropy.mean() + 0.5 * v_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
                
        # Logging
        avg_reward = b_rewards.mean().item()
        if global_step % 10240 == 0:
            print(f"Step {global_step} | Avg Reward: {avg_reward:.3f}")
            
        # Save Best
        if avg_reward > best_rew and global_step > 20000:
            best_rew = avg_reward
            torch.save(agent.state_dict(), checkpoint_path)

    print(f"PPO Training Finished. Final model: {checkpoint_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true", help="Train single snake mode")
    parser.add_argument("--load", type=str, default=None, help="Pretrained weights path")
    parser.add_argument("--steps", type=int, default=1_000_000)
    args = parser.parse_args()
    
    num = 1 if args.single else 4
    ckpt = "agent/checkpoints/ppo_best.pth" if args.single else "agent/checkpoints/ppo_battle_best.pth"
    
    train_ppo(num_snakes=num, total_timesteps=args.steps, load_path=args.load, checkpoint_path=ckpt)
