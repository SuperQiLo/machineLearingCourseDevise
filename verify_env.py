
import os
import sys
import numpy as np

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.multi_snake_env import MultiSnakeEnv

def test_env_dims():
    print("Initializing environment...")
    env = MultiSnakeEnv(num_snakes=4, width=20, height=20)
    
    print("Resetting environment...")
    obs_list = env.reset()
    
    print(f"Num snakes: {len(obs_list)}")
    print(f"Observation shape: {obs_list[0].shape}")
    print(f"Observation dtype: {obs_list[0].dtype}")
    print(f"Observation max value: {np.max(obs_list[0])}")
    print(f"Observation min value: {np.min(obs_list[0])}")

    expected_shape = (3, 84, 84)
    if obs_list[0].shape != expected_shape:
        print(f"ERROR: Expected shape {expected_shape}, got {obs_list[0].shape}")
        sys.exit(1)
    
    print("Stepping environment...")
    actions = [0] * 4
    next_obs, rewards, dones, info = env.step(actions)
    
    print("Step successful.")
    print(f"Reward shape: {len(rewards)}")
    print(f"Done shape: {len(dones)}")
    
    print("Environment Verification Passed!")

if __name__ == "__main__":
    test_env_dims()
