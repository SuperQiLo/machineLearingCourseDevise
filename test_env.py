from env.multi_snake_env import MultiSnakeEnv
import random
import time
import os

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    env = MultiSnakeEnv(width=10, height=10, num_snakes=2, num_food=2)
    obs = env.reset()
    
    print("Initial State:")
    env.render_text()
    time.sleep(1)
    
    for step in range(50):
        clear_screen()
        print(f"Step: {step}")
        
        # Random actions for all snakes
        # 0: Straight, 1: Left, 2: Right
        actions = [random.randint(0, 2) for _ in range(env.num_snakes)]
        
        obs, rewards, dones, info = env.step(actions)
        
        env.render_text()
        print(f"Rewards: {rewards}")
        print(f"Dones: {dones}")
        
        if all(dones):
            print("All snakes dead!")
            break
            
        time.sleep(0.2)

if __name__ == "__main__":
    main()
