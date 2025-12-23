"""
Snake Game GUI (Standalone).
Refactored to use shared Renderer and Agents.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import QTimer

# Import Refactored Modules
sys.path.append(str(Path(__file__).parent))
from env.battle_snake_env import BattleSnakeEnv, BattleSnakeConfig, Direction
from utils.renderer import GameRenderer
from agent import get_agent

class GameWindow(QMainWindow):
    def __init__(self, mode="single", algo="dqn", model_path=None, grid_size=20, fps=15, human=False, food_count=None):
        super().__init__()
        self.setWindowTitle(f"Snake Local - {mode.upper()}")
        self.resize(800, 600)
        self.setStyleSheet("background-color: #2b2b2b;")
        
        self.mode = mode
        self.human = human
        
        # 1. Setup Env (Unified)
        num_snakes = 1 if mode == "single" else 4
        # If food_count not provided, default to num_snakes
        if food_count is None:
            food_count = num_snakes
            
        self.env = BattleSnakeEnv(BattleSnakeConfig(
            width=grid_size, height=grid_size, 
            num_snakes=num_snakes,
            min_food=food_count,
            max_steps=1000
        ))
        
        # 2. Setup Renderer
        self.board = GameRenderer(self, grid_size)
        self.setCentralWidget(self.board)
        
        # 3. Setup Agent
        self.agent = None
        if not human and model_path:
             try:
                self.agent = get_agent(algo, 15, str(model_path))
             except Exception as e:
                print(f"Error loading agent: {e}")
                 
        self.fps = fps
        self.timer = QTimer()
        self.timer.timeout.connect(self.game_step)
        self.timer.start(int(1000/fps))
        
        self.reset_game()

    def reset_game(self):
        self.env.reset()
        self.update_ui()

    def update_ui(self):
        self.board.update_state(
            self.env.snakes,
            self.env.foods,
            self.env.dead,
            player_id=0 # Always highlight P0
        )

    def game_step(self):
        # 1. Collect Actions
        actions = []
        obs_list = self.env._get_observations()
        
        for i in range(self.env.config.num_snakes):
            if self.env.dead[i]:
                actions.append(0)
                continue
            
            # P0: Human or AI
            if i == 0 and self.human:
                 actions.append(self.get_human_action(i))
            else:
                 if self.agent:
                     actions.append(self.agent.act(obs_list[i]))
                 else:
                     actions.append(0) # Random/Straight if no agent
                     
        # 2. Step Env
        _, _, dones, info = self.env.step(actions)
        self.update_ui()
        
        if all(dones):
            print(f"Game Over. Scores: {self.env.scores}")
            self.reset_game()

    def keyPressEvent(self, event):
        from PyQt6.QtCore import Qt
        key = event.key()
        target = None
        if key == Qt.Key.Key_Up: target = Direction.UP
        elif key == Qt.Key.Key_Down: target = Direction.DOWN
        elif key == Qt.Key.Key_Left: target = Direction.LEFT
        elif key == Qt.Key.Key_Right: target = Direction.RIGHT
        
        if target is not None:
             self.human_target = target

    def get_human_action(self, agent_idx):
         curr = self.env.directions[agent_idx]
         target = getattr(self, 'human_target', None)
         if target is None: return 0
         if target == curr: return 0
         if (curr - 1) % 4 == target: return 1
         if (curr + 1) % 4 == target: return 2
         return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="single", choices=["single", "battle"])
    parser.add_argument("--algo", type=str, default="dqn") 
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--human", action="store_true")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--food", type=int, default=None, help="Number of food items")
    args = parser.parse_args()
    
    # Defaults
    if not args.human and args.model is None:
        if args.algo == "dqn":
            file_name = "dqn_best.pth" if args.mode == "single" else "dqn_battle_best.pth"
        else:
            file_name = "ppo_best.pth" if args.mode == "single" else "ppo_battle_best.pth"
        args.model = Path(f"agent/checkpoints/{file_name}")

    app = QApplication(sys.argv)
    win = GameWindow(args.mode, args.algo, args.model, 20, args.fps, args.human, args.food)
    win.show()
    sys.exit(app.exec())
