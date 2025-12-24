"""
Snake AI Battle GUI V3.
Enhanced with HUD, Dash Support, and V3 Observations.
"""

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QFrame)
from PyQt6.QtCore import QTimer, Qt

# Import Refactored Modules
from env.battle_snake_env import BattleSnakeEnv, BattleSnakeConfig, Direction, Action
from agent import get_agent
from utils.renderer import GameRenderer, COLORS_SNAKE

class GameWindow(QMainWindow):
    def __init__(self, mode="battle", algo="dqn", model_path=None, fps=10, 
                 grid_size=20, human=False, food_count=None):
        super().__init__()
        self.setWindowTitle(f"Battle Snake V3 - {mode.upper()} [{algo.upper()}]")
        self.resize(1000, 700) # Ensure window is large enough initially
        self.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4; font-family: 'Segoe UI', sans-serif;")
        
        # 1. Setup Env (Unified)
        num_snakes = 1 if mode == "single" else 4
        if food_count is None:
            food_count = max(2, num_snakes)
            
        self.env = BattleSnakeEnv(BattleSnakeConfig(
            width=grid_size, height=grid_size, 
            num_snakes=num_snakes,
            min_food=food_count,
            max_steps=2000
        ))
        
        # 2. Main Layout (Horizontal: Game | HUD)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Game Board
        self.board = GameRenderer(self, grid_size)
        layout.addWidget(self.board, stretch=4)
        
        # HUD Panel
        self.hud = QFrame()
        self.hud.setFixedWidth(200)
        self.hud.setStyleSheet("background-color: #313244; border-radius: 10px; padding: 10px;")
        hud_layout = QVBoxLayout(self.hud)
        
        title = QLabel("LEADERBOARD")
        title.setStyleSheet("font-weight: bold; font-size: 16px; color: #f5c2e7; margin-bottom: 10px;")
        hud_layout.addWidget(title)
        
        self.rank_labels = []
        for i in range(num_snakes):
            lbl = QLabel(f"P{i}: 0")
            lbl.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {COLORS_SNAKE[i % len(COLORS_SNAKE)].name()};")
            hud_layout.addWidget(lbl)
            self.rank_labels.append(lbl)
            
        hud_layout.addStretch()
        
        status_title = QLabel("GAME STATUS")
        status_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #94e2d5; margin-top: 20px;")
        hud_layout.addWidget(status_title)
        
        self.status_label = QLabel("Running...")
        hud_layout.addWidget(self.status_label)
        
        layout.addWidget(self.hud, stretch=1)
        
        # 3. Agents
        self.agents = [None] * num_snakes
        self.is_human = [False] * num_snakes
        if human:
            self.is_human[0] = True
            
        # Load AI for non-human slots
        for i in range(num_snakes):
            if not self.is_human[i] and model_path:
                try:
                    # In V3, input_dim for vector part is 24
                    self.agents[i] = get_agent(algo, 24, str(model_path))
                except Exception as e:
                    print(f"Error loading agent for P{i}: {e}")

        self.fps = fps
        self.timer = QTimer()
        self.timer.timeout.connect(self.game_step)
        self.timer.start(int(1000/fps))
        
        self.obs_list = self.env.reset()
        self.human_target = None
        
    def reset_game(self):
        self.obs_list = self.env.reset()
        self.status_label.setText("Restarted")

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Space:
            self.human_target = "DASH"
            return
            
        target = None
        if key == Qt.Key.Key_Up: target = Direction.UP
        elif key == Qt.Key.Key_Down: target = Direction.DOWN
        elif key == Qt.Key.Key_Left: target = Direction.LEFT
        elif key == Qt.Key.Key_Right: target = Direction.RIGHT
        
        if target is not None:
             self.human_target = target

    def get_human_action(self, agent_idx):
         target = self.human_target
         if target == "DASH":
             self.human_target = None 
             return 3 # Action.DASH
             
         curr = self.env.directions[agent_idx]
         if target is None or target == curr: return 0
         if (curr - 1) % 4 == target: return 1
         if (curr + 1) % 4 == target: return 2
         return 0

    def game_step(self):
        actions = []
        for i in range(self.env.config.num_snakes):
            if self.env.dead[i]:
                actions.append(0)
                continue
                
            if self.is_human[i]:
                actions.append(self.get_human_action(i))
            elif self.agents[i]:
                actions.append(self.agents[i].act(self.obs_list[i]))
            else:
                actions.append(0)
        
        self.obs_list, rewards, dones, info = self.env.step(actions)
        
        # Update UI
        self.board.update_state(self.env.snakes, self.env.foods, self.env.dead, 0 if self.is_human[0] else -1)
        
        # Update HUD Ranking
        scores = info.get("scores", [0]*len(self.env.snakes))
        # Sort indices by score
        ranked_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
        
        for i, idx in enumerate(ranked_indices):
            status = "DEAD" if self.env.dead[idx] else f"LEN: {len(self.env.snakes[idx])}"
            self.rank_labels[idx].setText(f"P{idx}: {scores[idx]} ({status})")
            # Highlight current head position in leaderboard? Or just order?
            # For simplicity, just update text.
            
        if all(dones):
            self.status_label.setText("Game Over!")
            QTimer.singleShot(2000, self.reset_game)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="battle", choices=["single", "battle"])
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ppo"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--grid", type=int, default=20)
    parser.add_argument("--human", action="store_true")
    parser.add_argument("--food", type=int, default=None)
    
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    win = GameWindow(mode=args.mode, algo=args.algo, model_path=args.model, 
                     fps=args.fps, grid_size=args.grid, human=args.human, food_count=args.food)
    win.show()
    sys.exit(app.exec())
