"""
Snake Game PyQt6 Client.
Refactored to use shared Renderer and Agents.
"""

import sys
import json
import socket
import argparse
from pathlib import Path
import glob

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QComboBox, QFileDialog, QMessageBox, QFrame)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread

# Import Refactored Modules
sys.path.append(str(Path(__file__).parent.parent))
from env.battle_snake_env import BattleSnakeEnv, BattleSnakeConfig, Direction
from utils.renderer import GameRenderer
from agent import AGENTS, get_agent

# Config
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5555

class NetworkThread(QThread):
    msg_received = pyqtSignal(dict)
    disconnected = pyqtSignal()
    
    def __init__(self, host, port):
        super().__init__()
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.running = True
        self.connected = False
        
    def run(self):
        try:
            self.sock.connect((self.host, self.port))
            self.connected = True
        except Exception as e:
            print(f"Connection error: {e}")
            self.disconnected.emit()
            return

        buffer = ""
        while self.running:
            try:
                data = self.sock.recv(4096).decode('utf-8')
                if not data: break
                
                buffer += data
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line: continue
                    try:
                        self.msg_received.emit(json.loads(line))
                    except:
                        pass
            except:
                break
        
        self.connected = False
        self.disconnected.emit()
        self.sock.close()

    def send(self, data: dict):
        if self.connected:
            try:
                msg = json.dumps(data) + "\n"
                self.sock.sendall(msg.encode('utf-8'))
            except:
                self.running = False

    def stop(self):
        self.running = False
        self.sock.close()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Snake AI Battle - Client")
        self.resize(800, 600)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")
        
        # Central Widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # Left: Settings
        settings_panel = QFrame()
        settings_panel.setFrameShape(QFrame.Shape.StyledPanel)
        settings_panel.setFixedWidth(250)
        settings_layout = QVBoxLayout(settings_panel)
        
        settings_layout.addWidget(QLabel("BATTLE SNAKE (v2)"))
        
        # Inputs...
        self.host_input = QLineEdit(DEFAULT_HOST)
        settings_layout.addWidget(QLabel("Host:"))
        settings_layout.addWidget(self.host_input)
        
        self.port_input = QLineEdit(str(DEFAULT_PORT))
        settings_layout.addWidget(QLabel("Port:"))
        settings_layout.addWidget(self.port_input)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Human", "AI", "Spectator"])
        settings_layout.addWidget(QLabel("Mode:"))
        settings_layout.addWidget(self.mode_combo)
        
        # AI Config
        self.algo_combo = QComboBox()
        # Dynamically load available agents
        self.algo_combo.addItems([name.upper() for name in AGENTS.keys()])
        settings_layout.addWidget(QLabel("Algo:"))
        settings_layout.addWidget(self.algo_combo)
        
        settings_layout.addWidget(QLabel("Model:"))
        
        # Model Selection (Scan + Browse)
        model_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True) # Allow custom input
        self.refresh_models()
        model_layout.addWidget(self.model_combo)
        
        btn_refresh = QPushButton("↻")
        btn_refresh.setFixedWidth(30)
        btn_refresh.clicked.connect(self.refresh_models)
        model_layout.addWidget(btn_refresh)
        
        btn_browse = QPushButton("...")
        btn_browse.setFixedWidth(30)
        btn_browse.clicked.connect(self.browse_model)
        model_layout.addWidget(btn_browse)
        
        settings_layout.addLayout(model_layout)
        
        # Connect Button
        settings_layout.addSpacing(20)
        self.btn_connect = QPushButton("CONNECT")
        self.btn_connect.setStyleSheet("background-color: #007acc; font-weight: bold; padding: 10px;")
        self.btn_connect.clicked.connect(self.connect_to_server)
        settings_layout.addWidget(self.btn_connect)
        
        self.score_label = QLabel("Scores...")
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        settings_layout.addWidget(self.score_label)
        settings_layout.addStretch()
        
        # Right: Renderer
        self.board = GameRenderer()
        main_layout.addWidget(settings_panel)
        main_layout.addWidget(self.board, 1)
        
        # Logic
        self.net_thread = None
        self.player_id = -1
        self.agent = None
        self.ai_timer = QTimer()
        self.ai_timer.timeout.connect(self.ai_step)
        
        # Init dummy env for AI observation calculation
        self.dummy_env = BattleSnakeEnv(BattleSnakeConfig(num_snakes=4)) # Max 4

    def refresh_models(self):
        """Scan agent/checkpoints for .pth files"""
        current_text = self.model_combo.currentText()
        self.model_combo.clear()
        
        # Find path relative to project root
        root = Path(__file__).parent.parent
        checkpoints_dir = root / "agent" / "checkpoints"
        
        files = []
        if checkpoints_dir.exists():
            files = list(checkpoints_dir.glob("*.pth"))
            
        for f in files:
            # Add relative path for cleaner display, but handle full path loading logic later
            # Or just absolute? Let's store absolute but show name? 
            # Simple: Store path as user data.
            self.model_combo.addItem(f.name, str(f))
            
        # Add basic defaults if empty
        if not files:
            self.model_combo.addItem("No models found", "")
            
        if current_text:
            self.model_combo.setCurrentText(current_text)

    def browse_model(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Model", str(Path(__file__).parent.parent), "Model Files (*.pth)")
        if file:
            # Check if likely in list
            path = Path(file)
            self.model_combo.insertItem(0, path.name, str(path))
            self.model_combo.setCurrentIndex(0)

    def connect_to_server(self):
        if self.net_thread:
            self.net_thread.stop()
            self.net_thread = None
            self.btn_connect.setText("CONNECT")
            self.btn_connect.setStyleSheet("background-color: #007acc; font-weight: bold; padding: 10px;")
            self.player_id = -1
            self.board.player_id = -1
            self.board.update()
            self.ai_timer.stop()
            return
            
        # AI Setup
        if self.mode_combo.currentText() == "AI":
            # Get path from combo
            path_str = self.model_combo.currentData() 
            if not path_str: 
                # Maybe user typed manually?
                path_str = self.model_combo.currentText()
                # If typed manually relative?
                if not Path(path_str).exists():
                     # Try resolving relative to agent/checkpoints
                     root = Path(__file__).parent.parent
                     maybe_path = root / "agent" / "checkpoints" / path_str
                     if maybe_path.exists():
                         path_str = str(maybe_path)
            
            path = Path(path_str) if path_str else None
            
            if not path or not path.exists():
                QMessageBox.critical(self, "Error", f"Model not found: {path_str}")
                return

            algo = self.algo_combo.currentText().lower()
            try:
                # Use Factory
                self.agent = get_agent(algo, 15, str(path))
                self.ai_timer.start(50)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load agent: {e}")
                return
        
        self.net_thread = NetworkThread(self.host_input.text(), int(self.port_input.text()))
        self.net_thread.msg_received.connect(self.handle_message)
        self.net_thread.disconnected.connect(self.on_disconnect)
        self.net_thread.start()
        self.btn_connect.setText("DISCONNECT")
        self.btn_connect.setStyleSheet("background-color: #cc3333; font-weight: bold; padding: 10px;")


    def on_disconnect(self):
        self.net_thread = None
        self.btn_connect.setText("CONNECT")
        self.btn_connect.setStyleSheet("background-color: #007acc; font-weight: bold; padding: 10px;")
        self.ai_timer.stop()
        self.score_label.setText("Disconnected")

    def handle_message(self, msg):
        typ = msg.get("type")
        if typ == "WELCOME":
            self.player_id = msg.get("player_id")
            self.board.grid_size = msg.get("width")
        elif typ == "STATE":
            self.board.update_state(
                msg.get("snakes"),
                msg.get("food"),
                msg.get("dead"),
                self.player_id
            )
            # Scores
            scores = msg.get("scores", [])
            txt = ""
            for i, s in enumerate(scores):
                prefix = "> " if i == self.player_id else "  "
                dead = " (DEAD)" if i < len(self.board.dead) and self.board.dead[i] else ""
                txt += f"{prefix}P{i}: {s}{dead}\n"
            self.score_label.setText(txt)

    def keyPressEvent(self, event):
        if self.mode_combo.currentText() != "Human": return
        
        key = event.key()
        target = None
        if key == Qt.Key.Key_Up: target = Direction.UP
        elif key == Qt.Key.Key_Down: target = Direction.DOWN
        elif key == Qt.Key.Key_Left: target = Direction.LEFT
        elif key == Qt.Key.Key_Right: target = Direction.RIGHT
        
        if target is not None and self.player_id != -1:
             # Logic to infer relative action
             # Need current direction of OUR snake
             if self.player_id >= len(self.board.snakes): return # Not spawned yet or sync issue
             
             s = self.board.snakes[self.player_id]
             if len(s) >= 2:
                 h, n = s[0], s[1]
                 
                 curr = Direction.UP
                 if h[0] == n[0] and h[1] > n[1]: curr = Direction.DOWN
                 elif h[0] < n[0] and h[1] == n[1]: curr = Direction.LEFT
                 elif h[0] > n[0] and h[1] == n[1]: curr = Direction.RIGHT
                 
                 action = 0 # Straight
                 if (curr - 1) % 4 == target: action = 1 # Left
                 elif (curr + 1) % 4 == target: action = 2 # Right
                 elif abs(curr - target) == 2: return # No 180 turn
                 
                 self.net_thread.send({"type": "ACTION", "action": action})

    def ai_step(self):
        if self.player_id == -1 or not self.agent: return
        if self.player_id >= len(self.board.snakes): return
        if self.board.dead[self.player_id]: return # AI dead
        
        # Sync dummy env
        self.dummy_env.snakes = [ [tuple(x) for x in s] for s in self.board.snakes ]
        self.dummy_env.foods = self.board.food # renderer.py stores list in self.food
        self.dummy_env.dead = self.board.dead
        
        # Infer directions for obs
        # Note: renderer.py stores snakes simply, but environment needs directions to build Obs
        # We must re-infer directions for ALL snakes from their body positions
        for i, s in enumerate(self.dummy_env.snakes):
            if len(s) >= 2:
                h, n = s[0], s[1]
                if h[0] == n[0] and h[1] < n[1]: self.dummy_env.directions.append(Direction.UP)
                elif h[0] == n[0] and h[1] > n[1]: self.dummy_env.directions.append(Direction.DOWN)
                elif h[0] < n[0] and h[1] == n[1]: self.dummy_env.directions.append(Direction.LEFT)
                elif h[0] > n[0] and h[1] == n[1]: self.dummy_env.directions.append(Direction.RIGHT)
                else: self.dummy_env.directions.append(Direction.UP)
            else:
                self.dummy_env.directions.append(Direction.UP)
        
        # Only keep last N (since we append above inside loop, need to clear first?)
        # BattleSnakeEnv.__init__ inits directions as empty list.
        # But we reused instance. We should clear directions before appending.
        self.dummy_env.directions = []
        for i, s in enumerate(self.dummy_env.snakes):
             if len(s) >= 2:
                h, n = s[0], s[1]
                if h[0] == n[0] and h[1] < n[1]: self.dummy_env.directions.append(Direction.UP)
                elif h[0] == n[0] and h[1] > n[1]: self.dummy_env.directions.append(Direction.DOWN)
                elif h[0] < n[0] and h[1] == n[1]: self.dummy_env.directions.append(Direction.LEFT)
                elif h[0] > n[0] and h[1] == n[1]: self.dummy_env.directions.append(Direction.RIGHT)
                else: self.dummy_env.directions.append(Direction.UP)
             else:
                self.dummy_env.directions.append(Direction.UP)

        # Get Obs
        obs = self.dummy_env._get_agent_obs(self.player_id)
        action = self.agent.act(obs)
        self.net_thread.send({"type": "ACTION", "action": action})

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
