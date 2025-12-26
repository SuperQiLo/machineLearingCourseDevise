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
        self.setWindowTitle("Snake AI Battle - Neon Client")
        self.resize(1000, 700)
        
        # Neon Style Sheet
        self.setStyleSheet("""
            QMainWindow { background-color: #1a1a1a; }
            QWidget { background-color: #1a1a1a; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
            QFrame#Settings { 
                background-color: #252525; 
                border-right: 2px solid #333;
                border-radius: 0px;
            }
            QLabel { font-weight: bold; color: #00d4ff; font-size: 13px; }
            QLineEdit, QComboBox { 
                background-color: #333; 
                border: 1px solid #444; 
                border-radius: 4px; 
                padding: 5px; 
                color: #fff;
            }
            QPushButton { 
                background-color: #007acc; 
                color: white; 
                border-radius: 4px; 
                font-weight: bold; 
                padding: 8px;
            }
            QPushButton:hover { background-color: #0098ff; }
            QPushButton#ReadyBtn { background-color: #28a745; }
            QPushButton#ReadyBtn:checked { background-color: #dc3545; }
        """)
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Left Panel
        self.settings_panel = QFrame()
        self.settings_panel.setObjectName("Settings")
        self.settings_panel.setFixedWidth(280)
        settings_layout = QVBoxLayout(self.settings_panel)
        settings_layout.setContentsMargins(20, 20, 20, 20)
        settings_layout.setSpacing(15)
        
        title = QLabel("NEON BATTLE v7.2")
        title.setStyleSheet("font-size: 18px; color: #ff00ff; margin-bottom: 10px;")
        settings_layout.addWidget(title)
        
        self.host_input = QLineEdit(DEFAULT_HOST)
        settings_layout.addWidget(QLabel("SERVER HOST"))
        settings_layout.addWidget(self.host_input)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Human", "AI", "Spectator"])
        self.mode_combo.currentTextChanged.connect(self.toggle_ai_config)
        settings_layout.addWidget(QLabel("CONTROL MODE"))
        settings_layout.addWidget(self.mode_combo)
        
        # AI Config Container
        self.ai_config_widget = QWidget()
        ai_cfg_layout = QVBoxLayout(self.ai_config_widget)
        ai_cfg_layout.setContentsMargins(0, 0, 0, 0)
        ai_cfg_layout.setSpacing(10)
        
        self.algo_combo = QComboBox()
        # V7.2: Strip "AGENT" for cleaner look
        clean_algos = [name.replace("agent", "").upper() for name in AGENTS.keys()]
        self.algo_combo.addItems(clean_algos)
        ai_cfg_layout.addWidget(QLabel("ALGORITHM"))
        ai_cfg_layout.addWidget(self.algo_combo)
        
        ai_cfg_layout.addWidget(QLabel("MODEL CHECKPOINT"))
        model_row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.refresh_models()
        model_row.addWidget(self.model_combo)
        btn_browse = QPushButton("...")
        btn_browse.setFixedWidth(30)
        btn_browse.clicked.connect(self.browse_model)
        model_row.addWidget(btn_browse)
        ai_cfg_layout.addLayout(model_row)
        
        settings_layout.addWidget(self.ai_config_widget)
        self.ai_config_widget.setVisible(False)
        
        # Actions
        settings_layout.addStretch()
        self.btn_connect = QPushButton("CONNECT TO SERVER")
        self.btn_connect.setFixedHeight(45)
        self.btn_connect.clicked.connect(self.connect_to_server)
        settings_layout.addWidget(self.btn_connect)
        
        self.btn_ready = QPushButton("SET READY")
        self.btn_ready.setObjectName("ReadyBtn")
        self.btn_ready.setCheckable(True)
        self.btn_ready.setFixedHeight(45)
        self.btn_ready.setVisible(False)
        self.btn_ready.clicked.connect(self.send_ready)
        settings_layout.addWidget(self.btn_ready)
        
        self.status_label = QLabel("Ready to connect...")
        self.status_label.setStyleSheet("color: #888;")
        settings_layout.addWidget(self.status_label)
        
        # Right Panel
        self.board = GameRenderer()
        main_layout.addWidget(self.settings_panel)
        main_layout.addWidget(self.board, 1)
        
        # Logic
        self.net_thread = None
        self.player_id = -1
        self.agent = None
        self.ai_timer = QTimer()
        self.ai_timer.timeout.connect(self.ai_step)
        self.dummy_env = BattleSnakeEnv(BattleSnakeConfig(num_snakes=4))
        self.server_state = "WAITING"
        self.countdown = 0

    def toggle_ai_config(self, mode):
        self.ai_config_widget.setVisible(mode == "AI")

    def refresh_models(self):
        self.model_combo.clear()
        root = Path(__file__).parent.parent
        files = list((root / "agent" / "checkpoints").glob("*.pth"))
        for f in files: self.model_combo.addItem(f.name, str(f))

    def browse_model(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Model", str(Path(__file__).parent.parent), "Models (*.pth)")
        if file:
            p = Path(file); self.model_combo.insertItem(0, p.name, str(p))
            self.model_combo.setCurrentIndex(0)

    def connect_to_server(self):
        if self.net_thread:
            self.net_thread.stop(); self.net_thread = None
            self.btn_connect.setText("CONNECT TO SERVER")
            self.btn_ready.setVisible(False)
            self.status_label.setText("Disconnected")
            self.ai_timer.stop()
            return
            
        if self.mode_combo.currentText() == "AI":
            path_str = self.model_combo.currentData() or self.model_combo.currentText()
            path = Path(path_str)
            if not path.exists():
                QMessageBox.critical(self, "Error", "Model not found."); return
            algo = self.algo_combo.currentText().lower()
            if not algo.endswith("agent"): algo += "agent"
            try:
                self.agent = get_agent(algo, 25, str(path))
                self.ai_timer.start(50)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Load failed: {e}"); return
        
        self.net_thread = NetworkThread(self.host_input.text(), DEFAULT_PORT)
        self.net_thread.msg_received.connect(self.handle_message)
        self.net_thread.disconnected.connect(self.on_disconnect)
        self.net_thread.start()
        self.btn_connect.setText("DISCONNECT")
        self.btn_ready.setVisible(True)
        self.btn_ready.setChecked(False)
        self.btn_ready.setText("SET READY")

    def send_ready(self, checked):
        if self.net_thread:
            self.net_thread.send({"type": "READY", "ready": checked})
            self.btn_ready.setText("CANCEL READY" if checked else "SET READY")

    def on_disconnect(self):
        self.net_thread = None
        self.btn_connect.setText("CONNECT TO SERVER")
        self.btn_ready.setVisible(False)
        self.ai_timer.stop()
        self.status_label.setText("Connection Lost")

    def handle_message(self, msg):
        typ = msg.get("type")
        if typ == "WELCOME":
            self.player_id = msg.get("player_id")
            self.board.grid_size = msg.get("width")
            self.status_label.setText(f"Connected as P{self.player_id}")
        elif typ == "SYNC":
            self.server_state = msg.get("state")
            self.countdown = msg.get("countdown", 0)
            
            # Sync Ready State (V7.4 Fix)
            r_list = msg.get("ready_list", [])
            if self.player_id != -1:
                is_ready_on_server = self.player_id in r_list
                # If state is WAITING and we are out of sync, force toggle button
                if self.server_state == "WAITING" and self.btn_ready.isChecked() != is_ready_on_server:
                    self.btn_ready.blockSignals(True)
                    self.btn_ready.setChecked(is_ready_on_server)
                    self.btn_ready.setText("CANCEL READY" if is_ready_on_server else "SET READY")
                    self.btn_ready.blockSignals(False)

            # Prepare Renderer
            self.board.update_state(msg.get("snakes", []), msg.get("food", []), msg.get("dead", []), self.player_id)
            self.board.countdown = self.countdown if self.server_state == "COUNTDOWN" else 0
            
            # Lobby Status
            r_list = msg.get("ready_list", [])
            p_list = msg.get("player_list", [])
            scores = msg.get("scores", [])
            
            txt = f"SERVER: {self.server_state}\n"
            if self.server_state == "WAITING":
                txt += f"READY: {len(r_list)}/{len(p_list)}\n\n"
            elif self.server_state == "COUNTDOWN":
                txt += f"STARTING IN {self.countdown}...\n\n"
            
            for pid in p_list:
                mark = "[✓] " if pid in r_list else "[  ] "
                prefix = "★ " if pid == self.player_id else "  "
                dead = " (DEAD)" if pid < len(self.board.dead) and self.board.dead[pid] else ""
                s = scores[pid] if pid < len(scores) else 0
                txt += f"{prefix}{mark}P{pid}: {s}{dead}\n"
            self.status_label.setText(txt)
            
            # AI Logic
            if self.server_state == "PLAYING" and self.mode_combo.currentText() == "AI":
                self.dummy_env.snakes = [ [tuple(x) for x in s] for s in self.board.snakes ]
                self.dummy_env.foods = [tuple(f) for f in self.board.food]
                self.dummy_env.dead = self.board.dead
                self.dummy_env.dash_cooldowns = msg.get("dash_cooldowns", [0]*4)

    def keyPressEvent(self, event):
        if self.mode_combo.currentText() != "Human" or self.server_state != "PLAYING": return
        key = event.key()
        if key == Qt.Key.Key_Space:
            self.net_thread.send({"type": "ACTION", "action": 3}); return
        
        target = {Qt.Key.Key_Up: Direction.UP, Qt.Key.Key_Down: Direction.DOWN, 
                  Qt.Key.Key_Left: Direction.LEFT, Qt.Key.Key_Right: Direction.RIGHT}.get(key)
        
        if target is not None and self.player_id != -1 and self.player_id < len(self.board.snakes):
            s = self.board.snakes[self.player_id]
            if len(s) >= 2:
                h, n = s[0], s[1]
                curr = Direction.UP
                if h[0] == n[0] and h[1] > n[1]: curr = Direction.DOWN
                elif h[0] < n[0] and h[1] == n[1]: curr = Direction.LEFT
                elif h[0] > n[0] and h[1] == n[1]: curr = Direction.RIGHT
                
                action = 0
                if (curr - 1) % 4 == target: action = 1
                elif (curr + 1) % 4 == target: action = 2
                elif abs(curr - target) == 2: return
                self.net_thread.send({"type": "ACTION", "action": action})

    def ai_step(self):
        if self.player_id == -1 or not self.agent or self.server_state != "PLAYING": return
        if self.player_id >= len(self.board.snakes) or self.board.dead[self.player_id]: return
        
        self.dummy_env.directions = []
        for s in self.dummy_env.snakes:
            if len(s) >= 2:
                h, n = s[0], s[1]
                if h[0] == n[0] and h[1] < n[1]: self.dummy_env.directions.append(Direction.UP)
                elif h[0] == n[0] and h[1] > n[1]: self.dummy_env.directions.append(Direction.DOWN)
                elif h[0] < n[0] and h[1] == n[1]: self.dummy_env.directions.append(Direction.LEFT)
                else: self.dummy_env.directions.append(Direction.RIGHT)
            else: self.dummy_env.directions.append(Direction.UP)
            
        obs = self.dummy_env._get_agent_obs(self.player_id)
        self.net_thread.send({"type": "ACTION", "action": self.agent.act(obs)})

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
