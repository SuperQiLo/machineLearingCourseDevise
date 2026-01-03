"""net/game_client.py

【中文说明】
PyQt6 网络客户端：连接 `net/game_server.py`，渲染服务器广播的状态，并发送玩家输入。

- 支持三种模式：Human / AI / Spectator。
- AI 模式下会通过 `agent.get_agent()` 动态加载模型，并定时发送 ACTION。
- 协议：JSON 行协议（每条消息以 `\n` 结尾）。

历史说明：本文件早期包含英文模块介绍，已统一为中文说明。
"""

import sys
import json
import socket
import argparse
from pathlib import Path
import glob
import random

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QComboBox, QFileDialog, QMessageBox, QFrame)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread

# 引入项目模块（把项目根目录加入 sys.path）
sys.path.append(str(Path(__file__).parent.parent))
from env.battle_snake_env import BattleSnakeEnv, BattleSnakeConfig, Direction
from utils.renderer import GameRenderer
from agent import AGENTS, get_agent

# 默认连接配置
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5555

class NetworkThread(QThread):
    msg_received = pyqtSignal(dict)
    disconnected = pyqtSignal()
    
    def __init__(self, host, port):
        """网络收发线程（避免阻塞 UI 线程）。"""
        super().__init__()
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.running = True
        self.connected = False
        
    def run(self):
        """连接服务器并持续接收消息；每收到一条 JSON 行就发射 `msg_received`。"""
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
        """发送一条 JSON 消息给服务器。"""
        if self.connected:
            try:
                msg = json.dumps(data) + "\n"
                self.sock.sendall(msg.encode('utf-8'))
            except:
                self.running = False

    def stop(self):
        """请求线程停止并关闭 socket。"""
        self.running = False
        self.sock.close()

class MainWindow(QMainWindow):
    def __init__(self):
        """客户端主窗口：左侧配置面板 + 右侧棋盘渲染。"""
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
        
        self.name_input = QLineEdit(f"Player_{random.randint(100,999)}")
        self.name_input.editingFinished.connect(self.sync_name_now)
        settings_layout.addWidget(QLabel("YOUR NICKNAME"))
        settings_layout.addWidget(self.name_input)
        
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

        # Connection for real-time AI updates
        self.algo_combo.currentIndexChanged.connect(self.setup_ai)
        self.model_combo.editTextChanged.connect(self.setup_ai)
        self.model_combo.currentIndexChanged.connect(self.setup_ai)
        
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
        self.board.clicked.connect(self.request_reset)
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
        self.setup_ai()

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

    def setup_ai(self):
        """Dynamic AI Initialization/Reload."""
        # Stop timer first if we are not in AI mode or not connected
        if self.mode_combo.currentText() != "AI" or not self.net_thread:
            self.ai_timer.stop()
            self.agent = None
            return

        # Load Agent
        path_str = self.model_combo.currentData() or self.model_combo.currentText()
        path = Path(path_str)
        if not path.exists():
            self.status_label.setText(f"AI ERROR: Model not found at {path.name}")
            return
            
        algo = self.algo_combo.currentText().lower()
        if not algo.endswith("agent"): algo += "agent"
        
        try:
            # Re-load only if needed or just replace for safety
            self.agent = get_agent(algo, 28, str(path))
            if not self.ai_timer.isActive():
                self.ai_timer.start(50)
            self.status_label.setText(f"AI LOADED: {algo.upper()}")
        except Exception as e:
            self.status_label.setText(f"AI LOADED FAILED: {e}")
            self.agent = None
            self.ai_timer.stop()

    def connect_to_server(self):
        if self.net_thread:
            self.net_thread.stop(); self.net_thread = None
            self.btn_connect.setText("CONNECT TO SERVER")
            self.btn_ready.setVisible(False)
            self.status_label.setText("Disconnected")
            self.ai_timer.stop()
            return
            
        self.net_thread = NetworkThread(self.host_input.text(), DEFAULT_PORT)
        self.net_thread.msg_received.connect(self.handle_message)
        self.net_thread.disconnected.connect(self.on_disconnect)
        self.net_thread.start()
        
        # Send Name immediately after connection
        QTimer.singleShot(500, lambda: self.net_thread.send({"type": "JOIN", "name": self.name_input.text()}))
        
        self.btn_connect.setText("DISCONNECT")
        self.btn_ready.setVisible(True)
        self.btn_ready.setChecked(False)
        self.btn_ready.setText("SET READY")
        
        # Trigger AI setup after connection is established
        self.setup_ai()

    def send_ready(self, checked):
        if self.net_thread:
            self.net_thread.send({"type": "READY", "ready": checked})
            self.btn_ready.setText("CANCEL READY" if checked else "SET READY")

    def request_reset(self):
        if self.net_thread and self.server_state == "RESULT":
            self.net_thread.send({"type": "RESET"})

    def sync_name_now(self):
        """Send current nickname to server."""
        if self.net_thread:
            self.net_thread.send({"type": "JOIN", "name": self.name_input.text()})

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
            self.board.countdown = self.countdown if self.server_state in ["COUNTDOWN", "RESULT"] else 0
            self.board.server_state = self.server_state
            self.board.winner_id = msg.get("winner_id", -1)
            self.board.player_names = msg.get("names", [])
            
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
                self.dummy_env.dash_durations = msg.get("dash_durations", [0]*4)

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
