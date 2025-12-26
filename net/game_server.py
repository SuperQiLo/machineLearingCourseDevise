"""
Snake Game LAN Server.
Hosts the BattleSnake environment and handles client connections.
"""

import json
import socket
import threading
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from env.battle_snake_env import BattleSnakeEnv, BattleSnakeConfig

HOST = '0.0.0.0'
PORT = 5555
MAX_PLAYERS = 4

from enum import Enum

class GameState(Enum):
    WAITING = "WAITING"
    COUNTDOWN = "COUNTDOWN"
    PLAYING = "PLAYING"

class GameServer:
    def __init__(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((HOST, PORT))
        self.server.listen(MAX_PLAYERS + 10)
        print(f"Server started on {HOST}:{PORT}")
        
        # Game State
        self.env = BattleSnakeEnv(config=BattleSnakeConfig(num_snakes=MAX_PLAYERS, dash_cooldown_steps=15))
        self.env.reset()
        
        self.state = GameState.WAITING
        self.countdown = 0
        
        # Clients: {conn: player_id}
        self.clients: Dict[socket.socket, int] = {}
        self.ready_pids = set()
        self.player_inputs = [0] * MAX_PLAYERS
        self.pid_to_idx = {} # V7.8: Mapping from player_id to env index
        self.lock = threading.Lock()
        
    def broadcast(self, data: dict):
        msg = json.dumps(data) + "\n"
        to_remove = []
        for conn in self.clients:
            try:
                conn.sendall(msg.encode('utf-8'))
            except:
                to_remove.append(conn)
        
        for conn in to_remove:
            self.remove_client(conn)
            
    def remove_client(self, conn):
        if conn in self.clients:
            pid = self.clients[conn]
            print(f"Client disconnected: Player {pid}")
            with self.lock:
                if pid in self.ready_pids:
                    self.ready_pids.remove(pid)
                del self.clients[conn]
            conn.close()

    def handle_client(self, conn, addr):
        print(f"New connection from {addr}")
        player_id = -1
        with self.lock:
            occupied = set(self.clients.values())
            for i in range(MAX_PLAYERS):
                if i not in occupied:
                    player_id = i; break
        
        self.clients[conn] = player_id
        
        # Send Welcome
        welcome = {"type": "WELCOME", "player_id": player_id, "width": self.env.width, "height": self.env.height}
        conn.sendall((json.dumps(welcome) + "\n").encode('utf-8'))
        
        buffer = ""
        while True:
            try:
                data = conn.recv(1024).decode('utf-8')
                if not data: break
                buffer += data
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line: continue
                    try:
                        msg = json.loads(line)
                        if msg["type"] == "READY" and player_id != -1:
                            with self.lock:
                                if msg["ready"]: self.ready_pids.add(player_id)
                                else: self.ready_pids.discard(player_id)
                        elif msg["type"] == "ACTION" and player_id != -1:
                            if self.state == GameState.PLAYING:
                                self.player_inputs[player_id] = msg["action"]
                    except: pass
            except: break
        self.remove_client(conn)

    def run(self):
        threading.Thread(target=self.accept_loop, daemon=True).start()
        print("Game loop started (V7.2 Formalized)")
        
        while True:
            start_time = time.time()
            with self.lock:
                active_players = [pid for pid in self.clients.values() if pid != -1]
                num_active = len(active_players)
                num_ready = len(self.ready_pids)
                
                # Logic: Lobby -> Countdown
                if self.state == GameState.WAITING:
                    # num_active only counts non-spectators (player_id != -1)
                    if num_active > 0 and num_ready == num_active:
                        self.state = GameState.COUNTDOWN
                        self.countdown = 3.0
                        
                        # V7.8 Dynamic Environment Reconstruction
                        ready_list_sorted = sorted(list(self.ready_pids))
                        self.pid_to_idx = {pid: i for i, pid in enumerate(ready_list_sorted)}
                        num_snakes = len(ready_list_sorted)
                        
                        self.env = BattleSnakeEnv(config=BattleSnakeConfig(num_snakes=num_snakes, dash_cooldown_steps=15))
                        self.env.reset()
                        self.player_inputs = [0] * MAX_PLAYERS
                        print(f">>> {num_snakes} players ready. Rebuilding env and starting countdown...")
                
                # Logic: Countdown -> Playing
                elif self.state == GameState.COUNTDOWN:
                    self.countdown -= 0.1
                    if self.countdown <= 0:
                        self.state = GameState.PLAYING
                        print(">>> GOGO! Game playing.")
                
                # Logic: Playing
                elif self.state == GameState.PLAYING:
                    # Prepare actions for the dynamic env
                    env_actions = [0] * len(self.pid_to_idx)
                    for pid, idx in self.pid_to_idx.items():
                        env_actions[idx] = self.player_inputs[pid]
                    
                    obs, rewards, dones, info = self.env.step(env_actions)
                    self.player_inputs = [0] * MAX_PLAYERS
                    
                    # End game if all active participants are done
                    if all(dones):
                        print(">>> Game ended. Back to Lobby.")
                        self.state = GameState.WAITING
                        self.ready_pids.clear()
                        self.pid_to_idx = {}
                        self.player_inputs = [0] * MAX_PLAYERS
                
                # Broadcast Full State (V7.8 Mapping)
                try:
                    snakes_full = [[] for _ in range(MAX_PLAYERS)]
                    dead_full = [True for _ in range(MAX_PLAYERS)]
                    scores_full = [0 for _ in range(MAX_PLAYERS)]
                    dash_full = [0 for _ in range(MAX_PLAYERS)]
                    
                    if self.state != GameState.WAITING:
                        for pid, idx in self.pid_to_idx.items():
                            if pid < MAX_PLAYERS:
                                if idx < len(self.env.snakes): snakes_full[pid] = self.env.snakes[idx]
                                if idx < len(self.env.dead): dead_full[pid] = self.env.dead[idx]
                                if idx < len(self.env.scores): scores_full[pid] = self.env.scores[idx]
                                if idx < len(self.env.dash_cooldowns): dash_full[pid] = self.env.dash_cooldowns[idx]

                    sync_msg = {
                        "type": "SYNC",
                        "state": self.state.value,
                        "countdown": int(self.countdown + 0.9) if self.state == GameState.COUNTDOWN else 0,
                        "ready_list": list(self.ready_pids),
                        "player_list": active_players,
                        "snakes": snakes_full,
                        "food": self.env.foods if self.state != GameState.WAITING else [],
                        "dead": dead_full,
                        "scores": scores_full,
                        "dash_cooldowns": dash_full
                    }
                    self.broadcast(sync_msg)
                except Exception as e:
                    print(f">>> Broadcast Error (Suppressed): {e}")
            
            time.sleep(max(0, 0.1 - (time.time() - start_time)))

    def accept_loop(self):
        while True:
            conn, addr = self.server.accept()
            threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":
    GameServer().run()
