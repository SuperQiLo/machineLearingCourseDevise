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

class GameServer:
    def __init__(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((HOST, PORT))
        self.server.listen(MAX_PLAYERS + 10)  # Allow spectators
        print(f"Server started on {HOST}:{PORT}")
        
        # Game State
        self.env = BattleSnakeEnv(config=BattleSnakeConfig(num_snakes=MAX_PLAYERS))
        self.env.reset()
        
        # Clients: {conn: player_id} (player_id -1 for spectator)
        self.clients: Dict[socket.socket, int] = {}
        self.player_inputs = [0] * MAX_PLAYERS
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
            del self.clients[conn]
            conn.close()

    def handle_client(self, conn, addr):
        print(f"New connection from {addr}")
        
        # Assign Player ID
        player_id = -1
        with self.lock:
            occupied = set(self.clients.values())
            for i in range(MAX_PLAYERS):
                if i not in occupied:
                    player_id = i
                    break
        
        self.clients[conn] = player_id
        
        # Send Welcome Message
        welcome = {
            "type": "WELCOME",
            "player_id": player_id,
            "width": self.env.width,
            "height": self.env.height
        }
        conn.sendall((json.dumps(welcome) + "\n").encode('utf-8'))
        
        # Listen for actions
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
                        if msg["type"] == "ACTION" and player_id != -1:
                            action = msg["action"]
                            self.player_inputs[player_id] = action
                    except:
                        pass
            except:
                break
        
        self.remove_client(conn)

    def run(self):
        # Accept thread
        threading.Thread(target=self.accept_loop, daemon=True).start()
        
        # Game Loop
        print("Game loop started")
        while True:
            start_time = time.time()
            
            with self.lock:
                # Step Environment
                obs, rewards, dones, info = self.env.step(self.player_inputs)
                
                state = {
                    "type": "STATE",
                    "snakes": self.env.snakes,
                    "food": self.env.foods, # Changed to list
                    "dead": self.env.dead,
                    "scores": self.env.scores
                }
                
                # Reset input to 0 (Straight) ? 
                # Or keep last action? Snake standard is keep last direction.
                # But Step takes relative action (0=Straight).
                # So we reset input to 0.
                self.player_inputs = [0] * MAX_PLAYERS
                
                if all(dones):
                    self.env.reset()
            
            self.broadcast(state)
            
            # FPS control (10 FPS)
            dt = time.time() - start_time
            sleep_time = max(0, 0.1 - dt)
            time.sleep(sleep_time)

    def accept_loop(self):
        while True:
            conn, addr = self.server.accept()
            threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":
    server = GameServer()
    server.run()
