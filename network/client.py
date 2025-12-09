"""基于 PyQt6 的云端客户端，内置 AI 代理加载与模型选择。"""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
import socket
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from agent.base_agent import BaseAgent
from agent.rainbow_config import RainbowConfig
from agent.rainbow_agent import RainbowAgent
from env.multi_snake_env import Direction, build_observation_from_snapshot
from network.utils import normalize_port
from network.renderer import BattleRenderer


ROLE_LABELS = {
    "human": "人工操作",
    "ai": "AI 模式",
    "spectator": "观战",
}


@dataclass
class RoomInfo:
    room_id: str
    owner: str
    owner_name: str
    members: int
    joinable: bool
    config: Dict


class CloudClient:
    """轻量级 TCP 客户端，负责与服务器通信并缓存状态。"""

    def __init__(
        self,
        host: str,
        port: int,
        name: str,
        *,
        auto_render_on_join: bool = False,
        log_handler: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.name = name
        self.auto_render_on_join = auto_render_on_join
        self._log_handler = log_handler

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener: Optional[threading.Thread] = None
        self.running = False

        self.client_id: Optional[str] = None
        self.rooms: List[RoomInfo] = []
        self.state_cache: Dict[str, Dict] = {}
        self.room_states: Dict[str, Dict] = {}
        self.current_room_id: Optional[str] = None
        self.current_slot: Optional[int] = None
        self.room_members: List[Dict] = []
        self.room_can_start: bool = False
        self.room_in_progress: bool = False
        self.room_owner: Optional[str] = None
        self.lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        return self.running and self.client_id is not None

    # --------------------------------------------------------------
    # 基础通信
    # --------------------------------------------------------------
    def _log(self, message: str) -> None:
        if self._log_handler:
            try:
                self._log_handler(message)
                return
            except Exception:
                pass
        print(message)

    def connect(self) -> bool:
        if self.running:
            self._log("已连接，若需重新连接请先断开。")
            return True

        if self.socket.fileno() == -1:
            # socket 已关闭时重新创建
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            self.socket.connect((self.host, self.port))
        except OSError as exc:
            self._log(f"无法连接服务器：{exc}")
            return False

        self.running = True
        self.listener = threading.Thread(target=self._listen_loop, daemon=True)
        self.listener.start()
        self.send({"type": "hello", "name": self.name})
        self.send({"type": "list_rooms"})
        return True

    def close(self) -> None:
        self.running = False
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.socket.close()
        self.client_id = None

    def send(self, payload: Dict) -> None:
        try:
            self.socket.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        except OSError:
            self.running = False

    def _listen_loop(self) -> None:
        buffer = ""
        while self.running:
            try:
                data = self.socket.recv(4096)
                if not data:
                    break
                buffer += data.decode("utf-8")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line:
                        continue
                    try:
                        message = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    try:
                        self._handle_message(message)
                    except Exception as exc:  # noqa: BLE001
                        self._log(f"处理消息失败: {exc}")
            except OSError:
                break
        self.running = False
        self._log("连接已断开。")

    # --------------------------------------------------------------
    # 消息处理
    # --------------------------------------------------------------
    def _handle_message(self, msg: Dict) -> None:
        mtype = msg.get("type")
        if mtype == "welcome":
            self.client_id = msg.get("client_id")
            self._log(f"已连接，分配的 ID 为 {self.client_id}")
        elif mtype == "rooms":
            rooms = [
                RoomInfo(
                    room_id=entry.get("room_id"),
                    owner=entry.get("owner", "-"),
                    owner_name=entry.get("owner_name", "-"),
                    members=int(entry.get("members", 0)),
                    joinable=bool(entry.get("joinable", True)),
                    config=entry.get("config", {}),
                )
                for entry in msg.get("rooms", [])
            ]
            with self.lock:
                self.rooms = rooms
        elif mtype == "room_joined":
            self.current_room_id = msg.get("room_id")
            self.current_slot = msg.get("slot")
            self._log(f"已加入房间 {self.current_room_id}")
            self._apply_cached_room_state()
            self.send({"type": "list_rooms"})
        elif mtype == "room_closed":
            if msg.get("room_id") == self.current_room_id:
                self.current_room_id = None
                self.current_slot = None
            self._log("房间已关闭。")
        elif mtype == "start":
            room_id = msg.get("room_id")
            if room_id:
                countdown_val = float(msg.get("countdown", 3.0))
                with self.lock:
                    previous = self.state_cache.get(room_id, {})
                    cached = dict(previous) if isinstance(previous, dict) else {}
                    cached.update({
                        "room_id": room_id,
                        "countdown": countdown_val,
                        "game_over": False,
                    })
                    self.state_cache[room_id] = cached
                if room_id == self.current_room_id:
                    self.room_in_progress = True
                self._log(f"房间 {room_id} 开始倒计时 {countdown_val:.1f}s")
        elif mtype == "state":
            room_id = msg.get("room_id")
            if room_id:
                with self.lock:
                    self.state_cache[room_id] = msg
        elif mtype == "room_state":
            self._cache_room_state(msg)
        elif mtype == "error":
            self._log(f"[服务器错误] {msg.get('message')}")
        elif mtype == "log":
            notice = msg.get("message")
            if notice:
                self._log(notice)

    def _cache_room_state(self, payload: Dict) -> None:
        room_id = payload.get("room_id")
        if not room_id:
            return
        with self.lock:
            self.room_states[room_id] = payload
            if room_id != self.current_room_id:
                return
            self.room_members = payload.get("members", [])
            self.room_can_start = bool(payload.get("can_start", False))
            self.room_in_progress = bool(payload.get("in_progress", False))
            self.room_owner = payload.get("owner") or payload.get("owner_id")

    def _apply_cached_room_state(self) -> None:
        if not self.current_room_id:
            return
        with self.lock:
            payload = self.room_states.get(self.current_room_id)
            if not payload:
                return
            self.room_members = payload.get("members", [])
            self.room_can_start = bool(payload.get("can_start", False))
            self.room_in_progress = bool(payload.get("in_progress", False))
            self.room_owner = payload.get("owner") or payload.get("owner_id")

    # --------------------------------------------------------------
    # Renderer/AI 接口
    # --------------------------------------------------------------
    def get_state(self, room_id: str) -> Optional[Dict]:
        with self.lock:
            return self.state_cache.get(room_id)

    def send_action(self, action: int) -> None:
        if not self.current_room_id:
            return
        self.send({"type": "action", "value": int(action)})

    # --------------------------------------------------------------
    # 房间管理
    # --------------------------------------------------------------
    def set_role(self, role: str) -> None:
        if not self.current_room_id:
            return
        self.send({"type": "set_role", "role": role})

    def set_ready(self, ready: bool) -> None:
        if not self.current_room_id:
            return
        self.send({"type": "set_ready", "ready": bool(ready)})

    def start_game(self) -> None:
        if not self.current_room_id:
            return
        self.send({"type": "start_game"})

    def leave_room(self) -> None:
        if not self.current_room_id:
            return
        self.send({"type": "leave_room"})
        self.current_room_id = None
        self.current_slot = None
        self.room_members = []
        self.room_can_start = False
        self.room_in_progress = False
        self.room_owner = None

    def refresh_rooms(self) -> None:
        self.send({"type": "list_rooms"})

    def create_room(self, config: Dict) -> None:
        self.send(
            {
                "type": "create_room",
                "config": config,
            }
        )

    def get_members(self) -> List[Dict]:
        with self.lock:
            return list(self.room_members)

    def get_self_member(self) -> Optional[Dict]:
        with self.lock:
            for member in self.room_members:
                if member.get("client_id") == self.client_id:
                    return member
        return None


# --------------------------------------------------------------
# Agent discovery helpers
# --------------------------------------------------------------

def discover_agent_classes() -> List[Tuple[str, Type[BaseAgent]]]:
    """动态扫描 agent 目录，返回 BaseAgent 子类列表。"""

    classes: List[Tuple[str, Type[BaseAgent]]] = []
    base_path = Path(__file__).resolve().parent.parent / "agent"
    for module_info in pkgutil.iter_modules([str(base_path)]):
        if module_info.name in {"base_agent", "__pycache__"}:
            continue
        try:
            module = importlib.import_module(f"agent.{module_info.name}")
        except Exception:
            continue
        for _, obj in inspect.getmembers(module, inspect.isclass):
            try:
                if issubclass(obj, BaseAgent) and obj is not BaseAgent:
                    classes.append((obj.__name__, obj))
            except Exception:
                continue
    unique: Dict[str, Tuple[str, Type[BaseAgent]]] = {}
    for name, cls in classes:
        key = name.lower()
        if key in unique:
            continue
        unique[key] = (name, cls)
    sorted_classes = sorted(unique.values(), key=lambda x: x[0].lower())
    return sorted_classes


def instantiate_agent(agent_cls: Type[BaseAgent], grid_size: int) -> Optional[BaseAgent]:
    """尽可能自动构造代理实例，优先兼容 RainbowAgent。"""

    try:
        sig = inspect.signature(agent_cls.__init__)
        params = list(sig.parameters.values())[1:]  # 跳过 self
        kwargs: Dict = {}
        if any(p.name == "config" for p in params):
            kwargs["config"] = RainbowConfig(grid_size=grid_size)
        if agent_cls is RainbowAgent and "config" not in kwargs:
            kwargs["config"] = RainbowConfig(grid_size=grid_size)
        return agent_cls(**kwargs)
    except Exception as exc:
        print(f"无法实例化代理 {agent_cls.__name__}: {exc}")
        return None


# --------------------------------------------------------------
# PyQt UI
# --------------------------------------------------------------


class ClientWindow(QMainWindow):
    log_signal = pyqtSignal(str)
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("多蛇云客户端 (PyQt6)")
        self.setMinimumSize(980, 640)

        self.client: Optional[CloudClient] = None
        self.agent_classes = discover_agent_classes()
        self.agent_instance: Optional[BaseAgent] = None
        self.agent_checkpoint: Optional[Path] = None
        self.current_role: str = "human"
        self.renderer_thread: Optional[threading.Thread] = None
        self.renderer_running = False
        self.renderer_blocked = False
        self.last_room_in_progress = False

        self._build_ui()
        self.log_signal.connect(self._on_log_message)

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._tick_refresh)
        self.refresh_timer.start(400)

        self.ai_timer = QTimer(self)
        self.ai_timer.timeout.connect(self._drive_ai)
        self.ai_timer.start(120)

    # ------------------------- UI 构建 -------------------------
    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)

        # 连接配置
        connection_group = QGroupBox("连接")
        conn_form = QFormLayout(connection_group)
        self.host_edit = QLineEdit("127.0.0.1")
        self.port_edit = QLineEdit("5555")
        self.name_edit = QLineEdit("访客")
        self.connect_btn = QPushButton("连接")
        self.disconnect_btn = QPushButton("断开")
        self.connect_btn.clicked.connect(self._connect)
        self.disconnect_btn.clicked.connect(self._disconnect)
        conn_form.addRow("Host", self.host_edit)
        conn_form.addRow("Port", self.port_edit)
        conn_form.addRow("昵称", self.name_edit)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.connect_btn)
        btn_row.addWidget(self.disconnect_btn)
        conn_form.addRow(btn_row)
        layout.addWidget(connection_group)

        # 房间列表
        room_group = QGroupBox("房间")
        room_layout = QHBoxLayout(room_group)

        table_column = QVBoxLayout()
        self.room_table = QTableWidget(0, 4)
        self.room_table.setHorizontalHeaderLabels(["ID", "房主", "成员", "网格"])
        self.room_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table_column.addWidget(self.room_table)
        room_btns = QHBoxLayout()
        room_btns.addStretch(1)
        self.refresh_btn = QPushButton("刷新")
        self.join_btn = QPushButton("加入")
        room_btns.addWidget(self.refresh_btn)
        room_btns.addWidget(self.join_btn)
        self.refresh_btn.clicked.connect(self._refresh_rooms)
        self.join_btn.clicked.connect(self._join_selected)
        table_column.addLayout(room_btns)

        create_panel = QGroupBox("房间参数")
        create_panel.setMaximumWidth(360)
        create_panel_layout = QVBoxLayout(create_panel)
        create_form = QFormLayout()
        create_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        create_form.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
        create_form.setHorizontalSpacing(12)
        create_form.setVerticalSpacing(10)
        self.grid_spin = QSpinBox()
        self.grid_spin.setRange(4, 80)
        self.grid_spin.setValue(30)
        self.snakes_spin = QSpinBox()
        self.snakes_spin.setRange(1, 12)
        self.snakes_spin.setValue(4)
        self.food_spin = QSpinBox()
        self.food_spin.setRange(1, 64)
        self.food_spin.setValue(6)
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(10, 5000)
        self.steps_spin.setValue(1500)
        self.tick_spin = QSpinBox()
        self.tick_spin.setRange(1, 500)
        self.tick_spin.setValue(120)
        self.tick_spin.setSuffix(" ms")
        self.create_btn = QPushButton("创建房间")
        self.create_btn.clicked.connect(self._create_room)
        controls = [
            ("网格", self.grid_spin),
            ("蛇数", self.snakes_spin),
            ("食物", self.food_spin),
            ("步数", self.steps_spin),
            ("Tick", self.tick_spin),
        ]
        for label_text, widget in controls:
            widget.setFixedHeight(34)
            create_form.addRow(label_text, widget)
        create_panel_layout.addLayout(create_form)
        self.create_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.create_btn.setFixedWidth(100)
        self.create_btn.setMinimumHeight(36)
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(self.create_btn)
        btn_row.addStretch(1)
        create_panel_layout.addSpacing(16)
        create_panel_layout.addLayout(btn_row)

        room_layout.addLayout(table_column, stretch=3)
        room_layout.addWidget(create_panel, stretch=1, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(room_group)

        # 玩家列表与操作
        player_group = QGroupBox("玩家")
        player_layout = QVBoxLayout(player_group)
        self.player_table = QTableWidget(0, 5)
        self.player_table.setHorizontalHeaderLabels(["槽位", "名称", "角色", "准备", "颜色"])
        self.player_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        player_layout.addWidget(self.player_table)

        player_btns = QHBoxLayout()
        self.leave_btn = QPushButton("离开")
        self.ready_btn = QPushButton("准备/取消")
        self.start_btn = QPushButton("开始游戏")
        self.leave_btn.clicked.connect(self._leave_room)
        self.ready_btn.clicked.connect(self._toggle_ready)
        self.start_btn.clicked.connect(self._start_game)
        player_btns.addWidget(self.leave_btn)
        player_btns.addWidget(self.ready_btn)
        player_btns.addWidget(self.start_btn)
        player_layout.addLayout(player_btns)
        layout.addWidget(player_group)

        # 身份与 AI
        role_group = QGroupBox("身份/代理")
        role_layout = QHBoxLayout(role_group)
        self.role_combo = QComboBox()
        self.role_combo.addItems(["human", "ai", "spectator"])
        self.role_combo.currentTextChanged.connect(self._role_changed)
        role_layout.addWidget(QLabel("角色"))
        role_layout.addWidget(self.role_combo)

        self.agent_combo = QComboBox()
        self.agent_combo.addItems([name for name, _ in self.agent_classes] or ["<未发现代理>"])
        self.model_path_label = QLabel("未选择模型")
        load_model_btn = QPushButton("加载模型")
        load_model_btn.clicked.connect(self._load_model)
        role_layout.addWidget(QLabel("代理"))
        role_layout.addWidget(self.agent_combo)
        role_layout.addWidget(load_model_btn)
        role_layout.addWidget(self.model_path_label)
        layout.addWidget(role_group)

        # 日志
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view, stretch=1)

        self.status_label = QLabel("未连接")
        layout.addWidget(self.status_label)

        self._update_connection_buttons(False, False)
        self._update_room_buttons(False, False, False)

        self.setCentralWidget(root)

    def _update_connection_buttons(self, connected: bool, in_room: bool) -> None:
        self.connect_btn.setEnabled(not connected)
        self.disconnect_btn.setEnabled(connected)
        self.refresh_btn.setEnabled(connected)
        self.join_btn.setEnabled(connected and not in_room)
        self.create_btn.setEnabled(connected and not in_room)

    def _update_room_buttons(self, enabled: bool, can_start: bool, is_owner: bool) -> None:
        for btn in (self.leave_btn, self.ready_btn):
            btn.setEnabled(enabled)
        self.start_btn.setEnabled(enabled and can_start and is_owner)

    # ------------------------- 事件 -------------------------
    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._disconnect()
        super().closeEvent(event)

    def _connect(self) -> None:
        host = self.host_edit.text().strip() or "127.0.0.1"
        port = normalize_port(self.port_edit.text().strip(), default=5555)
        name = self.name_edit.text().strip() or "访客"

        if self.client and self.client.running:
            QMessageBox.information(self, "已连接", "当前已连接，如需重新连接请先断开。")
            return

        if self.client:
            self.client.close()

        self.client = CloudClient(host=host, port=port, name=name, log_handler=self._append_log)
        if self.client.connect():
            self._append_log("连接成功，正在获取房间列表…")
            self.status_label.setText(f"已连接 {host}:{port}")
            self._update_connection_buttons(True, False)
            self._update_room_buttons(False, False, False)
        else:
            QMessageBox.warning(self, "连接失败", "无法连接到服务器")

    def _disconnect(self) -> None:
        if self.client:
            self.client.close()
            self.client = None
        self.status_label.setText("未连接")
        self._append_log("已断开连接")
        self.room_table.setRowCount(0)
        self.player_table.setRowCount(0)
        self._update_connection_buttons(False, False)
        self._update_room_buttons(False, False, False)
        self.renderer_blocked = False
        self.last_room_in_progress = False

    def _append_log(self, message: str) -> None:
        text = str(message)
        if threading.current_thread() is threading.main_thread():
            self._on_log_message(text)
        else:
            self.log_signal.emit(text)

    def _on_log_message(self, message: str) -> None:
        if not hasattr(self, "log_view") or self.log_view is None:
            print(message)
            return
        self.log_view.append(message)

    def _refresh_rooms(self) -> None:
        if self.client:
            self.client.refresh_rooms()

    def _join_selected(self) -> None:
        if not self.client:
            QMessageBox.information(self, "提示", "请先连接服务器。")
            return
        row = self.room_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "提示", "请先选择房间。")
            return
        room_id_item = self.room_table.item(row, 0)
        if not room_id_item:
            return
        self.client.send({"type": "join_room", "room_id": room_id_item.text()})

    def _leave_room(self) -> None:
        connected = bool(self.client and self.client.running)
        if self.client:
            self.client.leave_room()
        self.renderer_blocked = False
        self.last_room_in_progress = False
        self._update_room_buttons(False, False, False)
        self._update_connection_buttons(connected, False)

    def _toggle_ready(self) -> None:
        if not self.client:
            QMessageBox.information(self, "提示", "请先连接服务器。")
            return
        member = self.client.get_self_member()
        if not member:
            QMessageBox.information(self, "提示", "请先加入房间。")
            return
        if member.get("role") == "spectator":
            QMessageBox.information(self, "提示", "观战模式无需准备。")
            return
        if member.get("role") == "ai" and self.agent_instance is None:
            QMessageBox.information(self, "提示", "AI 模式需先加载模型才能准备。")
            return
        new_state = not member.get("ready", False)
        self.client.set_ready(new_state)

    def _start_game(self) -> None:
        if not self.client:
            QMessageBox.information(self, "提示", "请先连接服务器。")
            return
        if not self.client.current_room_id:
            QMessageBox.information(self, "提示", "请先加入房间。")
            return
        if not self.client.room_can_start:
            playable = [m for m in self.client.room_members if m.get("role") != "spectator"]
            playable_non_owner = [m for m in playable if not m.get("is_owner")]
            not_ready = [m.get("name", "?") for m in playable_non_owner if not m.get("ready", False)]
            msg = "仍有玩家未准备。"
            if not_ready:
                msg += " 未准备: " + ", ".join(not_ready)
            if len(playable) < 2:
                msg = "至少需要两名参战玩家才能开始。"
            QMessageBox.information(self, "无法开始", msg)
            return
        if self.client.room_owner and self.client.client_id != self.client.room_owner:
            QMessageBox.information(self, "提示", "只有房主可以开始游戏。")
            return
        self._append_log("发送开始游戏请求…")
        try:
            self.client.start_game()
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"开始游戏发送失败: {exc}")
            QMessageBox.warning(self, "发送失败", f"无法发送开始指令: {exc}")

    def _create_room(self) -> None:
        if not self.client:
            QMessageBox.information(self, "提示", "请先连接服务器。")
            return
        if not self.client.running:
            QMessageBox.warning(self, "未连接", "连接已断开，请重新连接后再创建房间。")
            return
        tick_seconds = max(0.02, self.tick_spin.value() / 1000.0)
        config: Dict = {
            "grid_size": self.grid_spin.value(),
            "num_food": self.food_spin.value(),
            "max_steps": self.steps_spin.value(),
            "tick_rate": tick_seconds,
        }
        config["num_snakes"] = self.snakes_spin.value()
        try:
            self.client.create_room(config=config)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "创建失败", f"创建房间时出错: {exc}")
            self._append_log(f"创建房间失败: {exc}")

    def _role_changed(self, role: str) -> None:
        if role == "ai" and self.agent_instance is None:
            QMessageBox.information(self, "提示", "请先加载 AI 模型后再切换到 AI 角色。")
            if hasattr(self, "role_combo"):
                self.role_combo.blockSignals(True)
                self.role_combo.setCurrentText(self.current_role)
                self.role_combo.blockSignals(False)
            return
        self.current_role = role or self.current_role
        if self.client and role:
            self.client.set_role(role)

    def _load_model(self) -> None:
        if not self.agent_classes:
            QMessageBox.information(self, "提示", "未发现可用代理")
            return
        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", str(Path("../agent/checkpoints")), "*.pth;*.pt;*.*")
        if not file_path:
            return
        self.agent_checkpoint = Path(file_path)
        self.model_path_label.setText(self.agent_checkpoint.name)

        # 实例化代理
        agent_name = self.agent_combo.currentText()
        agent_cls = next((cls for name, cls in self.agent_classes if name == agent_name), None)
        if not agent_cls:
            QMessageBox.warning(self, "加载失败", "未找到所选代理类")
            return

        grid_guess = self.grid_spin.value()
        state = self.client.get_state(self.client.current_room_id) if self.client and self.client.current_room_id else None
        if state:
            grid_guess = int(state.get("grid", grid_guess))

        agent = instantiate_agent(agent_cls, grid_guess)
        if not agent:
            QMessageBox.warning(self, "加载失败", "无法实例化代理类")
            return

        try:
            agent.load(self.agent_checkpoint)
        except Exception as exc:
            QMessageBox.warning(self, "加载失败", f"模型载入失败: {exc}")
            return

        self.agent_instance = agent
        self._append_log(f"已加载代理 {agent_name} @ {self.agent_checkpoint.name}")

    # ------------------------- 定时任务 -------------------------
    def _tick_refresh(self) -> None:
        try:
            client = self.client
            connected = bool(client and client.running)
            in_room = bool(connected and client and client.current_room_id)
            self._update_connection_buttons(connected, in_room)

            if not client:
                self.status_label.setText("未连接")
                self.room_table.setRowCount(0)
                self.player_table.setRowCount(0)
                self._update_room_buttons(False, False, False)
                self.last_room_in_progress = False
                return

            if not connected:
                self.status_label.setText("连接已断开")
                self.room_table.setRowCount(0)
                self.player_table.setRowCount(0)
                self._update_room_buttons(False, False, False)
                self.last_room_in_progress = False
                return

            rooms = list(client.rooms)
            self.room_table.setRowCount(len(rooms))
            for row, room in enumerate(rooms):
                self._set_center_item(self.room_table, row, 0, room.room_id)
                self._set_center_item(self.room_table, row, 1, room.owner_name)
                self._set_center_item(self.room_table, row, 2, str(room.members))
                self._set_center_item(self.room_table, row, 3, str(room.config.get("grid_size", 30)))

            if client.current_room_id:
                in_progress = bool(client.room_in_progress)
                if in_progress and not self.last_room_in_progress:
                    self.renderer_blocked = False
                self.last_room_in_progress = in_progress
                self.status_label.setText(
                    f"房间 {client.current_room_id} | 槽位 {client.current_slot} | 准备: {client.room_can_start} | 进行中: {client.room_in_progress}"
                )
                self._update_player_table()
                self._maybe_launch_renderer()
            else:
                self.status_label.setText("未加入房间")
                self.player_table.setRowCount(0)
                self.last_room_in_progress = False

            room_controls_enabled = bool(client.current_room_id)
            can_start = bool(client.room_can_start) if room_controls_enabled else False
            is_owner = bool(client.room_owner and client.client_id == client.room_owner) if room_controls_enabled else False
            self._update_room_buttons(room_controls_enabled, can_start, is_owner)
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"刷新界面时出错: {exc}")

    def _drive_ai(self) -> None:
        try:
            if not self.client or not self.agent_instance:
                return
            if self.role_combo.currentText() != "ai":
                return
            if not self.client.current_room_id or self.client.current_slot is None:
                return
            if not self.client.room_in_progress:
                return
            state = self.client.get_state(self.client.current_room_id)
            if not state:
                return

            try:
                obs = self._state_to_observation(state, self.client.current_slot)
            except Exception as exc:
                self._append_log(f"构造观测失败: {exc}")
                return

            action = int(self.agent_instance.act(obs, epsilon=0.0))
            self.client.send_action(action)
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"AI 驱动出错: {exc}")

    def _maybe_launch_renderer(self) -> None:
        if self.renderer_running:
            return
        if not self.client or not self.client.current_room_id:
            return
        if not self.client.room_in_progress:
            return
        if self.renderer_blocked:
            return
        # find role and slot
        role = self.role_combo.currentText() if hasattr(self, "role_combo") else "human"
        member = self.client.get_self_member()
        if member:
            role = member.get("role", role)
        slot = self.client.current_slot
        if slot is None and member is not None:
            try:
                slot = int(member.get("slot")) if member.get("slot") is not None else None
            except (TypeError, ValueError):
                slot = None
        role_at_launch = role

        def _run() -> None:
            self.renderer_running = True
            try:
                renderer = BattleRenderer(self.client, self.client.current_room_id or "", slot, role=role)
                renderer.run()
            except Exception as exc:  # noqa: BLE001
                self._append_log(f"渲染器启动失败: {exc}")
            finally:
                self.renderer_running = False
                self.renderer_blocked = True
                self.renderer_thread = None
                client = self.client
                if not client or not client.current_room_id:
                    return
                if role_at_launch not in {"human", "ai"}:
                    return
                try:
                    in_progress = bool(client.room_in_progress)
                except Exception:
                    in_progress = False
                target_role = "spectator" if in_progress else role_at_launch
                if target_role == self.current_role:
                    return
                try:
                    client.set_role(target_role)
                    self.current_role = target_role
                except Exception:
                    pass

        self.renderer_thread = threading.Thread(target=_run, daemon=True)
        self.renderer_thread.start()

    # ------------------------- 工具 -------------------------
    def _set_center_item(self, table: QTableWidget, row: int, col: int, text: str) -> None:
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        table.setItem(row, col, item)

    def _update_player_table(self) -> None:
        members = list(self.client.room_members)
        self.player_table.setRowCount(len(members))
        for row, member in enumerate(members):
            slot = member.get("slot", row)
            name = member.get("name") or member.get("client_id", "?")
            role = member.get("role", "?")
            ready = "是" if member.get("ready", False) else "否"
            color_raw = member.get("color", "-")
            color = str(color_raw)
            if member.get("is_owner"):
                color = f"{color_raw} (房主)"
            self._set_center_item(self.player_table, row, 0, str(slot))
            self._set_center_item(self.player_table, row, 1, str(name))
            self._set_center_item(self.player_table, row, 2, str(role))
            self._set_center_item(self.player_table, row, 3, ready)
            self._set_center_item(self.player_table, row, 4, str(color))

    def _state_to_observation(self, state: Dict, slot: int) -> np.ndarray:
        grid = int(state.get("grid", 30))
        snakes_raw = state.get("snakes", [])
        snakes_snapshot = []
        for idx, snake in enumerate(snakes_raw):
            direction_name = snake.get("direction", "RIGHT")
            try:
                direction = Direction[direction_name]
            except KeyError:
                direction = Direction.RIGHT
            body = [tuple(seg) if not isinstance(seg, tuple) else seg for seg in snake.get("body", [])]
            snakes_snapshot.append(
                {
                    "id": idx,
                    "body": body,
                    "direction": direction,
                    "alive": snake.get("alive", True),
                }
            )

        food = [tuple(f) if not isinstance(f, tuple) else f for f in state.get("food", [])]
        return build_observation_from_snapshot(
            width=grid,
            height=grid,
            snakes=snakes_snapshot,
            food=food,
            slot=slot,
        )


def main() -> None:
    app = QApplication([])
    window = ClientWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
