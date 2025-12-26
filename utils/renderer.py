"""
Unified Game Renderer (PyQt6).
Provides a reusable widget for rendering the Snake Game state.
Used by both local GUI and network clients.
"""

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QRadialGradient

# Define Colors centrally
COLOR_BG = QColor(25, 30, 45)
COLOR_GRID = QColor(40, 45, 60)
COLOR_FOOD = QColor(255, 0, 85)
COLORS_SNAKE = [
    QColor(50, 255, 50),    # Green
    QColor(50, 50, 255),    # Blue
    QColor(255, 255, 50),   # Yellow
    QColor(50, 255, 255),   # Cyan
    QColor(255, 50, 255)    # Magenta
]

class GameRenderer(QWidget):
    """
    Reusable Widget that accepts game state and paints it.
    Input state structure:
    {
        "snakes": [[(x,y), ...], ...],
        "food": [(x,y), ...],
        "dead": [bool, ...],
    }
    """
    def __init__(self, parent=None, grid_size=20):
        super().__init__(parent)
        self.grid_size = grid_size
        
        # State
        self.snakes = []
        self.food = []
        self.dead = []
        self.player_id = -1 # ID to highlight
        self.countdown = 0  # V7.2: Countdown overlay
        
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
    def update_state(self, snakes, foods, dead, player_id=-1):
        self.snakes = snakes
        # ... (normalize food logic same)
        self.food = [tuple(f) for f in foods] if foods else []
        self.dead = dead if dead else [False] * len(self.snakes)
        self.player_id = player_id
        self.update() 

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 1. Background & Grid
        painter.fillRect(self.rect(), COLOR_BG)
        cell_w, cell_h = self.width() / self.grid_size, self.height() / self.grid_size
        
        pen = QPen(COLOR_GRID)
        painter.setPen(pen)
        for x in range(self.grid_size + 1):
            painter.drawLine(int(x*cell_w), 0, int(x*cell_w), self.height())
        for y in range(self.grid_size + 1):
            painter.drawLine(0, int(y*cell_h), self.width(), int(y*cell_h))
            
        # 2. Food
        for fx, fy in self.food:
            cx, cy = int((fx + 0.5) * cell_w), int((fy + 0.5) * cell_h)
            radius = min(cell_w, cell_h) * 0.35
            grad = QRadialGradient(cx, cy, radius * 2.5)
            grad.setColorAt(0, QColor(255, 0, 85, 150)); grad.setColorAt(1, QColor(255, 0, 85, 0))
            painter.setBrush(grad); painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPoint(cx, cy), int(radius*2.5), int(radius*2.5))
            painter.setBrush(COLOR_FOOD); painter.setPen(QPen(Qt.GlobalColor.white, 2))
            painter.drawEllipse(QPoint(cx, cy), int(radius), int(radius))
            
        # 3. Snakes
        for i, snake in enumerate(self.snakes):
            if i < len(self.dead) and self.dead[i]: continue
            if not snake: continue
            base_color = COLORS_SNAKE[i % len(COLORS_SNAKE)]
            
            # Body Segments
            for idx in range(len(snake) - 1):
                p1, p2 = snake[idx], snake[idx+1]
                p1x, p1y = int((p1[0] + 0.5) * cell_w), int((p1[1] + 0.5) * cell_h)
                p2x, p2y = int((p2[0] + 0.5) * cell_w), int((p2[1] + 0.5) * cell_h)
                alpha = int(255 * max(0.4, 1.0 - (idx / len(snake)) * 0.6))
                color = QColor(base_color); color.setAlpha(alpha)
                pen = QPen(color); pen.setWidth(int(min(cell_w, cell_h) * 0.8))
                pen.setCapStyle(Qt.PenCapStyle.RoundCap); painter.setPen(pen)
                painter.drawLine(p1x, p1y, p2x, p2y)
                
            # Head Glow & Solid
            hx, hy = snake[0]
            hcx, hcy = int((hx + 0.5) * cell_w), int((hy + 0.5) * cell_h)
            hr = min(cell_w, cell_h) * 0.45
            grad = QRadialGradient(hcx, hcy, hr * 2.5)
            grad.setColorAt(0, QColor(base_color.red(), base_color.green(), base_color.blue(), 100))
            grad.setColorAt(1, QColor(base_color.red(), base_color.green(), base_color.blue(), 0))
            painter.setBrush(grad); painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPoint(hcx, hcy), int(hr*2.5), int(hr*2.5))
            painter.setBrush(Qt.GlobalColor.white); painter.setPen(QPen(base_color, 3))
            painter.drawEllipse(QPoint(hcx, hcy), int(hr), int(hr))
            
            if i == self.player_id:
                painter.setPen(Qt.GlobalColor.white)
                font = painter.font(); font.setBold(True); painter.setFont(font)
                painter.drawText(hcx - 12, hcy - 20, "YOU")

        # 4. Countdown Overlay (V7.2)
        if self.countdown > 0:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 150))
            painter.drawRect(self.rect())
            
            painter.setPen(QColor(0, 212, 255))
            font = painter.font()
            font.setPointSize(72); font.setBold(True)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, str(self.countdown))
            
            # Sub-text
            font.setPointSize(24)
            painter.setFont(font)
            painter.drawText(self.rect().adjusted(0, 150, 0, 0), Qt.AlignmentFlag.AlignCenter, "GET READY!")
