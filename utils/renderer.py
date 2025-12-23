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
        
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
    def update_state(self, snakes, foods, dead, player_id=-1):
        self.snakes = snakes
        # Normalize food to list of tuples
        if not foods:
            self.food = []
        elif isinstance(foods, list):
            # Check if it's a list of [x, y] or list of (x, y)
            if len(foods) > 0 and (isinstance(foods[0], int) or isinstance(foods[0], float)):
                # Likely a single [x, y]
                self.food = [tuple(foods)]
            else:
                # Likely a list of points
                self.food = [tuple(f) for f in foods]
        elif isinstance(foods, tuple):
             # Single (x, y)
             self.food = [foods]
        else:
            self.food = []
            
        self.dead = dead if dead else [False] * len(self.snakes)
        self.player_id = player_id
        self.update() # Trigger repaint

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw Background
        painter.fillRect(self.rect(), COLOR_BG)
        
        # Calculate cell size
        w = self.width()
        h = self.height()
        cell_w = w / self.grid_size
        cell_h = h / self.grid_size
        
        # Draw Grid
        pen = QPen(COLOR_GRID)
        pen.setWidth(1)
        painter.setPen(pen)
        
        for x in range(self.grid_size + 1):
            painter.drawLine(int(x*cell_w), 0, int(x*cell_w), h)
        for y in range(self.grid_size + 1):
            painter.drawLine(0, int(y*cell_h), w, int(y*cell_h))
            
        # Draw Food
        for fx, fy in self.food:
            cx = int((fx + 0.5) * cell_w)
            cy = int((fy + 0.5) * cell_h)
            radius = min(cell_w, cell_h) * 0.35
            
            # Glow
            gradient = QRadialGradient(cx, cy, radius * 2.5)
            gradient.setColorAt(0, QColor(255, 0, 85, 150))
            gradient.setColorAt(1, QColor(255, 0, 85, 0))
            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPoint(cx, cy), int(radius*2.5), int(radius*2.5))
            
            # Solid
            painter.setBrush(QBrush(COLOR_FOOD))
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            painter.drawEllipse(QPoint(cx, cy), int(radius), int(radius))
            
        # Draw Snakes
        for i, snake in enumerate(self.snakes):
            if i < len(self.dead) and self.dead[i]: continue
            if not snake: continue
            
            base_color = COLORS_SNAKE[i % len(COLORS_SNAKE)]
            
            # Body
            for idx in range(len(snake) - 1):
                p1 = snake[idx]
                p2 = snake[idx+1]
                
                p1x = int((p1[0] + 0.5) * cell_w)
                p1y = int((p1[1] + 0.5) * cell_h)
                p2x = int((p2[0] + 0.5) * cell_w)
                p2y = int((p2[1] + 0.5) * cell_h)
                
                # Gradient opacity
                alpha = int(255 * max(0.4, 1.0 - (idx / len(snake)) * 0.6))
                color = QColor(base_color)
                color.setAlpha(alpha)
                
                pen = QPen(color)
                pen.setWidth(int(min(cell_w, cell_h) * 0.8))
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawLine(p1x, p1y, p2x, p2y)
                
            # Head
            hx, hy = snake[0]
            cx = int((hx + 0.5) * cell_w)
            cy = int((hy + 0.5) * cell_h)
            radius = min(cell_w, cell_h) * 0.4
            
            # Head Glow
            gradient = QRadialGradient(cx, cy, radius * 2.5)
            gradient.setColorAt(0, QColor(base_color.red(), base_color.green(), base_color.blue(), 100))
            gradient.setColorAt(1, QColor(base_color.red(), base_color.green(), base_color.blue(), 0))
            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPoint(cx, cy), int(radius*2.5), int(radius*2.5))
            
            painter.setBrush(QBrush(Qt.GlobalColor.white))
            painter.setPen(QPen(base_color, 2))
            painter.drawEllipse(QPoint(cx, cy), int(radius), int(radius))
            
            # ID
            if i == self.player_id:
                painter.setPen(Qt.GlobalColor.white)
                painter.drawText(cx - 5, cy - 15, "YOU")
