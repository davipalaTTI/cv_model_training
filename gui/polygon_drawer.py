import cv2
import numpy as np
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QMessageBox)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush
from PyQt6.QtCore import Qt, QPoint


class ClickableImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.zones = []  # Completed zones (List of lists of QPoints)
        self.current_zone = []  # Points for the zone currently being drawn
        self.undo_stack = []
        self.redo_stack = []
        self.dragging_idx = None  # Tracks which point is being dragged
        self.drag_threshold = 10  # How close the mouse needs to be to grab a point

    def save_state(self):
        """Saves a deep copy of the current state for Undo/Redo."""
        state = (
            [[QPoint(p.x(), p.y()) for p in zone] for zone in self.zones],
            [QPoint(p.x(), p.y()) for p in self.current_zone]
        )
        self.undo_stack.append(state)
        self.redo_stack.clear()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()
            self.save_state()

            # 1. Check if clicking an existing point in the current active zone
            for i, p in enumerate(self.current_zone):
                if (p - pos).manhattanLength() < self.drag_threshold:
                    self.dragging_idx = (False, -1, i)
                    return

            # 2. Check if clicking an existing point in a finished zone
            for z_idx, zone in enumerate(self.zones):
                for p_idx, p in enumerate(zone):
                    if (p - pos).manhattanLength() < self.drag_threshold:
                        self.dragging_idx = (True, z_idx, p_idx)
                        return

            # 3. If clicking empty space, add a new point sequentially
            self.current_zone.append(pos)
            self.update()

    def mouseMoveEvent(self, event):
        """Allows dragging the point if one was clicked."""
        if self.dragging_idx is not None:
            is_completed, z_idx, p_idx = self.dragging_idx
            if is_completed:
                self.zones[z_idx][p_idx] = event.pos()
            else:
                self.current_zone[p_idx] = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging_idx = None

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw all Completed Zones (Closed and filled)
        for zone in self.zones:
            if not zone: continue
            painter.setPen(QPen(QColor(255, 50, 50), 3))
            painter.setBrush(QBrush(QColor(255, 50, 50, 100)))  # Semi-transparent red
            painter.drawPolygon(*zone)

            painter.setBrush(QColor(0, 255, 0))  # Green dots for corners
            for point in zone:
                painter.drawEllipse(point, 5, 5)

        # Draw Current Active Zone (Sequential, Open lines)
        if self.current_zone:
            painter.setPen(QPen(QColor(50, 150, 255), 3))  # Blue active line
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for i in range(1, len(self.current_zone)):
                painter.drawLine(self.current_zone[i - 1], self.current_zone[i])

            painter.setBrush(QColor(255, 255, 0))  # Yellow dots
            for point in self.current_zone:
                painter.drawEllipse(point, 5, 5)

    def finish_current_zone(self):
        if len(self.current_zone) >= 3:
            self.save_state()
            self.zones.append(self.current_zone)
            self.current_zone = []
            self.update()
            return True
        return False

    def undo(self):
        if self.undo_stack:
            current_state = (
                [[QPoint(p.x(), p.y()) for p in z] for z in self.zones],
                [QPoint(p.x(), p.y()) for p in self.current_zone]
            )
            self.redo_stack.append(current_state)
            self.zones, self.current_zone = self.undo_stack.pop()
            self.update()

    def redo(self):
        if self.redo_stack:
            current_state = (
                [[QPoint(p.x(), p.y()) for p in z] for z in self.zones],
                [QPoint(p.x(), p.y()) for p in self.current_zone]
            )
            self.undo_stack.append(current_state)
            self.zones, self.current_zone = self.redo_stack.pop()
            self.update()


class PolygonDrawerDialog(QDialog):
    def __init__(self, first_frame_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Draw Exclusion Zones")
        self.final_zones = []

        layout = QVBoxLayout(self)
        self.instructions = QLabel(
            "• Click to draw points sequentially.\n"
            "• Drag points to adjust.\n"
            "• Click 'Finish Zone' to close the shape.\n"
            "• Draw as many separate zones as needed."
        )
        self.instructions.setStyleSheet("font-weight: bold; color: #cdd6f4;")
        layout.addWidget(self.instructions)

        self.image_label = ClickableImageLabel()
        cv_img = cv2.imread(first_frame_path)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w

        self.image_label.setPixmap(
            QPixmap.fromImage(QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)))
        self.image_label.setFixedSize(w, h)
        layout.addWidget(self.image_label)

        btn_layout = QHBoxLayout()
        self.btn_undo = QPushButton("↺ Undo")
        self.btn_undo.clicked.connect(self.image_label.undo)

        self.btn_redo = QPushButton("↻ Redo")
        self.btn_redo.clicked.connect(self.image_label.redo)

        self.btn_finish = QPushButton("Finish Zone")
        self.btn_finish.clicked.connect(self.finish_click)

        self.btn_clear = QPushButton("Clear All")
        self.btn_clear.clicked.connect(self.clear_all_click)

        self.btn_save = QPushButton("Save & Exit")
        self.btn_save.setStyleSheet("background-color: #a6e3a1; color: #11111b; font-weight: bold;")
        self.btn_save.clicked.connect(self.save_zones)

        btn_layout.addWidget(self.btn_undo)
        btn_layout.addWidget(self.btn_redo)
        btn_layout.addWidget(self.btn_finish)
        btn_layout.addWidget(self.btn_clear)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_save)
        layout.addLayout(btn_layout)

    def finish_click(self):
        if not self.image_label.finish_current_zone():
            QMessageBox.warning(self, "Invalid Zone", "A zone needs at least 3 points!")

    def clear_all_click(self):
        self.image_label.save_state()
        self.image_label.zones, self.image_label.current_zone = [], []
        self.image_label.update()

    def save_zones(self):
        if len(self.image_label.current_zone) >= 3:
            self.image_label.finish_current_zone()

        # Format output: List of multiple zones, each containing [x, y] lists
        self.final_zones = [[[p.x(), p.y()] for p in z] for z in self.image_label.zones]
        self.accept()