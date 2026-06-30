import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QListWidget, QSplitter, QSizePolicy)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt


class YOLOViewerWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.img_dir = ""
        self.lbl_dir = ""
        self.class_names = []
        self.image_files = []
        self.current_idx = 0

        # --- A vibrant color palette for different classes (BGR format for OpenCV) ---
        self.palette = [
            (56, 56, 255),  # Red
            (151, 157, 255),  # Orange
            (31, 112, 255),  # Coral
            (29, 178, 255),  # Yellow
            (49, 210, 207),  # Light Green
            (10, 249, 72),  # Green
            (255, 194, 0),  # Light Blue
            (236, 75, 54),  # Blue
            (255, 26, 130),  # Purple
            (94, 53, 255),  # Pink
        ]

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # Left side: File List
        self.file_list = QListWidget()
        self.file_list.setMaximumWidth(250)
        self.file_list.currentRowChanged.connect(self.display_image)

        # Right side: Image Display and Controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.image_label = QLabel("No images loaded.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #11111b; border: 1px solid #45475a; border-radius: 6px;")

        # --- THE GLITCH FIX ---
        # Force the layout to ignore the image's physical size so it stops expanding
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image_label.setMinimumSize(1, 1)
        # ----------------------

        # Controls
        controls_layout = QHBoxLayout()
        self.btn_prev = QPushButton("◀ Previous")
        self.btn_prev.clicked.connect(self.show_prev)
        self.btn_next = QPushButton("Next ▶")
        self.btn_next.clicked.connect(self.show_next)

        controls_layout.addWidget(self.btn_prev)
        controls_layout.addStretch()
        controls_layout.addWidget(self.btn_next)

        right_layout.addWidget(self.image_label, stretch=1)
        right_layout.addLayout(controls_layout)

        # Add to splitter for adjustable sizing
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.file_list)
        splitter.addWidget(right_panel)
        splitter.setSizes([200, 800])

        main_layout.addWidget(splitter)

    def load_dataset(self, img_dir, lbl_dir, class_names):
        """Called by main_gui.py when annotation finishes."""
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.class_names = class_names

        self.file_list.clear()
        if not os.path.exists(img_dir):
            return

        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        self.image_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(valid_exts)])

        self.file_list.addItems(self.image_files)
        if self.image_files:
            self.file_list.setCurrentRow(0)

    def show_prev(self):
        if self.current_idx > 0:
            self.file_list.setCurrentRow(self.current_idx - 1)

    def show_next(self):
        if self.current_idx < len(self.image_files) - 1:
            self.file_list.setCurrentRow(self.current_idx + 1)

    def display_image(self, idx):
        if idx < 0 or idx >= len(self.image_files):
            return

        self.current_idx = idx
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # Determine the matching .txt label file
        lbl_name = os.path.splitext(img_name)[0] + ".txt"
        lbl_path = os.path.join(self.lbl_dir, lbl_name)

        # 1. Read Image
        img = cv2.imread(img_path)
        if img is None:
            return

        # 2. Draw Annotations (Colors + Text)
        img = self.draw_yolo_annotations(img, lbl_path)

        # 3. Convert to PyQt format and display
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytes_per_line = ch * w

        qt_img = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)

        # Scale to fit window while keeping aspect ratio
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def draw_yolo_annotations(self, img, label_path):
        """Parses the YOLO text file and paints the boxes, polygons, and name tags."""
        if not os.path.exists(label_path):
            return img

        h, w, _ = img.shape

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if not parts: continue

            class_id = int(parts[0])

            # Assign color consistently based on class_id
            color = self.palette[class_id % len(self.palette)]

            # Safe class name retrieval
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"

            if len(parts) == 5:
                # Format: Standard YOLO Bounding Box (cx, cy, bw, bh)
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, max(1, int(w / 400)))
            else:
                # Format: YOLO Polygon / Segmentation
                points = np.array(list(map(float, parts[1:]))).reshape(-1, 2)
                points[:, 0] *= w
                points[:, 1] *= h
                pts = points.astype(np.int32)

                cv2.polylines(img, [pts], isClosed=True, color=color, thickness=max(1, int(w / 400)))

                # Calculate bounding box of the polygon to place the text label correctly
                x1, y1 = np.min(pts, axis=0)
                x2, y2 = np.max(pts, axis=0)

            # --- TEXT LABEL DRAWING ---
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.4, w / 1500)
            thickness = max(1, int(w / 1000))

            (tw, th), baseline = cv2.getTextSize(class_name, font, font_scale, thickness)

            # Draw solid background rectangle for the text
            cv2.rectangle(img, (x1, y1 - th - baseline - 2), (x1 + tw, y1), color, -1)

            # Draw the text in white
            cv2.putText(img, class_name, (x1, y1 - baseline), font, font_scale, (255, 255, 255), thickness)

        return img

    def resizeEvent(self, event):
        """Ensure the image scales dynamically when the window is resized."""
        super().resizeEvent(event)
        if self.image_files and self.current_idx < len(self.image_files):
            self.display_image(self.current_idx)