import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QComboBox, QMessageBox)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt


class YOLOViewerWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.image_paths = []
        self.current_index = 0
        self.img_dir = ""
        self.lbl_dir = ""
        self.classes = []

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # --- Controls Layout ---
        controls_layout = QHBoxLayout()

        self.btn_prev = QPushButton("◀ Previous")
        self.btn_prev.clicked.connect(self.show_prev)

        self.lbl_info = QLabel("No images loaded")
        self.lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.btn_next = QPushButton("Next ▶")
        self.btn_next.clicked.connect(self.show_next)

        controls_layout.addWidget(self.btn_prev)
        controls_layout.addWidget(self.lbl_info)
        controls_layout.addWidget(self.btn_next)

        # --- Image Display ---
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #222; border: 1px solid #555;")
        self.image_label.setMinimumSize(640, 480)  # Minimum viewing size

        layout.addWidget(self.image_label, stretch=1)
        layout.addLayout(controls_layout)

    def load_dataset(self, img_dir, lbl_dir, classes):
        """Loads the dataset paths from the inference engine output."""
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.classes = classes
        self.current_index = 0

        if not os.path.exists(img_dir):
            return

        # Get all valid images
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        self.image_paths = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(valid_exts)])

        if self.image_paths:
            self.show_image(self.current_index)
        else:
            self.lbl_info.setText("Folder is empty")
            self.image_label.clear()

    def show_image(self, index):
        if not self.image_paths or index < 0 or index >= len(self.image_paths):
            return

        img_name = self.image_paths[index]
        img_path = os.path.join(self.img_dir, img_name)
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        lbl_path = os.path.join(self.lbl_dir, txt_name)

        # 1. Read Image using OpenCV
        image = cv2.imread(img_path)
        if image is None:
            return

        h, w = image.shape[:2]

        # 2. Draw YOLO Annotations if they exist
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if not parts: continue

                class_id = int(parts[0])
                color = (0, 165, 255)  # Orange box (BGR format for OpenCV)

                # Format 1: Bounding Box (5 values: id, cx, cy, w, h)
                if len(parts) == 5:
                    _, cx, cy, bw, bh = map(float, parts)
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                # Format 2: Polygon/OBB (>5 values)
                elif len(parts) > 5:
                    points = [float(p) for p in parts[1:]]
                    pts = np.array(points).reshape(-1, 2)
                    pts[:, 0] *= w
                    pts[:, 1] *= h
                    pts = pts.astype(np.int32)
                    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)

        # 3. Convert OpenCV BGR Image to PyQt RGB Pixmap
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * w
        q_img = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # 4. Scale it to fit the window nicely while keeping aspect ratio
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

        # 5. Update Label
        self.lbl_info.setText(f"Image {index + 1} of {len(self.image_paths)} : {img_name}")

    def show_prev(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image(self.current_index)

    def show_next(self):
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.show_image(self.current_index)