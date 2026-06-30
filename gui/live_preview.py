import os
import cv2
import torch
import queue
import numpy as np
from PIL import Image
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QSlider, QPushButton, QSizePolicy)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap
import logging

logger = logging.getLogger(__name__)


class PersistentSAMWorker(QThread):
    """A background thread that holds SAM in VRAM to allow rapid, real-time testing."""
    result_ready = pyqtSignal(object)  # Returns the annotated cv2 image
    status_update = pyqtSignal(str)

    def __init__(self, model_path, device):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.request_queue = queue.Queue()
        self._is_running = True
        self.model = None

        # Re-use the color palette from your viewer
        self.palette = [
            (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
            (49, 210, 207), (10, 249, 72), (255, 194, 0), (236, 75, 54),
            (255, 26, 130), (94, 53, 255)
        ]

    def run(self):
        self.status_update.emit("Loading SAM 3.1 into VRAM (Please wait)...")
        try:
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            self.model = build_sam3_image_model(checkpoint_path=self.model_path).to(self.device)
            self.status_update.emit("Ready. Adjust settings to preview.")
        except Exception as e:
            self.status_update.emit(f"Error loading model: {e}")
            return

        # The Persistent Idle Loop
        while self._is_running:
            if not self.request_queue.empty():
                try:
                    # Grab the latest parameters from the queue
                    img_path, conf_thresh, class_mapping = self.request_queue.get()
                    self.status_update.emit("Processing frame...")

                    # Re-instantiate processor with new threshold (very fast)
                    processor = Sam3Processor(self.model, confidence_threshold=conf_thresh)

                    cv_img = cv2.imread(img_path)
                    pil_image = Image.open(img_path).convert("RGB")
                    w, h = pil_image.size

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        inference_state = processor.set_image(pil_image)

                        for class_id, raw_keywords in enumerate(class_mapping):
                            search_terms = [t.strip() for t in raw_keywords.split('|') if t.strip()]
                            color = self.palette[class_id % len(self.palette)]
                            class_name = search_terms[0] if search_terms else f"Class {class_id}"

                            for term in search_terms:
                                output = processor.set_text_prompt(state=inference_state, prompt=f"a {term}")
                                masks = output.get("masks")

                                if masks is not None and len(masks) > 0:
                                    masks_np = masks.cpu().numpy()
                                    for mask in masks_np:
                                        if mask.ndim == 3: mask = mask[0]
                                        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                                       cv2.CHAIN_APPROX_SIMPLE)
                                        if contours:
                                            contour = max(contours, key=cv2.contourArea)
                                            # Draw Bounding Box
                                            x, y, bw, bh = cv2.boundingRect(contour)
                                            cv2.rectangle(cv_img, (x, y), (x + bw, y + bh), color, max(1, int(w / 400)))

                                            # Draw Label
                                            font = cv2.FONT_HERSHEY_SIMPLEX
                                            font_scale = max(0.4, w / 1500)
                                            thickness = max(1, int(w / 1000))
                                            (tw, th), baseline = cv2.getTextSize(class_name, font, font_scale,
                                                                                 thickness)
                                            cv2.rectangle(cv_img, (x, y - th - baseline - 2), (x + tw, y), color, -1)
                                            cv2.putText(cv_img, class_name, (x, y - baseline), font, font_scale,
                                                        (255, 255, 255), thickness)

                    self.result_ready.emit(cv_img)
                    self.status_update.emit("Preview Updated.")

                except Exception as e:
                    self.status_update.emit(f"Inference Error: {e}")

            # Sleep briefly to prevent maxing out CPU while idling
            self.msleep(100)

    def stop(self):
        self._is_running = False
        # Crucial: Nuke the model from VRAM when closed
        del self.model
        torch.cuda.empty_cache()


class LivePreviewDialog(QDialog):
    def __init__(self, frames, model_path, device, initial_conf, class_mapping, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Live Parameters Preview")
        self.resize(1000, 700)

        self.frames = frames
        self.class_mapping = class_mapping
        self.current_frame_idx = 0

        # --- The Debouncer ---
        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.setInterval(600)  # Wait 0.6 seconds after sliding stops
        self.debounce_timer.timeout.connect(self.trigger_inference)

        # UI Layout
        layout = QVBoxLayout(self)

        self.lbl_status = QLabel("Initializing...")
        self.lbl_status.setStyleSheet("color: #f9e2af; font-weight: bold;")
        layout.addWidget(self.lbl_status)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image_label.setMinimumSize(1, 1)
        layout.addWidget(self.image_label, stretch=1)

        # Controls
        controls = QHBoxLayout()

        controls.addWidget(QLabel("Frame:"))
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, len(frames) - 1)
        self.frame_slider.valueChanged.connect(self.on_setting_changed)
        controls.addWidget(self.frame_slider)

        controls.addWidget(QLabel("Confidence:"))
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(int(initial_conf * 100))
        self.conf_slider.valueChanged.connect(self.on_setting_changed)
        controls.addWidget(self.conf_slider)

        layout.addLayout(controls)

        # Spin up the background VRAM thread
        self.worker = PersistentSAMWorker(model_path, device)
        self.worker.status_update.connect(self.lbl_status.setText)
        self.worker.result_ready.connect(self.display_image)
        self.worker.start()

        # Trigger the first image
        self.trigger_inference()

    def on_setting_changed(self):
        """Restarts the timer every time the slider moves. Delays the AI check."""
        self.lbl_status.setText("Waiting for adjustment to finish...")
        self.debounce_timer.start()

    def trigger_inference(self):
        """Fires only when the timer completely finishes."""
        img_path = self.frames[self.frame_slider.value()]
        conf = self.conf_slider.value() / 100.0

        # Clear queue to drop old requests and push the newest one
        while not self.worker.request_queue.empty():
            self.worker.request_queue.get()
        self.worker.request_queue.put((img_path, conf, self.class_mapping))

    def display_image(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = cv_img.shape
        qt_img = QImage(cv_img.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                 Qt.TransformationMode.SmoothTransformation))

    def closeEvent(self, event):
        """Safely shut down the heavy AI thread when the window is closed."""
        self.worker.stop()
        self.worker.wait()
        super().closeEvent(event)