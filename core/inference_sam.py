import os
import cv2
import torch
import numpy as np
import logging
from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


class SAM3InferenceThread(QThread):
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    # --- NEW: Added exclusion_coords to the initialization ---
    def __init__(self, model_path, input_path, output_paths, class_mapping,
                 batch_size, device, conf_thresh, prompt_thresh, nms_thresh, output_format,
                 exclusion_coords=None):
        super().__init__()
        self.model_path = model_path
        self.input_path = input_path
        self.output_paths = output_paths
        self.class_mapping = class_mapping

        self.batch_size = max(1, batch_size)
        self.device = device
        self.conf_thresh = conf_thresh
        self.prompt_thresh = prompt_thresh
        self.nms_thresh = nms_thresh
        self.output_format = output_format

        # Save the GUI's polygon coordinates
        self.exclusion_coords = exclusion_coords

        self._is_running = True

    def stop(self):
        """Method to safely stop the thread from the GUI."""
        self._is_running = False

    def run(self):
        try:
            self.status_updated.emit("Initializing SAM 3.1 Processor...")

            # --- 1. LOAD MODEL ---
            try:
                from sam3 import build_sam3_image_model
                from sam3.model.sam3_image_processor import Sam3Processor

                logger.info(f"Loading weights from: {self.model_path}")
                model = build_sam3_image_model(checkpoint_path=self.model_path).to(self.device)

                processor = Sam3Processor(
                    model,
                    confidence_threshold=self.conf_thresh
                )
                logger.info("SAM 3.1 Processor loaded successfully into VRAM.")
            except Exception as e:
                self.error_occurred.emit(f"Failed to load SAM 3.1:\n{str(e)}")
                return

            # --- 2. PARSE INPUTS ---
            self.status_updated.emit("Parsing inputs...")
            image_filepaths = self._gather_inputs(self.input_path)

            total_imgs = len(image_filepaths)
            if total_imgs == 0:
                self.error_occurred.emit("No valid images or videos found at the input path.")
                return

            # --- 3. BATCH CHUNKING LOGIC ---
            batches = [image_filepaths[i:i + self.batch_size]
                       for i in range(0, total_imgs, self.batch_size)]

            processed_count = 0

            # --- 4. MAIN INFERENCE LOOP ---
            for batch_idx, batch_paths in enumerate(batches):
                for img_path in batch_paths:

                    # --- CANCEL CHECK ---
                    if not self._is_running:
                        logger.info("Inference cancelled by user.")
                        self.status_updated.emit("Process Cancelled.")
                        self.finished.emit("Annotation was cancelled by user.")
                        return

                    try:
                        img_name = os.path.basename(img_path)
                        self.status_updated.emit(f"Annotating: {img_name} (Batch {batch_idx + 1}/{len(batches)})")

                        is_valid = (processed_count % 5 == 0)
                        target_img_dir = self.output_paths['val_img'] if is_valid else self.output_paths['train_img']
                        target_lbl_dir = self.output_paths['val_lbl'] if is_valid else self.output_paths['train_lbl']

                        pil_image = Image.open(img_path).convert("RGB")
                        pil_image.save(os.path.join(target_img_dir, img_name))

                        w, h = pil_image.size
                        label_lines = []

                        # ==========================================
                        # MASK GENERATION: Build the exclusion canvas
                        # ==========================================
                        exclusion_mask = None
                        if self.exclusion_coords:  # Now this is a list of multiple zones
                            exclusion_mask = np.zeros((h, w), dtype=np.uint8)

                            # Convert multiple zones into a list of NumPy integer arrays
                            pts_list = [np.array(zone, dtype=np.int32) for zone in self.exclusion_coords]

                            # fillPoly inherently accepts a LIST of polygons and draws all of them!
                            cv2.fillPoly(exclusion_mask, pts_list, 1)

                        # --- THE MAGIC FORWARD PASS ---
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            inference_state = processor.set_image(pil_image)

                            for class_id, raw_keywords in enumerate(self.class_mapping):
                                search_terms = [t.strip() for t in raw_keywords.split('|') if t.strip()]

                                for term in search_terms:
                                    prompt = f"a {term}"
                                    output = processor.set_text_prompt(state=inference_state, prompt=prompt)
                                    masks = output.get("masks")

                                    if masks is not None and len(masks) > 0:
                                        masks_np = masks.cpu().numpy()

                                        for mask in masks_np:
                                            # Ensure mask is 2D
                                            if mask.ndim == 3:
                                                mask = mask[0]

                                            # ==========================================
                                            # EXCLUSION CHECK: Discard overlapping items
                                            # ==========================================
                                            if exclusion_mask is not None:
                                                overlap = cv2.bitwise_and(mask.astype(np.uint8), exclusion_mask)
                                                if np.any(overlap):
                                                    continue  # Skip to the next mask!

                                            # If it survived, format and save it
                                            yolo_line = self.convert_to_yolo(mask, class_id, w, h)
                                            if yolo_line:
                                                label_lines.append(yolo_line)

                        # Save the .txt label file
                        txt_name = os.path.splitext(img_name)[0] + ".txt"
                        with open(os.path.join(target_lbl_dir, txt_name), "w") as f:
                            f.write("\n".join(label_lines))

                        # Update Progress
                        processed_count += 1
                        self.progress_updated.emit(int((processed_count / total_imgs) * 100))

                    except Exception as e:
                        logger.error(f"Error processing {img_name}: {e}")
                        processed_count += 1
                        continue

            # --- 5. SUCCESS ---
            self.progress_updated.emit(100)
            self.finished.emit(f"Successfully auto-annotated {total_imgs} images/frames into YOLO format.")

        except Exception as e:
            logger.critical(f"Thread Crash: {e}")
            self.error_occurred.emit(f"System Error: {e}")

    # ==========================================
    # INPUT PARSING HELPERS
    # ==========================================
    def _gather_inputs(self, path_string):
        """Determines if input is a folder, single file, or list of files and extracts accordingly."""
        inputs = []
        for path in path_string.split(";"):
            path = path.strip()

            # If the user selected a Folder
            if os.path.isdir(path):
                for f in os.listdir(path):
                    full_path = os.path.join(path, f)
                    ext = os.path.splitext(f)[1].lower()

                    # Check for images
                    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                        inputs.append(full_path)
                    # Check for videos
                    elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
                        inputs.extend(self._extract_video_frames(full_path))

            # If the user selected specific Files
            elif os.path.isfile(path):
                ext = os.path.splitext(path)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    inputs.append(path)
                elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
                    inputs.extend(self._extract_video_frames(path))

        return inputs

    def _extract_video_frames(self, video_path):
        self.status_updated.emit("Extracting video frames...")
        frames_dir = os.path.join(os.path.dirname(self.output_paths['train_img']), "temp_extracted_frames")
        os.makedirs(frames_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        frame_paths = []
        frame_count = 0
        saved_count = 0
        skip_interval = 15

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            if frame_count % skip_interval == 0:
                frame_name = f"frame_{saved_count:06d}.jpg"
                frame_path = os.path.join(frames_dir, frame_name)
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                saved_count += 1
            frame_count += 1

        cap.release()
        logger.info(f"Extracted {saved_count} frames from video.")
        return frame_paths

    # ==========================================
    # MATH HELPERS
    # ==========================================
    def convert_to_yolo(self, mask, class_id, img_w, img_h):
        try:
            if mask.ndim == 3: mask = mask[0]

            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: return ""
            contour = max(contours, key=cv2.contourArea)

            if "BBox" in self.output_format:
                x, y, w, h = cv2.boundingRect(contour)
                return f"{class_id} {(x + w / 2) / img_w:.6f} {(y + h / 2) / img_h:.6f} {w / img_w:.6f} {h / img_h:.6f}"
            elif "OBB" in self.output_format:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                points = [val for p in box for val in (p[0] / img_w, p[1] / img_h)]
                return f"{class_id} " + " ".join([f"{p:.6f}" for p in points])
            else:
                polygon = contour.reshape(-1, 2)
                norm_poly = [val for p in polygon for val in (p[0] / img_w, p[1] / img_h)]
                if len(norm_poly) < 6: return ""
                return f"{class_id} " + " ".join([f"{p:.6f}" for p in norm_poly])

        except Exception as e:
            logger.error(f"Math conversion failed: {e}")
            return ""