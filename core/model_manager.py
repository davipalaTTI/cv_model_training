import os
import requests
from PyQt6.QtCore import QThread, pyqtSignal

import os
import requests
import logging
from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


class ModelDownloaderThread(QThread):
    """
    Refactored into a Model Validator.
    Checks if the required SAM 3.1 weights exist locally.
    """
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    download_complete = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, model_type):
        super().__init__()
        self.model_type = model_type

        # We only need the filenames now, not the URLs
        self.model_registry = {
            "SAM 3.1 Multiplex (Standard)": "sam3.1_multiplex.pt"
        }

    def run(self):
        try:
            filename = self.model_registry.get(self.model_type)

            # --- THE BULLETPROOF PATH FIX ---
            # 1. Get the exact path to this model_manager.py file (.../core/)
            core_dir = os.path.dirname(os.path.abspath(__file__))

            # 2. Go up one level to the project root (.../cv_model_training/)
            project_root = os.path.dirname(core_dir)

            # 3. Explicitly lock onto the weights folder
            weights_dir = os.path.join(project_root, "weights")
            os.makedirs(weights_dir, exist_ok=True)

            filepath = os.path.join(weights_dir, filename)

            self.status_updated.emit("Validating weights...")
            logger.info(f"Looking for weights at absolute path: {filepath}")

            # --- VALIDATION CHECK ---
            if os.path.exists(filepath):
                logger.info(f"Local weights validated: {filepath}")
                self.progress_updated.emit(100)

                # We send the ABSOLUTE path to the inference engine so it doesn't get lost either!
                self.download_complete.emit(filepath)
            else:
                error_msg = (
                    f"Missing Model Weights: {filename}\n\n"
                    f"I looked exactly here:\n{filepath}\n\n"
                    f"Please ensure it is placed there and named correctly."
                )
                logger.error(f"Validation failed: {filepath} not found.")
                self.error_occurred.emit(error_msg)

        except Exception as e:
            logger.error(f"Validator failed: {e}")
            self.error_occurred.emit(str(e))