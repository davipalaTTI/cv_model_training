import os
import logging
from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


class ModelDownloaderThread(QThread):
    """
    Refactored into a Model Validator.
    Checks if the required SAM 3.1 weights AND the CLIP vocab exist locally.
    """
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    download_complete = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, model_type):
        super().__init__()
        self.model_type = model_type

        # --- UPDATED REGISTRY: Now expects a LIST of required files ---
        self.model_registry = {
            "SAM 3.1 Multiplex (Standard)": [
                "sam3.1_multiplex.pt",
                "bpe_simple_vocab_16e6.txt.gz"
            ]
        }

    def run(self):
        try:
            # Grab the list of files needed for the selected model
            required_files = self.model_registry.get(self.model_type, [])

            # --- THE BULLETPROOF PATH FIX ---
            core_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(core_dir)
            weights_dir = os.path.join(project_root, "weights")
            os.makedirs(weights_dir, exist_ok=True)

            self.status_updated.emit("Validating model assets...")

            # --- VALIDATION LOOP ---
            for filename in required_files:
                filepath = os.path.join(weights_dir, filename)
                logger.info(f"Looking for asset at absolute path: {filepath}")

                if not os.path.exists(filepath):
                    error_msg = (
                        f"Missing Required Model Asset: {filename}\n\n"
                        f"I looked exactly here:\n{filepath}\n\n"
                        f"Please ensure it is placed in the 'weights/' folder."
                    )
                    logger.error(f"Validation failed: {filepath} not found.")
                    self.error_occurred.emit(error_msg)
                    return  # Stop the thread immediately if ANY file is missing

            # --- SUCCESS ---
            # If the loop finishes without returning, all files exist!
            logger.info("All local model assets validated successfully.")
            self.progress_updated.emit(100)

            # We send the ABSOLUTE path of the MAIN .pt file to the inference engine
            # (SAM 3 will automatically find the vocab file sitting next to it)
            main_filepath = os.path.join(weights_dir, required_files[0])
            self.download_complete.emit(main_filepath)

        except Exception as e:
            logger.error(f"Validator failed: {e}")
            self.error_occurred.emit(str(e))