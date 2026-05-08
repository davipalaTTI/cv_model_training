import sys
import logging
from PyQt6.QtWidgets import QApplication, QMessageBox
from gui.main_gui import DataEngineGUI

# ==========================================
# 0. GLOBAL LOGGING CONFIGURATION
# ==========================================
# This ensures that ALL modules (gui, core, utils) use the same log file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cv_data_engine.log"),  # Persistent log file
        logging.StreamHandler(sys.stdout)  # Terminal output
    ]
)
logger = logging.getLogger(__name__)


def main():
    try:
        # Initialize the PyQt6 Application
        app = QApplication(sys.argv)
        app.setStyle("Fusion")

        logger.info("--- Starting YOLO Data Engine ---")

        # 1. Initialize the Main GUI
        # If there is a bug in setup_model_section or elsewhere,
        # this try block will catch it.
        window = DataEngineGUI()
        window.show()

        # 2. Start the Event Loop
        sys.exit(app.exec())

    except Exception as e:
        # This is the 'Safety Net'
        # If the app crashes before the window even opens, we log it here.
        error_msg = f"Application failed to initialize: {e}"
        logger.critical(error_msg)

        # If we can, show a popup to the user so they aren't left guessing
        if QApplication.instance():
            QMessageBox.critical(None, "Startup Error", error_msg)

        sys.exit(1)


if __name__ == "__main__":
    main()