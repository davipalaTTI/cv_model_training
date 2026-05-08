import sys
import logging
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGroupBox, QLabel, QComboBox,
                             QPushButton, QTableWidget, QTableWidgetItem,
                             QHeaderView, QLineEdit, QProgressBar, QFileDialog,
                             QSlider, QAbstractItemView, QStackedWidget, QStyledItemDelegate,
                             QMessageBox, QTabWidget, QSizePolicy)
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui import QIntValidator
from gui.viewer_widget import YOLOViewerWidget

# Import our custom background downloader/validator
try:
    from core.model_manager import ModelDownloaderThread
except ImportError:
    print("CRITICAL: core/model_manager.py not found. Pipeline will fail.")

# ==========================================
# 0. LOGGING CONFIGURATION
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cv_data_engine.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ==========================================
# 1. BASE CLASSES
# ==========================================

class BaseAppWindow(QMainWindow):
    def __init__(self, title="CV Tool", width=800, height=700):
        super().__init__()
        try:
            self.setWindowTitle(title)
            self.resize(width, height)
            self.apply_standard_styling()
            logger.info(f"Initialized Base Window: {title}")
        except Exception as e:
            logger.error(f"Failed to initialize Base Window: {e}")

    def apply_standard_styling(self):
        """Applies a universal, modern dark theme with enhanced Progress Bar."""
        try:
            dark_stylesheet = """
                QWidget { background-color: #1e1e2e; color: #cdd6f4; font-family: 'Segoe UI'; font-size: 10pt; }
                QGroupBox { font-weight: bold; border: 1px solid #313244; border-radius: 6px; margin-top: 15px; padding-top: 15px; }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #89b4fa; }
                QPushButton { background-color: #313244; border: 1px solid #45475a; border-radius: 4px; padding: 6px 12px; }
                QPushButton:hover { background-color: #45475a; }
                QPushButton#start_button { background-color: #a6e3a1; color: #11111b; font-weight: bold; }
                QPushButton#cancel_button { background-color: #eba0ac; color: #11111b; font-weight: bold; }
                QPushButton#cancel_button:disabled { background-color: #313244; color: #585b70; }
                QLineEdit, QComboBox, QTableWidget { background-color: #11111b; border: 1px solid #45475a; border-radius: 4px; padding: 4px; }
                QComboBox QListView { background-color: #000000; border: 1px solid #45475a; color: #cdd6f4; }

                /* --- UPGRADED PROGRESS BAR --- */
                QProgressBar { 
                    border: 2px solid #45475a; 
                    border-radius: 6px; 
                    text-align: center; 
                    background-color: #11111b; 
                    color: #11111b; /* Dark text for contrast when filled */
                    font-weight: bold;
                    font-size: 11pt;
                    min-height: 25px; /* Makes it thicker and more readable */
                }
                QProgressBar::chunk { 
                    background-color: #a6e3a1; 
                    border-radius: 4px; 
                }

                QSlider::groove:horizontal { border: 1px solid #45475a; height: 8px; background: #11111b; border-radius: 4px; }
                QSlider::handle:horizontal { background: #89b4fa; border: 1px solid #89b4fa; width: 14px; margin-top: -4px; border-radius: 7px; }
            """
            self.setStyleSheet(dark_stylesheet)
        except Exception as e:
            logger.error(f"Error applying stylesheet: {e}")


# ==========================================
# 2. MAIN APPLICATION CLASS
# ==========================================

class DataEngineGUI(BaseAppWindow):
    def __init__(self):
        super().__init__(title="YOLO Data Engine: Auto-Annotator", width=950, height=800)

        try:
            self.deleted_rows_history = []
            self.current_output_paths = {}

            # --- TABS SETUP ---
            self.tabs = QTabWidget()
            self.setCentralWidget(self.tabs)

            # Tab 1: Engine
            self.engine_tab = QWidget()
            self.main_layout = QVBoxLayout(self.engine_tab)
            self.main_layout.setSpacing(15)

            self.setup_model_section()
            self.setup_parameters_section()
            self.setup_mapping_section()
            self.setup_io_section()

            self.tabs.addTab(self.engine_tab, "🚀 Processing Engine")

            # Tab 2: Results Gallery
            self.viewer_tab = YOLOViewerWidget()
            self.tabs.addTab(self.viewer_tab, "🖼️ Results Gallery")

            # Initialize Keybinds
            self.installEventFilter(self)
            self.table.installEventFilter(self)

            logger.info("GUI Components built successfully.")
        except Exception as e:
            logger.critical(f"Critical Failure during GUI Setup: {e}")
            QMessageBox.critical(self, "Setup Error", f"The app failed to start: {e}")

    # --- UI Setup Modules ---

    def setup_model_section(self):
        try:
            group_box = QGroupBox("1. Pipeline Configuration")
            main_vbox = QVBoxLayout()

            pipeline_layout = QHBoxLayout()
            pipeline_layout.addWidget(QLabel("Pipeline Mode:"))
            self.combo_pipeline = QComboBox()
            self.combo_pipeline.setItemDelegate(QStyledItemDelegate())
            self.combo_pipeline.addItems(["Zero-Shot Annotation (SAM 3.1)"])
            pipeline_layout.addWidget(self.combo_pipeline)
            main_vbox.addLayout(pipeline_layout)

            self.model_stack = QStackedWidget()
            self.page_zero_shot = QWidget()
            layout_zero_shot = QHBoxLayout(self.page_zero_shot)
            layout_zero_shot.setContentsMargins(0, 10, 0, 0)

            layout_zero_shot.addWidget(QLabel("Model:"))
            self.combo_sam = QComboBox()
            self.combo_sam.setItemDelegate(QStyledItemDelegate())
            self.combo_sam.addItems(["SAM 3.1 Multiplex (Standard)"])
            layout_zero_shot.addWidget(self.combo_sam)

            layout_zero_shot.addWidget(QLabel("Format:"))
            self.combo_format = QComboBox()
            self.combo_format.setItemDelegate(QStyledItemDelegate())
            self.combo_format.addItems(["YOLO BBox", "YOLO OBB", "YOLO Seg"])
            layout_zero_shot.addWidget(self.combo_format)

            layout_zero_shot.addWidget(QLabel("Device:"))
            self.combo_device = QComboBox()
            self.combo_device.setItemDelegate(QStyledItemDelegate())
            self.combo_device.addItems(["cuda:0", "cpu", "mps"])
            layout_zero_shot.addWidget(self.combo_device)

            # --- CUSTOM EDITABLE BATCH SIZE ---
            batch_container = QWidget()
            batch_hbox = QHBoxLayout(batch_container)
            batch_hbox.setContentsMargins(0, 0, 0, 0)
            batch_hbox.addWidget(QLabel("Batch Size:"))

            self.combo_batch = QComboBox()
            self.combo_batch.setEditable(True)
            self.combo_batch.setValidator(QIntValidator(1, 128))
            self.combo_batch.addItems(["1", "4", "8", "16", "32"])
            self.combo_batch.currentTextChanged.connect(self.check_batch_warning)
            batch_hbox.addWidget(self.combo_batch)

            self.lbl_batch_warning = QLabel("⚠ High VRAM")
            self.lbl_batch_warning.setStyleSheet("color: #f9e2af; font-weight: bold;")

            sp = self.lbl_batch_warning.sizePolicy()
            sp.setRetainSizeWhenHidden(True)
            self.lbl_batch_warning.setSizePolicy(sp)
            self.lbl_batch_warning.hide()

            batch_hbox.addWidget(self.lbl_batch_warning)
            layout_zero_shot.addWidget(batch_container)

            self.model_stack.addWidget(self.page_zero_shot)
            main_vbox.addWidget(self.model_stack)
            group_box.setLayout(main_vbox)
            self.main_layout.addWidget(group_box)
        except Exception as e:
            logger.error(f"Error setting up model section: {e}")

    def setup_parameters_section(self):
        """Simplified to only show the relevant Confidence Threshold slider."""
        try:
            group_box = QGroupBox("2. Model Sensitivity")
            layout = QVBoxLayout()

            def create_slider(label_text, default, tip):
                row = QHBoxLayout()
                lbl = QLabel(f"{label_text}: {default / 100.0:.2f}")
                lbl.setMinimumWidth(180)
                lbl.setToolTip(tip)
                sld = QSlider(Qt.Orientation.Horizontal)
                sld.setRange(1, 100)
                sld.setValue(default)
                sld.setToolTip(tip)
                row.addWidget(lbl)
                row.addWidget(sld)
                return row, lbl, sld

            # Only keeping the Confidence Threshold
            r1, self.lbl_conf_thresh, self.slider_conf = create_slider(
                "Confidence Threshold", 25, "How sure the model must be to label an object."
            )
            self.slider_conf.valueChanged.connect(
                lambda v: self.lbl_conf_thresh.setText(f"Confidence Threshold: {v / 100.0:.2f}")
            )
            layout.addLayout(r1)

            group_box.setLayout(layout)
            self.main_layout.addWidget(group_box)
        except Exception as e:
            logger.error(f"Error setting up parameter section: {e}")

    def setup_mapping_section(self):
        try:
            group_box = QGroupBox("3. Class Configuration")
            layout = QVBoxLayout()
            self.lbl_mapping_instructions = QLabel(
                "Map YOLO names to Keywords. Use | to separate multiple keywords (e.g. plate | vanity).")
            layout.addWidget(self.lbl_mapping_instructions)
            self.table = QTableWidget(0, 2)
            self.table.setHorizontalHeaderLabels(["YOLO Class Name", "Search Keywords (Prompt)"])
            self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            layout.addWidget(self.table)

            btn_layout = QHBoxLayout()
            self.btn_add_class = QPushButton("+ Add Class")
            self.btn_add_class.clicked.connect(self.add_class_row)
            self.btn_remove_class = QPushButton("- Remove Selected")
            self.btn_remove_class.clicked.connect(self.remove_class_row)
            self.btn_undo = QPushButton("↺ Undo")
            self.btn_undo.setEnabled(False)
            self.btn_undo.clicked.connect(self.undo_remove)
            btn_layout.addWidget(self.btn_add_class)
            btn_layout.addWidget(self.btn_remove_class)
            btn_layout.addWidget(self.btn_undo)
            btn_layout.addStretch()
            layout.addLayout(btn_layout)

            self.add_class_row()
            group_box.setLayout(layout)
            self.main_layout.addWidget(group_box)
        except Exception as e:
            logger.error(f"Error setting up mapping section: {e}")

    def setup_io_section(self):
        try:
            group_box = QGroupBox("4. Input / Output & Processing")
            layout = QVBoxLayout()

            row_in = QHBoxLayout()
            self.input_path_display = QLineEdit()
            self.input_path_display.setReadOnly(True)

            btn_folder = QPushButton("Select Folder")
            btn_folder.clicked.connect(self.browse_input_folder)

            btn_files = QPushButton("Select File(s)")
            btn_files.clicked.connect(self.browse_input_files)

            row_in.addWidget(QLabel("Input Images:"))
            row_in.addWidget(self.input_path_display)
            row_in.addWidget(btn_folder)
            row_in.addWidget(btn_files)
            layout.addLayout(row_in)

            row_out = QHBoxLayout()
            self.output_path_display = QLineEdit()
            self.output_path_display.setReadOnly(True)
            btn_out = QPushButton("Browse")
            btn_out.clicked.connect(self.browse_output)

            row_out.addWidget(QLabel("Output Dataset:"))
            row_out.addWidget(self.output_path_display)
            row_out.addWidget(btn_out)
            layout.addLayout(row_out)

            btn_layout = QHBoxLayout()
            self.btn_start = QPushButton("START AUTO-ANNOTATION")
            self.btn_start.clicked.connect(self.start_annotation_pipeline)
            self.btn_start.setObjectName("start_button")
            self.btn_start.setMinimumHeight(40)

            self.btn_cancel = QPushButton("CANCEL")
            self.btn_cancel.clicked.connect(self.request_cancel)
            self.btn_cancel.setObjectName("cancel_button")
            self.btn_cancel.setMinimumHeight(40)
            self.btn_cancel.setEnabled(False)

            btn_layout.addWidget(self.btn_start, stretch=3)
            btn_layout.addWidget(self.btn_cancel, stretch=1)
            layout.addLayout(btn_layout)

            self.progress_bar = QProgressBar()
            self.progress_bar.setValue(0)
            # Center the text format percentage
            self.progress_bar.setFormat("%p%")
            layout.addWidget(self.progress_bar)

            group_box.setLayout(layout)
            self.main_layout.addWidget(group_box)
        except Exception as e:
            logger.error(f"Error setting up IO section: {e}")

    # ==========================================
    # 3. INTERFACE LOGIC
    # ==========================================

    def start_annotation_pipeline(self):
        try:
            logger.info("Pipeline Start requested.")

            try:
                batch_size = int(self.combo_batch.currentText())
            except ValueError:
                batch_size = 1

            device = self.combo_device.currentText()
            conf_thresh = self.slider_conf.value() / 100.0

            # --- DUMMY VALUES FOR BACKEND SIGNATURE ---
            # Even though we removed the sliders, the SAM3InferenceThread __init__
            # still expects 10 arguments. We pass these silently so we don't break it.
            prompt_thresh_dummy = 0.25
            nms_thresh_dummy = 0.40

            if not self.input_path_display.text() or not self.output_path_display.text():
                raise ValueError("Input or Output path is empty.")

            self.btn_start.setEnabled(False)
            self.btn_start.setText("INITIALIZING...")
            self.btn_cancel.setEnabled(True)
            self.btn_cancel.setText("CANCEL")
            self.progress_bar.setValue(0)

            selected_model = self.combo_sam.currentText()
            self.downloader_thread = ModelDownloaderThread(selected_model)
            self.downloader_thread.progress_updated.connect(self.progress_bar.setValue)
            self.downloader_thread.status_updated.connect(lambda msg: self.btn_start.setText(msg))
            self.downloader_thread.error_occurred.connect(self.handle_pipeline_error)

            self.downloader_thread.download_complete.connect(
                lambda path: self.start_ai_inference(path, batch_size, device, conf_thresh, prompt_thresh_dummy,
                                                     nms_thresh_dummy)
            )
            self.downloader_thread.start()

        except Exception as e:
            logger.error(f"Pipeline failed to start: {e}")
            self.handle_pipeline_error(str(e))

    def start_ai_inference(self, model_filepath, batch, device, conf, prompt, nms):
        try:
            input_folder = self.input_path_display.text()
            output_folder = self.output_path_display.text()

            class_list = []
            for row in range(self.table.rowCount()):
                item = self.table.item(row, 1)
                if item and item.text().strip():
                    class_list.append(item.text().strip())

            out_format = self.combo_format.currentText()

            from core.dataset_utils import prepare_output_folders, generate_yaml
            output_paths = prepare_output_folders(output_folder)

            class_names = []
            for row in range(self.table.rowCount()):
                item = self.table.item(row, 0)
                class_names.append(item.text().strip() if item else f"class_{row}")
            generate_yaml(output_folder, class_names)

            self.current_output_paths = output_paths

            from core.inference_sam import SAM3InferenceThread
            self.inference_thread = SAM3InferenceThread(
                model_filepath, input_folder, output_paths, class_list,
                batch, device, conf, prompt, nms, out_format
            )

            self.inference_thread.status_updated.connect(lambda m: self.btn_start.setText(m))
            self.inference_thread.progress_updated.connect(self.progress_bar.setValue)
            self.inference_thread.error_occurred.connect(self.handle_pipeline_error)
            self.inference_thread.finished.connect(self.finalize_pipeline)
            self.inference_thread.start()
        except Exception as e:
            logger.error(f"Inference initialization failed: {e}")
            self.handle_pipeline_error(f"Inference Error: {e}")

    def handle_pipeline_error(self, error_message):
        logger.error(f"PIPELINE CRASH: {error_message}")
        self.btn_start.setText("ERROR. CHECK LOG.")
        self.btn_start.setStyleSheet("background-color: #f38ba8; color: #11111b; font-weight: bold;")
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setText("CANCEL")
        QMessageBox.critical(self, "Pipeline Error", f"Something went wrong:\n\n{error_message}")

    def finalize_pipeline(self, message):
        logger.info(f"Pipeline complete: {message}")
        self.btn_start.setEnabled(True)
        self.btn_start.setText("START AUTO-ANNOTATION")
        self.btn_start.setStyleSheet("")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setText("CANCEL")

        if "cancel" in message.lower():
            QMessageBox.warning(self, "Process Cancelled", "The auto-annotation process was stopped early.")
            return

        self.progress_bar.setValue(100)

        if self.current_output_paths:
            img_dir = self.current_output_paths.get('train_img', '')
            lbl_dir = self.current_output_paths.get('train_lbl', '')

            class_names = []
            for row in range(self.table.rowCount()):
                item = self.table.item(row, 0)
                class_names.append(item.text().strip() if item else f"class_{row}")

            self.viewer_tab.load_dataset(img_dir, lbl_dir, class_names)
            self.tabs.setCurrentWidget(self.viewer_tab)

        QMessageBox.information(self, "Success", message)

    def check_batch_warning(self, text):
        try:
            val = int(text)
            if val >= 16:
                self.lbl_batch_warning.show()
            else:
                self.lbl_batch_warning.hide()
        except ValueError:
            self.lbl_batch_warning.hide()

    def browse_input_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if path:
            self.input_path_display.setText(path)

    def browse_input_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Image(s) or Video", "",
                                                "Media Files (*.jpg *.jpeg *.png *.mp4 *.avi)")
        if paths:
            self.input_path_display.setText(";".join(paths))

    def browse_output(self):
        f = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if f: self.output_path_display.setText(f)

    def add_class_row(self):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(""))
        self.table.setItem(r, 1, QTableWidgetItem(""))

    def remove_class_row(self):
        row = self.table.currentRow()
        if row >= 0:
            y = self.table.item(row, 0).text() if self.table.item(row, 0) else ""
            p = self.table.item(row, 1).text() if self.table.item(row, 1) else ""
            self.deleted_rows_history.append((row, y, p))
            self.table.removeRow(row)
            self.btn_undo.setEnabled(True)

    def undo_remove(self):
        if self.deleted_rows_history:
            idx, y, p = self.deleted_rows_history.pop()
            self.table.insertRow(idx)
            self.table.setItem(idx, 0, QTableWidgetItem(y))
            self.table.setItem(idx, 1, QTableWidgetItem(p))
            if not self.deleted_rows_history: self.btn_undo.setEnabled(False)

    def eventFilter(self, source, event):
        if event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and source is self.table:
                if self.table.state() != QAbstractItemView.State.EditingState: self.remove_class_row(); return True
            if event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_Z:
                fw = QApplication.focusWidget()
                if not isinstance(fw, (QLineEdit, QComboBox, QSlider)): self.undo_remove(); return True
        return super().eventFilter(source, event)

    def request_cancel(self):
        if hasattr(self, 'inference_thread') and self.inference_thread.isRunning():
            self.btn_cancel.setEnabled(False)
            self.btn_cancel.setText("STOPPING...")
            self.inference_thread.stop()
            logger.info("User requested pipeline cancellation.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    try:
        window = DataEngineGUI()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        logger.critical(f"The application crashed on startup: {e}")