import sys
import os
from pathlib import Path
from types import SimpleNamespace
from PySide6.QtCore import Qt, Slot, QSize, QThread, Signal, QByteArray
from PySide6.QtGui import QIcon, QShortcut, QKeySequence, QPainter, QPen, QPixmap, QImage, QIntValidator, QColor, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QTextEdit, QSpinBox,
    QDoubleSpinBox, QCheckBox, QMessageBox, QDialog, QComboBox,
    QProgressBar, QButtonGroup, QLineEdit, QInputDialog
)
import numpy as np
import cv2
from PIL import Image
import io

from draw import DrawingHandler
from inference import InferenceHandler

# Add a model loading thread to prevent UI freezing
class ModelLoaderThread(QThread):
    finished = Signal(bool)
    
    def __init__(self, inference_handler, model_path, scheduler_name):
        super().__init__()
        self.inference_handler = inference_handler
        self.model_path = model_path
        self.scheduler_name = scheduler_name
        
    def run(self):
        success = self.inference_handler.load_model(self.model_path, self.scheduler_name)
        self.finished.emit(success)

# Add image generation thread with streaming support
class GenerationThread(QThread):
    finished = Signal(Image.Image)
    progress = Signal(int, int, QByteArray)
    
    def __init__(self, inference_handler, drawing, props, stream_updates=False):
        super().__init__()
        self.inference_handler = inference_handler
        self.drawing = drawing
        self.props = props
        self.stream_updates = stream_updates
        
    def run(self):
        try:
            if self.stream_updates:
                # Define callback for streaming updates
                def update_callback(step, total_steps, img_bytes):
                    self.progress.emit(step, total_steps, QByteArray(img_bytes))
                
                # Generate with streaming
                result = self.inference_handler.generate_image(
                    self.drawing, 
                    self.props,
                    callback=update_callback
                )
            else:
                # Generate without streaming
                result = self.inference_handler.generate_image(
                    self.drawing, 
                    self.props
                )
            
            self.finished.emit(result)
        except Exception as e:
            print(f"Error in generation thread: {e}")

# Add Canny Edge processing thread
class CannyEdgeThread(QThread):
    finished = Signal(np.ndarray)
    
    def __init__(self, image, low_threshold, high_threshold):
        super().__init__()
        self.image = image
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        
    def run(self):
        try:
            # Convert PIL Image to cv2 format
            cv_image = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            # Detect edges
            edges = cv2.Canny(blurred, self.low_threshold, self.high_threshold)
            # Convert back to RGB
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            self.finished.emit(edges_rgb)
        except Exception as e:
            print(f"Error in Canny edge detection: {e}")

class CannyEdgeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Canny Edge Detection")
        self.setMinimumSize(400, 500)
        self.source_image = None
        self.edge_image = None
        self.edge_thread = None
        self.loading_indicator = LoadingIndicator(self)
        self.setup_ui()
        
    def closeEvent(self, event):
        # Clean up any running thread before closing
        if self.edge_thread and self.edge_thread.isRunning():
            self.edge_thread.wait()
        super().closeEvent(event)
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Drop area for image
        self.drop_label = QLabel("Drag and drop image here")
        self.drop_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #666;
                border-radius: 5px;
                padding: 20px;
                background: #2A2A2A;
            }
        """)
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setMinimumSize(300, 200)
        self.drop_label.setAcceptDrops(True)
        
        # Override drop events
        self.drop_label.dragEnterEvent = self.dragEnterEvent
        self.drop_label.dropEvent = self.dropEvent
        
        layout.addWidget(self.drop_label)
        
        # Threshold controls with loading indicator
        thresholds_widget = QWidget()
        thresholds_layout = QVBoxLayout(thresholds_widget)
        
        # Low threshold
        low_layout = QHBoxLayout()
        low_layout.addWidget(QLabel("Low Threshold:"))
        self.low_threshold = QSlider(Qt.Horizontal)
        self.low_threshold.setRange(0, 255)
        self.low_threshold.setValue(100)
        self.low_value = QLabel("100")
        low_layout.addWidget(self.low_threshold)
        low_layout.addWidget(self.low_value)
        thresholds_layout.addLayout(low_layout)
        
        # High threshold
        high_layout = QHBoxLayout()
        high_layout.addWidget(QLabel("High Threshold:"))
        self.high_threshold = QSlider(Qt.Horizontal)
        self.high_threshold.setRange(0, 255)
        self.high_threshold.setValue(200)
        self.high_value = QLabel("200")
        high_layout.addWidget(self.high_threshold)
        high_layout.addWidget(self.high_value)
        thresholds_layout.addLayout(high_layout)
        
        # Add loading indicator to thresholds widget
        self.loading_indicator.setParent(thresholds_widget)
        self.loading_indicator.move(10, 10)
        self.loading_indicator.hide()
        
        layout.addWidget(thresholds_widget)
        
        # Result preview
        self.result_label = QLabel()
        self.result_label.setStyleSheet("border: 1px solid #333;")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumSize(300, 200)
        layout.addWidget(self.result_label)
        
        # Apply button
        self.apply_btn = QPushButton("Apply to Drawing")
        self.apply_btn.setEnabled(False)
        layout.addWidget(self.apply_btn)
        
        # Connect signals with debouncing
        from PySide6.QtCore import QTimer
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.update_edge_detection)
        
        self.low_threshold.valueChanged.connect(
            lambda v: (self.low_value.setText(str(v)), self.schedule_update())
        )
        self.high_threshold.valueChanged.connect(
            lambda v: (self.high_value.setText(str(v)), self.schedule_update())
        )
        
    def schedule_update(self):
        """Debounce the edge detection update"""
        self.update_timer.start(300)  # Wait for 300ms of no changes before updating
        
    def center_loading_indicator(self):
        x = (self.width() - self.loading_indicator.width()) // 2
        y = (self.height() - self.loading_indicator.height()) // 2
        self.loading_indicator.move(x, y)
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.center_loading_indicator()
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()
            
    def dropEvent(self, event: QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            self.load_image(files[0])
            
    def load_image(self, path):
        try:
            # Load image and convert to RGB
            self.source_image = Image.open(path).convert('RGB')
            
            # Calculate scaling factor to fit within 512x512 while maintaining aspect ratio
            scale = min(512 / self.source_image.width, 512 / self.source_image.height)
            new_width = int(self.source_image.width * scale)
            new_height = int(self.source_image.height * scale)
            
            # Resize image using bilinear interpolation
            self.source_image = self.source_image.resize((new_width, new_height), Image.BILINEAR)
            
            # Update preview
            self.update_edge_detection()
            self.apply_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
            
    def update_edge_detection(self):
        if not self.source_image:
            return
            
        # Clean up previous thread if it exists
        if self.edge_thread and self.edge_thread.isRunning():
            self.edge_thread.wait()
            
        self.loading_indicator.start()
        
        # Create and start the edge detection thread
        self.edge_thread = CannyEdgeThread(
            self.source_image,
            self.low_threshold.value(),
            self.high_threshold.value()
        )
        self.edge_thread.finished.connect(self.on_edge_detection_complete)
        self.edge_thread.start()
        
    def on_edge_detection_complete(self, edges):
        self.loading_indicator.stop()
        self.edge_image = Image.fromarray(edges)
        
        # Display result
        qimage = QImage(
            edges.data,
            edges.shape[1],
            edges.shape[0],
            edges.shape[1] * 3,
            QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qimage)
        self.result_label.setPixmap(pixmap.scaled(
            self.result_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
        
    def get_edge_image(self):
        return self.edge_image if self.edge_image else None

class BrushSettingsDialog(QDialog):
    def __init__(self, drawing_handler, parent=None):
        super().__init__(parent)
        self.drawing_handler = drawing_handler
        self.setWindowTitle("Brush Settings")
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Brush size
        size_layout = QHBoxLayout()
        size_label = QLabel("Size:")
        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 100)
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(1, 100)
        self.size_slider.setValue(self.drawing_handler.drawing_area.brush_size)
        
        size_layout.addWidget(size_label)
        size_layout.addWidget(self.size_slider)
        size_layout.addWidget(self.size_spin)
        layout.addLayout(size_layout)
        
        # Connect both controls
        self.size_slider.valueChanged.connect(self.size_spin.setValue)
        self.size_spin.valueChanged.connect(self.size_slider.setValue)
        self.size_slider.valueChanged.connect(self.drawing_handler.set_brush_size)
        
        # Brush opacity
        opacity_layout = QHBoxLayout()
        opacity_label = QLabel("Opacity:")
        self.opacity_spin = QDoubleSpinBox()
        self.opacity_spin.setRange(0, 1)
        self.opacity_spin.setSingleStep(0.1)
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(int(self.drawing_handler.drawing_area.brush_opacity / 255 * 100))
        
        opacity_layout.addWidget(opacity_label)
        opacity_layout.addWidget(self.opacity_slider)
        opacity_layout.addWidget(self.opacity_spin)
        layout.addLayout(opacity_layout)
        
        # Connect opacity controls
        self.opacity_slider.valueChanged.connect(
            lambda v: self.opacity_spin.setValue(v / 100)
        )
        self.opacity_spin.valueChanged.connect(
            lambda v: self.opacity_slider.setValue(int(v * 100))
        )
        self.opacity_slider.valueChanged.connect(
            lambda v: self.drawing_handler.set_brush_opacity(v / 100)
        )

class LoadingIndicator(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(0, 0)  # Smaller size for inline use
        self.angle = 0
        self.timer = None
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
    def start(self):
        from PySide6.QtCore import QTimer
        self.setFixedSize(48, 48)  # Smaller size for inline use
        if not self.timer:
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_rotation)
        self.timer.start(50)  # Update every 50ms
        self.show()
        
    def stop(self):
        if self.timer:
            self.timer.stop()
        self.hide()
        
    def update_rotation(self):
        self.angle = (self.angle + 10) % 360
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Set up the painter
        painter.translate(24, 24)  # Center of widget
        painter.rotate(self.angle)
        
        # Draw the spinner
        pen = QPen(Qt.white)
        pen.setWidth(3)  # Thinner lines for smaller spinner
        painter.setPen(pen)
        
        for i in range(8):
            painter.rotate(45)
            opacity = 0.3 + (i / 8) * 0.7
            pen.setColor(QColor(255, 255, 255, int(opacity * 255)))
            painter.setPen(pen)
            painter.drawLine(0, 6, 0, 12)  # Shorter lines for smaller spinner

class SaveSettingsDialog(QDialog):
    def __init__(self, save_dir, base_filename, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save Settings")
        self.setMinimumWidth(400)
        
        self.save_dir = save_dir
        self.base_filename = base_filename
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Output directory
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Output Directory:"))
        
        self.dir_input = QLineEdit(self.save_dir)
        self.dir_input.setReadOnly(True)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_directory)
        
        dir_layout.addWidget(self.dir_input)
        dir_layout.addWidget(browse_btn)
        layout.addLayout(dir_layout)
        
        # Base filename
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Base Filename:"))
        
        self.name_input = QLineEdit(self.base_filename)
        name_layout.addWidget(self.name_input)
        layout.addLayout(name_layout)
        
        # Example of how files will be named
        self.example_label = QLabel(f"Example: {self.base_filename}0.png, {self.base_filename}1.png, ...")
        layout.addWidget(self.example_label)
        
        # Connect signals
        self.name_input.textChanged.connect(self.update_example)
        
        # Buttons
        button_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
    def browse_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.dir_input.text())
        if dir_path:
            self.dir_input.setText(dir_path)
            self.save_dir = dir_path
            
    def update_example(self, text):
        self.base_filename = text
        self.example_label.setText(f"Example: {text}0.png, {text}1.png, ...")
        
    def get_settings(self):
        return self.dir_input.text(), self.name_input.text()

class DoodlePixUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DoodlePix")
        self.setStyleSheet("background-color: black; color: white;")
        
        # Initialize handlers
        self.drawing_handler = DrawingHandler()
        self.inference_handler = InferenceHandler()
        
        # Initialize image history
        self.generated_images = []
        self.current_image_index = -1
        
        # Initialize status and progress BEFORE setup_ui
        self.status_label = QLabel()
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        
        # Initialize loading indicator
        self.loading_indicator = LoadingIndicator()
        
        # Initialize Canny Edge dialog
        self.canny_dialog = None
        
        # Initialize generation thread
        self.generation_thread = None
        
        # Initialize save settings
        self.save_counter = 0
        self.save_dir = str(Path(__file__).parent / "output")
        self.base_filename = "image_"
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Setup UI and shortcuts
        self.setup_ui()
        self.setup_shortcuts()
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(0)  # Reduced spacing between panels
        
        # Left panel - Drawing tools and canvas
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(8)
        
        # Tools row
        tools_widget = QWidget()
        tools_layout = QHBoxLayout(tools_widget)
        tools_layout.setSpacing(8)
        tools_layout.setContentsMargins(8, 8, 8, 8)
        
        # Style for tool buttons
        button_style = """
            QPushButton {
                background-color: #2A2A2A;
                border: none;
                border-radius: 5px;
                padding: 5px;
                font-size: 24px;  /* Increase emoji size */
            }
            QPushButton:hover {
                background-color: #f3d44b;
            }
            QPushButton:checked {
                background-color: #55fca8;
            }
        """
        
        # Create tool buttons with consistent styling
        for btn_data in [
            ("✏️" if not self.drawing_handler.drawing_area.eraser_mode else "🩹", "Draw/Erase Toggle", self.drawing_handler.toggle_eraser, True),
            ("🎨", "Color Picker", self.drawing_handler.pick_color, False),
            ("📦", "Bounding Box", self.drawing_handler.toggle_shape_drawing, True),
            ("↩️", "Undo", self.drawing_handler.undo, False),
            ("↪️", "Redo", self.drawing_handler.redo, False),
            ("🗑️", "Clear Canvas", self.drawing_handler.clear_canvas, False),
            ("🖼️", "Load Background", self.load_background_image, False),
            ("🕸️", "Canny Edge", self.show_canny_dialog, False),
            ("💾", "Save Drawing", self.save_drawing, False),
            ("🖌️", "Brush Settings", self.show_brush_settings, False),
        ]:
            btn = QPushButton(btn_data[0])
            btn.setToolTip(btn_data[1])
            btn.setCheckable(btn_data[3])
            btn.setFixedSize(64, 64)
            btn.setStyleSheet(button_style)
            btn.clicked.connect(btn_data[2])
            tools_layout.addWidget(btn)
            
        tools_layout.addStretch()  # Add stretch at end
        left_layout.addWidget(tools_widget)
        
        # Model loading section with loading indicator
        model_widget = QWidget()
        model_layout = QHBoxLayout(model_widget)
        model_layout.setContentsMargins(5, 0, 5, 5)
        
        # Style for larger buttons with bigger icons
        large_button_style = """
            QPushButton {
                background-color: #2A2A2A;
                border: none;
                border-radius: 4px;
                padding: 8px;
                font-size: 24px;  /* Larger emoji size */
                min-width: 64px;
                min-height: 32px;
            }
            QPushButton:hover {
                background-color: #f3d44b;
            }
        """
        
        # Navigation buttons with larger size
        nav_button_style = """
            QPushButton {
                background-color: #000000;
                border: none;
                border-radius: 8px;
                padding: 8px;
                font-size: 24px;
                min-width: 32px;
                min-height: 32px;
            }
            QPushButton:hover {
                font-weight: bold;
                font-size: 32px;
            }
        """
        
        browse_btn = QPushButton("📂")
        browse_btn.setToolTip("Browse Model Path")
        browse_btn.clicked.connect(self.browse_model)
        browse_btn.setStyleSheet(nav_button_style)
        
        self.reload_cb = QCheckBox("Reload")
        self.reload_cb.setToolTip("Reload Model")
        self.reload_cb.setStyleSheet("QCheckBox { font-size: 16px; }")
        
        # Add loading indicator next to the load button
        model_layout.addWidget(browse_btn)
        model_layout.addWidget(self.loading_indicator)
        model_layout.addWidget(self.reload_cb)
        model_layout.setContentsMargins(0, -10, 0, 0)
        model_layout.addStretch()
        
        left_layout.addWidget(model_widget)
        
        # Status bar
        status_layout = QHBoxLayout()
        status_layout.addWidget(self.progress_bar)
        status_layout.addWidget(self.status_label)
        left_layout.addLayout(status_layout)
        
        # Generate button and navigation controls above drawing area
        gen_nav_widget = QWidget()
        gen_nav_layout = QHBoxLayout(gen_nav_widget)
        gen_nav_layout.setContentsMargins(4, 32, 4, 32)
        
        # Generate button
        self.generate_btn = QPushButton("🚀RUN🚀")
        self.generate_btn.setToolTip("Generate Image (Ctrl+G)")
        self.generate_btn.clicked.connect(self.generate_image)
        self.generate_btn.setStyleSheet(nav_button_style)
        gen_nav_layout.addWidget(self.generate_btn)
        
        
        prev_btn = QPushButton("⬅️")
        prev_btn.setToolTip("Previous Image")
        prev_btn.clicked.connect(self.prev_image)
        prev_btn.setStyleSheet(nav_button_style)
        
        next_btn = QPushButton("➡️")
        next_btn.setToolTip("Next Image")
        next_btn.clicked.connect(self.next_image)
        next_btn.setStyleSheet(nav_button_style)
        
        gen_nav_layout.addWidget(prev_btn)
        gen_nav_layout.addWidget(next_btn)
        
        left_layout.addWidget(gen_nav_widget)
        
        # Create a container for the drawing handler with minimal padding
        drawing_container = QWidget()
        drawing_container_layout = QVBoxLayout(drawing_container)
        drawing_container_layout.setContentsMargins(88,0,0,10)
        drawing_container_layout.addWidget(self.drawing_handler)
        
        left_layout.addWidget(drawing_container)
        
        # Right panel - Generation settings and results
        right_panel = QWidget()
        right_panel.setStyleSheet("""
            QLabel { color: #CCCCCC; }
            QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
                background-color: #2A2A2A;
                border: 1px solid #3A3A3A;
                border-radius: 5px;
                color: white;
                padding: 5px;
            }
            QPushButton {
                background-color: #2A2A2A;
                border: none;
                border-radius: 5px;
                padding: 8px;
                color: white;
            }
            QPushButton:hover {
                background-color: #3A3A3A;
            }
            QPushButton:checked {
                background-color: #4A4A4A;
            }
            QPushButton:checked:disabled {
                background-color: #5A5A5A;
            }
        """)
        right_layout = QVBoxLayout(right_panel)
        
        # Prompt input
        right_layout.addWidget(QLabel("Prompt:"))
        self.prompt_input = QTextEdit()
        self.prompt_input.setMaximumHeight(40)
        self.prompt_input.setPlaceholderText("write the things you want in the image, followed by the colors")
        right_layout.addWidget(self.prompt_input)
        
        # Negative prompt
        right_layout.addWidget(QLabel("Negative Prompt:"))
        self.neg_prompt_input = QTextEdit()
        self.neg_prompt_input.setMaximumHeight(40)
        self.neg_prompt_input.setPlainText("")
        self.neg_prompt_input.setPlaceholderText("write what you don't want")
        right_layout.addWidget(self.neg_prompt_input)
        
        # Fidelity slider
        fidelity_layout = QHBoxLayout()
        fidelity_layout.addWidget(QLabel("Fidelity:"))
        self.fidelity_slider = QSlider(Qt.Horizontal)
        self.fidelity_slider.setRange(0, 9)
        self.fidelity_slider.setValue(5)
        self.fidelity_slider.setTickPosition(QSlider.TicksBelow)
        self.fidelity_slider.setTickInterval(1)
        self.fidelity_slider.setToolTip("Controls the fidelity of the image. 0 : the model is gonna be more creative, 9 : the model is gonna be extremely faithful to the drawing")
        self.fidelity_value_label = QLabel("4")
        self.fidelity_slider.valueChanged.connect(lambda v: self.fidelity_value_label.setText(str(v)))
        fidelity_layout.addWidget(self.fidelity_slider)
        fidelity_layout.addWidget(self.fidelity_value_label)
        right_layout.addLayout(fidelity_layout)
        
        # Style buttons
        style_layout = QHBoxLayout()
        style_layout.addWidget(QLabel("Style:"))
        self.style_buttons = []
        
        # Style button group for mutual exclusivity
        self.style_button_group = QButtonGroup(self)
        
        # Style button style
        style_button_style = """
            QPushButton {
                background-color: #f87e4e;
                border: 1px solid #ffffff;
                border-radius: 4px;
                padding: 4px;
                color: white;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #fded60;
                color: #f87e4e;
            }
            QPushButton:checked {
                background-color: #4eff92;
                color: #389fff;                
            }
        """
        
        for i, style_name in enumerate(["Normal⛺", "3D⏹️", "Outline🔲", "Flat⬜"]):
            btn = QPushButton(style_name)
            btn.setCheckable(True)
            btn.setStyleSheet(style_button_style)
            style_layout.addWidget(btn)
            self.style_buttons.append(btn)
            self.style_button_group.addButton(btn, i)
            
        # Set the first button (Normal) as checked by default
        self.style_buttons[0].setChecked(True)
        
        right_layout.addLayout(style_layout)
        
        # Generation parameters
        params_layout = QVBoxLayout()
        
        # Steps slider
        steps_layout = QHBoxLayout()
        steps_layout.addWidget(QLabel("Steps:"))
        self.steps_slider = QSlider(Qt.Horizontal)
        self.steps_slider.setRange(1, 60)
        self.steps_slider.setValue(20)
        self.steps_slider.setTickPosition(QSlider.TicksBelow)
        self.steps_slider.setTickInterval(5)
        self.steps_value_label = QLabel("24")
        self.steps_slider.valueChanged.connect(lambda v: self.steps_value_label.setText(str(v)))
        steps_layout.addWidget(self.steps_slider)
        steps_layout.addWidget(self.steps_value_label)
        params_layout.addLayout(steps_layout)
        
        # Guidance scale slider
        guidance_layout = QHBoxLayout()
        guidance_layout.addWidget(QLabel("Guidance:"))
        self.guidance_slider = QSlider(Qt.Horizontal)
        self.guidance_slider.setRange(0, 40)  # 0 to 20 with 0.5 steps (40 steps total)
        self.guidance_slider.setValue(15)     # 7.5 default
        self.guidance_value_label = QLabel("5.5")
        self.guidance_slider.valueChanged.connect(
            lambda v: self.guidance_value_label.setText(str(v / 2))
        )
        guidance_layout.addWidget(self.guidance_slider)
        guidance_layout.addWidget(self.guidance_value_label)
        params_layout.addLayout(guidance_layout)
        
        # Image guidance slider
        img_guidance_layout = QHBoxLayout()
        img_guidance_layout.addWidget(QLabel("Image Guidance:"))
        self.img_guidance_slider = QSlider(Qt.Horizontal)
        self.img_guidance_slider.setRange(2, 12)  # 1 to 6 with 0.5 steps (10 steps total)
        self.img_guidance_slider.setValue(3)      # 1.5 default
        self.img_guidance_value_label = QLabel("1.5")
        self.img_guidance_slider.valueChanged.connect(
            lambda v: self.img_guidance_value_label.setText(str(v / 2))
        )
        img_guidance_layout.addWidget(self.img_guidance_slider)
        img_guidance_layout.addWidget(self.img_guidance_value_label)
        params_layout.addLayout(img_guidance_layout)
        
        # Seed
        seed_layout = QHBoxLayout()
        seed_layout.addWidget(QLabel("Seed:"))
        self.seed_input = QLineEdit()
        self.seed_input.setValidator(QIntValidator())
        self.seed_input.setText("42")
        self.random_seed_cb = QCheckBox("Random")
        self.random_seed_cb.setChecked(True)
        seed_layout.addWidget(self.seed_input)
        seed_layout.addWidget(self.random_seed_cb)
        params_layout.addLayout(seed_layout)
        
        # Stream updates checkbox
        stream_layout = QHBoxLayout()
        stream_layout.addWidget(QLabel("Stream Updates:"))
        self.stream_updates_cb = QCheckBox("Show intermediate steps")
        self.stream_updates_cb.setChecked(True)
        self.stream_updates_cb.setToolTip("Show generation progress in real-time")
        stream_layout.addWidget(self.stream_updates_cb)
        params_layout.addLayout(stream_layout)
        
        right_layout.addLayout(params_layout)
         # Save generated image controls
        save_controls = QWidget()
        save_layout = QHBoxLayout(save_controls)
        save_layout.setContentsMargins(5, 5, 5, 5)
        
        self.save_result_btn = QPushButton("💾 Save Result")
        self.save_result_btn.setToolTip("Save generated image with auto-incrementing filename")
        self.save_result_btn.clicked.connect(self.save_generated_image)
        self.save_result_btn.setEnabled(False)  # Disabled until an image is generated
        
        self.save_settings_btn = QPushButton("⚙️")
        self.save_settings_btn.setToolTip("Configure save settings")
        self.save_settings_btn.clicked.connect(self.configure_save_settings)
        self.save_settings_btn.setFixedWidth(30)
        
        save_layout.addWidget(self.save_result_btn)
        save_layout.addWidget(self.save_settings_btn)
        
        right_layout.addWidget(save_controls)
        
        # Result display
        self.result_label = QLabel()
        self.result_label.setMinimumSize(512, 512)
        self.result_label.setMaximumSize(520, 520)
        self.result_label.setStyleSheet("border: 2px solid #333333; padding: 2px;")
        self.result_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.result_label)
        
        
        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
    def resizeEvent(self, event):
        # Center the loading indicator when window is resized
        super().resizeEvent(event)
        self.center_loading_indicator()
        
    def center_loading_indicator(self):
        # Center the loading indicator in the window
        x = (self.width() - self.loading_indicator.width()) // 2
        y = (self.height() - self.loading_indicator.height()) // 2
        self.loading_indicator.move(x, y)
        
    def setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.save_drawing)
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self.drawing_handler.undo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self).activated.connect(self.drawing_handler.redo)
        QShortcut(QKeySequence("Ctrl+G"), self).activated.connect(self.generate_image)
        
    @Slot()
    def show_brush_settings(self):
        dialog = BrushSettingsDialog(self.drawing_handler, self)
        dialog.exec()
        
    @Slot()
    def load_background_image(self):
        """Load an image as background for drawing"""
        success = self.drawing_handler.load_background_image()
        if success:
            self.status_label.setText("Background image loaded")
        
    @Slot()
    def browse_model(self):
        path = QFileDialog.getExistingDirectory(self, "Select Model Directory")
        if path and (self.reload_cb.isChecked() or not self.inference_handler.pipeline):
            # Show loading indicator
            self.status_label.setText("Loading model...")
            self.loading_indicator.start()
            
            # Create and start the loader thread
            self.loader_thread = ModelLoaderThread(
                self.inference_handler, 
                path, 
                "Euler Ancestral"
            )
            self.loader_thread.finished.connect(self.on_model_loaded)
            self.loader_thread.start()
            
    @Slot(bool)
    def on_model_loaded(self, success):
        # Hide loading indicator
        self.loading_indicator.stop()
        
        if success:
            self.status_label.setText("Model loaded successfully!")
        else:
            self.status_label.setText("Failed to load model")
            QMessageBox.critical(self, "Error", "Failed to load model")
            
    @Slot()
    def save_drawing(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Drawing", "", "PNG Files (*.png)"
        )
        if file_path:
            self.drawing_handler.get_image().save(file_path)
            
    @Slot()
    def generate_image(self):
        if not self.inference_handler.pipeline and not self.reload_cb.isChecked():
            QMessageBox.warning(self, "Error", "Please load a model first!")
            return
            
        # Disable generate button during generation
        self.generate_btn.setEnabled(False)
        
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.status_label.setText("Generating...")
        
        try:
            drawing = self.drawing_handler.get_image()
            
            # Build prompt with fidelity and style
            base_prompt = self.prompt_input.toPlainText()
            fidelity_value = self.fidelity_slider.value()
            
            # Add fidelity prefix
            prompt = f"f{fidelity_value}, {base_prompt}"
            
            # Add style if not Normal
            style_index = self.style_button_group.checkedId()
            if style_index == 1:  # 3D
                prompt = f"f{fidelity_value}, [3D] {base_prompt}"
            elif style_index == 2:  # Outline
                prompt = f"f{fidelity_value}, [outline] {base_prompt}"
            elif style_index == 3:  # Flat
                prompt = f"f{fidelity_value}, [flat] {base_prompt}"
            
            # Handle seed
            if self.random_seed_cb.isChecked():
                # Generate a random seed
                import random
                seed = random.randint(0, 999999)
                # Update the seed input field
                self.seed_input.setText(str(seed))
            else:
                # Use the seed from the input field
                try:
                    seed = int(self.seed_input.text())
                except ValueError:
                    seed = 42  # Default if invalid
                    self.seed_input.setText("42")
            
            props = SimpleNamespace(
                prompt=prompt,
                negative_prompt=self.neg_prompt_input.toPlainText(),
                num_inference_steps=self.steps_slider.value(),
                guidance_scale=self.guidance_slider.value() / 2,  # Convert from slider value
                image_guidance_scale=self.img_guidance_slider.value() / 2,  # Convert from slider value
                seed=seed
            )
            
            # Check if streaming is enabled
            stream_updates = self.stream_updates_cb.isChecked()
            
            # Create and start the generation thread
            self.generation_thread = GenerationThread(
                self.inference_handler,
                drawing,
                props,
                stream_updates
            )
            
            # Connect signals
            self.generation_thread.finished.connect(self.on_generation_complete)
            if stream_updates:
                self.generation_thread.progress.connect(self.on_generation_progress)
            
            # Start generation
            self.generation_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Generation failed: {str(e)}")
            self.status_label.setText("Generation failed!")
            self.progress_bar.hide()
            self.generate_btn.setEnabled(True)
    
    @Slot(int, int, QByteArray)
    def on_generation_progress(self, step, total_steps, img_bytes):
        """Handle intermediate generation updates"""
        # Update progress bar
        progress_percent = int((step / total_steps) * 100)
        self.progress_bar.setValue(progress_percent)
        
        # Update status
        self.status_label.setText(f"Generating... Step {step}/{total_steps}")
        
        # Convert bytes to image and display
        try:
            # Convert QByteArray to bytes
            bytes_data = img_bytes.data()
            
            # Load image from bytes
            buffer = io.BytesIO(bytes_data)
            pil_image = Image.open(buffer)
            
            # Convert to QPixmap and display
            image_array = np.array(pil_image)
            height, width, channels = image_array.shape
            qimage = QImage(
                image_array.data,
                width,
                height,
                width * channels,  # bytes per line
                QImage.Format_RGB888
            )
            pixmap = QPixmap.fromImage(qimage)
            self.result_label.setPixmap(pixmap.scaled(
                self.result_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
            
            # Process events to update UI
            QApplication.processEvents()
            
        except Exception as e:
            print(f"Error displaying intermediate result: {e}")
    
    @Slot(object)
    def on_generation_complete(self, result):
        """Handle generation completion"""
        # Add to history
        self.generated_images.append(result)
        self.current_image_index = len(self.generated_images) - 1
        
        # Display final result
        self.display_current_image()
        
        # Update UI
        self.progress_bar.setValue(100)
        self.status_label.setText("Generation complete!")
        self.progress_bar.hide()
        
        # Enable save button
        self.save_result_btn.setEnabled(True)
        
        # Re-enable generate button
        self.generate_btn.setEnabled(True)
        
    @Slot()
    def prev_image(self):
        """Navigate to the previous image in history"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_current_image()
            self.status_label.setText(f"Showing image {self.current_image_index + 1}/{len(self.generated_images)}")
            
    @Slot()
    def next_image(self):
        """Navigate to the next image in history"""
        if self.current_image_index < len(self.generated_images) - 1:
            self.current_image_index += 1
            self.display_current_image()
            self.status_label.setText(f"Showing image {self.current_image_index + 1}/{len(self.generated_images)}")
            
    def display_current_image(self):
        """Display the current image from history"""
        if 0 <= self.current_image_index < len(self.generated_images):
            image = self.generated_images[self.current_image_index]
            # Convert PIL Image to QPixmap using numpy as intermediate
            image_array = np.array(image)
            height, width, channels = image_array.shape
            qimage = QImage(
                image_array.data,
                width,
                height,
                width * channels,  # bytes per line
                QImage.Format_RGB888
            )
            pixmap = QPixmap.fromImage(qimage)
            self.result_label.setPixmap(pixmap.scaled(
                self.result_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))

    @Slot()
    def show_canny_dialog(self):
        if not self.canny_dialog:
            self.canny_dialog = CannyEdgeDialog(self)
            self.canny_dialog.apply_btn.clicked.connect(self.apply_canny_edge)
        self.canny_dialog.show()
        
    @Slot()
    def apply_canny_edge(self):
        edge_image = self.canny_dialog.get_edge_image()
        if edge_image:
            # Use the edge image directly without inversion
            self.drawing_handler.load_background_image_from_pil(edge_image)
            self.status_label.setText("Canny edge applied as background")
            self.canny_dialog.hide()

    @Slot()
    def save_generated_image(self):
        """Save the current generated image with auto-incrementing filename"""
        if 0 <= self.current_image_index < len(self.generated_images):
            # Get current image
            image = self.generated_images[self.current_image_index]
            
            # Create filename with counter
            filename = f"{self.base_filename}{self.save_counter}.png"
            filepath = os.path.join(self.save_dir, filename)
            
            try:
                # Ensure directory exists
                os.makedirs(self.save_dir, exist_ok=True)
                
                # Save image
                image.save(filepath)
                
                # Increment counter for next save
                self.save_counter += 1
                
                # Update status
                self.status_label.setText(f"Saved to {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "No image to save!")
            
    @Slot()
    def configure_save_settings(self):
        """Configure save directory and base filename"""
        dialog = SaveSettingsDialog(self.save_dir, self.base_filename, self)
        if dialog.exec() == QDialog.Accepted:
            self.save_dir, self.base_filename = dialog.get_settings()
            # Reset counter when settings change
            self.save_counter = 0

    def closeEvent(self, event):
        # Clean up any running threads before closing
        if hasattr(self, 'loader_thread') and self.loader_thread.isRunning():
            self.loader_thread.wait()
        if hasattr(self, 'generation_thread') and self.generation_thread and self.generation_thread.isRunning():
            self.generation_thread.wait()
        if self.canny_dialog:
            self.canny_dialog.close()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DoodlePixUI()
    window.show()
    sys.exit(app.exec())