from PySide6.QtCore import Qt, Signal, QPoint, QRectF, QSize, QPointF
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QPainterPath, QTransform, QBrush
from PySide6.QtWidgets import QWidget, QColorDialog, QScrollArea, QFrame, QFileDialog, QVBoxLayout, QHBoxLayout, QPushButton
from PIL import Image
import numpy as np
import math

class BoundingBox:
    def __init__(self, x, y, width, height):
        self.rect = QRectF(x, y, width, height)
        self.initial_rect = QRectF(x, y, width, height)  # Store initial size for scaling
        self.rotation = 0
        self.handle_size = 8
        self.rotate_handle_size = 12
        self.drag_handle_size = 16  # Size for the drag handle
        self.selected_handle = None
        self.dragging = False
        self.drag_start = None
        self.original_rect = None
        self.original_rotation = None
        self.scale_x = 1.0
        self.scale_y = 1.0
        
    def contains(self, point):
        # Transform point based on rotation
        center = self.rect.center()
        transform = QTransform()
        transform.translate(center.x(), center.y())
        transform.rotate(-self.rotation)
        transform.translate(-center.x(), -center.y())
        transformed_point = transform.map(point)
        return self.rect.contains(transformed_point)
        
    def get_handle_at(self, point):
        # Check drag handle first
        drag_handle_pos = self.get_drag_handle_pos()
        drag_handle_rect = QRectF(
            drag_handle_pos.x() - self.drag_handle_size/2,
            drag_handle_pos.y() - self.drag_handle_size/2,
            self.drag_handle_size,
            self.drag_handle_size
        )
        if drag_handle_rect.contains(point):
            return "drag"
            
        # Check rotate handle
        rotate_handle_pos = self.get_rotate_handle_pos()
        rotate_handle_rect = QRectF(
            rotate_handle_pos.x() - self.rotate_handle_size/2,
            rotate_handle_pos.y() - self.rotate_handle_size/2,
            self.rotate_handle_size,
            self.rotate_handle_size
        )
        if rotate_handle_rect.contains(point):
            return "rotate"
            
        # Transform point based on rotation
        center = self.rect.center()
        transform = QTransform()
        transform.translate(center.x(), center.y())
        transform.rotate(-self.rotation)
        transform.translate(-center.x(), -center.y())
        transformed_point = transform.map(point)
        
        # Check resize handles
        handles = {
            "top-left": QRectF(self.rect.left() - self.handle_size/2, self.rect.top() - self.handle_size/2, 
                              self.handle_size, self.handle_size),
            "top-right": QRectF(self.rect.right() - self.handle_size/2, self.rect.top() - self.handle_size/2,
                               self.handle_size, self.handle_size),
            "bottom-left": QRectF(self.rect.left() - self.handle_size/2, self.rect.bottom() - self.handle_size/2,
                                 self.handle_size, self.handle_size),
            "bottom-right": QRectF(self.rect.right() - self.handle_size/2, self.rect.bottom() - self.handle_size/2,
                                  self.handle_size, self.handle_size),
            "top": QRectF(self.rect.center().x() - self.handle_size/2, self.rect.top() - self.handle_size/2,
                         self.handle_size, self.handle_size),
            "bottom": QRectF(self.rect.center().x() - self.handle_size/2, self.rect.bottom() - self.handle_size/2,
                            self.handle_size, self.handle_size),
            "left": QRectF(self.rect.left() - self.handle_size/2, self.rect.center().y() - self.handle_size/2,
                          self.handle_size, self.handle_size),
            "right": QRectF(self.rect.right() - self.handle_size/2, self.rect.center().y() - self.handle_size/2,
                           self.handle_size, self.handle_size)
        }
        
        for handle_name, handle_rect in handles.items():
            if handle_rect.contains(transformed_point):
                return handle_name
        return None
        
    def get_rotate_handle_pos(self):
        # Get the top-right corner
        corner = QPointF(self.rect.right(), self.rect.top())
        # Move it up by 20 pixels
        corner.setY(corner.y() - 20)
        # Rotate the point around the rect center
        center = self.rect.center()
        transform = QTransform()
        transform.translate(center.x(), center.y())
        transform.rotate(self.rotation)
        transform.translate(-center.x(), -center.y())
        return transform.map(corner)
        
    def get_drag_handle_pos(self):
        # Get the top-middle position
        pos = QPointF(self.rect.center().x(), self.rect.top())
        # Move it up by 20 pixels
        pos.setY(pos.y() - 20)
        # Rotate the point around the rect center
        center = self.rect.center()
        transform = QTransform()
        transform.translate(center.x(), center.y())
        transform.rotate(self.rotation)
        transform.translate(-center.x(), -center.y())
        return transform.map(pos)
        
    def start_transform(self, point, handle):
        self.selected_handle = handle
        self.drag_start = point
        self.original_rect = QRectF(self.rect)
        self.original_rotation = self.rotation
        
    def transform(self, point):
        if not self.selected_handle or not self.drag_start:
            return
            
        if self.selected_handle == "rotate":
            # Calculate angle between center and points
            center = self.rect.center()
            start_angle = math.degrees(math.atan2(self.drag_start.y() - center.y(),
                                                self.drag_start.x() - center.x()))
            current_angle = math.degrees(math.atan2(point.y() - center.y(),
                                                  point.x() - center.x()))
            self.rotation = self.original_rotation + (current_angle - start_angle)
            return
            
        # For resizing and moving, transform the point based on rotation
        transform = QTransform()
        center = self.rect.center()
        transform.translate(center.x(), center.y())
        transform.rotate(-self.rotation)
        transform.translate(-center.x(), -center.y())
        transformed_point = transform.map(point)
        transformed_start = transform.map(self.drag_start)
        
        dx = transformed_point.x() - transformed_start.x()
        dy = transformed_point.y() - transformed_start.y()
        
        new_rect = QRectF(self.original_rect)
        
        if self.selected_handle == "drag" or self.selected_handle == None:  # Moving (either by drag handle or clicking inside)
            new_rect.translate(dx, dy)
        else:  # Resizing
            old_width = new_rect.width()
            old_height = new_rect.height()
            
            if "left" in self.selected_handle:
                new_rect.setLeft(min(new_rect.right(), new_rect.left() + dx))
            if "right" in self.selected_handle:
                new_rect.setRight(max(new_rect.left(), new_rect.right() + dx))
            if "top" in self.selected_handle:
                new_rect.setTop(min(new_rect.bottom(), new_rect.top() + dy))
            if "bottom" in self.selected_handle:
                new_rect.setBottom(max(new_rect.top(), new_rect.bottom() + dy))
                
            # Calculate new scale factors
            self.scale_x = new_rect.width() / self.initial_rect.width()
            self.scale_y = new_rect.height() / self.initial_rect.height()
                
        self.rect = new_rect
        
    def end_transform(self):
        self.selected_handle = None
        self.drag_start = None
        self.original_rect = None
        self.original_rotation = None
        
    def draw(self, painter):
        # Save current transform
        painter.save()
        
        # Set up transform for rotation
        center = self.rect.center()
        painter.translate(center.x(), center.y())
        painter.rotate(self.rotation)
        painter.translate(-center.x(), -center.y())
        
        # Draw the box
        pen = QPen(Qt.white)
        pen.setWidth(1)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.drawRect(self.rect)
        
        # Draw the handles
        painter.setBrush(QBrush(Qt.white))
        painter.setPen(QPen(Qt.black, 1))
        
        # Corner handles
        corners = [
            (self.rect.left(), self.rect.top()),
            (self.rect.right(), self.rect.top()),
            (self.rect.left(), self.rect.bottom()),
            (self.rect.right(), self.rect.bottom())
        ]
        for x, y in corners:
            painter.drawRect(QRectF(x - self.handle_size/2, y - self.handle_size/2,
                                  self.handle_size, self.handle_size))
                                  
        # Middle handles
        midpoints = [
            (self.rect.center().x(), self.rect.top()),
            (self.rect.center().x(), self.rect.bottom()),
            (self.rect.left(), self.rect.center().y()),
            (self.rect.right(), self.rect.center().y())
        ]
        for x, y in midpoints:
            painter.drawRect(QRectF(x - self.handle_size/2, y - self.handle_size/2,
                                  self.handle_size, self.handle_size))
                                  
        # Restore transform for special handles
        painter.restore()
        
        # Draw rotate handle
        rotate_pos = self.get_rotate_handle_pos()
        painter.setBrush(QBrush(Qt.white))
        painter.setPen(QPen(Qt.black, 1))
        painter.drawEllipse(rotate_pos, self.rotate_handle_size/2, self.rotate_handle_size/2)

        # Draw drag handle (hand cursor icon)
        drag_pos = self.get_drag_handle_pos()
        painter.setBrush(QBrush(Qt.white))
        painter.setPen(QPen(Qt.black, 1))
        # Draw a grip-like handle
        painter.drawRect(QRectF(
            drag_pos.x() - self.drag_handle_size/2,
            drag_pos.y() - self.drag_handle_size/2,
            self.drag_handle_size,
            self.drag_handle_size
        ))
        # Add grip lines
        pen = QPen(Qt.black, 2)
        painter.setPen(pen)
        for i in range(3):
            y = drag_pos.y() - self.drag_handle_size/4 + i * self.drag_handle_size/4
            painter.drawLine(
                drag_pos.x() - self.drag_handle_size/3,
                y,
                drag_pos.x() + self.drag_handle_size/3,
                y
            )

class DrawingArea(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas_size = (512, 512)
        self.drawing_layer = QPixmap(*self.canvas_size)
        self.drawing_layer.fill(Qt.black)
        
        # Drawing properties
        self.last_point = None
        self.drawing = False
        self.brush_size = 5
        self.brush_color = QColor(Qt.white)
        self.brush_opacity = 255
        self.brush_hardness = 1.0
        self.eraser_mode = False
        self.drawing_shape = False
        self.temp_layer = None
        
        # Bounding box properties
        self.bounding_box = None
        self.creating_box = False
        self.box_start = None
        self.selected_content = None
        
        # Undo/redo history
        self.history = []
        self.redo_stack = []
        self.max_history = 20
        
        # Save initial state
        self._save_state()
        
        # Zoom properties
        self.scale = 1.0
        self.min_scale = 0.1
        self.max_scale = 5.0
        
        # Set minimum size to prevent collapse
        self.setMinimumSize(*self.canvas_size)
        
        # Enable mouse tracking for smooth drawing
        self.setMouseTracking(True)
        
        # Create flip buttons
        self.setup_flip_buttons()
        
        # Add transformation properties
        self.original_content_size = None
        
    def setup_flip_buttons(self):
        # Create container widget for flip buttons
        self.flip_buttons = QWidget(self)
        layout = QHBoxLayout(self.flip_buttons)
        
        # Create flip buttons
        flip_h_btn = QPushButton("↔️")
        flip_v_btn = QPushButton("↕️")
        
        flip_h_btn.setFixedSize(30, 30)
        flip_v_btn.setFixedSize(30, 30)
        
        flip_h_btn.clicked.connect(self.flip_horizontal)
        flip_v_btn.clicked.connect(self.flip_vertical)
        
        layout.addWidget(flip_h_btn)
        layout.addWidget(flip_v_btn)
        
        # Hide buttons initially
        self.flip_buttons.hide()
        
    def update_flip_buttons_position(self):
        if self.bounding_box:
            # Position buttons below the bounding box
            box_bottom = self.bounding_box.rect.bottom()
            box_center = self.bounding_box.rect.center().x()
            
            # Apply rotation transform to get actual position
            center = self.bounding_box.rect.center()
            transform = QTransform()
            transform.translate(center.x(), center.y())
            transform.rotate(self.bounding_box.rotation)
            transform.translate(-center.x(), -center.y())
            
            point = transform.map(QPointF(box_center, box_bottom + 10))
            
            # Position buttons
            self.flip_buttons.move(
                int(point.x() - self.flip_buttons.width() / 2),
                int(point.y())
            )
            self.flip_buttons.show()
        else:
            self.flip_buttons.hide()
            
    def flip_horizontal(self):
        if self.bounding_box and self.selected_content:
            # Create a QImage from the selected content
            image = self.selected_content.toImage()
            # Flip the image horizontally
            flipped = image.mirrored(True, False)
            # Update the selected content
            self.selected_content = QPixmap.fromImage(flipped)
            self.update()
            
    def flip_vertical(self):
        if self.bounding_box and self.selected_content:
            # Create a QImage from the selected content
            image = self.selected_content.toImage()
            # Flip the image vertically
            flipped = image.mirrored(False, True)
            # Update the selected content
            self.selected_content = QPixmap.fromImage(flipped)
            self.update()

    def _save_state(self):
        """Save current state to history"""
        if len(self.history) >= self.max_history:
            self.history.pop(0)  # Remove oldest state if at max capacity
        self.history.append(self.drawing_layer.copy())
        self.redo_stack.clear()  # Clear redo stack when new action is performed
        
    def wheelEvent(self, event):
        # Handle zoom with mouse wheel
        if event.modifiers() == Qt.ControlModifier:
            delta = event.angleDelta().y()
            zoom_factor = 1.1 if delta > 0 else 0.9
            new_scale = self.scale * zoom_factor
            
            # Constrain scale within bounds
            self.scale = max(self.min_scale, min(self.max_scale, new_scale))
            
            # Update widget size
            self.updateGeometry()
            self.update()
            event.accept()
            
    def sizeHint(self):
        # Return scaled size
        return QSize(
            int(self.canvas_size[0] * self.scale),
            int(self.canvas_size[1] * self.scale)
        )
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # Apply zoom transform
        painter.scale(self.scale, self.scale)
        
        # Draw the canvas
        painter.drawPixmap(0, 0, self.drawing_layer)
        
        # Draw the selected content if exists
        if self.bounding_box and self.selected_content:
            painter.save()
            
            # Set up transform for the selected content
            center = self.bounding_box.rect.center()
            transform = QTransform()
            transform.translate(center.x(), center.y())
            transform.rotate(self.bounding_box.rotation)
            transform.scale(self.bounding_box.scale_x, self.bounding_box.scale_y)
            transform.translate(-center.x(), -center.y())
            
            painter.setTransform(transform, True)
            
            # Draw the content
            painter.drawPixmap(
                self.bounding_box.rect.topLeft(),
                self.selected_content
            )
            
            painter.restore()
        
        # Draw the bounding box if exists
        if self.bounding_box:
            self.bounding_box.draw(painter)

    def mousePressEvent(self, event):
        pos = event.pos() / self.scale
        if event.button() == Qt.LeftButton:
            if self.bounding_box:
                handle = self.bounding_box.get_handle_at(pos)
                if handle is not None:
                    self.bounding_box.start_transform(pos, handle)
                    if not self.original_content_size and self.selected_content:
                        self.original_content_size = self.selected_content.size()
                    return
                elif self.bounding_box.contains(pos):
                    self.bounding_box.start_transform(pos, None)  # None means moving
                    return
                else:
                    # Click outside box - apply transformation and clear selection
                    if self.selected_content:
                        self.apply_transformation()
                    self.bounding_box = None
                    self.selected_content = None
                    self.original_content_size = None
                    self.update_flip_buttons_position()
                    
            # Start new box or drawing
            if self.drawing_shape:  # Bounding box mode
                self.creating_box = True
                self.box_start = pos
            else:  # Normal drawing mode
                self.drawing = True
                self.last_point = pos

    def apply_transformation(self):
        if not self.bounding_box or not self.selected_content:
            return
            
        # Create a new pixmap for the transformed content
        transformed = QPixmap(self.drawing_layer.size())
        transformed.fill(Qt.transparent)
        
        painter = QPainter(transformed)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Set up the transformation
        center = self.bounding_box.rect.center()
        transform = QTransform()
        transform.translate(center.x(), center.y())
        transform.rotate(self.bounding_box.rotation)
        transform.scale(self.bounding_box.scale_x, self.bounding_box.scale_y)
        transform.translate(-center.x(), -center.y())
        
        painter.setTransform(transform)
        
        # Draw the transformed content
        painter.drawPixmap(self.bounding_box.rect.topLeft(), self.selected_content)
        painter.end()
        
        # Draw the transformed content onto the main layer
        painter = QPainter(self.drawing_layer)
        painter.drawPixmap(0, 0, transformed)
        painter.end()
        
        self._save_state()

    def mouseMoveEvent(self, event):
        pos = event.pos() / self.scale
        if self.creating_box:
            # Create temporary box for preview
            x = min(self.box_start.x(), pos.x())
            y = min(self.box_start.y(), pos.y())
            width = abs(pos.x() - self.box_start.x())
            height = abs(pos.y() - self.box_start.y())
            self.bounding_box = BoundingBox(x, y, width, height)
            self.update()
        elif self.bounding_box and (self.bounding_box.selected_handle is not None or self.bounding_box.dragging):
            self.bounding_box.transform(pos)
            self.update_flip_buttons_position()
            self.update()
        elif self.drawing and self.last_point:
            painter = QPainter(self.drawing_layer)
            painter.setRenderHint(QPainter.Antialiasing)
            
            pen = QPen()
            color = QColor(Qt.black if self.eraser_mode else self.brush_color)
            color.setAlpha(self.brush_opacity)
            pen.setColor(color)
            pen.setWidth(self.brush_size)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            
            painter.setPen(pen)
            painter.drawLine(self.last_point, pos)
            
            self.last_point = pos
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.creating_box:
                self.creating_box = False
                if self.bounding_box and self.bounding_box.rect.width() > 10 and self.bounding_box.rect.height() > 10:
                    # Capture the content within the box
                    self.selected_content = QPixmap(self.bounding_box.rect.size().toSize())
                    self.selected_content.fill(Qt.transparent)
                    painter = QPainter(self.selected_content)
                    painter.drawPixmap(
                        0, 0,
                        self.drawing_layer.copy(self.bounding_box.rect.toRect())
                    )
                    painter.end()
                    
                    # Store original size for scaling reference
                    self.original_content_size = self.selected_content.size()
                    
                    # Clear the area in the original layer
                    painter = QPainter(self.drawing_layer)
                    painter.setCompositionMode(QPainter.CompositionMode_Clear)
                    painter.fillRect(self.bounding_box.rect, Qt.transparent)
                    painter.end()
                    
                    self.update_flip_buttons_position()
                else:
                    self.bounding_box = None
                    self.selected_content = None
                    self.original_content_size = None
                    
            elif self.bounding_box:
                self.bounding_box.end_transform()
                
            elif self.drawing:
                self.drawing = False
                self.last_point = None
                self._save_state()
                
            self.update()

    def set_brush_size(self, size):
        self.brush_size = size

    def set_brush_opacity(self, opacity):
        self.brush_opacity = int(opacity * 255)

    def set_brush_hardness(self, hardness):
        self.brush_hardness = hardness

    def toggle_eraser(self):
        self.eraser_mode = not self.eraser_mode

    def toggle_shape_drawing(self):
        self.drawing_shape = not self.drawing_shape

    def pick_color(self):
        color = QColorDialog.getColor(self.brush_color)
        if color.isValid():
            self.brush_color = color

    def clear_canvas(self):
        # Save current state before clearing
        self._save_state()
        self.drawing_layer.fill(Qt.black)
        self.update()

    def undo(self):
        """Undo the last drawing action"""
        if len(self.history) > 1:  # Keep at least one state (initial)
            # Save current state to redo stack
            self.redo_stack.append(self.drawing_layer.copy())
            # Pop the current state
            self.history.pop()
            # Restore previous state
            self.drawing_layer = self.history[-1].copy()
            self.update()

    def redo(self):
        """Redo the last undone action"""
        if self.redo_stack:
            # Save current state to history
            self.history.append(self.drawing_layer.copy())
            # Restore state from redo stack
            self.drawing_layer = self.redo_stack.pop()
            self.update()

    def load_background_image(self):
        """Load an image as background"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            try:
                # Save current state before loading
                self._save_state()
                
                # Load and resize image using PIL for better control
                pil_image = Image.open(file_path).convert('RGB')
                
                # Calculate scaling factor to fit within 512x512 while maintaining aspect ratio
                scale = min(512 / pil_image.width, 512 / pil_image.height)
                new_width = int(pil_image.width * scale)
                new_height = int(pil_image.height * scale)
                
                # Resize image using bilinear interpolation
                pil_image = pil_image.resize((new_width, new_height), Image.BILINEAR)
                
                # Convert to QPixmap
                img_data = pil_image.tobytes("raw", "RGB")
                qimage = QImage(img_data, pil_image.width, pil_image.height, QImage.Format_RGB888)
                background = QPixmap.fromImage(qimage)
                
                # Create a new layer with black background
                new_layer = QPixmap(*self.canvas_size)
                new_layer.fill(Qt.black)
                
                # Draw the background centered
                painter = QPainter(new_layer)
                x = (self.canvas_size[0] - background.width()) // 2
                y = (self.canvas_size[1] - background.height()) // 2
                painter.drawPixmap(x, y, background)
                painter.end()
                
                # Set as current drawing layer
                self.drawing_layer = new_layer
                self.update()
                
                return True
            except Exception as e:
                print(f"Error loading image: {e}")
                return False
        return False

    def get_image(self):
        """Convert current drawing to PIL Image"""
        # Get the QImage from QPixmap
        qimage = self.drawing_layer.toImage()
        
        # Convert to RGB format
        qimage = qimage.convertToFormat(QImage.Format_RGB888)
        
        # Get the image data
        width = qimage.width()
        height = qimage.height()
        
        # Create numpy array directly from the bits
        ptr = qimage.constBits()
        arr = np.frombuffer(ptr, np.uint8).reshape((512, 512, 3))
        
        # Convert to PIL Image
        pil_image = Image.fromarray(arr, 'RGB')
        return pil_image

class DrawingHandler(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create drawing area
        self.drawing_area = DrawingArea(self)
        
        # Set layout to position drawing area
        layout = QVBoxLayout(self)
        margin=4
        layout.setContentsMargins(margin, margin, margin, margin)  # Margins for border + bottom offset
        layout.addWidget(self.drawing_area)

        self.setFixedSize(520, 520)
        layout.setAlignment(Qt.AlignCenter)
        
        # Add frame/border
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(4)

        self.setStyleSheet("""
            QFrame {
                border: 2px solid #f2f2f2;
                background-color: #1a1a1a;
            }
            
            DrawingArea {
                border: 0px solid #808080;
            }
        """)
        
    def get_image(self):
        return self.drawing_area.get_image()

    # Forward methods to drawing area
    def set_brush_size(self, size):
        self.drawing_area.set_brush_size(size)

    def set_brush_opacity(self, opacity):
        self.drawing_area.set_brush_opacity(opacity)

    def set_brush_hardness(self, hardness):
        self.drawing_area.set_brush_hardness(hardness)

    def toggle_eraser(self):
        self.drawing_area.toggle_eraser()

    def toggle_shape_drawing(self):
        self.drawing_area.toggle_shape_drawing()

    def pick_color(self):
        self.drawing_area.pick_color()

    def clear_canvas(self):
        self.drawing_area.clear_canvas()

    def undo(self):
        self.drawing_area.undo()

    def redo(self):
        self.drawing_area.redo()
        
    def load_background_image(self):
        return self.drawing_area.load_background_image()

    def load_background_image_from_pil(self, pil_image):
        """Load a PIL Image directly as background"""
        try:
            # Save current state before loading
            self.drawing_area._save_state()
            
            # Get target dimensions (512x512)
            target_width = target_height = 512
            
            # Calculate dimensions to fit within target while preserving aspect ratio
            aspect_ratio = pil_image.width / pil_image.height
            if aspect_ratio > 1:  # Wider than tall
                new_height = target_height
                new_width = int(new_height * aspect_ratio)
            else:  # Taller than wide
                new_width = target_width 
                new_height = int(new_width / aspect_ratio)
                
            # Resize image preserving aspect ratio
            pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
            
            # Create a new image with padding to center
            padded_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            padded_image.paste(pil_image, (paste_x, paste_y))
            
            # Convert to QPixmap
            img_data = padded_image.tobytes("raw", "RGB")
            qimage = QImage(img_data, target_width, target_height, target_width * 3, QImage.Format_RGB888)
            new_layer = QPixmap.fromImage(qimage)
            
            # Set as current drawing layer
            self.drawing_area.drawing_layer = new_layer
            self.drawing_area.update()
            
            return True
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return False