#!/usr/bin/env python3
"""
Face Landmark Viewer - 3D visualization of face landmarks from .npy files using OpenGL
"""

import sys
import argparse
import logging
from typing import Optional
import numpy as np
import numpy.typing as npt
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QLabel,
    QFileDialog,
    QCheckBox,
    QSpinBox,
)
from PySide6.QtCore import Qt, QPoint
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat
import OpenGL.GL as gl
import OpenGL.GLU as glu


class LandmarkGLWidget(QOpenGLWidget):
    """OpenGL widget for rendering 3D landmarks"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.data: Optional[npt.NDArray[np.float64]] = None
        self.base_frame: int = 0
        self.current_frame: int = 0
        self.show_vectors: bool = False

        # Camera controls
        self.rotation_x: float = 20.0
        self.rotation_y: float = 45.0
        self.zoom: float = 3.0
        self.last_pos: Optional[QPoint] = None

    def set_data(self, data: npt.NDArray[np.float64]) -> None:
        """Set the landmark data"""
        logger.info(f"Setting data with shape: {data.shape}")
        self.data = data
        self.update()

    def set_base_frame(self, frame: int) -> None:
        """Set the base frame"""
        logger.debug(f"Setting base frame to: {frame}")
        self.base_frame = frame
        self.update()

    def set_current_frame(self, frame: int) -> None:
        """Set the current frame"""
        logger.debug(f"Setting current frame to: {frame}")
        self.current_frame = frame
        self.update()

    def set_show_vectors(self, show: bool) -> None:
        """Set whether to show vectors"""
        logger.debug(f"Setting show_vectors to: {show}")
        self.show_vectors = show
        self.update()

    def initializeGL(self) -> None:
        """Initialize OpenGL"""
        logger.info("Initializing OpenGL context")
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_POINT_SMOOTH)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glHint(gl.GL_POINT_SMOOTH_HINT, gl.GL_NICEST)
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)

    def resizeGL(self, w: int, h: int) -> None:
        """Handle window resize"""
        logger.info(f"Resize GL: width={w}, height={h}")
        gl.glViewport(0, 0, w, h)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        aspect = w / h if h > 0 else 1.0
        logger.debug(f"Aspect ratio: {aspect}")
        glu.gluPerspective(45.0, aspect, 0.1, 100.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def paintGL(self) -> None:
        """Render the scene"""
        logger.debug("paintGL called")
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glLoadIdentity()

        # Set camera position
        gl.glTranslatef(0.0, 0.0, -self.zoom)
        gl.glRotatef(self.rotation_x, 1.0, 0.0, 0.0)
        gl.glRotatef(self.rotation_y, 0.0, 1.0, 0.0)
        logger.debug(f"Camera: zoom={self.zoom}, rotation_x={self.rotation_x}, rotation_y={self.rotation_y}")

        if self.data is None:
            logger.warning("paintGL: No data to render")
            return

        # Get landmark data
        base_landmarks = self.data[self.base_frame]
        current_landmarks = self.data[self.current_frame]
        logger.info(f"Rendering frame {self.current_frame} (base: {self.base_frame})")
        logger.debug(f"Base landmarks shape: {base_landmarks.shape}, Current landmarks shape: {current_landmarks.shape}")

        # Filter out NaN values
        base_valid_mask = ~np.isnan(base_landmarks).any(axis=1)
        current_valid_mask = ~np.isnan(current_landmarks).any(axis=1)
        
        base_landmarks_valid = base_landmarks[base_valid_mask]
        current_landmarks_valid = current_landmarks[current_valid_mask]
        
        nan_count_base = (~base_valid_mask).sum()
        nan_count_current = (~current_valid_mask).sum()
        
        if nan_count_base > 0 or nan_count_current > 0:
            logger.warning(f"Filtered out NaN landmarks: base={nan_count_base}, current={nan_count_current}")
        
        if len(base_landmarks_valid) == 0 and len(current_landmarks_valid) == 0:
            logger.error("No valid landmarks to render (all contain NaN)")
            return

        # Center the data (using only valid points)
        valid_points = []
        if len(base_landmarks_valid) > 0:
            valid_points.append(base_landmarks_valid)
        if len(current_landmarks_valid) > 0:
            valid_points.append(current_landmarks_valid)
        
        if len(valid_points) == 0:
            logger.error("No valid points to calculate center")
            return
            
        all_points = np.vstack(valid_points)
        center = all_points.mean(axis=0)

        # Calculate scale to fit in view
        extent = all_points.max(axis=0) - all_points.min(axis=0)
        max_extent = extent.max()
        scale = 2.0 / max_extent if max_extent > 0 else 1.0
        logger.info(f"Data center: {center}, extent: {extent}, max_extent: {max_extent}, scale: {scale}")
        logger.debug(f"Valid landmarks: base={len(base_landmarks_valid)}, current={len(current_landmarks_valid)}")

        # Draw base frame landmarks (blue circles)
        if len(base_landmarks_valid) > 0:
            logger.debug(f"Drawing {len(base_landmarks_valid)} base frame landmarks (blue)")
            gl.glPointSize(8.0)
            gl.glColor4f(0.0, 0.0, 1.0, 0.6)
            gl.glBegin(gl.GL_POINTS)
            for i, point in enumerate(base_landmarks_valid):
                scaled_point = (point - center) * scale
                if i == 0:  # Log first point as example
                    logger.debug(f"First base landmark: original={point}, scaled={scaled_point}")
                gl.glVertex3f(scaled_point[0], scaled_point[1], scaled_point[2])
            gl.glEnd()

        # Draw current frame landmarks (red triangles - simulated with larger points)
        if len(current_landmarks_valid) > 0:
            logger.debug(f"Drawing {len(current_landmarks_valid)} current frame landmarks (red)")
            gl.glPointSize(12.0)
            gl.glColor4f(1.0, 0.0, 0.0, 0.8)
            gl.glBegin(gl.GL_POINTS)
            for i, point in enumerate(current_landmarks_valid):
                scaled_point = (point - center) * scale
                if i == 0:  # Log first point as example
                    logger.debug(f"First current landmark: original={point}, scaled={scaled_point}")
                gl.glVertex3f(scaled_point[0], scaled_point[1], scaled_point[2])
            gl.glEnd()

        # Draw vectors if enabled (only for landmarks that are valid in both frames)
        if self.show_vectors and len(base_landmarks_valid) > 0 and len(current_landmarks_valid) > 0:
            # Match valid landmarks from both frames
            both_valid_mask = base_valid_mask & current_valid_mask
            base_landmarks_both = base_landmarks[both_valid_mask]
            current_landmarks_both = current_landmarks[both_valid_mask]
            
            if len(base_landmarks_both) > 0:
                logger.debug(f"Drawing {len(base_landmarks_both)} vectors (green)")
                gl.glLineWidth(1.0)
                gl.glColor4f(0.0, 0.8, 0.0, 0.3)
                gl.glBegin(gl.GL_LINES)
                for base_pt, curr_pt in zip(base_landmarks_both, current_landmarks_both):
                    scaled_base = (base_pt - center) * scale
                    scaled_curr = (curr_pt - center) * scale
                    gl.glVertex3f(scaled_base[0], scaled_base[1], scaled_base[2])
                    gl.glVertex3f(scaled_curr[0], scaled_curr[1], scaled_curr[2])
                gl.glEnd()

        # Draw coordinate axes
        self._draw_axes()

    def _draw_axes(self) -> None:
        """Draw coordinate axes"""
        gl.glLineWidth(2.0)
        axis_length = 1.5

        gl.glBegin(gl.GL_LINES)
        # X axis (red)
        gl.glColor3f(1.0, 0.0, 0.0)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(axis_length, 0.0, 0.0)

        # Y axis (green)
        gl.glColor3f(0.0, 1.0, 0.0)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.0, axis_length, 0.0)

        # Z axis (blue)
        gl.glColor3f(0.0, 0.0, 1.0)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.0, 0.0, axis_length)
        gl.glEnd()

    def mousePressEvent(self, event) -> None:
        """Handle mouse press"""
        self.last_pos = event.pos()

    def mouseMoveEvent(self, event) -> None:
        """Handle mouse move for rotation"""
        if self.last_pos is not None:
            dx = event.pos().x() - self.last_pos.x()
            dy = event.pos().y() - self.last_pos.y()

            if event.buttons() & Qt.LeftButton:
                self.rotation_x += dy * 0.5
                self.rotation_y += dx * 0.5
                self.update()

            self.last_pos = event.pos()

    def wheelEvent(self, event) -> None:
        """Handle mouse wheel for zoom"""
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom *= 0.9
        else:
            self.zoom *= 1.1
        self.zoom = max(1.0, min(10.0, self.zoom))
        self.update()


class FaceLandmarkViewer(QMainWindow):
    """Main window for the face landmark viewer application"""

    def __init__(
        self, initial_file: Optional[Path] = None, initial_base_frame: int = 0
    ) -> None:
        super().__init__()
        self.data: Optional[npt.NDArray[np.float64]] = None
        self.base_frame: int = initial_base_frame
        self.current_frame: int = 0
        self.show_vectors: bool = False
        self.initial_file: Optional[Path] = initial_file

        self.setWindowTitle("Face Landmark Viewer (OpenGL)")
        self.setGeometry(100, 100, 1200, 800)

        self.init_ui()

        # Load initial file if provided
        if self.initial_file is not None:
            self.load_file_from_path(self.initial_file)

    def init_ui(self) -> None:
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # Control panel
        control_layout = QHBoxLayout()

        # Load button
        self.load_button = QPushButton("Load .npy File")
        self.load_button.clicked.connect(self.load_file)
        control_layout.addWidget(self.load_button)

        # Base frame selector
        control_layout.addWidget(QLabel("Base Frame:"))
        self.base_frame_spinbox = QSpinBox()
        self.base_frame_spinbox.setMinimum(0)
        self.base_frame_spinbox.setValue(0)
        self.base_frame_spinbox.valueChanged.connect(self.on_base_frame_changed)
        control_layout.addWidget(self.base_frame_spinbox)

        # Show vectors checkbox
        self.show_vectors_checkbox = QCheckBox("Show Vectors")
        self.show_vectors_checkbox.stateChanged.connect(self.on_show_vectors_changed)
        control_layout.addWidget(self.show_vectors_checkbox)

        control_layout.addStretch()

        main_layout.addLayout(control_layout)

        # Frame slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Frame:"))

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.on_frame_changed)
        slider_layout.addWidget(self.frame_slider)

        self.frame_label = QLabel("0 / 0")
        slider_layout.addWidget(self.frame_label)

        main_layout.addLayout(slider_layout)

        # OpenGL widget for 3D visualization
        self.gl_widget = LandmarkGLWidget()
        main_layout.addWidget(self.gl_widget)

        # Info label
        self.info_label = QLabel(
            "Load a .npy file to begin. Use mouse to rotate (drag) and zoom (wheel)."
        )
        main_layout.addWidget(self.info_label)

    def load_file(self) -> None:
        """Load a .npy file via file dialog"""
        file_path_str, _ = QFileDialog.getOpenFileName(
            self, "Open .npy File", "", "NumPy Files (*.npy);;All Files (*)"
        )

        if not file_path_str:
            return

        file_path = Path(file_path_str)
        self.load_file_from_path(file_path)

    def load_file_from_path(self, file_path: Path) -> None:
        """Load a .npy file from a given path"""
        logger.info(f"Loading file: {file_path}")
        try:
            self.data = np.load(file_path)
            logger.info(f"Loaded data shape: {self.data.shape}")

            # Validate data shape
            if len(self.data.shape) != 3 or self.data.shape[2] != 3:
                logger.error(f"Invalid data shape: {self.data.shape}")
                self.info_label.setText(
                    f"Error: Invalid data shape {self.data.shape}. "
                    "Expected (n_frames, n_landmarks, 3)"
                )
                self.data = None
                return

            n_frames, n_landmarks, _ = self.data.shape
            logger.info(f"Valid data: {n_frames} frames, {n_landmarks} landmarks")

            # Check for NaN values
            nan_count = np.isnan(self.data).sum()
            if nan_count > 0:
                nan_landmarks = np.isnan(self.data).any(axis=2).sum()
                logger.warning(f"Data contains {nan_count} NaN values across {nan_landmarks} landmark positions")
                logger.warning("NaN landmarks will be filtered out during rendering")

            # Log data range for debugging
            logger.debug(f"Data min: {np.nanmin(self.data)}, max: {np.nanmax(self.data)}")
            logger.debug(f"Data mean: {np.nanmean(self.data, axis=(0,1))}")

            # Update UI
            self.frame_slider.setMaximum(n_frames - 1)
            self.frame_slider.setEnabled(True)
            self.base_frame_spinbox.setMaximum(n_frames - 1)

            # Set base frame (use the value from initialization or 0)
            if self.base_frame < n_frames:
                self.base_frame_spinbox.setValue(self.base_frame)
                logger.debug(f"Base frame set to: {self.base_frame}")
            else:
                self.base_frame = 0
                self.base_frame_spinbox.setValue(0)
                logger.debug("Base frame reset to 0")

            self.current_frame = 0
            self.frame_slider.setValue(0)

            self.info_label.setText(
                f"Loaded: {file_path.name} - "
                f"{n_frames} frames, {n_landmarks} landmarks. "
                "Use mouse to rotate (drag) and zoom (wheel)."
            )

            # Update OpenGL widget
            logger.info("Updating OpenGL widget with data")
            self.gl_widget.set_data(self.data)

        except Exception as e:
            logger.exception(f"Error loading file: {str(e)}")
            self.info_label.setText(f"Error loading file: {str(e)}")
            self.data = None

    def on_frame_changed(self, value: int) -> None:
        """Handle frame slider change"""
        self.current_frame = value
        if self.data is not None:
            self.frame_label.setText(f"{value} / {self.data.shape[0] - 1}")
            self.gl_widget.set_current_frame(value)

    def on_base_frame_changed(self, value: int) -> None:
        """Handle base frame spinbox change"""
        self.base_frame = value
        if self.data is not None:
            self.gl_widget.set_base_frame(value)

    def on_show_vectors_changed(self, state: int) -> None:
        """Handle show vectors checkbox change"""
        self.show_vectors = state == Qt.Checked
        if self.data is not None:
            self.gl_widget.set_show_vectors(self.show_vectors)


def main() -> None:
    """Main entry point"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Face Landmark Viewer - 3D visualization of face landmarks from .npy files"
    )
    parser.add_argument(
        "file", type=Path, nargs="?", help="Path to .npy file to load on startup"
    )
    parser.add_argument(
        "--base-frame",
        type=int,
        default=0,
        help="Base frame index to use as reference (default: 0)",
    )

    args = parser.parse_args()

    # Set up OpenGL format
    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)  # Enable multisampling for better quality
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    viewer = FaceLandmarkViewer(
        initial_file=args.file, initial_base_frame=args.base_frame
    )
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
