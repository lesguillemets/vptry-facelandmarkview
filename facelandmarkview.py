#!/usr/bin/env python3
"""
Face Landmark Viewer - 3D visualization of face landmarks from .npy files using OpenGL
"""

import sys
import argparse
from typing import Optional
import numpy as np
import numpy.typing as npt
from pathlib import Path
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
from OpenGL.GL import *
from OpenGL.GLU import *


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
        self.data = data
        self.update()

    def set_base_frame(self, frame: int) -> None:
        """Set the base frame"""
        self.base_frame = frame
        self.update()

    def set_current_frame(self, frame: int) -> None:
        """Set the current frame"""
        self.current_frame = frame
        self.update()

    def set_show_vectors(self, show: bool) -> None:
        """Set whether to show vectors"""
        self.show_vectors = show
        self.update()

    def initializeGL(self) -> None:
        """Initialize OpenGL"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glClearColor(1.0, 1.0, 1.0, 1.0)

    def resizeGL(self, w: int, h: int) -> None:
        """Handle window resize"""
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = w / h if h > 0 else 1.0
        gluPerspective(45.0, aspect, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self) -> None:
        """Render the scene"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Set camera position
        glTranslatef(0.0, 0.0, -self.zoom)
        glRotatef(self.rotation_x, 1.0, 0.0, 0.0)
        glRotatef(self.rotation_y, 0.0, 1.0, 0.0)

        if self.data is None:
            return

        # Get landmark data
        base_landmarks = self.data[self.base_frame]
        current_landmarks = self.data[self.current_frame]

        # Center the data
        all_points = np.vstack([base_landmarks, current_landmarks])
        center = all_points.mean(axis=0)

        # Calculate scale to fit in view
        extent = all_points.max(axis=0) - all_points.min(axis=0)
        max_extent = extent.max()
        scale = 2.0 / max_extent if max_extent > 0 else 1.0

        # Draw base frame landmarks (blue circles)
        glPointSize(8.0)
        glColor4f(0.0, 0.0, 1.0, 0.6)
        glBegin(GL_POINTS)
        for point in base_landmarks:
            scaled_point = (point - center) * scale
            glVertex3f(scaled_point[0], scaled_point[1], scaled_point[2])
        glEnd()

        # Draw current frame landmarks (red triangles - simulated with larger points)
        glPointSize(12.0)
        glColor4f(1.0, 0.0, 0.0, 0.8)
        glBegin(GL_POINTS)
        for point in current_landmarks:
            scaled_point = (point - center) * scale
            glVertex3f(scaled_point[0], scaled_point[1], scaled_point[2])
        glEnd()

        # Draw vectors if enabled
        if self.show_vectors:
            glLineWidth(1.0)
            glColor4f(0.0, 0.8, 0.0, 0.3)
            glBegin(GL_LINES)
            for base_pt, curr_pt in zip(base_landmarks, current_landmarks):
                scaled_base = (base_pt - center) * scale
                scaled_curr = (curr_pt - center) * scale
                glVertex3f(scaled_base[0], scaled_base[1], scaled_base[2])
                glVertex3f(scaled_curr[0], scaled_curr[1], scaled_curr[2])
            glEnd()

        # Draw coordinate axes
        self._draw_axes()

    def _draw_axes(self) -> None:
        """Draw coordinate axes"""
        glLineWidth(2.0)
        axis_length = 1.5

        glBegin(GL_LINES)
        # X axis (red)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(axis_length, 0.0, 0.0)

        # Y axis (green)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, axis_length, 0.0)

        # Z axis (blue)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, axis_length)
        glEnd()

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
        try:
            self.data = np.load(file_path)

            # Validate data shape
            if len(self.data.shape) != 3 or self.data.shape[2] != 3:
                self.info_label.setText(
                    f"Error: Invalid data shape {self.data.shape}. "
                    "Expected (n_frames, n_landmarks, 3)"
                )
                self.data = None
                return

            n_frames, n_landmarks, _ = self.data.shape

            # Update UI
            self.frame_slider.setMaximum(n_frames - 1)
            self.frame_slider.setEnabled(True)
            self.base_frame_spinbox.setMaximum(n_frames - 1)

            # Set base frame (use the value from initialization or 0)
            if self.base_frame < n_frames:
                self.base_frame_spinbox.setValue(self.base_frame)
            else:
                self.base_frame = 0
                self.base_frame_spinbox.setValue(0)

            self.current_frame = 0
            self.frame_slider.setValue(0)

            self.info_label.setText(
                f"Loaded: {file_path.name} - "
                f"{n_frames} frames, {n_landmarks} landmarks. "
                "Use mouse to rotate (drag) and zoom (wheel)."
            )

            # Update OpenGL widget
            self.gl_widget.set_data(self.data)

        except Exception as e:
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
