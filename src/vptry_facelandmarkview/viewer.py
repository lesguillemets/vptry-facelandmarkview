"""
Main window for the Face Landmark Viewer application.
"""

import logging
from typing import Optional
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PySide6.QtWidgets import (
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
from PySide6.QtCore import Qt

from vptry_facelandmarkview.gl_widget import LandmarkGLWidget

logger = logging.getLogger(__name__)


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

        main_layout.addLayout(control_layout, stretch=0)

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

        main_layout.addLayout(slider_layout, stretch=0)

        # OpenGL widget for 3D visualization
        self.gl_widget = LandmarkGLWidget()
        main_layout.addWidget(self.gl_widget, stretch=1)

        # Info label
        self.info_label = QLabel(
            "Load a .npy file to begin. Use mouse to rotate (drag) and zoom (wheel)."
        )
        main_layout.addWidget(self.info_label, stretch=0)

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
                logger.warning(
                    f"Data contains {nan_count} NaN values across {nan_landmarks} landmark positions"
                )
                logger.warning("NaN landmarks will be filtered out during rendering")

            # Log data range for debugging
            logger.debug(
                f"Data min: {np.nanmin(self.data)}, max: {np.nanmax(self.data)}"
            )
            logger.debug(f"Data mean: {np.nanmean(self.data, axis=(0, 1))}")

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
        # state is Qt.CheckState enum value: 0=Unchecked, 2=Checked
        self.show_vectors = state == Qt.CheckState.Checked.value
        logger.debug(f"Show vectors changed to: {self.show_vectors} (state={state})")
        if self.data is not None:
            self.gl_widget.set_show_vectors(self.show_vectors)
