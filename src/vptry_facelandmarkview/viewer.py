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
    QGridLayout,
    QPushButton,
    QSlider,
    QLabel,
    QFileDialog,
    QCheckBox,
    QSpinBox,
)
from PySide6.QtCore import Qt

from vptry_facelandmarkview.gl_widget import LandmarkGLWidget
from vptry_facelandmarkview.projection_widget import ProjectionWidget
from vptry_facelandmarkview.constants import ProjectionType, PROJECTION_SIZE_PX

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
        self.align_faces: bool = False
        self.use_static_points: bool = False
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

        # Align faces checkbox
        self.align_faces_checkbox = QCheckBox("Align Faces")
        self.align_faces_checkbox.stateChanged.connect(self.on_align_faces_changed)
        control_layout.addWidget(self.align_faces_checkbox)

        # Use static points checkbox (for alignment)
        self.use_static_points_checkbox = QCheckBox("Limit to Static Points")
        self.use_static_points_checkbox.stateChanged.connect(
            self.on_use_static_points_changed
        )
        control_layout.addWidget(self.use_static_points_checkbox)

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

        # Grid layout for visualization widgets
        # Layout structure:
        #   [x-z plot]     [empty/reserved]
        #   [3D main plot] [y-z plot]
        viz_grid = QGridLayout()

        # X-Z projection (top view) - row 0, column 0
        self.xz_widget = ProjectionWidget(projection_type=ProjectionType.XZ)
        self.xz_widget.setFixedHeight(PROJECTION_SIZE_PX)
        viz_grid.addWidget(self.xz_widget, 0, 0)

        # Top-right corner - reserved for future use (row 0, column 1)
        self.top_right_placeholder = QLabel()
        self.top_right_placeholder.setFixedHeight(PROJECTION_SIZE_PX)
        self.top_right_placeholder.setFixedWidth(PROJECTION_SIZE_PX)
        viz_grid.addWidget(self.top_right_placeholder, 0, 1)

        # Main 3D OpenGL widget (row 1, column 0)
        self.gl_widget = LandmarkGLWidget()
        viz_grid.addWidget(self.gl_widget, 1, 0)

        # Y-Z projection (side view) - row 1, column 1
        self.yz_widget = ProjectionWidget(projection_type=ProjectionType.YZ)
        self.yz_widget.setFixedWidth(PROJECTION_SIZE_PX)
        viz_grid.addWidget(self.yz_widget, 1, 1)

        # Set stretch factors so main plot takes up most space
        viz_grid.setRowStretch(0, 0)  # Top row (x-z) doesn't stretch
        viz_grid.setRowStretch(1, 1)  # Bottom row (main + y-z) stretches
        viz_grid.setColumnStretch(0, 1)  # Left column (x-z + main) stretches
        viz_grid.setColumnStretch(
            1, 0
        )  # Right column (y-z + placeholder) doesn't stretch

        main_layout.addLayout(viz_grid, stretch=1)

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
        # Ensure file_path is a Path object
        if isinstance(file_path, str):
            file_path = Path(file_path)

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

            # Update OpenGL widgets
            logger.info("Updating OpenGL widgets with data")
            self.gl_widget.set_data(self.data)
            self.xz_widget.set_data(self.data)
            self.yz_widget.set_data(self.data)

            # Update projections with center and scale from main widget
            self._update_projection_center_scale()

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
            self.xz_widget.set_current_frame(value)
            self.yz_widget.set_current_frame(value)

    def on_base_frame_changed(self, value: int) -> None:
        """Handle base frame spinbox change"""
        self.base_frame = value
        if self.data is not None:
            self.gl_widget.set_base_frame(value)
            self.xz_widget.set_base_frame(value)
            self.yz_widget.set_base_frame(value)
            # Update center and scale for projections
            self._update_projection_center_scale()

    def on_show_vectors_changed(self, state: int) -> None:
        """Handle show vectors checkbox change"""
        # state is Qt.CheckState enum value: 0=Unchecked, 2=Checked
        self.show_vectors = state == Qt.CheckState.Checked.value
        logger.debug(f"Show vectors changed to: {self.show_vectors} (state={state})")
        if self.data is not None:
            self.gl_widget.set_show_vectors(self.show_vectors)
            self.xz_widget.set_show_vectors(self.show_vectors)
            self.yz_widget.set_show_vectors(self.show_vectors)

    def on_align_faces_changed(self, state: int) -> None:
        """Handle align faces checkbox change"""
        # state is Qt.CheckState enum value: 0=Unchecked, 2=Checked
        self.align_faces = state == Qt.CheckState.Checked.value
        logger.debug(f"Align faces changed to: {self.align_faces} (state={state})")
        if self.data is not None:
            self.gl_widget.set_align_faces(self.align_faces)
            self.xz_widget.set_align_faces(self.align_faces)
            self.yz_widget.set_align_faces(self.align_faces)

    def on_use_static_points_changed(self, state: int) -> None:
        """Handle use static points checkbox change"""
        # state is Qt.CheckState enum value: 0=Unchecked, 2=Checked
        self.use_static_points = state == Qt.CheckState.Checked.value
        logger.debug(
            f"Use static points changed to: {self.use_static_points} (state={state})"
        )
        if self.data is not None:
            self.gl_widget.set_use_static_points(self.use_static_points)
            self.xz_widget.set_use_static_points(self.use_static_points)
            self.yz_widget.set_use_static_points(self.use_static_points)

    def _update_projection_center_scale(self) -> None:
        """Update projection widgets with center and scale from base frame"""
        if self.data is None:
            return

        # Import here to avoid circular dependency
        from vptry_facelandmarkview.utils import (
            filter_nan_landmarks,
            calculate_center_and_scale,
        )

        # Get base frame landmarks and calculate center/scale
        base_landmarks = self.data[self.base_frame]
        base_landmarks_valid, _ = filter_nan_landmarks(base_landmarks)

        if len(base_landmarks_valid) > 0:
            center, scale = calculate_center_and_scale(base_landmarks_valid)
            self.xz_widget.set_center_and_scale(center, scale)
            self.yz_widget.set_center_and_scale(center, scale)
