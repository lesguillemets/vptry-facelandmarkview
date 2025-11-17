"""
OpenGL widget for rendering 2D projections of face landmarks.
"""

import logging
from functools import partial
from typing import Optional

import numpy as np
import numpy.typing as npt
from PySide6.QtWidgets import QWidget
from PySide6.QtOpenGLWidgets import QOpenGLWidget
import OpenGL.GL as gl

from vptry_facelandmarkview.constants import (
    DEFAULT_ALIGNMENT_LANDMARKS,
    ProjectionType,
    PROJECTION_VIEWPORT_FILL,
    PROJECTION_Z_SCALE,
    BASE_LANDMARK_COLOR,
    CURRENT_LANDMARK_COLOR,
    VECTOR_COLOR,
)
from vptry_facelandmarkview.utils import (
    filter_nan_landmarks,
    align_landmarks_to_base,
)

logger = logging.getLogger(__name__)


class ProjectionWidget(QOpenGLWidget):
    """OpenGL widget for rendering 2D projections of landmarks"""

    def __init__(
        self,
        projection_type: ProjectionType = ProjectionType.XY,
        parent: Optional[QWidget] = None,
    ) -> None:
        """
        Initialize projection widget

        Args:
            projection_type: Type of projection (ProjectionType enum)
            parent: Parent widget
        """
        super().__init__(parent)
        self.projection_type = projection_type
        self.data: Optional[npt.NDArray[np.float64]] = None
        self.base_frame: int = 0
        self.current_frame: int = 0
        self.show_vectors: bool = False
        self.align_faces: bool = False
        self.use_static_points: bool = False

        # Store shared center and scale from main widget
        self.center: Optional[npt.NDArray[np.float64]] = None
        self.scale: Optional[float] = None

    def set_data(self, data: npt.NDArray[np.float64]) -> None:
        """Set the landmark data"""
        logger.debug(
            f"{self.projection_type} projection: Setting data with shape: {data.shape}"
        )
        self.data = data
        self.update()

    def set_base_frame(self, frame: int) -> None:
        """Set the base frame"""
        logger.debug(
            f"{self.projection_type} projection: Setting base frame to: {frame}"
        )
        self.base_frame = frame
        self.update()

    def set_current_frame(self, frame: int) -> None:
        """Set the current frame"""
        logger.debug(
            f"{self.projection_type} projection: Setting current frame to: {frame}"
        )
        self.current_frame = frame
        self.update()

    def set_show_vectors(self, show: bool) -> None:
        """Set whether to show vectors"""
        logger.debug(
            f"{self.projection_type} projection: Setting show_vectors to: {show}"
        )
        self.show_vectors = show
        self.update()

    def set_align_faces(self, align: bool) -> None:
        """Set whether to align faces to base frame"""
        logger.debug(
            f"{self.projection_type} projection: Setting align_faces to: {align}"
        )
        self.align_faces = align
        self.update()

    def set_use_static_points(self, use_static: bool) -> None:
        """Set whether to use only static points for alignment"""
        logger.debug(
            f"{self.projection_type} projection: Setting use_static_points to: {use_static}"
        )
        self.use_static_points = use_static
        self.update()

    def set_center_and_scale(
        self, center: npt.NDArray[np.float64], scale: float
    ) -> None:
        """Set the center and scale from the main widget"""
        self.center = center
        self.scale = scale
        self.update()

    def _project_to_2d(
        self, scaled_point: npt.NDArray[np.float64]
    ) -> tuple[float, float]:
        """Project a scaled 3D point to 2D based on projection type
        
        Args:
            scaled_point: 3D point already scaled and centered
            
        Returns:
            Tuple of (x, y) coordinates in 2D projection space
        """
        if self.projection_type == ProjectionType.XZ:
            # X-Z projection (top view) - x horizontal, z vertical (negated) with additional z-scale
            return scaled_point[0], -scaled_point[2] * PROJECTION_Z_SCALE
        elif self.projection_type == ProjectionType.YZ:
            # Y-Z projection (side view) - z horizontal with additional z-scale, y vertical (negated, top is -y)
            return scaled_point[2] * PROJECTION_Z_SCALE, -scaled_point[1]
        else:  # xy
            # X-Y projection (not used currently)
            return scaled_point[0], -scaled_point[1]

    def initializeGL(self) -> None:
        """Initialize OpenGL"""
        logger.info(
            f"Initializing OpenGL context for {self.projection_type} projection"
        )
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_POINT_SMOOTH)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glHint(gl.GL_POINT_SMOOTH_HINT, gl.GL_NICEST)
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)

    def resizeGL(self, w: int, h: int) -> None:
        """Handle window resize"""
        logger.debug(
            f"{self.projection_type} projection: Resize GL: width={w}, height={h}"
        )
        gl.glViewport(0, 0, w, h)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()

        # Set up orthographic projection for 2D view
        # Scale to use configured percentage of axis span (data spans roughly from -1 to 1 after scaling)
        # Don't maintain aspect ratio - fill the available space to maximize landmark visibility
        view_range = 1.0 / PROJECTION_VIEWPORT_FILL
        gl.glOrtho(-view_range, view_range, -view_range, view_range, -1.0, 1.0)

        gl.glMatrixMode(gl.GL_MODELVIEW)

    def paintGL(self) -> None:
        """Render the scene"""
        logger.debug(f"{self.projection_type} projection: paintGL called")
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glLoadIdentity()

        if self.data is None or self.center is None or self.scale is None:
            logger.debug(f"{self.projection_type} projection: No data to render")
            return

        # Get landmark data
        base_landmarks = self.data[self.base_frame]
        current_landmarks = self.data[self.current_frame]

        # Filter out NaN values
        base_landmarks_valid, base_valid_mask = filter_nan_landmarks(base_landmarks)
        current_landmarks_valid, current_valid_mask = filter_nan_landmarks(
            current_landmarks
        )

        if len(base_landmarks_valid) == 0:
            logger.error(
                f"{self.projection_type} projection: No valid base landmarks to render"
            )
            return

        # Draw base frame landmarks (blue)
        self._draw_projection_landmarks(
            base_landmarks_valid, BASE_LANDMARK_COLOR, "base"
        )

        # Create alignment function if enabled
        alignment_fn = None
        if self.align_faces and len(current_landmarks_valid) > 0:
            alignment_indices = None
            if self.use_static_points:
                alignment_indices = DEFAULT_ALIGNMENT_LANDMARKS

            alignment_fn = partial(
                align_landmarks_to_base,
                base_landmarks=base_landmarks_valid,
                alignment_indices=alignment_indices,
            )

        # Draw current frame landmarks (red)
        self._draw_projection_landmarks(
            current_landmarks_valid,
            CURRENT_LANDMARK_COLOR,
            "current",
            alignment_fn=alignment_fn,
        )

        # Draw vectors if enabled
        if self.show_vectors and len(current_landmarks_valid) > 0:
            both_valid_mask = base_valid_mask & current_valid_mask
            base_landmarks_both = base_landmarks[both_valid_mask]
            current_landmarks_both = current_landmarks[both_valid_mask]

            if self.align_faces and len(current_landmarks_both) > 0:
                vector_alignment_indices = (
                    DEFAULT_ALIGNMENT_LANDMARKS if self.use_static_points else None
                )
                current_landmarks_both = align_landmarks_to_base(
                    current_landmarks_both,
                    base_landmarks_both,
                    alignment_indices=vector_alignment_indices,
                )

            if len(base_landmarks_both) > 0:
                self._draw_projection_vectors(
                    base_landmarks_both, current_landmarks_both
                )

    def _draw_projection_landmarks(
        self,
        landmarks: npt.NDArray[np.float64],
        color: tuple[float, float, float, float],
        label: str,
        alignment_fn: Optional = None,
    ) -> None:
        """Draw landmarks as 2D projection"""
        if len(landmarks) == 0:
            return

        # Apply alignment if provided
        if alignment_fn is not None:
            landmarks = alignment_fn(landmarks)

        logger.debug(
            f"{self.projection_type} projection: Drawing {len(landmarks)} {label} landmarks"
        )
        gl.glPointSize(2.0)
        gl.glColor4f(*color)
        gl.glBegin(gl.GL_POINTS)

        for point in landmarks:
            # Project to 2D based on projection type
            scaled_point = (point - self.center) * self.scale
            x, y = self._project_to_2d(scaled_point)
            gl.glVertex2f(x, y)

        gl.glEnd()

    def _draw_projection_vectors(
        self,
        base_landmarks: npt.NDArray[np.float64],
        current_landmarks: npt.NDArray[np.float64],
    ) -> None:
        """Draw vectors as 2D projection"""
        logger.debug(
            f"{self.projection_type} projection: Drawing {len(base_landmarks)} vectors (green)"
        )
        gl.glLineWidth(1.0)
        gl.glColor4f(*VECTOR_COLOR)
        gl.glBegin(gl.GL_LINES)

        for base_pt, curr_pt in zip(base_landmarks, current_landmarks):
            scaled_base = (base_pt - self.center) * self.scale
            scaled_curr = (curr_pt - self.center) * self.scale

            base_x, base_y = self._project_to_2d(scaled_base)
            curr_x, curr_y = self._project_to_2d(scaled_curr)

            gl.glVertex2f(base_x, base_y)
            gl.glVertex2f(curr_x, curr_y)

        gl.glEnd()
