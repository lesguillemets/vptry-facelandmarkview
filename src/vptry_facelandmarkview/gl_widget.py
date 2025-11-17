"""
OpenGL widget for rendering 3D face landmarks.
"""

import logging
from functools import partial
from typing import Optional

import numpy as np
import numpy.typing as npt
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QPoint
from PySide6.QtOpenGLWidgets import QOpenGLWidget
import OpenGL.GL as gl
import OpenGL.GLU as glu

from vptry_facelandmarkview.constants import (
    SCALE_MARGIN,
    DEFAULT_ALIGNMENT_LANDMARKS,
    BASE_LANDMARK_COLOR,
    CURRENT_LANDMARK_COLOR,
    VECTOR_COLOR,
    AXIS_X_COLOR,
    AXIS_Y_COLOR,
    AXIS_Z_COLOR,
    DisplayState,
    DEFAULT_ROTATION_X,
    DEFAULT_ROTATION_Y,
    DEFAULT_ZOOM,
    MIN_ZOOM,
    MAX_ZOOM,
    ZOOM_IN_FACTOR,
    ZOOM_OUT_FACTOR,
    ROTATION_SENSITIVITY,
    AXIS_LENGTH,
    VECTOR_LINE_WIDTH,
    AXIS_LINE_WIDTH,
    BACKGROUND_COLOR,
    PERSPECTIVE_FOV,
    PERSPECTIVE_NEAR,
    PERSPECTIVE_FAR,
)
from vptry_facelandmarkview.utils import (
    filter_nan_landmarks,
    calculate_center_and_scale,
    draw_landmarks,
    align_landmarks_to_base,
)

logger = logging.getLogger(__name__)


class LandmarkGLWidget(QOpenGLWidget):
    """OpenGL widget for rendering 3D landmarks"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.data: Optional[npt.NDArray[np.float64]] = None
        self.state = DisplayState()

        # Camera controls
        self.rotation_x: float = DEFAULT_ROTATION_X
        self.rotation_y: float = DEFAULT_ROTATION_Y
        self.zoom: float = DEFAULT_ZOOM
        self.last_pos: Optional[QPoint] = None

    def set_data(self, data: npt.NDArray[np.float64]) -> None:
        """Set the landmark data"""
        logger.info(f"Setting data with shape: {data.shape}")
        self.data = data
        self.update()

    def set_base_frame(self, frame: int) -> None:
        """Set the base frame"""
        logger.debug(f"Setting base frame to: {frame}")
        self.state.base_frame = frame
        self.update()

    def set_current_frame(self, frame: int) -> None:
        """Set the current frame"""
        logger.debug(f"Setting current frame to: {frame}")
        self.state.current_frame = frame
        self.update()

    def set_show_vectors(self, show: bool) -> None:
        """Set whether to show vectors"""
        logger.debug(f"Setting show_vectors to: {show}")
        self.state.show_vectors = show
        self.update()

    def set_align_faces(self, align: bool) -> None:
        """Set whether to align faces to base frame"""
        logger.debug(f"Setting align_faces to: {align}")
        self.state.align_faces = align
        self.update()

    def set_use_static_points(self, use_static: bool) -> None:
        """Set whether to use only static points for alignment"""
        logger.debug(f"Setting use_static_points to: {use_static}")
        self.state.use_static_points = use_static
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
        gl.glClearColor(*BACKGROUND_COLOR)

    def resizeGL(self, w: int, h: int) -> None:
        """Handle window resize"""
        logger.info(f"Resize GL: width={w}, height={h}")
        gl.glViewport(0, 0, w, h)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        aspect = w / h if h > 0 else 1.0
        logger.debug(f"Aspect ratio: {aspect}")
        glu.gluPerspective(PERSPECTIVE_FOV, aspect, PERSPECTIVE_NEAR, PERSPECTIVE_FAR)
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
        logger.debug(
            f"Camera: zoom={self.zoom}, rotation_x={self.rotation_x}, rotation_y={self.rotation_y}"
        )

        if self.data is None:
            logger.warning("paintGL: No data to render")
            return

        # Get landmark data
        base_landmarks = self.data[self.state.base_frame]
        current_landmarks = self.data[self.state.current_frame]
        logger.info(
            f"Rendering frame {self.state.current_frame} (base: {self.state.base_frame})"
        )
        logger.debug(
            f"Base landmarks shape: {base_landmarks.shape}, Current landmarks shape: {current_landmarks.shape}"
        )

        # Filter out NaN values using module-level function
        base_landmarks_valid, base_valid_mask = filter_nan_landmarks(base_landmarks)
        current_landmarks_valid, current_valid_mask = filter_nan_landmarks(
            current_landmarks
        )

        nan_count_base = (~base_valid_mask).sum()
        nan_count_current = (~current_valid_mask).sum()

        if nan_count_base > 0 or nan_count_current > 0:
            logger.warning(
                f"Filtered out NaN landmarks: base={nan_count_base}, current={nan_count_current}"
            )

        if len(base_landmarks_valid) == 0:
            logger.error("No valid base landmarks to render (all contain NaN)")
            return

        # Calculate center and scale from base frame only (with 20% margin)
        center, scale = calculate_center_and_scale(base_landmarks_valid)

        logger.info(
            f"Data center: {center}, scale: {scale} (calculated from base frame with {SCALE_MARGIN}x margin)"
        )
        logger.debug(
            f"Valid landmarks: base={len(base_landmarks_valid)}, current={len(current_landmarks_valid)}"
        )

        # Draw base frame landmarks (blue)
        draw_landmarks(base_landmarks_valid, center, scale, BASE_LANDMARK_COLOR, "base")

        # Create alignment function if enabled
        alignment_fn = None
        if self.state.align_faces and len(current_landmarks_valid) > 0:
            # Determine which landmarks to use for alignment
            alignment_indices = None
            if self.state.use_static_points:
                # Use only nose + forehead landmarks for stable alignment
                alignment_indices = DEFAULT_ALIGNMENT_LANDMARKS
                logger.debug(
                    f"Using {len(DEFAULT_ALIGNMENT_LANDMARKS)} static points for alignment"
                )

            # Create a partial function that aligns to base landmarks
            alignment_fn = partial(
                align_landmarks_to_base,
                base_landmarks=base_landmarks_valid,
                alignment_indices=alignment_indices,
            )

        # Draw current frame landmarks (red)
        draw_landmarks(
            current_landmarks_valid,
            center,
            scale,
            CURRENT_LANDMARK_COLOR,
            "current",
            alignment_fn=alignment_fn,
        )

        # Draw vectors if enabled (only for landmarks that are valid in both frames)
        if self.state.show_vectors and len(current_landmarks_valid) > 0:
            # Match valid landmarks from both frames
            both_valid_mask = base_valid_mask & current_valid_mask
            base_landmarks_both = base_landmarks[both_valid_mask]
            current_landmarks_both = current_landmarks[both_valid_mask]

            # Apply alignment to current landmarks if enabled
            if self.state.align_faces and len(current_landmarks_both) > 0:
                # Use same alignment indices for vectors
                vector_alignment_indices = (
                    DEFAULT_ALIGNMENT_LANDMARKS
                    if self.state.use_static_points
                    else None
                )
                current_landmarks_both = align_landmarks_to_base(
                    current_landmarks_both,
                    base_landmarks_both,
                    alignment_indices=vector_alignment_indices,
                )

            if len(base_landmarks_both) > 0:
                logger.debug(f"Drawing {len(base_landmarks_both)} vectors (green)")
                gl.glLineWidth(VECTOR_LINE_WIDTH)
                gl.glColor4f(*VECTOR_COLOR)
                gl.glBegin(gl.GL_LINES)
                for base_pt, curr_pt in zip(
                    base_landmarks_both, current_landmarks_both
                ):
                    scaled_base = (base_pt - center) * scale
                    scaled_curr = (curr_pt - center) * scale
                    # Flip Y coordinates
                    scaled_base[1] = -scaled_base[1]
                    scaled_curr[1] = -scaled_curr[1]
                    gl.glVertex3f(scaled_base[0], scaled_base[1], scaled_base[2])
                    gl.glVertex3f(scaled_curr[0], scaled_curr[1], scaled_curr[2])
                gl.glEnd()

        # Draw coordinate axes
        self._draw_axes()

    def _draw_axes(self) -> None:
        """Draw coordinate axes"""
        gl.glLineWidth(AXIS_LINE_WIDTH)

        gl.glBegin(gl.GL_LINES)
        # X axis (red)
        gl.glColor3f(*AXIS_X_COLOR)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(AXIS_LENGTH, 0.0, 0.0)

        # Y axis (green)
        gl.glColor3f(*AXIS_Y_COLOR)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.0, AXIS_LENGTH, 0.0)

        # Z axis (blue)
        gl.glColor3f(*AXIS_Z_COLOR)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.0, 0.0, AXIS_LENGTH)
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
                self.rotation_x += dy * ROTATION_SENSITIVITY
                self.rotation_y += dx * ROTATION_SENSITIVITY
                self.update()

            self.last_pos = event.pos()

    def wheelEvent(self, event) -> None:
        """Handle mouse wheel for zoom"""
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom *= ZOOM_IN_FACTOR
        else:
            self.zoom *= ZOOM_OUT_FACTOR
        self.zoom = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom))
        self.update()
