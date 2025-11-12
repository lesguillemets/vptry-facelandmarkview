"""
Utility functions for Face Landmark Viewer.
"""

import logging
import numpy as np
import numpy.typing as npt
import OpenGL.GL as gl

from vptry_facelandmarkview.constants import POINT_SIZE, SCALE_MARGIN

logger = logging.getLogger(__name__)


def filter_nan_landmarks(
    landmarks: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Filter out landmarks with NaN values

    Args:
        landmarks: Landmark array to filter

    Returns:
        Tuple of (valid_landmarks, valid_mask)
    """
    valid_mask = ~np.isnan(landmarks).any(axis=1)
    valid_landmarks = landmarks[valid_mask]
    return valid_landmarks, valid_mask


def calculate_center_and_scale(
    base_landmarks: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], float]:
    """Calculate center and scale from base frame landmarks with margin

    Args:
        base_landmarks: Base frame landmarks (only valid ones)

    Returns:
        Tuple of (center, scale)
    """
    center = base_landmarks.mean(axis=0)
    extent = base_landmarks.max(axis=0) - base_landmarks.min(axis=0)
    max_extent = extent.max()
    # Apply margin to give 20% extra space
    scale = (2.0 / SCALE_MARGIN) / max_extent if max_extent > 0 else 1.0
    return center, scale


def draw_landmarks(
    landmarks: npt.NDArray[np.float64],
    center: npt.NDArray[np.float64],
    scale: float,
    color: tuple[float, float, float, float],
    label: str,
) -> None:
    """Draw landmarks as points

    Args:
        landmarks: Valid landmarks to draw
        center: Data center for transformation
        scale: Scale factor for transformation
        color: RGBA color tuple
        label: Label for logging
    """
    if len(landmarks) == 0:
        return

    logger.debug(f"Drawing {len(landmarks)} {label} landmarks")
    gl.glPointSize(POINT_SIZE)
    gl.glColor4f(*color)
    gl.glBegin(gl.GL_POINTS)
    for i, point in enumerate(landmarks):
        # Flip Y coordinate to fix upside-down display
        scaled_point = (point - center) * scale
        scaled_point[1] = -scaled_point[1]  # Flip Y
        if i == 0:  # Log first point as example
            logger.debug(
                f"First {label} landmark: original={point}, scaled={scaled_point}"
            )
        gl.glVertex3f(scaled_point[0], scaled_point[1], scaled_point[2])
    gl.glEnd()
