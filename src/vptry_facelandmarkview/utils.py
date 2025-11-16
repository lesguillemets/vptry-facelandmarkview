"""
Utility functions for Face Landmark Viewer.
"""

import logging
from typing import Callable, Optional
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


def align_landmarks_to_base(
    landmarks: npt.NDArray[np.float64],
    base_landmarks: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Align landmarks to base landmarks using Procrustes alignment

    This removes the effects of face position and rotation by computing
    the optimal rigid transformation (translation + rotation) to align
    the landmarks to the base landmarks.

    Args:
        landmarks: Landmarks to align (n_points, 3)
        base_landmarks: Base landmarks to align to (n_points, 3)

    Returns:
        Aligned landmarks (n_points, 3)
    """
    if len(landmarks) == 0 or len(base_landmarks) == 0:
        return landmarks

    if len(landmarks) != len(base_landmarks):
        logger.warning(
            f"Landmark count mismatch: {len(landmarks)} vs {len(base_landmarks)}. "
            "Returning unaligned landmarks."
        )
        return landmarks

    # Center both sets of landmarks
    landmarks_center = landmarks.mean(axis=0)
    base_center = base_landmarks.mean(axis=0)

    landmarks_centered = landmarks - landmarks_center
    base_centered = base_landmarks - base_center

    # Compute optimal rotation using SVD (Kabsch algorithm)
    # H = X^T * Y where X is source (centered landmarks) and Y is target (centered base)
    H = landmarks_centered.T @ base_centered
    U, _, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    # Need to handle reflection case
    d = np.linalg.det(Vt.T @ U.T)
    rotation = Vt.T @ np.diag([1, 1, d]) @ U.T

    # Apply rotation and translation
    aligned = (rotation @ landmarks_centered.T).T + base_center

    logger.debug(
        f"Alignment: translation={base_center - landmarks_center}, "
        f"rotation_det={np.linalg.det(rotation):.3f}"
    )

    return aligned


def draw_landmarks(
    landmarks: npt.NDArray[np.float64],
    center: npt.NDArray[np.float64],
    scale: float,
    color: tuple[float, float, float, float],
    label: str,
    alignment_fn: Optional[
        Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    ] = None,
) -> None:
    """Draw landmarks as points

    Args:
        landmarks: Valid landmarks to draw
        center: Data center for transformation
        scale: Scale factor for transformation
        color: RGBA color tuple
        label: Label for logging
        alignment_fn: Optional function to align landmarks before drawing
    """
    if len(landmarks) == 0:
        return

    # Apply alignment if provided
    if alignment_fn is not None:
        landmarks = alignment_fn(landmarks)

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
