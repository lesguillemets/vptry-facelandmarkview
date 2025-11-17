"""
Default alignment method using Kabsch algorithm (Procrustes alignment).

This is the original alignment implementation that was in utils.py.
"""

import logging
from typing import Optional
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def align_landmarks_default(
    landmarks: npt.NDArray[np.float64],
    base_landmarks: npt.NDArray[np.float64],
    alignment_indices: Optional[set[int] | list[int]] = None,
) -> npt.NDArray[np.float64]:
    """Align landmarks to base landmarks using Procrustes alignment (Kabsch algorithm)

    This removes the effects of face position and rotation by computing
    the optimal rigid transformation (translation + rotation) to align
    the landmarks to the base landmarks.

    Args:
        landmarks: Landmarks to align (n_points, 3)
        base_landmarks: Base landmarks to align to (n_points, 3)
        alignment_indices: Optional set or list of landmark indices to use for
            calculating alignment. If provided, only these landmarks are used
            to compute the transformation, which is then applied to all landmarks.
            If None, all landmarks are used for alignment calculation.

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

    # If alignment_indices is provided, use only those landmarks for computing alignment
    if alignment_indices is not None:
        # Convert to list if it's a set for indexing
        indices_list = (
            list(alignment_indices)
            if isinstance(alignment_indices, set)
            else alignment_indices
        )

        # Validate indices
        max_idx = len(landmarks)
        if any(idx < 0 or idx >= max_idx for idx in indices_list):
            logger.warning(
                f"Invalid alignment indices provided (range: 0-{max_idx - 1}). "
                "Using all landmarks for alignment."
            )
            landmarks_for_alignment = landmarks
            base_for_alignment = base_landmarks
        else:
            # Use only specified landmarks for alignment calculation
            landmarks_for_alignment = landmarks[indices_list]
            base_for_alignment = base_landmarks[indices_list]
            logger.debug(
                f"Using {len(indices_list)} landmarks for alignment calculation"
            )
    else:
        # Use all landmarks for alignment
        landmarks_for_alignment = landmarks
        base_for_alignment = base_landmarks

    # Center both sets of landmarks (for alignment calculation)
    landmarks_center = landmarks_for_alignment.mean(axis=0)
    base_center = base_for_alignment.mean(axis=0)

    landmarks_centered = landmarks_for_alignment - landmarks_center
    base_centered = base_for_alignment - base_center

    # Compute optimal rotation using SVD (Kabsch algorithm)
    # H = X^T * Y where X is source (centered landmarks) and Y is target (centered base)
    H = landmarks_centered.T @ base_centered
    U, _, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    # Need to handle reflection case
    d = np.linalg.det(Vt.T @ U.T)
    rotation = Vt.T @ np.diag([1, 1, d]) @ U.T

    # Apply rotation and translation to ALL landmarks
    all_landmarks_centered = landmarks - landmarks_center
    aligned = (rotation @ all_landmarks_centered.T).T + base_center

    logger.debug(
        f"Alignment: translation={base_center - landmarks_center}, "
        f"rotation_det={np.linalg.det(rotation):.3f}"
    )

    return aligned
