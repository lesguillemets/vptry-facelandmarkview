"""
Scipy Procrustes alignment method.

Uses scipy.spatial.procrustes for aligning landmarks.
"""

import logging
from typing import Optional
import numpy as np
import numpy.typing as npt
from scipy.spatial import procrustes

logger = logging.getLogger(__name__)


def align_landmarks_scipy_procrustes(
    landmarks: npt.NDArray[np.float64],
    base_landmarks: npt.NDArray[np.float64],
    alignment_indices: Optional[set[int] | list[int]] = None,
) -> npt.NDArray[np.float64]:
    """Align landmarks to base landmarks using scipy's procrustes analysis

    This uses scipy.spatial.procrustes to compute the optimal transformation.
    Unlike the default method, scipy's procrustes includes scaling in addition
    to rotation and translation.

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

    # Use scipy's procrustes to compute transformation
    # Note: procrustes returns (mtx1, mtx2, disparity)
    # mtx1 is the standardized version of the first matrix (target)
    # mtx2 is the transformed version of the second matrix (aligned to mtx1)
    # We want to transform landmarks to match base_landmarks
    _, aligned_subset, disparity = procrustes(
        base_for_alignment, landmarks_for_alignment
    )

    logger.debug(f"Scipy procrustes disparity: {disparity:.6f}")

    # If we used a subset for alignment, we need to apply the transformation to all landmarks
    if alignment_indices is not None and len(indices_list) < len(landmarks):
        # Compute the transformation parameters from the subset
        # Center both sets
        landmarks_center = landmarks_for_alignment.mean(axis=0)
        base_center = base_for_alignment.mean(axis=0)
        aligned_subset_center = aligned_subset.mean(axis=0)

        # Compute scale
        landmarks_norm = np.linalg.norm(landmarks_for_alignment - landmarks_center)
        aligned_norm = np.linalg.norm(aligned_subset - aligned_subset_center)
        scale = aligned_norm / landmarks_norm if landmarks_norm > 0 else 1.0

        # Compute rotation from the subset
        landmarks_centered = (landmarks_for_alignment - landmarks_center) / (
            landmarks_norm if landmarks_norm > 0 else 1.0
        )
        base_centered = (base_for_alignment - base_center) / (
            np.linalg.norm(base_for_alignment - base_center)
        )

        H = landmarks_centered.T @ base_centered
        U, _, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        rotation = Vt.T @ np.diag([1, 1, d]) @ U.T

        # Apply transformation to all landmarks
        all_landmarks_centered = landmarks - landmarks_center
        aligned = scale * (rotation @ all_landmarks_centered.T).T + base_center

        logger.debug(
            f"Applied computed transformation to all landmarks: scale={scale:.3f}"
        )
    else:
        # All landmarks were used for alignment, so aligned_subset is the result
        aligned = aligned_subset

    return aligned
