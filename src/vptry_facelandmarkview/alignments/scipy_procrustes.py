"""
Scipy Procrustes alignment method.

Implements the Procrustes alignment algorithm with scaling, following the
approach used by scipy.spatial.procrustes. Unlike scipy's function which
standardizes both input matrices (centering at origin), this implementation
preserves the base landmarks' position and scale in the output.
"""

import logging
from typing import Optional
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def align_landmarks_scipy_procrustes(
    landmarks: npt.NDArray[np.float64],
    base_landmarks: npt.NDArray[np.float64],
    alignment_indices: Optional[set[int] | list[int]] = None,
) -> npt.NDArray[np.float64]:
    """Align landmarks to base landmarks using Procrustes analysis with scaling

    This implements the Procrustes alignment algorithm following scipy's approach,
    which includes scaling in addition to rotation and translation. The transformation
    is computed to minimize the sum of squared differences between the aligned
    landmarks and the base landmarks, while preserving the base landmarks' position
    and scale in the output.

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

    # Important: scipy's procrustes standardizes BOTH input matrices
    # (centers at origin and scales to unit norm). We cannot use the returned
    # mtx2 directly as it would place landmarks at the origin instead of
    # at the base landmarks' position.
    #
    # Solution: Manually compute and apply the transformation while preserving
    # the base landmarks' position and scale.

    # Center both point sets
    base_center = base_for_alignment.mean(axis=0)
    landmarks_center = landmarks_for_alignment.mean(axis=0)

    base_centered = base_for_alignment - base_center
    landmarks_centered = landmarks_for_alignment - landmarks_center

    # Compute norms
    base_norm = np.linalg.norm(base_centered)
    landmarks_norm = np.linalg.norm(landmarks_centered)

    if base_norm == 0 or landmarks_norm == 0:
        logger.warning("Zero norm encountered in procrustes alignment")
        return landmarks

    # Normalize for computing optimal rotation
    base_normalized = base_centered / base_norm
    landmarks_normalized = landmarks_centered / landmarks_norm

    # Compute optimal rotation using SVD (Kabsch algorithm)
    H = landmarks_normalized.T @ base_normalized
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T

    # Compute optimal scale (this is what procrustes does)
    scale = np.trace(base_normalized.T @ (R @ landmarks_normalized.T).T)

    # Compute disparity for logging (same as scipy's procrustes would return)
    disparity = np.sum((base_normalized - scale * (R @ landmarks_normalized.T).T) ** 2)
    logger.debug(f"Scipy procrustes disparity: {disparity:.6f}")

    # Apply transformation to the specified subset or all landmarks
    if alignment_indices is not None and len(indices_list) < len(landmarks):
        # Apply to all landmarks using computed transformation
        all_landmarks_centered = landmarks - landmarks_center
        # Transform: center -> rotate -> scale -> translate to base position
        aligned = (
            scale * (all_landmarks_centered @ R.T) * (base_norm / landmarks_norm)
            + base_center
        )
        logger.debug(
            f"Applied transformation to all landmarks: scale={scale:.3f}, "
            f"base_norm/landmarks_norm={base_norm/landmarks_norm:.3f}"
        )
    else:
        # Apply to landmarks_for_alignment (which is same as landmarks when no indices)
        aligned = (
            scale * (landmarks_centered @ R.T) * (base_norm / landmarks_norm)
            + base_center
        )

    return aligned
