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
    # - mtx1 is the standardized version of base_for_alignment (centered at origin, unit norm)
    # - mtx2 is the aligned version of landmarks_for_alignment (also standardized)
    # - disparity is the sum of squared differences after alignment
    #
    # Both matrices are standardized (centered at origin and scaled to unit Frobenius norm).
    # To display in the base landmarks' coordinate frame, we transform mtx2 back to
    # the base's original position and scale.
    _, aligned_subset_std, disparity = procrustes(
        base_for_alignment, landmarks_for_alignment
    )

    logger.debug(f"Scipy procrustes disparity: {disparity:.6f}")

    # Get the original base frame parameters for back-transformation
    base_center = base_for_alignment.mean(axis=0)
    base_centered = base_for_alignment - base_center
    base_norm = np.linalg.norm(base_centered)

    if base_norm == 0:
        logger.warning("Base landmarks have zero norm, returning original landmarks")
        return landmarks

    # Transform the aligned subset from standardized space back to base frame
    # aligned_std is at origin with unit norm, so: aligned = aligned_std * base_norm + base_center
    aligned_subset = aligned_subset_std * base_norm + base_center

    logger.debug(
        f"Transformed from standardized space (norm=1, center=0) back to base frame "
        f"(norm={base_norm:.3f}, center={base_center})"
    )

    # If we used a subset for alignment, apply the same transformation to all landmarks
    if alignment_indices is not None and len(indices_list) < len(landmarks):
        # We need to apply the same transformation that scipy.procrustes computed
        # to all landmarks, not just the subset.

        # The transformation from landmarks_for_alignment to aligned_subset_std involves:
        # 1. Centering landmarks
        # 2. Scaling to unit norm
        # 3. Rotation/reflection to align with base
        # 4. Scaling by optimal scale factor

        # To apply this to all landmarks, we need to compute the transformation parameters
        landmarks_center = landmarks_for_alignment.mean(axis=0)
        landmarks_centered = landmarks_for_alignment - landmarks_center
        landmarks_norm = np.linalg.norm(landmarks_centered)

        if landmarks_norm == 0:
            logger.warning("Landmarks have zero norm, returning original landmarks")
            return landmarks

        # Normalize to match what scipy.procrustes does internally
        landmarks_normalized = landmarks_centered / landmarks_norm
        base_normalized = base_centered / base_norm

        # Compute rotation matrix from the normalized sets (using Kabsch algorithm)
        H = landmarks_normalized.T @ base_normalized
        U, _, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        R = Vt.T @ np.diag([1, 1, d]) @ U.T

        # Compute the optimal scale (what scipy.procrustes computes)
        scale = np.trace(base_normalized.T @ (R @ landmarks_normalized.T).T)

        # Now apply the full transformation to ALL landmarks:
        # 1. Center at landmarks_center
        # 2. Normalize by landmarks_norm
        # 3. Rotate by R
        # 4. Scale by optimal scale
        # 5. Scale back by base_norm
        # 6. Translate to base_center
        all_landmarks_centered = landmarks - landmarks_center
        all_landmarks_normalized = all_landmarks_centered / landmarks_norm
        aligned = scale * (all_landmarks_normalized @ R.T) * base_norm + base_center

        logger.debug(
            f"Applied transformation to all {len(landmarks)} landmarks "
            f"(scale={scale:.3f}, rotation det={np.linalg.det(R):.3f})"
        )
    else:
        # All landmarks were used for alignment
        aligned = aligned_subset

    return aligned
