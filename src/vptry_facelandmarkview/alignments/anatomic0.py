"""
Anatomic0 alignment method using nose landmarks and specific midpoints.

This method uses stable anatomic landmarks for alignment:
- All nose landmarks (defined in NOSE_LANDMARKS)
- Midpoint of landmarks 33 and 133
- Midpoint of landmarks 362 and 263
"""

import logging
from typing import Optional
import numpy as np
import numpy.typing as npt

from vptry_facelandmarkview.constants import NOSE_LANDMARKS, ANATOMIC0_MIDPOINT_PAIRS

logger = logging.getLogger(__name__)


def align_landmarks_anatomic0(
    landmarks: npt.NDArray[np.float64],
    base_landmarks: npt.NDArray[np.float64],
    alignment_indices: Optional[set[int] | list[int]] = None,
) -> npt.NDArray[np.float64]:
    """Align landmarks using anatomic0 method with nose landmarks and midpoints.

    This alignment method uses stable anatomic features:
    - All nose landmarks from NOSE_LANDMARKS constant
    - Midpoints of specific landmark pairs (e.g., 33-133, 362-263)

    The alignment uses Procrustes alignment (Kabsch algorithm) for rigid transformation.

    Args:
        landmarks: Landmarks to align (n_points, 3)
        base_landmarks: Base landmarks to align to (n_points, 3)
        alignment_indices: Optional set or list of landmark indices to use for
            calculating alignment. If provided, overrides the default anatomic0
            landmarks. If None, uses nose landmarks and computed midpoints.

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

    # If alignment_indices is provided, use those instead of anatomic0 defaults
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
                "Using anatomic0 landmarks for alignment."
            )
            # Fall through to use anatomic0 defaults
        else:
            # Use only specified landmarks for alignment calculation
            landmarks_for_alignment = landmarks[indices_list]
            base_for_alignment = base_landmarks[indices_list]
            logger.debug(
                f"Using {len(indices_list)} custom landmarks for alignment calculation"
            )
            # Skip to alignment computation
            use_custom_indices = True
    else:
        use_custom_indices = False

    # Use anatomic0 method: nose landmarks + midpoints
    if not use_custom_indices or "use_custom_indices" not in locals():
        # Start with nose landmarks, but validate them first
        max_idx = len(landmarks)
        anatomic_indices = [idx for idx in NOSE_LANDMARKS if 0 <= idx < max_idx]

        if len(anatomic_indices) < len(NOSE_LANDMARKS):
            logger.warning(
                f"Some nose landmarks are out of bounds for landmarks with {max_idx} points. "
                f"Using {len(anatomic_indices)} valid nose landmarks."
            )

        # Add midpoints by computing them and appending to the point sets
        # We'll create extended arrays with the original points plus computed midpoints
        midpoint_landmarks = []
        midpoint_base = []

        for idx1, idx2 in ANATOMIC0_MIDPOINT_PAIRS:
            # Validate landmark indices
            if idx1 < 0 or idx1 >= max_idx or idx2 < 0 or idx2 >= max_idx:
                logger.warning(
                    f"Invalid midpoint indices ({idx1}, {idx2}) for landmarks with "
                    f"{max_idx} points. Skipping this midpoint."
                )
                continue

            # Compute midpoints
            midpoint_current = (landmarks[idx1] + landmarks[idx2]) / 2.0
            midpoint_base_pt = (base_landmarks[idx1] + base_landmarks[idx2]) / 2.0

            midpoint_landmarks.append(midpoint_current)
            midpoint_base.append(midpoint_base_pt)

        # Validate that we have enough points
        if len(anatomic_indices) == 0 and len(midpoint_landmarks) == 0:
            logger.warning(
                "No valid anatomic0 landmarks found. Returning unaligned landmarks."
            )
            return landmarks

        # Create the point sets for alignment
        # Use nose landmarks if any are valid
        if len(anatomic_indices) > 0:
            landmarks_for_alignment = landmarks[anatomic_indices].copy()
            base_for_alignment = base_landmarks[anatomic_indices].copy()
        else:
            # Start with empty arrays if no valid nose landmarks
            landmarks_for_alignment = np.empty((0, 3), dtype=np.float64)
            base_for_alignment = np.empty((0, 3), dtype=np.float64)

        # Append computed midpoints if any
        if len(midpoint_landmarks) > 0:
            landmarks_for_alignment = np.vstack(
                [landmarks_for_alignment, np.array(midpoint_landmarks)]
            )
            base_for_alignment = np.vstack(
                [base_for_alignment, np.array(midpoint_base)]
            )

        logger.debug(
            f"Using anatomic0 method with {len(anatomic_indices)} nose landmarks "
            f"and {len(midpoint_landmarks)} midpoints "
            f"(total: {len(landmarks_for_alignment)} points)"
        )
        use_custom_indices = False

    # Now perform Procrustes alignment (Kabsch algorithm)
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

    # Apply rotation and translation to ALL original landmarks
    all_landmarks_centered = landmarks - landmarks_center
    aligned = (rotation @ all_landmarks_centered.T).T + base_center

    logger.debug(
        f"Anatomic0 alignment: translation={base_center - landmarks_center}, "
        f"rotation_det={np.linalg.det(rotation):.3f}"
    )

    return aligned
