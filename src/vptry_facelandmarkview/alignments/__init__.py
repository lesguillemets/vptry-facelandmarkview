"""
Alignment functions for face landmarks.

This module provides different methods for aligning face landmarks,
allowing users to experiment with different alignment strategies.
"""

from typing import Callable, Optional
import numpy as np
import numpy.typing as npt

# Type alias for alignment function
AlignmentFunction = Callable[
    [
        npt.NDArray[np.float64],  # landmarks to align
        npt.NDArray[np.float64],  # base landmarks
        Optional[set[int] | list[int]],  # optional alignment indices
    ],
    npt.NDArray[np.float64],  # aligned landmarks
]

from vptry_facelandmarkview.alignments.default import align_landmarks_default
from vptry_facelandmarkview.alignments.scipy_procrustes import (
    align_landmarks_scipy_procrustes,
)

# Registry of available alignment methods
ALIGNMENT_METHODS: dict[str, AlignmentFunction] = {
    "default": align_landmarks_default,
    "scipy procrustes": align_landmarks_scipy_procrustes,
}

# Default alignment method
DEFAULT_ALIGNMENT_METHOD = "default"


def get_alignment_method(name: str) -> AlignmentFunction:
    """Get alignment function by name
    
    Args:
        name: Name of the alignment method
        
    Returns:
        Alignment function
        
    Raises:
        KeyError: If alignment method not found
    """
    return ALIGNMENT_METHODS[name]


def get_available_alignment_methods() -> list[str]:
    """Get list of available alignment method names
    
    Returns:
        List of alignment method names
    """
    return list(ALIGNMENT_METHODS.keys())


__all__ = [
    "AlignmentFunction",
    "ALIGNMENT_METHODS",
    "DEFAULT_ALIGNMENT_METHOD",
    "get_alignment_method",
    "get_available_alignment_methods",
    "align_landmarks_default",
    "align_landmarks_scipy_procrustes",
]
