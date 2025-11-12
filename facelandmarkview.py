#!/usr/bin/env python3
"""
Face Landmark Viewer - Backward compatibility wrapper

This module provides backward compatibility by re-exporting all functionality
from the new src/vptry_facelandmarkview package structure.
"""

# Re-export all public classes and functions for backward compatibility
from vptry_facelandmarkview import FaceLandmarkViewer, LandmarkGLWidget, main
from vptry_facelandmarkview.constants import POINT_SIZE, SCALE_MARGIN
from vptry_facelandmarkview.utils import (
    filter_nan_landmarks,
    calculate_center_and_scale,
    draw_landmarks,
)

__all__ = [
    "FaceLandmarkViewer",
    "LandmarkGLWidget",
    "main",
    "POINT_SIZE",
    "SCALE_MARGIN",
    "filter_nan_landmarks",
    "calculate_center_and_scale",
    "draw_landmarks",
]


if __name__ == "__main__":
    main()
