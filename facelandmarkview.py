#!/usr/bin/env python3
"""
Face Landmark Viewer - Backward compatibility wrapper

This module provides backward compatibility by re-exporting all functionality
from the new src/vptry_facelandmarkview package structure.
"""

from typing import TYPE_CHECKING

# Import non-Qt dependencies eagerly
from vptry_facelandmarkview.constants import POINT_SIZE, SCALE_MARGIN
from vptry_facelandmarkview.utils import (
    filter_nan_landmarks,
    calculate_center_and_scale,
    draw_landmarks,
)

# Type checking imports (not executed at runtime)
if TYPE_CHECKING:
    from vptry_facelandmarkview import FaceLandmarkViewer, LandmarkGLWidget, main

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


def __getattr__(name: str):
    """Lazy import of Qt-dependent classes and functions"""
    if name == "FaceLandmarkViewer":
        from vptry_facelandmarkview import FaceLandmarkViewer
        return FaceLandmarkViewer
    elif name == "LandmarkGLWidget":
        from vptry_facelandmarkview import LandmarkGLWidget
        return LandmarkGLWidget
    elif name == "main":
        from vptry_facelandmarkview import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    from vptry_facelandmarkview import main
    main()
