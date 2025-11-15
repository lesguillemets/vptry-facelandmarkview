"""
Face Landmark Viewer - 3D visualization of face landmarks from .npy files using OpenGL
"""

# Lazy imports to avoid importing Qt dependencies on module import
# This allows importing utility modules without Qt
__all__ = ["FaceLandmarkViewer", "LandmarkGLWidget", "main"]


def __getattr__(name):
    """Lazy import of Qt-dependent modules"""
    if name == "FaceLandmarkViewer":
        from vptry_facelandmarkview.viewer import FaceLandmarkViewer

        return FaceLandmarkViewer
    elif name == "LandmarkGLWidget":
        from vptry_facelandmarkview.gl_widget import LandmarkGLWidget

        return LandmarkGLWidget
    elif name == "main":
        from vptry_facelandmarkview.main import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
