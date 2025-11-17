"""
Constants used throughout the Face Landmark Viewer application.
"""

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

POINT_SIZE = 2.0
SCALE_MARGIN = 1.2  # 20% margin for scaling

# Camera and view constants
DEFAULT_ROTATION_X = 20.0
DEFAULT_ROTATION_Y = 45.0
DEFAULT_ZOOM = 3.0
MIN_ZOOM = 1.0
MAX_ZOOM = 20.0
ZOOM_IN_FACTOR = 0.9
ZOOM_OUT_FACTOR = 1.1
ROTATION_SENSITIVITY = 0.5

# OpenGL rendering constants
AXIS_LENGTH = 1.5
VECTOR_LINE_WIDTH = 1.0
AXIS_LINE_WIDTH = 2.0
BACKGROUND_COLOR = (1.0, 1.0, 1.0, 1.0)  # White
PERSPECTIVE_FOV = 45.0
PERSPECTIVE_NEAR = 0.1
PERSPECTIVE_FAR = 100.0

# Projection widget constants
PROJECTION_SIZE_PX = (
    100  # Fixed size for projection plots (width for y-z, height for x-z)
)
PROJECTION_VIEWPORT_FILL = 0.8  # Landmarks should span 80% of the viewport
PROJECTION_Z_SCALE = (
    1.5  # Additional scaling factor for z-axis to enhance depth visibility
)


# Color constants for rendering (RGBA tuples)
class Color(NamedTuple):
    """RGBA color representation"""

    r: float
    g: float
    b: float
    a: float


# Landmark colors
BASE_LANDMARK_COLOR = Color(0.0, 0.0, 1.0, 0.6)  # Blue with transparency
CURRENT_LANDMARK_COLOR = Color(1.0, 0.0, 0.0, 0.8)  # Red with transparency
VECTOR_COLOR = Color(0.0, 0.8, 0.0, 0.3)  # Green with transparency

# Axis colors (RGB only)
AXIS_X_COLOR = (1.0, 0.0, 0.0)  # Red
AXIS_Y_COLOR = (0.0, 1.0, 0.0)  # Green
AXIS_Z_COLOR = (0.0, 0.0, 1.0)  # Blue


class ProjectionType(Enum):
    """Enum for projection types"""

    XY = "xy"
    XZ = "xz"
    YZ = "yz"


@dataclass
class DisplayState:
    """State for controlling landmark display across widgets"""

    base_frame: int = 0
    current_frame: int = 0
    show_vectors: bool = False
    align_faces: bool = False
    use_static_points: bool = False


# Landmark indices for alignment using stable facial features
# These indices correspond to MediaPipe Face Landmarker landmarks
# that remain relatively stable across facial expressions

# Nose landmarks (stable across expressions)
NOSE_LANDMARKS = [
    122,
    196,
    3,
    51,
    45,
    44,
    417,
    351,
    419,
    248,
    281,
    275,
    274,
    412,
    399,
    456,
    363,
    440,
    128,
    114,
    217,
    198,
    131,
    115,
]

# Forehead landmarks (very stable, minimal expression movement)
FOREHEAD_LANDMARKS = [
    109,
    10,
    338,
    108,
    151,
    357,
]

# Combined stable landmarks for default alignment
DEFAULT_ALIGNMENT_LANDMARKS = NOSE_LANDMARKS + FOREHEAD_LANDMARKS
