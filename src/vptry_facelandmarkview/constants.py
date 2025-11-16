"""
Constants used throughout the Face Landmark Viewer application.
"""

POINT_SIZE = 2.0
SCALE_MARGIN = 1.2  # 20% margin for scaling

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
