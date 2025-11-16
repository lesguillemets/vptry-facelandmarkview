# Face Landmark Viewer

A PySide6-based 3D visualization tool for viewing face landmark data from .npy files using OpenGL rendering.

## Features

- Load and visualize 3D face landmark data from NumPy (.npy) files
- Interactive frame navigation using a slider
- Display both current frame and base frame landmarks
- Optional vector visualization showing movement from base frame to current frame
- **Face alignment** to remove head movement effects and highlight facial expressions
- Hardware-accelerated 3D rendering using OpenGL
- Interactive mouse controls for rotation (drag) and zoom (wheel)

## Requirements

- Python 3.8+
- PySide6
- NumPy
- PyOpenGL
- PyOpenGL-accelerate (optional, for better performance)

## Installation

```bash
pip install -e .
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

**Using the installed command:**
```bash
facelandmarkview
```

**Or using Python module:**
```bash
python -m vptry_facelandmarkview.main
```

**Or using backward compatibility wrapper:**
```bash
python facelandmarkview.py
```

**Load a file directly from command line:**
```bash
facelandmarkview sample_landmarks.npy
```

**Load a file with a specific base frame:**
```bash
facelandmarkview sample_landmarks.npy --base-frame 10
```

**View help and options:**
```bash
facelandmarkview --help
```

### Data Format

The application expects .npy files containing a NumPy array of shape `(n_frames, n_landmarks, 3)` where:
- `n_frames`: Number of frames in the sequence
- `n_landmarks`: Number of landmarks per frame
- `3`: X, Y, Z coordinates

For each frame `fr` and landmark `p`:
- `dat[fr][p][0]`: X coordinate
- `dat[fr][p][1]`: Y coordinate
- `dat[fr][p][2]`: Z coordinate

### Generating Sample Data

To generate sample data for testing:

```bash
python tests/generate_sample_data.py
```

This creates `sample_landmarks.npy` with 50 frames and 68 landmarks.

## Controls

### UI Controls
- **Load .npy File**: Open a file dialog to load landmark data
- **Base Frame**: Select which frame to use as the base reference (blue points)
- **Show Vectors**: Toggle visualization of vectors from base frame to current frame
- **Align Faces**: Align current frame to base frame to remove head movement (translation and rotation), making facial expression changes easier to see
- **Limit to Static Points**: When enabled with "Align Faces", uses only stable facial landmarks (nose and forehead, 30 points total) for computing alignment. This provides more robust alignment when expressions vary significantly, as it focuses on features that don't move with expressions.
- **Frame Slider**: Navigate through frames to see animation

### Mouse Controls
- **Left Click + Drag**: Rotate the 3D view
- **Mouse Wheel**: Zoom in/out

## Visualization

- **Blue points**: Base frame landmarks (reference position)
- **Red points**: Current frame landmarks (larger)
- **Green lines**: Vectors from base to current frame (when enabled)
- **RGB axes**: Red (X), Green (Y), Blue (Z) coordinate axes

### Face Alignment

The "Align Faces" feature removes the effects of head movement (translation and rotation) to make facial expression changes more visible. When enabled:

- The current frame landmarks are aligned to the base frame using Procrustes alignment (Kabsch algorithm)
- This computes the optimal rigid transformation (rotation + translation) that best matches the two point sets
- Head position and orientation differences are removed
- Facial expression changes (relative movements of landmarks) are preserved
- Useful for analyzing subtle facial expressions without distraction from head movement

#### Static Points for Robust Alignment

The application includes preset landmark indices for stable facial features:
- **Nose landmarks** (24 points): Nose bridge, sides, and tip region
- **Forehead landmarks** (6 points): Upper forehead area
- **Combined default** (30 points): Nose + forehead for optimal stability

Enable "Limit to Static Points" in the UI to use these stable landmarks for alignment computation. This is particularly effective when:
- Facial expressions vary significantly (e.g., speaking, smiling)
- You want to focus on expression changes in the mouth or eye areas
- The full face alignment is affected by expression-related movements

#### Programmatic Usage with Specific Landmarks

The alignment function can be used programmatically with optional landmark selection:

```python
from vptry_facelandmarkview.utils import align_landmarks_to_base
from vptry_facelandmarkview.constants import DEFAULT_ALIGNMENT_LANDMARKS

# Align using all landmarks
aligned = align_landmarks_to_base(current_landmarks, base_landmarks)

# Align using preset static points (nose + forehead)
aligned = align_landmarks_to_base(
    current_landmarks, 
    base_landmarks, 
    alignment_indices=DEFAULT_ALIGNMENT_LANDMARKS
)

# Align using custom landmark indices
alignment_indices = [0, 1, 2, 5, 10]  # or a set: {0, 1, 2, 5, 10}
aligned = align_landmarks_to_base(
    current_landmarks, 
    base_landmarks, 
    alignment_indices=alignment_indices
)
```

This is useful when you want to align based on stable facial features while preserving movements in expression-active areas.

## Project Structure

The project follows a modular src-layout structure:

```
src/vptry_facelandmarkview/
├── __init__.py          # Package initialization and exports
├── constants.py         # Application constants
├── utils.py             # Utility functions for landmark processing
├── gl_widget.py         # OpenGL widget for 3D rendering
├── viewer.py            # Main window and UI components
└── main.py              # Application entry point
```

The old `facelandmarkview.py` is maintained as a backward compatibility wrapper that imports from the new package structure.

## Implementation Notes

The application uses:
- **Type annotations** throughout for better code clarity and IDE support
- **pathlib.Path** for cross-platform file path handling
- **OpenGL** for hardware-accelerated 3D rendering (instead of matplotlib)
- **Modular architecture** with separate modules for better organization and maintainability
- **Absolute imports** for clear dependency management