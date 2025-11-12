# Face Landmark Viewer

A PySide6-based 3D visualization tool for viewing face landmark data from .npy files using OpenGL rendering.

## Features

- Load and visualize 3D face landmark data from NumPy (.npy) files
- Interactive frame navigation using a slider
- Display both current frame and base frame landmarks
- Optional vector visualization showing movement from base frame to current frame
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
pip install -r requirements.txt
```

## Usage

### Running the Application

**Basic usage (with GUI file picker):**
```bash
python facelandmarkview.py
```

**Load a file directly from command line:**
```bash
python facelandmarkview.py sample_landmarks.npy
```

**Load a file with a specific base frame:**
```bash
python facelandmarkview.py sample_landmarks.npy --base-frame 10
```

**View help and options:**
```bash
python facelandmarkview.py --help
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
python generate_sample_data.py
```

This creates `sample_landmarks.npy` with 50 frames and 68 landmarks.

## Controls

### UI Controls
- **Load .npy File**: Open a file dialog to load landmark data
- **Base Frame**: Select which frame to use as the base reference (blue points)
- **Show Vectors**: Toggle visualization of vectors from base frame to current frame
- **Frame Slider**: Navigate through frames to see animation

### Mouse Controls
- **Left Click + Drag**: Rotate the 3D view
- **Mouse Wheel**: Zoom in/out

## Visualization

- **Blue points**: Base frame landmarks (reference position)
- **Red points**: Current frame landmarks (larger)
- **Green lines**: Vectors from base to current frame (when enabled)
- **RGB axes**: Red (X), Green (Y), Blue (Z) coordinate axes

## Implementation Notes

The application uses:
- **Type annotations** throughout for better code clarity and IDE support
- **pathlib.Path** for cross-platform file path handling
- **OpenGL** for hardware-accelerated 3D rendering (instead of matplotlib)