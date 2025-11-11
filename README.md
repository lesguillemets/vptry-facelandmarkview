# Face Landmark Viewer

A PySide6-based 3D visualization tool for viewing face landmark data from .npy files.

## Features

- Load and visualize 3D face landmark data from NumPy (.npy) files
- Interactive frame navigation using a slider
- Display both current frame and base frame landmarks
- Optional vector visualization showing movement from base frame to current frame
- 3D interactive plot with rotation and zoom capabilities

## Requirements

- Python 3.8+
- PySide6
- NumPy
- Matplotlib

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

```bash
python facelandmarkview.py
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

- **Load .npy File**: Open a file dialog to load landmark data
- **Base Frame**: Select which frame to use as the base reference (blue markers)
- **Show Vectors**: Toggle visualization of vectors from base frame to current frame
- **Frame Slider**: Navigate through frames to see animation

## Visualization

- **Blue circles**: Base frame landmarks (reference position)
- **Red triangles**: Current frame landmarks
- **Green lines**: Vectors from base to current frame (when enabled)