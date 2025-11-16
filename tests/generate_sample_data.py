#!/usr/bin/env python3
"""
Generate sample face landmark data for testing
"""

from typing import Union
import numpy as np
import numpy.typing as npt
from pathlib import Path


def generate_sample_data(
    n_frames: int = 50,
    n_landmarks: int = 68,
    output_file: Union[str, Path] = "sample_landmarks.npy",
) -> npt.NDArray[np.float64]:
    """
    Generate sample face landmark data with animation

    Args:
        n_frames: Number of frames
        n_landmarks: Number of landmarks (default 68 for face)
        output_file: Output filename
    """
    output_path = Path(output_file)
    # Create base landmarks in a face-like pattern
    # Arrange landmarks in a semi-circular pattern (simplified face)
    angles = np.linspace(0, 2 * np.pi, n_landmarks)
    radius = 1.0

    # Base face shape
    base_x = radius * np.cos(angles)
    base_y = radius * np.sin(angles)
    base_z = np.zeros(n_landmarks)

    # Create data array
    data = np.zeros((n_frames, n_landmarks, 3))

    # Generate animation: face moves and deforms over time
    for frame in range(n_frames):
        t = frame / n_frames

        # Add some motion: rotation and translation
        rotation = np.sin(2 * np.pi * t) * 0.3

        # Rotate around Z axis
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)

        x = base_x * cos_r - base_y * sin_r
        y = base_x * sin_r + base_y * cos_r
        z = base_z + 0.2 * np.sin(2 * np.pi * t)

        # Add some deformation (mouth opening, eyebrow raising, etc.)
        deformation = 0.1 * np.sin(4 * np.pi * t + angles)
        x += deformation * np.cos(angles)
        y += deformation * np.sin(angles)

        # Add some noise
        x += np.random.normal(0, 0.01, n_landmarks)
        y += np.random.normal(0, 0.01, n_landmarks)
        z += np.random.normal(0, 0.01, n_landmarks)

        data[frame, :, 0] = x
        data[frame, :, 1] = y
        data[frame, :, 2] = z

    # Save to file
    np.save(output_path, data)
    print(f"Generated sample data: {output_path}")
    print(f"Shape: {data.shape}")
    print(f"Frames: {n_frames}, Landmarks: {n_landmarks}")

    return data


if __name__ == "__main__":
    generate_sample_data()
