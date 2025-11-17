#!/usr/bin/env python3
"""
Demo script showing the alignment methods feature.

This script demonstrates:
1. The available alignment methods
2. How to use them programmatically
3. The difference between the methods

Note: This is a demonstration script, not a visual UI test.
The UI includes a dropdown menu labeled "Alignment Method:" in the top control panel
that allows users to switch between "default" and "scipy procrustes" methods.
"""

import numpy as np
from vptry_facelandmarkview.alignments import (
    get_available_alignment_methods,
    get_alignment_method,
)


def demo_alignment_methods():
    """Demonstrate the alignment methods feature"""
    print("=" * 70)
    print("Face Landmark Viewer - Alignment Methods Feature Demo")
    print("=" * 70)
    print()

    # Show available methods
    print("Available Alignment Methods:")
    print("-" * 70)
    methods = get_available_alignment_methods()
    for i, method in enumerate(methods, 1):
        print(f"  {i}. {method}")
    print()

    # Create test data
    print("Creating test landmark data...")
    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # Apply transformation (rotation + translation)
    angle = np.pi / 4  # 45 degrees
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
    translation = np.array([5.0, 3.0, -2.0])
    current = (rotation @ base.T).T + translation

    print(f"Base landmarks shape: {base.shape}")
    print(f"Current landmarks shape: {current.shape}")
    print()

    # Test each alignment method
    print("Comparing Alignment Methods:")
    print("-" * 70)

    original_distances = np.linalg.norm(current - base, axis=1)
    print(f"Original distances from base (per landmark):")
    print(f"  Mean: {original_distances.mean():.6f}")
    print(f"  Min:  {original_distances.min():.6f}")
    print(f"  Max:  {original_distances.max():.6f}")
    print()

    for method_name in methods:
        print(f"\nTesting '{method_name}' method:")
        print("-" * 35)

        # Get and apply alignment method
        align_func = get_alignment_method(method_name)
        aligned = align_func(current, base)

        # Calculate alignment quality metrics
        aligned_distances = np.linalg.norm(aligned - base, axis=1)

        print(f"  Aligned distances from base (per landmark):")
        print(f"    Mean: {aligned_distances.mean():.6f}")
        print(f"    Min:  {aligned_distances.min():.6f}")
        print(f"    Max:  {aligned_distances.max():.6f}")

        # Check if centers match
        base_center = base.mean(axis=0)
        aligned_center = aligned.mean(axis=0)
        center_diff = np.linalg.norm(aligned_center - base_center)
        print(f"  Center alignment error: {center_diff:.6f}")

        # Method-specific notes
        if method_name == "default":
            print(
                "\n  Notes: Uses Kabsch algorithm (rigid transformation only - "
                "rotation + translation)"
            )
            print("         Preserves relative scale of facial features")
        elif method_name == "scipy procrustes":
            print(
                "\n  Notes: Uses scipy.spatial.procrustes (includes scaling in "
                "addition to rotation + translation)"
            )
            print("         Can normalize differences in face sizes")

    print()
    print("=" * 70)
    print("UI Integration:")
    print("-" * 70)
    print("In the Face Landmark Viewer application, users can:")
    print("  1. Find the 'Alignment Method:' dropdown in the top control panel")
    print("  2. Select between 'default' and 'scipy procrustes' methods")
    print("  3. The alignment will update in real-time across all views:")
    print("     - Main 3D view")
    print("     - Top projection (X-Z)")
    print("     - Side projection (Y-Z)")
    print("     - Distance histogram")
    print()
    print("This allows for easy experimentation to find the best alignment")
    print("method for specific datasets and analysis goals.")
    print("=" * 70)


if __name__ == "__main__":
    demo_alignment_methods()
