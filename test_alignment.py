#!/usr/bin/env python3
"""
Test alignment functionality
"""

import numpy as np
from vptry_facelandmarkview.utils import align_landmarks_to_base


def test_alignment_identity():
    """Test that aligning identical landmarks returns the same landmarks"""
    print("Test: Alignment with identical landmarks")

    # Create base landmarks
    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # Align to itself - should return the same
    aligned = align_landmarks_to_base(base, base)

    # Check that they are approximately equal
    np.testing.assert_array_almost_equal(aligned, base, decimal=5)
    print("  ✓ Identity alignment works correctly")


def test_alignment_translation():
    """Test that alignment removes translation"""
    print("\nTest: Alignment removes translation")

    # Create base landmarks
    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # Create translated landmarks (moved by [5, 3, -2])
    translation = np.array([5.0, 3.0, -2.0])
    translated = base + translation

    # Align translated back to base
    aligned = align_landmarks_to_base(translated, base)

    # Should be approximately equal to base
    np.testing.assert_array_almost_equal(aligned, base, decimal=5)
    print("  ✓ Translation removed correctly")


def test_alignment_rotation():
    """Test that alignment removes rotation"""
    print("\nTest: Alignment removes rotation")

    # Create base landmarks - a simple square in xy plane
    base = np.array(
        [
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
        ]
    )

    # Create rotation matrix (45 degrees around Z axis)
    angle = np.pi / 4  # 45 degrees
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotation = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])

    # Rotate landmarks
    rotated = (rotation @ base.T).T

    # Align rotated back to base
    aligned = align_landmarks_to_base(rotated, base)

    # Should be approximately equal to base
    np.testing.assert_array_almost_equal(aligned, base, decimal=5)
    print("  ✓ Rotation removed correctly")


def test_alignment_combined():
    """Test that alignment removes both translation and rotation"""
    print("\nTest: Alignment removes translation + rotation")

    # Create base landmarks
    base = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )

    # Create rotation matrix (30 degrees around Z axis)
    angle = np.pi / 6  # 30 degrees
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotation = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])

    # Apply rotation and translation
    translation = np.array([10.0, -5.0, 3.0])
    transformed = (rotation @ base.T).T + translation

    # Align transformed back to base
    aligned = align_landmarks_to_base(transformed, base)

    # Should be approximately equal to base
    np.testing.assert_array_almost_equal(aligned, base, decimal=5)
    print("  ✓ Combined transformation removed correctly")


def test_alignment_empty():
    """Test alignment with empty arrays"""
    print("\nTest: Alignment with empty arrays")

    empty = np.array([]).reshape(0, 3)
    base = np.array([[1.0, 2.0, 3.0]])

    # Aligning empty should return empty
    result = align_landmarks_to_base(empty, base)
    assert len(result) == 0
    print("  ✓ Empty array handling works correctly")


def test_alignment_with_sample_data():
    """Test alignment with sample data"""
    print("\nTest: Alignment with real sample data")

    try:
        data = np.load("sample_landmarks.npy")
        print(f"  Loaded sample data: {data.shape}")

        # Get base and current frame
        base_frame = data[0]
        current_frame = data[10]

        print(f"  Base frame shape: {base_frame.shape}")
        print(f"  Current frame shape: {current_frame.shape}")

        # Align current to base
        aligned = align_landmarks_to_base(current_frame, base_frame)

        print(f"  Aligned shape: {aligned.shape}")
        assert aligned.shape == current_frame.shape

        # Check that center is close to base center after alignment
        base_center = base_frame.mean(axis=0)
        aligned_center = aligned.mean(axis=0)

        print(f"  Base center: {base_center}")
        print(f"  Aligned center: {aligned_center}")
        print(f"  Center difference: {np.linalg.norm(aligned_center - base_center)}")

        # Centers should be close (translation removed)
        np.testing.assert_array_almost_equal(aligned_center, base_center, decimal=5)
        print("  ✓ Sample data alignment works correctly")

    except FileNotFoundError:
        print("  ⚠ Sample data not found, skipping this test")


def test_alignment_with_indices():
    """Test alignment using specific landmark indices"""
    print("\nTest: Alignment with specific landmark indices")
    
    # Create base landmarks (5 points)
    base = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ])
    
    # Create translated and rotated landmarks
    angle = np.pi / 6  # 30 degrees
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotation = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    translation = np.array([5.0, 3.0, -2.0])
    current = (rotation @ base.T).T + translation
    
    # Align using only first 3 points (indices 0, 1, 2)
    alignment_indices = [0, 1, 2]
    aligned = align_landmarks_to_base(current, base, alignment_indices=alignment_indices)
    
    print(f"  Using indices {alignment_indices} for alignment")
    print(f"  Aligned shape: {aligned.shape}")
    assert aligned.shape == current.shape
    
    # Check that the alignment was computed using only specified points
    # The first 3 points should be very close to base
    for i in alignment_indices:
        diff = np.linalg.norm(aligned[i] - base[i])
        print(f"  Point {i} difference: {diff:.6f}")
        assert diff < 0.01, f"Point {i} should be well-aligned"
    
    print("  ✓ Alignment with indices works correctly")
    
    # Test with set instead of list
    alignment_indices_set = {1, 2, 3}
    aligned_set = align_landmarks_to_base(current, base, alignment_indices=alignment_indices_set)
    assert aligned_set.shape == current.shape
    print("  ✓ Alignment with set of indices works correctly")
    
    # Test with invalid indices (should use all landmarks)
    invalid_indices = [0, 1, 100]  # 100 is out of range
    aligned_invalid = align_landmarks_to_base(current, base, alignment_indices=invalid_indices)
    assert aligned_invalid.shape == current.shape
    print("  ✓ Invalid indices handled gracefully")


def main():
    """Run all alignment tests"""
    print("=" * 60)
    print("Face Alignment Tests")
    print("=" * 60)
    print()

    try:
        test_alignment_identity()
        test_alignment_translation()
        test_alignment_rotation()
        test_alignment_combined()
        test_alignment_empty()
        test_alignment_with_sample_data()
        test_alignment_with_indices()

        print()
        print("=" * 60)
        print("All alignment tests passed! ✓")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
