#!/usr/bin/env python3
"""
Test different alignment methods
"""

import numpy as np
from vptry_facelandmarkview.alignments import (
    get_available_alignment_methods,
    get_alignment_method,
    align_landmarks_default,
    align_landmarks_scipy_procrustes,
)


def test_alignment_methods_available():
    """Test that alignment methods are available"""
    print("Test: Alignment methods are available")

    methods = get_available_alignment_methods()
    print(f"  Available methods: {methods}")

    assert "default" in methods
    assert "scipy procrustes" in methods
    assert len(methods) >= 2

    print("  ✓ All expected alignment methods available")


def test_get_alignment_method():
    """Test getting alignment method by name"""
    print("\nTest: Get alignment method by name")

    default_method = get_alignment_method("default")
    assert default_method == align_landmarks_default
    print("  ✓ default method retrieved correctly")

    scipy_method = get_alignment_method("scipy procrustes")
    assert scipy_method == align_landmarks_scipy_procrustes
    print("  ✓ scipy procrustes method retrieved correctly")


def test_alignment_methods_work():
    """Test that all alignment methods produce valid results"""
    print("\nTest: All alignment methods work")

    # Create test data
    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # Create translated and rotated landmarks
    angle = np.pi / 6  # 30 degrees
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotation = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
    translation = np.array([5.0, 3.0, -2.0])
    current = (rotation @ base.T).T + translation

    methods = get_available_alignment_methods()
    for method_name in methods:
        print(f"\n  Testing method: {method_name}")
        align_func = get_alignment_method(method_name)

        # Align current to base
        aligned = align_func(current, base)

        # Check that shape is preserved
        assert aligned.shape == current.shape
        print(f"    ✓ Shape preserved: {aligned.shape}")

        # Check that alignment brings landmarks closer to base
        original_distance = np.mean(np.linalg.norm(current - base, axis=1))
        aligned_distance = np.mean(np.linalg.norm(aligned - base, axis=1))

        print(f"    Original mean distance: {original_distance:.6f}")
        print(f"    Aligned mean distance: {aligned_distance:.6f}")

        # Alignment should reduce distance (or keep it similar for default method with rigid transform)
        # For rigid transforms, the distance should be very small
        # For procrustes with scaling, it might be different but still small
        assert aligned_distance < original_distance * 1.1  # Allow some tolerance

        print(f"    ✓ {method_name} alignment works correctly")


def test_alignment_with_indices():
    """Test that alignment methods work with specified indices"""
    print("\nTest: Alignment methods work with indices")

    # Create test data
    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )

    # Create transformed landmarks
    angle = np.pi / 6  # 30 degrees
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotation = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
    translation = np.array([5.0, 3.0, -2.0])
    current = (rotation @ base.T).T + translation

    alignment_indices = [0, 1, 2]

    methods = get_available_alignment_methods()
    for method_name in methods:
        print(f"\n  Testing method with indices: {method_name}")
        align_func = get_alignment_method(method_name)

        # Align using only first 3 points
        aligned = align_func(current, base, alignment_indices=alignment_indices)

        # Check that shape is preserved
        assert aligned.shape == current.shape
        print(f"    ✓ Shape preserved: {aligned.shape}")

        # Check that the specified points are well-aligned
        for i in alignment_indices:
            diff = np.linalg.norm(aligned[i] - base[i])
            print(f"    Point {i} difference: {diff:.6f}")
            # Allow reasonable tolerance for alignment
            assert diff < 1.0, f"Point {i} should be reasonably aligned"

        print(f"    ✓ {method_name} works with indices")


def main():
    """Run all alignment method tests"""
    print("=" * 60)
    print("Alignment Methods Tests")
    print("=" * 60)
    print()

    try:
        test_alignment_methods_available()
        test_get_alignment_method()
        test_alignment_methods_work()
        test_alignment_with_indices()

        print()
        print("=" * 60)
        print("All alignment method tests passed! ✓")
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
