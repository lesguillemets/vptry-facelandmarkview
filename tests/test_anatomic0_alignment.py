#!/usr/bin/env python3
"""
Test the anatomic0 alignment method with realistic MediaPipe landmark data.
"""

import numpy as np
from vptry_facelandmarkview.alignments import get_alignment_method
from vptry_facelandmarkview.constants import (
    NOSE_LANDMARKS,
    ANATOMIC0_MIDPOINT_PAIRS,
)


def test_anatomic0_with_mediapipe_landmarks():
    """Test anatomic0 alignment with 478 MediaPipe landmarks"""
    print("Test: Anatomic0 alignment with MediaPipe landmarks")

    # MediaPipe Face Landmarker produces 478 landmarks
    n_landmarks = 478

    # Create synthetic base landmarks
    np.random.seed(42)
    base = np.random.randn(n_landmarks, 3) * 0.1
    # Make it look more like a face (centered around origin)
    base[:, 2] -= 0.5  # Push z back a bit

    # Create transformed current landmarks (rotation + translation)
    angle = np.pi / 4  # 45 degrees
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotation = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
    translation = np.array([0.5, 0.3, -0.2])
    current = (rotation @ base.T).T + translation

    # Get the anatomic0 alignment method
    align_func = get_alignment_method("anatomic0")

    # Align current to base
    aligned = align_func(current, base)

    # Verify shape is preserved
    assert aligned.shape == current.shape == (n_landmarks, 3)
    print(f"  ✓ Shape preserved: {aligned.shape}")

    # Check that alignment improves distance
    original_distance = np.mean(np.linalg.norm(current - base, axis=1))
    aligned_distance = np.mean(np.linalg.norm(aligned - base, axis=1))

    print(f"  Original mean distance: {original_distance:.6f}")
    print(f"  Aligned mean distance: {aligned_distance:.6f}")
    print(f"  Improvement: {(1 - aligned_distance / original_distance) * 100:.2f}%")

    # Alignment should significantly reduce distance
    assert aligned_distance < original_distance * 0.1, "Alignment should greatly reduce distance"

    print("  ✓ Anatomic0 alignment significantly improved alignment")


def test_anatomic0_uses_correct_landmarks():
    """Test that anatomic0 uses nose landmarks and midpoints correctly"""
    print("\nTest: Anatomic0 uses correct anatomic landmarks")

    n_landmarks = 478
    np.random.seed(42)
    base = np.random.randn(n_landmarks, 3) * 0.1

    # Create a copy with slight perturbations
    current = base.copy()
    current += np.random.randn(n_landmarks, 3) * 0.01

    # Apply large changes to non-anatomic landmarks
    # (to ensure alignment ignores these)
    non_anatomic_indices = [i for i in range(n_landmarks) if i not in NOSE_LANDMARKS]
    # Take a subset that doesn't include midpoint pairs
    excluded_from_midpoints = set()
    for idx1, idx2 in ANATOMIC0_MIDPOINT_PAIRS:
        excluded_from_midpoints.add(idx1)
        excluded_from_midpoints.add(idx2)

    perturbable = [i for i in non_anatomic_indices if i not in excluded_from_midpoints]
    if len(perturbable) > 10:
        # Add large noise to some non-anatomic landmarks
        current[perturbable[:10]] += np.random.randn(10, 3) * 2.0

    # Align using anatomic0
    align_func = get_alignment_method("anatomic0")
    aligned = align_func(current, base)

    # Check that nose landmarks are well-aligned
    nose_aligned_distance = np.mean(
        np.linalg.norm(aligned[NOSE_LANDMARKS] - base[NOSE_LANDMARKS], axis=1)
    )
    print(f"  Nose landmarks mean distance: {nose_aligned_distance:.6f}")
    assert nose_aligned_distance < 0.05, "Nose landmarks should be well-aligned"
    print("  ✓ Nose landmarks are well-aligned")

    # Check that midpoint landmarks are well-aligned
    for idx1, idx2 in ANATOMIC0_MIDPOINT_PAIRS:
        midpoint_aligned = (aligned[idx1] + aligned[idx2]) / 2.0
        midpoint_base = (base[idx1] + base[idx2]) / 2.0
        midpoint_distance = np.linalg.norm(midpoint_aligned - midpoint_base)
        print(f"  Midpoint ({idx1}, {idx2}) distance: {midpoint_distance:.6f}")
        assert midpoint_distance < 0.1, f"Midpoint ({idx1}, {idx2}) should be well-aligned"

    print("  ✓ Midpoint landmarks are well-aligned")


def test_anatomic0_landmark_constants():
    """Test that the anatomic0 constants are properly defined"""
    print("\nTest: Anatomic0 constants are properly defined")

    # Check that NOSE_LANDMARKS is defined and non-empty
    assert len(NOSE_LANDMARKS) > 0, "NOSE_LANDMARKS should not be empty"
    print(f"  ✓ NOSE_LANDMARKS defined with {len(NOSE_LANDMARKS)} landmarks")

    # Check that all nose landmarks are valid MediaPipe indices
    assert all(0 <= idx < 478 for idx in NOSE_LANDMARKS), (
        "All NOSE_LANDMARKS should be valid MediaPipe indices (0-477)"
    )
    print("  ✓ All nose landmarks are valid MediaPipe indices")

    # Check that ANATOMIC0_MIDPOINT_PAIRS is defined
    assert len(ANATOMIC0_MIDPOINT_PAIRS) > 0, "ANATOMIC0_MIDPOINT_PAIRS should not be empty"
    print(f"  ✓ ANATOMIC0_MIDPOINT_PAIRS defined with {len(ANATOMIC0_MIDPOINT_PAIRS)} pairs")

    # Check that all midpoint pairs contain valid indices
    for idx1, idx2 in ANATOMIC0_MIDPOINT_PAIRS:
        assert 0 <= idx1 < 478, f"Midpoint index {idx1} should be valid (0-477)"
        assert 0 <= idx2 < 478, f"Midpoint index {idx2} should be valid (0-477)"
        assert idx1 != idx2, f"Midpoint pair ({idx1}, {idx2}) should have different indices"

    print("  ✓ All midpoint pairs are valid")

    # Verify the specific pairs mentioned in the issue
    expected_pairs = [(33, 133), (362, 263)]
    for pair in expected_pairs:
        assert pair in ANATOMIC0_MIDPOINT_PAIRS, (
            f"Expected midpoint pair {pair} not found in ANATOMIC0_MIDPOINT_PAIRS"
        )
    print(f"  ✓ All required midpoint pairs present: {expected_pairs}")


def main():
    """Run all anatomic0 tests"""
    print("=" * 60)
    print("Anatomic0 Alignment Method Tests")
    print("=" * 60)
    print()

    try:
        test_anatomic0_landmark_constants()
        test_anatomic0_with_mediapipe_landmarks()
        test_anatomic0_uses_correct_landmarks()

        print()
        print("=" * 60)
        print("All anatomic0 tests passed! ✓")
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
