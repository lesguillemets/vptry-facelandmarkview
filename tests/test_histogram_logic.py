#!/usr/bin/env python3
"""
Test the histogram calculation logic without requiring GUI.
"""

import sys
import numpy as np


def test_distance_calculation():
    """Test that distance calculation works correctly"""
    print("Test: Distance calculation")
    
    # Create simple test data
    # 3 frames, 5 landmarks, 3 coordinates
    data = np.array([
        # Frame 0 (base frame)
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
        # Frame 1 (moved slightly)
        [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 1.1], [1.0, 1.0, 1.0]],
        # Frame 2 (moved more)
        [[0.5, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.5], [1.0, 1.0, 1.0]],
    ])
    
    base_frame = 0
    current_frame = 1
    
    base_landmarks = data[base_frame]
    current_landmarks = data[current_frame]
    
    # Calculate distances manually
    distances = np.linalg.norm(current_landmarks - base_landmarks, axis=1)
    
    print(f"  Base landmarks:\n{base_landmarks}")
    print(f"  Current landmarks:\n{current_landmarks}")
    print(f"  Distances: {distances}")
    
    # Check expected distances
    # Landmarks 0-3 moved by 0.1, landmark 4 didn't move
    expected = np.array([0.1, 0.1, 0.1, 0.1, 0.0])
    
    assert np.allclose(distances, expected), f"Expected {expected}, got {distances}"
    print("  ✓ Distance calculation correct")
    
    return True


def test_histogram_creation():
    """Test histogram creation with percentile-based outlier detection"""
    print("\nTest: Histogram creation with outliers")
    
    # Create data with some outliers
    np.random.seed(42)
    
    # Most values are small (0-1), with a few outliers (>2)
    distances = np.concatenate([
        np.random.uniform(0, 1, 95),  # 95 normal values
        np.random.uniform(2, 5, 5),   # 5 outliers
    ])
    
    print(f"  Total points: {len(distances)}")
    print(f"  Distance range: [{distances.min():.3f}, {distances.max():.3f}]")
    
    # Calculate 95th percentile
    percentile_95 = np.percentile(distances, 95)
    print(f"  95th percentile: {percentile_95:.3f}")
    
    # Count outliers
    outlier_mask = distances > percentile_95
    outlier_count = outlier_mask.sum()
    print(f"  Outlier count: {outlier_count}")
    
    # Create histogram for non-outliers
    non_outlier_distances = distances[~outlier_mask]
    hist_values, bin_edges = np.histogram(
        non_outlier_distances,
        bins=20,
        range=(0, percentile_95)
    )
    
    print(f"  Histogram bins: {len(hist_values)}")
    print(f"  Histogram range: [0, {percentile_95:.3f}]")
    print(f"  Non-outlier points: {len(non_outlier_distances)}")
    
    # Verify that approximately 5% are outliers (with some tolerance due to discrete percentiles)
    outlier_percentage = (outlier_count / len(distances)) * 100
    print(f"  Outlier percentage: {outlier_percentage:.1f}%")
    
    assert outlier_count >= 5, f"Expected at least 5 outliers, got {outlier_count}"
    assert outlier_count <= 10, f"Expected at most 10 outliers (allowing some variance), got {outlier_count}"
    assert len(hist_values) == 20, f"Expected 20 bins, got {len(hist_values)}"
    assert hist_values.sum() == len(non_outlier_distances), "Histogram should contain all non-outlier points"
    
    print("  ✓ Histogram creation correct")
    print("  ✓ Outliers detected and separated correctly")
    
    return True


def test_nan_handling():
    """Test that NaN values are properly filtered"""
    print("\nTest: NaN handling")
    
    # Create data with NaN values
    data = np.array([
        # Frame 0 (base)
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [np.nan, np.nan, np.nan], [0.0, 0.0, 1.0]],
        # Frame 1 (current)
        [[0.1, 0.0, 0.0], [np.nan, np.nan, np.nan], [0.5, 0.5, 0.0], [0.0, 0.0, 1.1]],
    ])
    
    base_landmarks = data[0]
    current_landmarks = data[1]
    
    # Filter out NaN values
    base_valid_mask = ~np.isnan(base_landmarks).any(axis=1)
    current_valid_mask = ~np.isnan(current_landmarks).any(axis=1)
    both_valid_mask = base_valid_mask & current_valid_mask
    
    base_both = base_landmarks[both_valid_mask]
    current_both = current_landmarks[both_valid_mask]
    
    print(f"  Base landmarks valid: {base_valid_mask}")
    print(f"  Current landmarks valid: {current_valid_mask}")
    print(f"  Both valid: {both_valid_mask}")
    print(f"  Valid pairs: {len(base_both)}")
    
    # Only landmarks 0 and 3 are valid in both frames
    assert len(base_both) == 2, f"Expected 2 valid pairs, got {len(base_both)}"
    
    # Calculate distances for valid pairs
    distances = np.linalg.norm(current_both - base_both, axis=1)
    print(f"  Distances: {distances}")
    
    assert len(distances) == 2, f"Expected 2 distances, got {len(distances)}"
    assert np.allclose(distances[0], 0.1), f"Expected distance 0.1, got {distances[0]}"
    assert np.allclose(distances[1], 0.1), f"Expected distance 0.1, got {distances[1]}"
    
    print("  ✓ NaN values filtered correctly")
    print("  ✓ Distances calculated only for valid pairs")
    
    return True


def main():
    """Run histogram logic tests"""
    print("=" * 60)
    print("Testing Histogram Logic (no GUI required)")
    print("=" * 60)
    print()
    
    try:
        if not test_distance_calculation():
            return 1
        if not test_histogram_creation():
            return 1
        if not test_nan_handling():
            return 1
        
        print()
        print("=" * 60)
        print("All histogram logic tests passed! ✓")
        print("=" * 60)
        return 0
    
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
