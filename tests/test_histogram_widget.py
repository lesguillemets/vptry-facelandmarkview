#!/usr/bin/env python3
"""
Test the histogram widget functionality without requiring full GUI rendering.
"""

import sys
import numpy as np


def test_histogram_widget_creation():
    """Test that histogram widget can be created and data set"""
    print("Test: Histogram widget creation and data handling")
    
    try:
        # Skip if Qt is not available
        from PySide6.QtWidgets import QApplication
        from vptry_facelandmarkview.histogram_widget import HistogramWidget
    except ImportError as e:
        print(f"  ⚠ Skipping test (Qt not available): {e}")
        return True
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Create histogram widget
    widget = HistogramWidget()
    print("  ✓ Histogram widget created")
    
    # Create test data
    data = np.array([
        # Frame 0 (base)
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        # Frame 1 (moved slightly)
        [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 1.1]],
        # Frame 2 (moved more)
        [[0.5, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.5]],
    ])
    
    # Set data
    widget.set_data(data)
    print("  ✓ Data set successfully")
    
    # Verify internal state
    assert widget.data is not None, "Data should be set"
    assert widget.distances is not None, "Distances should be calculated"
    assert len(widget.distances) == 4, f"Expected 4 distances, got {len(widget.distances)}"
    
    print(f"  ✓ Calculated {len(widget.distances)} distances")
    print(f"    Distance range: [{widget.distances.min():.3f}, {widget.distances.max():.3f}]")
    
    # Verify histogram data
    assert widget.hist_values is not None, "Histogram values should be set"
    assert widget.bin_edges is not None, "Bin edges should be set"
    assert len(widget.hist_values) == 20, f"Expected 20 bins, got {len(widget.hist_values)}"
    
    print(f"  ✓ Histogram created with {len(widget.hist_values)} bins")
    print(f"    Outliers: {widget.outlier_count}")
    
    # Test frame change
    widget.set_current_frame(2)
    print("  ✓ Frame change handled")
    
    assert widget.current_frame == 2, "Current frame should be updated"
    # Distances should be recalculated
    assert widget.distances is not None, "Distances should be recalculated"
    
    expected_distances = np.array([0.5, 0.5, 0.5, 0.5])
    assert np.allclose(widget.distances, expected_distances), \
        f"Expected distances {expected_distances}, got {widget.distances}"
    
    print(f"  ✓ Distances recalculated: {widget.distances}")
    
    # Test alignment mode
    widget.set_align_faces(True)
    print("  ✓ Alignment mode enabled")
    
    widget.set_use_static_points(True)
    print("  ✓ Static points mode enabled")
    
    return True


def test_histogram_with_nan():
    """Test histogram widget with NaN values"""
    print("\nTest: Histogram widget with NaN values")
    
    try:
        from PySide6.QtWidgets import QApplication
        from vptry_facelandmarkview.histogram_widget import HistogramWidget
    except ImportError as e:
        print(f"  ⚠ Skipping test (Qt not available): {e}")
        return True
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    widget = HistogramWidget()
    
    # Create data with NaN values
    data = np.array([
        # Frame 0 (base)
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [np.nan, np.nan, np.nan], [0.0, 0.0, 1.0]],
        # Frame 1 (current)
        [[0.1, 0.0, 0.0], [np.nan, np.nan, np.nan], [0.5, 0.5, 0.0], [0.0, 0.0, 1.1]],
    ])
    
    widget.set_data(data)
    widget.set_current_frame(1)  # Compare frame 0 (base) to frame 1 (current)
    
    # Only landmarks 0 and 3 are valid in both frames
    assert widget.distances is not None, "Distances should be calculated"
    assert len(widget.distances) == 2, f"Expected 2 valid distances, got {len(widget.distances)}"
    
    print("  ✓ NaN values filtered correctly")
    print(f"  ✓ Calculated {len(widget.distances)} valid distances")
    
    return True


def test_histogram_no_data():
    """Test histogram widget with no data"""
    print("\nTest: Histogram widget with no data")
    
    try:
        from PySide6.QtWidgets import QApplication
        from vptry_facelandmarkview.histogram_widget import HistogramWidget
    except ImportError as e:
        print(f"  ⚠ Skipping test (Qt not available): {e}")
        return True
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    widget = HistogramWidget()
    
    # Verify initial state
    assert widget.data is None, "Initial data should be None"
    assert widget.distances is None, "Initial distances should be None"
    assert widget.hist_values is None, "Initial hist_values should be None"
    
    print("  ✓ Widget handles no data gracefully")
    
    return True


def main():
    """Run histogram widget tests"""
    print("=" * 60)
    print("Testing Histogram Widget")
    print("=" * 60)
    print()
    
    try:
        if not test_histogram_widget_creation():
            return 1
        if not test_histogram_with_nan():
            return 1
        if not test_histogram_no_data():
            return 1
        
        print()
        print("=" * 60)
        print("All histogram widget tests passed! ✓")
        print("=" * 60)
        return 0
    
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
