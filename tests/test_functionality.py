#!/usr/bin/env python3
"""
Comprehensive functionality test for Face Landmark Viewer
"""

import sys
import numpy as np
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from vptry_facelandmarkview import FaceLandmarkViewer


def test_load_data_programmatically():
    """Test loading data programmatically"""
    print("Test: Load data programmatically")

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    viewer = FaceLandmarkViewer()

    # Load data
    data = np.load("sample_landmarks.npy")
    viewer.data = data
    n_frames, n_landmarks, _ = data.shape

    # Simulate UI updates that would happen in load_file
    viewer.frame_slider.setMaximum(n_frames - 1)
    viewer.frame_slider.setEnabled(True)
    viewer.base_frame_spinbox.setMaximum(n_frames - 1)
    viewer.current_frame = 0
    viewer.frame_slider.setValue(0)

    print(f"  ✓ Data loaded: {n_frames} frames, {n_landmarks} landmarks")

    # Test OpenGL widget update
    viewer.gl_widget.set_data(data)
    print("  ✓ OpenGL widget updated successfully")

    return viewer


def test_frame_navigation(viewer):
    """Test frame navigation"""
    print("\nTest: Frame navigation")

    n_frames = viewer.data.shape[0]

    # Test moving to middle frame
    mid_frame = n_frames // 2
    viewer.frame_slider.setValue(mid_frame)
    assert viewer.current_frame == mid_frame, f"Current frame should be {mid_frame}"
    print(f"  ✓ Moved to frame {mid_frame}")

    # Test moving to last frame
    viewer.frame_slider.setValue(n_frames - 1)
    assert viewer.current_frame == n_frames - 1, (
        f"Current frame should be {n_frames - 1}"
    )
    print(f"  ✓ Moved to frame {n_frames - 1}")

    # Test moving back to first frame
    viewer.frame_slider.setValue(0)
    assert viewer.current_frame == 0, "Current frame should be 0"
    print("  ✓ Moved back to frame 0")


def test_base_frame_change(viewer):
    """Test base frame changes"""
    print("\nTest: Base frame changes")

    # Test changing base frame
    viewer.base_frame_spinbox.setValue(10)
    assert viewer.base_frame == 10, "Base frame should be 10"
    print("  ✓ Base frame changed to 10")

    # Test OpenGL widget updates with different base frame
    assert viewer.gl_widget.base_frame == 10, "OpenGL widget base frame should be 10"
    print("  ✓ OpenGL widget updated with new base frame")

    # Reset base frame
    viewer.base_frame_spinbox.setValue(0)
    assert viewer.base_frame == 0, "Base frame should be 0"
    print("  ✓ Base frame reset to 0")


def test_vector_display(viewer):
    """Test vector display toggle"""
    print("\nTest: Vector display")

    # Test enabling vectors - manually call the handler with integer value
    viewer.show_vectors_checkbox.setChecked(True)
    viewer.on_show_vectors_changed(Qt.CheckState.Checked.value)
    assert viewer.show_vectors, "Vectors should be enabled"
    assert viewer.gl_widget.show_vectors, "OpenGL widget vectors should be enabled"
    print("  ✓ Vectors enabled")

    print("  ✓ OpenGL widget updated with vectors")

    # Test disabling vectors - manually call the handler with integer value
    viewer.show_vectors_checkbox.setChecked(False)
    viewer.on_show_vectors_changed(Qt.CheckState.Unchecked.value)
    assert not viewer.show_vectors, "Vectors should be disabled"
    assert not viewer.gl_widget.show_vectors, "OpenGL widget vectors should be disabled"
    print("  ✓ Vectors disabled")

    print("  ✓ OpenGL widget updated without vectors")


def test_data_access():
    """Test that data access patterns work correctly"""
    print("\nTest: Data access patterns")

    data = np.load("sample_landmarks.npy")
    n_frames, n_landmarks, _ = data.shape

    # Test accessing coordinates as specified in requirements
    fr = 0
    p = 0
    x = data[fr][p][0]
    y = data[fr][p][1]
    z = data[fr][p][2]

    print(f"  ✓ dat[{fr}][{p}][0] (x) = {x:.3f}")
    print(f"  ✓ dat[{fr}][{p}][1] (y) = {y:.3f}")
    print(f"  ✓ dat[{fr}][{p}][2] (z) = {z:.3f}")

    # Test base frame access
    base_frame = 0
    base_landmarks = data[base_frame][:][:]
    assert base_landmarks.shape == (n_landmarks, 3), (
        "Base landmarks shape should be (n_landmarks, 3)"
    )
    print(f"  ✓ Base frame data accessed correctly: shape {base_landmarks.shape}")

    # Test vector calculation
    current_frame = 10
    vectors = data[current_frame] - data[base_frame]
    assert vectors.shape == (n_landmarks, 3), "Vectors shape should be (n_landmarks, 3)"
    print(f"  ✓ Vectors calculated correctly: shape {vectors.shape}")


def main():
    """Run all functionality tests"""
    print("=" * 60)
    print("Face Landmark Viewer - Comprehensive Functionality Tests")
    print("=" * 60)
    print()

    try:
        # Test data access patterns
        test_data_access()

        # Test loading and visualization
        viewer = test_load_data_programmatically()

        # Test frame navigation
        test_frame_navigation(viewer)

        # Test base frame changes
        test_base_frame_change(viewer)

        # Test vector display
        test_vector_display(viewer)

        print()
        print("=" * 60)
        print("All functionality tests passed! ✓")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
