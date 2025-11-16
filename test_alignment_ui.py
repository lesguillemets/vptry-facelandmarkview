#!/usr/bin/env python3
"""
Test alignment UI integration
"""

import sys
import numpy as np
from vptry_facelandmarkview import FaceLandmarkViewer
from PySide6.QtWidgets import QApplication


def test_alignment_ui():
    """Test that the alignment UI works correctly"""
    print("Test: Alignment UI integration")

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Create viewer
    viewer = FaceLandmarkViewer()

    # Load sample data
    data = np.load("sample_landmarks.npy")
    viewer.data = data
    n_frames, n_landmarks, _ = data.shape

    # Simulate UI updates
    viewer.frame_slider.setMaximum(n_frames - 1)
    viewer.frame_slider.setEnabled(True)
    viewer.base_frame_spinbox.setMaximum(n_frames - 1)
    viewer.gl_widget.set_data(data)

    print(f"  ✓ Data loaded: {n_frames} frames, {n_landmarks} landmarks")

    # Test that align_faces checkbox exists
    assert hasattr(viewer, "align_faces_checkbox"), "Align faces checkbox should exist"
    print("  ✓ Align faces checkbox exists")

    # Test that gl_widget has align_faces attribute
    assert hasattr(viewer.gl_widget, "align_faces"), (
        "GL widget should have align_faces attribute"
    )
    assert viewer.gl_widget.align_faces == False, (
        "Alignment should be disabled by default"
    )
    print("  ✓ GL widget has align_faces attribute (default: False)")

    # Test enabling alignment via UI
    from PySide6.QtCore import Qt

    viewer.align_faces_checkbox.setChecked(True)
    viewer.on_align_faces_changed(Qt.CheckState.Checked.value)

    assert viewer.align_faces == True, "Viewer align_faces should be True"
    assert viewer.gl_widget.align_faces == True, "GL widget align_faces should be True"
    print("  ✓ Alignment enabled via UI")

    # Test disabling alignment via UI
    viewer.align_faces_checkbox.setChecked(False)
    viewer.on_align_faces_changed(Qt.CheckState.Unchecked.value)

    assert viewer.align_faces == False, "Viewer align_faces should be False"
    assert viewer.gl_widget.align_faces == False, (
        "GL widget align_faces should be False"
    )
    print("  ✓ Alignment disabled via UI")

    print()
    print("All alignment UI tests passed! ✓")


if __name__ == "__main__":
    test_alignment_ui()
