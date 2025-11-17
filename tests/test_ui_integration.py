#!/usr/bin/env python3
"""
Test UI integration with alignment methods
"""

import sys
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vptry_facelandmarkview.viewer import FaceLandmarkViewer
from vptry_facelandmarkview.alignments import get_available_alignment_methods


def test_ui_integration():
    """Test that the UI integrates properly with alignment methods"""
    print("Test: UI integration with alignment methods")

    app = QApplication(sys.argv)

    # Create sample data file if it doesn't exist
    sample_file = Path("sample_landmarks.npy")
    if not sample_file.exists():
        print("  Generating sample data...")
        # Create simple sample data
        data = np.random.randn(10, 20, 3) * 10
        np.save(sample_file, data)

    # Create viewer with sample data
    viewer = FaceLandmarkViewer(initial_file=sample_file)

    # Test that alignment method dropdown exists and is populated
    assert hasattr(viewer, "alignment_method_combo")
    print("  ✓ Alignment method dropdown exists")

    # Check dropdown items
    methods = get_available_alignment_methods()
    combo_items = [
        viewer.alignment_method_combo.itemText(i)
        for i in range(viewer.alignment_method_combo.count())
    ]
    print(f"  Dropdown items: {combo_items}")
    assert set(combo_items) == set(methods)
    print("  ✓ Dropdown contains all alignment methods")

    # Test changing alignment method
    for method in methods:
        index = combo_items.index(method)
        viewer.alignment_method_combo.setCurrentIndex(index)
        assert viewer.alignment_method == method
        print(f"  ✓ Changed to method: {method}")

    # Test that widgets have the set_alignment_method method
    for widget_name in ["gl_widget", "xz_widget", "yz_widget", "histogram_widget"]:
        widget = getattr(viewer, widget_name)
        assert hasattr(widget, "set_alignment_method")
        print(f"  ✓ {widget_name} has set_alignment_method")

    # Close the viewer
    viewer.close()

    print("\n✓ All UI integration tests passed!")


if __name__ == "__main__":
    test_ui_integration()
