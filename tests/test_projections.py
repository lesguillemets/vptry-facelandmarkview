#!/usr/bin/env python3
"""
Test script to validate the projection widgets work correctly
"""

import sys
import numpy as np
from PySide6.QtWidgets import QApplication
from vptry_facelandmarkview import FaceLandmarkViewer
from vptry_facelandmarkview.projection_widget import ProjectionWidget
from vptry_facelandmarkview.constants import ProjectionType


def test_projection_widgets_exist():
    """Test that projection widgets are created"""
    print("Test: Projection widgets exist")

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    viewer = FaceLandmarkViewer()

    # Check that projection widgets exist
    assert hasattr(viewer, 'xz_widget'), "xz_widget should exist"
    assert hasattr(viewer, 'yz_widget'), "yz_widget should exist"
    print("  ✓ X-Z and Y-Z projection widgets exist")

    # Check that they are ProjectionWidget instances
    assert isinstance(viewer.xz_widget, ProjectionWidget), "xz_widget should be ProjectionWidget"
    assert isinstance(viewer.yz_widget, ProjectionWidget), "yz_widget should be ProjectionWidget"
    print("  ✓ Widgets are ProjectionWidget instances")

    # Check projection types
    assert viewer.xz_widget.projection_type == ProjectionType.XZ, "xz_widget should have projection_type XZ"
    assert viewer.yz_widget.projection_type == ProjectionType.YZ, "yz_widget should have projection_type YZ"
    print("  ✓ Widgets have correct projection types")

    return True


def test_projection_sync():
    """Test that projection widgets sync with main widget"""
    print("Test: Projection widgets sync with main widget")

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    viewer = FaceLandmarkViewer()

    # Create sample data
    sample_data = np.random.randn(10, 68, 3)
    
    # Set data on viewer (which should update all widgets)
    viewer.data = sample_data
    viewer.gl_widget.set_data(sample_data)
    viewer.xz_widget.set_data(sample_data)
    viewer.yz_widget.set_data(sample_data)
    viewer._update_projection_center_scale()

    # Check that all widgets have data
    assert viewer.gl_widget.data is not None, "Main widget should have data"
    assert viewer.xz_widget.data is not None, "X-Z widget should have data"
    assert viewer.yz_widget.data is not None, "Y-Z widget should have data"
    print("  ✓ All widgets have data")

    # Test frame sync
    viewer.on_frame_changed(5)
    assert viewer.gl_widget.current_frame == 5, "Main widget frame should be 5"
    assert viewer.xz_widget.current_frame == 5, "X-Z widget frame should be 5"
    assert viewer.yz_widget.current_frame == 5, "Y-Z widget frame should be 5"
    print("  ✓ Frame changes sync across all widgets")

    # Test base frame sync
    viewer.on_base_frame_changed(3)
    assert viewer.gl_widget.base_frame == 3, "Main widget base frame should be 3"
    assert viewer.xz_widget.base_frame == 3, "X-Z widget base frame should be 3"
    assert viewer.yz_widget.base_frame == 3, "Y-Z widget base frame should be 3"
    print("  ✓ Base frame changes sync across all widgets")

    # Test show vectors sync
    viewer.on_show_vectors_changed(2)  # 2 = Checked
    assert viewer.gl_widget.show_vectors is True, "Main widget should show vectors"
    assert viewer.xz_widget.show_vectors is True, "X-Z widget should show vectors"
    assert viewer.yz_widget.show_vectors is True, "Y-Z widget should show vectors"
    print("  ✓ Show vectors setting syncs across all widgets")

    # Test align faces sync
    viewer.on_align_faces_changed(2)  # 2 = Checked
    assert viewer.gl_widget.align_faces is True, "Main widget should align faces"
    assert viewer.xz_widget.align_faces is True, "X-Z widget should align faces"
    assert viewer.yz_widget.align_faces is True, "Y-Z widget should align faces"
    print("  ✓ Align faces setting syncs across all widgets")

    # Test static points sync
    viewer.on_use_static_points_changed(2)  # 2 = Checked
    assert viewer.gl_widget.use_static_points is True, "Main widget should use static points"
    assert viewer.xz_widget.use_static_points is True, "X-Z widget should use static points"
    assert viewer.yz_widget.use_static_points is True, "Y-Z widget should use static points"
    print("  ✓ Static points setting syncs across all widgets")

    # Check that center and scale are shared
    assert viewer.xz_widget.center is not None, "X-Z widget should have center"
    assert viewer.xz_widget.scale is not None, "X-Z widget should have scale"
    assert viewer.yz_widget.center is not None, "Y-Z widget should have center"
    assert viewer.yz_widget.scale is not None, "Y-Z widget should have scale"
    print("  ✓ Center and scale are shared with projection widgets")

    return True


def test_widget_dimensions():
    """Test that projection widgets have correct fixed dimensions"""
    print("Test: Projection widget dimensions")

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    viewer = FaceLandmarkViewer()

    # Check X-Z widget height
    assert viewer.xz_widget.maximumHeight() == 100, "X-Z widget should have max height of 100px"
    print("  ✓ X-Z widget has fixed height (max: 100px)")

    # Check Y-Z widget width
    assert viewer.yz_widget.maximumWidth() == 100, "Y-Z widget should have max width of 100px"
    print("  ✓ Y-Z widget has fixed width (max: 100px)")

    # Check placeholder dimensions
    assert viewer.top_right_placeholder.maximumHeight() == 100, "Placeholder should have max height of 100px"
    assert viewer.top_right_placeholder.maximumWidth() == 100, "Placeholder should have max width of 100px"
    print("  ✓ Top-right placeholder has correct dimensions (100x100px)")

    return True


def main():
    """Run projection widget tests"""
    print("=" * 60)
    print("Testing Projection Widgets")
    print("=" * 60)
    print()

    try:
        if not test_projection_widgets_exist():
            return 1
        
        print()
        
        if not test_projection_sync():
            return 1
        
        print()
        
        if not test_widget_dimensions():
            return 1

        print()
        print("=" * 60)
        print("Projection widget tests passed! ✓")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
