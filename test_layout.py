#!/usr/bin/env python3
"""
Test script to validate the GUI layout improvements
"""

import sys
from PySide6.QtWidgets import QApplication, QVBoxLayout
from vptry_facelandmarkview import FaceLandmarkViewer


def test_layout_stretch_factors():
    """Test that layout components have correct stretch factors"""
    print("Test: Layout stretch factors")

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    viewer = FaceLandmarkViewer()

    # Get the main layout
    central_widget = viewer.centralWidget()
    main_layout = central_widget.layout()

    assert isinstance(main_layout, QVBoxLayout), "Main layout should be a QVBoxLayout"
    print("  ✓ Main layout is QVBoxLayout")

    # Check the number of items in the layout
    # Should be: control_layout(0), slider_layout(1), gl_widget(2), info_label(3)
    item_count = main_layout.count()
    assert item_count == 4, f"Expected 4 items in layout, got {item_count}"
    print("  ✓ Layout has 4 items")

    # Get stretch factors
    # Item 0: control_layout
    control_stretch = main_layout.stretch(0)
    print(f"  ✓ control_layout stretch factor: {control_stretch}")
    assert control_stretch == 0, (
        f"control_layout should have stretch 0, got {control_stretch}"
    )

    # Item 1: slider_layout
    slider_stretch = main_layout.stretch(1)
    print(f"  ✓ slider_layout stretch factor: {slider_stretch}")
    assert slider_stretch == 0, (
        f"slider_layout should have stretch 0, got {slider_stretch}"
    )

    # Item 2: gl_widget
    gl_stretch = main_layout.stretch(2)
    print(f"  ✓ gl_widget stretch factor: {gl_stretch}")
    assert gl_stretch == 1, f"gl_widget should have stretch 1, got {gl_stretch}"

    # Item 3: info_label
    info_stretch = main_layout.stretch(3)
    print(f"  ✓ info_label stretch factor: {info_stretch}")
    assert info_stretch == 0, f"info_label should have stretch 0, got {info_stretch}"

    print("  ✓ All stretch factors are correct!")
    print("  ✓ gl_widget will occupy maximum available vertical space (stretch=1)")
    print(
        "  ✓ control_layout, slider_layout, info_label will use minimal space (stretch=0)"
    )

    return True


def main():
    """Run layout tests"""
    print("=" * 60)
    print("Testing GUI Layout Improvements")
    print("=" * 60)
    print()

    try:
        if not test_layout_stretch_factors():
            return 1

        print()
        print("=" * 60)
        print("Layout tests passed! ✓")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
