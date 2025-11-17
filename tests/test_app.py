#!/usr/bin/env python3
"""
Test script to validate the application loads and functions correctly
"""

import sys
import numpy as np


# Test data loading
def test_data_loading():
    """Test that sample data can be loaded"""
    data = np.load("sample_landmarks.npy")
    print("✓ Data loaded successfully")
    print(f"  Shape: {data.shape}")

    assert len(data.shape) == 3, "Data should be 3D"
    assert data.shape[2] == 3, "Last dimension should be 3 (x, y, z)"
    print("✓ Data shape is valid")

    return data


def test_imports():
    """Test that all required imports work"""
    try:
        from PySide6.QtWidgets import QApplication

        print("✓ PySide6.QtWidgets imported")

        import vptry_facelandmarkview

        print("✓ vptry_facelandmarkview module imported")

        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_app_instantiation():
    """Test that the application can be instantiated"""
    try:
        from PySide6.QtWidgets import QApplication
        from vptry_facelandmarkview import FaceLandmarkViewer

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        viewer = FaceLandmarkViewer()
        print("✓ Application instantiated successfully")
        print(f"  Window title: {viewer.windowTitle()}")

        # Test initial state
        assert viewer.data is None, "Initial data should be None"
        assert viewer.base_frame == 0, "Initial base frame should be 0"
        assert viewer.current_frame == 0, "Initial current frame should be 0"
        assert not viewer.show_vectors, "Initial show_vectors should be False"
        print("✓ Initial state is correct")

        return True
    except Exception as e:
        print(f"✗ App instantiation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 50)
    print("Testing Face Landmark Viewer")
    print("=" * 50)
    print()

    print("Test 1: Module imports")
    if not test_imports():
        return 1
    print()

    print("Test 2: Sample data loading")
    try:
        test_data_loading()
    except Exception as e:
        print(f"✗ Failed: {e}")
        return 1
    print()

    print("Test 3: Application instantiation")
    if not test_app_instantiation():
        return 1
    print()

    print("=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
