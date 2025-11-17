#!/usr/bin/env python3
"""
Visual test for the histogram widget - loads sample data and takes a screenshot.
"""

import sys
import time
from pathlib import Path

from PySide6.QtWidgets import QApplication

from vptry_facelandmarkview import FaceLandmarkViewer


def take_screenshot(viewer, filename="histogram_test.png"):
    """Take a screenshot of the viewer window"""
    # Wait a bit for rendering to complete
    QApplication.processEvents()
    time.sleep(0.5)

    # Grab the window
    pixmap = viewer.grab()

    # Save the screenshot
    output_path = Path(filename)
    pixmap.save(str(output_path))
    print(f"Screenshot saved to: {output_path.absolute()}")

    return output_path


def main():
    """Run visual test"""
    print("=" * 60)
    print("Visual Test: Histogram Widget")
    print("=" * 60)

    QApplication(sys.argv)

    # Create viewer
    sample_file = Path("sample_landmarks.npy")
    if not sample_file.exists():
        print(f"Error: {sample_file} not found. Run generate_sample_data.py first.")
        return 1

    print(f"Loading sample data from: {sample_file}")
    viewer = FaceLandmarkViewer(initial_file=sample_file, initial_base_frame=0)
    viewer.show()

    # Wait for window to be fully rendered
    QApplication.processEvents()
    time.sleep(0.2)

    # Set to a frame with some movement to see the histogram
    if viewer.data is not None and viewer.data.shape[0] > 10:
        viewer.frame_slider.setValue(10)
        QApplication.processEvents()
        time.sleep(0.2)

    # Take screenshot with default view
    take_screenshot(viewer, "histogram_test_default.png")
    print("✓ Screenshot 1: Default view")

    # Enable alignment mode
    viewer.align_faces_checkbox.setChecked(True)
    QApplication.processEvents()
    time.sleep(0.2)

    take_screenshot(viewer, "histogram_test_aligned.png")
    print("✓ Screenshot 2: With alignment enabled")

    # Move to a different frame with more movement
    if viewer.data is not None and viewer.data.shape[0] > 25:
        viewer.frame_slider.setValue(25)
        QApplication.processEvents()
        time.sleep(0.2)

        take_screenshot(viewer, "histogram_test_frame25.png")
        print("✓ Screenshot 3: Frame 25 with alignment")

    print()
    print("=" * 60)
    print("Visual test completed!")
    print("Screenshots saved. Please review them manually.")
    print("=" * 60)

    # Close the application
    viewer.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
