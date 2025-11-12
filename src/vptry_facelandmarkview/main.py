"""
Main entry point for Face Landmark Viewer application.
"""

import sys
import argparse
import logging
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QSurfaceFormat

from vptry_facelandmarkview.viewer import FaceLandmarkViewer

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main() -> None:
    """Main entry point"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Face Landmark Viewer - 3D visualization of face landmarks from .npy files"
    )
    parser.add_argument(
        "file", type=Path, nargs="?", help="Path to .npy file to load on startup"
    )
    parser.add_argument(
        "--base-frame",
        type=int,
        default=0,
        help="Base frame index to use as reference (default: 0)",
    )

    args = parser.parse_args()

    # Set up OpenGL format
    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)  # Enable multisampling for better quality
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    viewer = FaceLandmarkViewer(
        initial_file=args.file, initial_base_frame=args.base_frame
    )
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
