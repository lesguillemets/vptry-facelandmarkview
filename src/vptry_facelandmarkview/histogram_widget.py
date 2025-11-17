"""
Widget for displaying a histogram of landmark distances from base frame to current frame.
"""

import logging
from typing import Optional

import numpy as np
import numpy.typing as npt
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QColor, QPen, QFont

from vptry_facelandmarkview.constants import DEFAULT_ALIGNMENT_LANDMARKS
from vptry_facelandmarkview.utils import filter_nan_landmarks, align_landmarks_to_base

logger = logging.getLogger(__name__)

# Histogram configuration
HISTOGRAM_BINS = 20
OUTLIER_PERCENTILE = 95  # 5% of points can be outside
HISTOGRAM_BG_COLOR = QColor(255, 255, 255)  # White background
BAR_COLOR = QColor(100, 100, 100)  # Gray bars
OUTLIER_BAR_COLOR = QColor(200, 50, 50)  # Red for outliers
AXIS_COLOR = QColor(0, 0, 0)  # Black axes
TEXT_COLOR = QColor(0, 0, 0)  # Black text
GRID_COLOR = QColor(200, 200, 200)  # Light gray grid


class HistogramWidget(QWidget):
    """Widget for displaying a histogram of distances between base and current frame landmarks"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.data: Optional[npt.NDArray[np.float64]] = None
        self.base_frame: int = 0
        self.current_frame: int = 0
        self.align_faces: bool = False
        self.use_static_points: bool = False

        # Cached histogram data
        self.distances: Optional[npt.NDArray[np.float64]] = None
        self.hist_values: Optional[npt.NDArray[np.int_]] = None
        self.bin_edges: Optional[npt.NDArray[np.float64]] = None
        self.outlier_count: int = 0
        self.max_distance: float = 0.0

        self.setMinimumSize(100, 100)

    def set_data(self, data: npt.NDArray[np.float64]) -> None:
        """Set the landmark data"""
        logger.debug(f"Histogram: Setting data with shape: {data.shape}")
        self.data = data
        self._update_histogram()
        self.update()

    def set_base_frame(self, frame: int) -> None:
        """Set the base frame"""
        logger.debug(f"Histogram: Setting base frame to: {frame}")
        self.base_frame = frame
        self._update_histogram()
        self.update()

    def set_current_frame(self, frame: int) -> None:
        """Set the current frame"""
        logger.debug(f"Histogram: Setting current frame to: {frame}")
        self.current_frame = frame
        self._update_histogram()
        self.update()

    def set_show_vectors(self, show: bool) -> None:
        """Set whether to show vectors (not used for histogram, but part of protocol)"""
        pass

    def set_align_faces(self, align: bool) -> None:
        """Set whether to align faces to base frame"""
        logger.debug(f"Histogram: Setting align_faces to: {align}")
        self.align_faces = align
        self._update_histogram()
        self.update()

    def set_use_static_points(self, use_static: bool) -> None:
        """Set whether to use only static points for alignment"""
        logger.debug(f"Histogram: Setting use_static_points to: {use_static}")
        self.use_static_points = use_static
        self._update_histogram()
        self.update()

    def _calculate_distances(self) -> Optional[npt.NDArray[np.float64]]:
        """Calculate Euclidean distances between base and current frame landmarks

        Returns:
            Array of distances for valid landmarks, or None if no valid data
        """
        if self.data is None:
            return None

        base_landmarks = self.data[self.base_frame]
        current_landmarks = self.data[self.current_frame]

        # Filter out NaN values
        base_landmarks_valid, base_valid_mask = filter_nan_landmarks(base_landmarks)
        current_landmarks_valid, current_valid_mask = filter_nan_landmarks(
            current_landmarks
        )

        # Get landmarks that are valid in both frames
        both_valid_mask = base_valid_mask & current_valid_mask
        base_landmarks_both = base_landmarks[both_valid_mask]
        current_landmarks_both = current_landmarks[both_valid_mask]

        if len(base_landmarks_both) == 0:
            logger.warning("No valid landmarks to calculate distances")
            return None

        # Apply alignment if enabled
        if self.align_faces:
            alignment_indices = None
            if self.use_static_points:
                alignment_indices = DEFAULT_ALIGNMENT_LANDMARKS

            current_landmarks_both = align_landmarks_to_base(
                current_landmarks_both,
                base_landmarks_both,
                alignment_indices=alignment_indices,
            )

        # Calculate Euclidean distances
        distances = np.linalg.norm(current_landmarks_both - base_landmarks_both, axis=1)
        logger.debug(
            f"Calculated {len(distances)} distances, range: [{distances.min():.4f}, {distances.max():.4f}]"
        )

        return distances

    def _update_histogram(self) -> None:
        """Update histogram data based on current state"""
        distances = self._calculate_distances()

        if distances is None or len(distances) == 0:
            self.distances = None
            self.hist_values = None
            self.bin_edges = None
            self.outlier_count = 0
            self.max_distance = 0.0
            return

        self.distances = distances
        self.max_distance = distances.max()

        # Calculate the 95th percentile for the histogram range
        percentile_95 = np.percentile(distances, OUTLIER_PERCENTILE)

        # Count outliers (values above 95th percentile)
        outlier_mask = distances > percentile_95
        self.outlier_count = outlier_mask.sum()

        # Create histogram for non-outliers
        non_outlier_distances = distances[~outlier_mask]

        if len(non_outlier_distances) > 0:
            # Create histogram with bins from 0 to 95th percentile
            self.hist_values, self.bin_edges = np.histogram(
                non_outlier_distances, bins=HISTOGRAM_BINS, range=(0, percentile_95)
            )
            logger.debug(
                f"Histogram created: {len(non_outlier_distances)} values, "
                f"{self.outlier_count} outliers, "
                f"range: [0, {percentile_95:.4f}]"
            )
        else:
            # All values are outliers (rare case)
            self.hist_values = np.zeros(HISTOGRAM_BINS, dtype=np.int_)
            self.bin_edges = np.linspace(0, percentile_95, HISTOGRAM_BINS + 1)
            logger.debug("All distances are outliers")

    def paintEvent(self, event) -> None:
        """Paint the histogram"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Fill background
        painter.fillRect(self.rect(), HISTOGRAM_BG_COLOR)

        if self.hist_values is None or self.bin_edges is None:
            # No data to display
            painter.setPen(TEXT_COLOR)
            painter.drawText(self.rect(), Qt.AlignCenter, "No data")
            return

        # Define margins
        margin_left = 30  # Space for y-axis tick label (a few characters)
        margin_right = 30  # Extra space for outlier bar
        margin_top = 10
        margin_bottom = 40  # Space for x-axis labels and mean/variance text

        width = self.width() - margin_left - margin_right
        height = self.height() - margin_top - margin_bottom

        if width <= 0 or height <= 0:
            return

        # Calculate bar width
        n_bins = len(self.hist_values)
        bar_width = width / (n_bins + 1)  # +1 for outlier bar

        # Find max count for scaling (include outliers in scaling calculation)
        max_count = (
            max(self.hist_values.max(), self.outlier_count)
            if len(self.hist_values) > 0
            else 1
        )
        if max_count == 0:
            max_count = 1

        # Draw histogram bars
        for i, count in enumerate(self.hist_values):
            if count > 0:
                bar_height = (count / max_count) * height
                x = margin_left + i * bar_width
                y = margin_top + height - bar_height

                painter.fillRect(
                    int(x), int(y), int(bar_width - 1), int(bar_height), BAR_COLOR
                )

        # Draw outlier bar (red) if there are any outliers
        if self.outlier_count > 0:
            bar_height = (self.outlier_count / max_count) * height
            x = margin_left + n_bins * bar_width
            y = margin_top + height - bar_height

            painter.fillRect(
                int(x), int(y), int(bar_width - 1), int(bar_height), OUTLIER_BAR_COLOR
            )

        # Draw axes
        painter.setPen(QPen(AXIS_COLOR, 2))
        # X-axis
        painter.drawLine(
            margin_left, margin_top + height, margin_left + width, margin_top + height
        )
        # Y-axis
        painter.drawLine(margin_left, margin_top, margin_left, margin_top + height)

        # Draw y-axis tick at the top showing max count
        painter.drawLine(margin_left - 3, margin_top, margin_left + 3, margin_top)

        # Draw x-axis labels
        painter.setPen(TEXT_COLOR)
        font = QFont()
        font.setPointSize(7)
        painter.setFont(font)

        # Label for y-axis max (at the top)
        painter.drawText(margin_left - 25, margin_top + 5, f"{max_count}")

        # Label at start (0)
        # Note: Multiply by 100 to convert from decimal to more readable scale
        painter.drawText(margin_left - 5, margin_top + height + 15, "0")

        # Label at end (95th percentile value)
        # Note: Multiply by 100 to convert from decimal to more readable scale
        if self.bin_edges is not None and len(self.bin_edges) > 0:
            max_label = f"{self.bin_edges[-1] * 100:.2f}"
            painter.drawText(
                margin_left + width - 20, margin_top + height + 15, max_label
            )

            # Label for outliers if present
            if self.outlier_count > 0:
                painter.setPen(OUTLIER_BAR_COLOR)
                painter.drawText(
                    margin_left + width + 5,
                    margin_top + height + 15,
                    f">{OUTLIER_PERCENTILE}%",
                )

        # Draw mean and variance below the histogram
        # Note: Display values before ×100 scaling
        if self.distances is not None and len(self.distances) > 0:
            mean_dist = (100 * self.distances).mean()
            var_dist = (100 * self.distances).var()

            painter.setPen(TEXT_COLOR)
            stats_text = f"Mean: {mean_dist:.3f}  Var: {var_dist:.3f}"
            painter.drawText(margin_left, margin_top + height + 30, stats_text)
