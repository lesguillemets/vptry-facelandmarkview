"""
Dialog for selecting which landmarks to use for alignment.
"""

import logging
from typing import Optional

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QWidget,
    QGridLayout,
    QCheckBox,
    QLabel,
)
from PySide6.QtCore import Qt

from vptry_facelandmarkview.constants import DEFAULT_ALIGNMENT_LANDMARKS

logger = logging.getLogger(__name__)

# Total number of landmarks in MediaPipe Face Landmarker
TOTAL_LANDMARKS = 478


class LandmarkSelectorDialog(QDialog):
    """Dialog for selecting which landmarks to use for alignment"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Choose Alignment Landmarks")
        self.setModal(True)

        # Initialize selected landmarks to the default set
        self.selected_landmarks: set[int] = set(DEFAULT_ALIGNMENT_LANDMARKS)

        # Store checkboxes for each landmark
        self.checkboxes: list[QCheckBox] = []

        self.init_ui()

        # Set initial state based on selected_landmarks
        self._update_checkboxes_from_selection()

    def init_ui(self) -> None:
        """Initialize the user interface"""
        layout = QVBoxLayout(self)

        # Add instruction label
        instruction = QLabel(
            f"Select landmarks to use for alignment (0-{TOTAL_LANDMARKS - 1}).\n"
            "Selected landmarks will be used to compute the transformation when 'Limit to Static Points' is enabled."
        )
        instruction.setWordWrap(True)
        layout.addWidget(instruction)

        # Button row: All, None, Default
        button_layout = QHBoxLayout()

        all_button = QPushButton("Select All")
        all_button.clicked.connect(self.select_all)
        button_layout.addWidget(all_button)

        none_button = QPushButton("Select None")
        none_button.clicked.connect(self.select_none)
        button_layout.addWidget(none_button)

        default_button = QPushButton("Select Default")
        default_button.clicked.connect(self.select_default)
        button_layout.addWidget(default_button)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Create scrollable area for checkboxes
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(400)
        scroll_area.setMinimumWidth(600)

        # Container widget for the grid
        container = QWidget()
        grid_layout = QGridLayout(container)

        # Create checkboxes in a grid (e.g., 10 columns)
        columns = 10
        for i in range(TOTAL_LANDMARKS):
            checkbox = QCheckBox(str(i))
            checkbox.setChecked(i in self.selected_landmarks)
            checkbox.stateChanged.connect(
                lambda state, idx=i: self._on_checkbox_changed(idx, state)
            )
            self.checkboxes.append(checkbox)

            row = i // columns
            col = i % columns
            grid_layout.addWidget(checkbox, row, col)

        scroll_area.setWidget(container)
        layout.addWidget(scroll_area)

        # OK and Cancel buttons
        button_box_layout = QHBoxLayout()

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        button_box_layout.addWidget(ok_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_box_layout.addWidget(cancel_button)

        button_box_layout.addStretch()
        layout.addLayout(button_box_layout)

    def _on_checkbox_changed(self, landmark_idx: int, state: int) -> None:
        """Handle checkbox state change

        Args:
            landmark_idx: Index of the landmark
            state: Qt.CheckState value
        """
        if state == Qt.CheckState.Checked.value:
            self.selected_landmarks.add(landmark_idx)
        else:
            self.selected_landmarks.discard(landmark_idx)

        logger.debug(
            f"Landmark {landmark_idx} {'added to' if state == Qt.CheckState.Checked.value else 'removed from'} selection. "
            f"Total: {len(self.selected_landmarks)}"
        )

    def _update_checkboxes_from_selection(self) -> None:
        """Update all checkboxes based on current selected_landmarks set"""
        for i, checkbox in enumerate(self.checkboxes):
            # Temporarily block signals to avoid triggering _on_checkbox_changed
            checkbox.blockSignals(True)
            checkbox.setChecked(i in self.selected_landmarks)
            checkbox.blockSignals(False)

    def select_all(self) -> None:
        """Select all landmarks"""
        logger.info("Selecting all landmarks")
        self.selected_landmarks = set(range(TOTAL_LANDMARKS))
        self._update_checkboxes_from_selection()

    def select_none(self) -> None:
        """Deselect all landmarks"""
        logger.info("Deselecting all landmarks")
        self.selected_landmarks = set()
        self._update_checkboxes_from_selection()

    def select_default(self) -> None:
        """Select default landmarks (nose and forehead)"""
        logger.info(
            f"Selecting default landmarks ({len(DEFAULT_ALIGNMENT_LANDMARKS)} landmarks)"
        )
        self.selected_landmarks = set(DEFAULT_ALIGNMENT_LANDMARKS)
        self._update_checkboxes_from_selection()

    def get_selected_landmarks(self) -> list[int]:
        """Get the list of selected landmark indices

        Returns:
            Sorted list of selected landmark indices
        """
        return sorted(list(self.selected_landmarks))
