#!/usr/bin/env python3
"""
Face Landmark Viewer - 3D visualization of face landmarks from .npy files
"""

import sys
import numpy as np
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QFileDialog, QCheckBox, QSpinBox
)
from PySide6.QtCore import Qt
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - Required for 3D projection


class FaceLandmarkViewer(QMainWindow):
    """Main window for the face landmark viewer application"""
    
    def __init__(self):
        super().__init__()
        self.data = None
        self.base_frame = 0
        self.current_frame = 0
        self.show_vectors = False
        
        self.setWindowTitle("Face Landmark Viewer")
        self.setGeometry(100, 100, 1200, 800)
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Control panel
        control_layout = QHBoxLayout()
        
        # Load button
        self.load_button = QPushButton("Load .npy File")
        self.load_button.clicked.connect(self.load_file)
        control_layout.addWidget(self.load_button)
        
        # Base frame selector
        control_layout.addWidget(QLabel("Base Frame:"))
        self.base_frame_spinbox = QSpinBox()
        self.base_frame_spinbox.setMinimum(0)
        self.base_frame_spinbox.setValue(0)
        self.base_frame_spinbox.valueChanged.connect(self.on_base_frame_changed)
        control_layout.addWidget(self.base_frame_spinbox)
        
        # Show vectors checkbox
        self.show_vectors_checkbox = QCheckBox("Show Vectors")
        self.show_vectors_checkbox.stateChanged.connect(self.on_show_vectors_changed)
        control_layout.addWidget(self.show_vectors_checkbox)
        
        control_layout.addStretch()
        
        main_layout.addLayout(control_layout)
        
        # Frame slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Frame:"))
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.on_frame_changed)
        slider_layout.addWidget(self.frame_slider)
        
        self.frame_label = QLabel("0 / 0")
        slider_layout.addWidget(self.frame_label)
        
        main_layout.addLayout(slider_layout)
        
        # Matplotlib canvas for 3D plot
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')
        main_layout.addWidget(self.canvas)
        
        # Info label
        self.info_label = QLabel("Load a .npy file to begin")
        main_layout.addWidget(self.info_label)
    
    def load_file(self):
        """Load a .npy file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open .npy File",
            "",
            "NumPy Files (*.npy);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            self.data = np.load(file_path)
            
            # Validate data shape
            if len(self.data.shape) != 3 or self.data.shape[2] != 3:
                self.info_label.setText(
                    f"Error: Invalid data shape {self.data.shape}. "
                    "Expected (n_frames, n_landmarks, 3)"
                )
                self.data = None
                return
            
            n_frames, n_landmarks, _ = self.data.shape
            
            # Update UI
            self.frame_slider.setMaximum(n_frames - 1)
            self.frame_slider.setEnabled(True)
            self.base_frame_spinbox.setMaximum(n_frames - 1)
            
            self.current_frame = 0
            self.frame_slider.setValue(0)
            
            self.info_label.setText(
                f"Loaded: {Path(file_path).name} - "
                f"{n_frames} frames, {n_landmarks} landmarks"
            )
            
            self.update_plot()
            
        except Exception as e:
            self.info_label.setText(f"Error loading file: {str(e)}")
            self.data = None
    
    def on_frame_changed(self, value):
        """Handle frame slider change"""
        self.current_frame = value
        if self.data is not None:
            self.frame_label.setText(f"{value} / {self.data.shape[0] - 1}")
            self.update_plot()
    
    def on_base_frame_changed(self, value):
        """Handle base frame spinbox change"""
        self.base_frame = value
        if self.data is not None:
            self.update_plot()
    
    def on_show_vectors_changed(self, state):
        """Handle show vectors checkbox change"""
        self.show_vectors = (state == Qt.Checked)
        if self.data is not None:
            self.update_plot()
    
    def update_plot(self):
        """Update the 3D plot"""
        if self.data is None:
            return
        
        self.ax.clear()
        
        # Get current frame and base frame data
        current_landmarks = self.data[self.current_frame]
        base_landmarks = self.data[self.base_frame]
        
        # Plot base frame landmarks (in blue)
        self.ax.scatter(
            base_landmarks[:, 0],
            base_landmarks[:, 1],
            base_landmarks[:, 2],
            c='blue',
            marker='o',
            s=30,
            alpha=0.6,
            label=f'Base Frame {self.base_frame}'
        )
        
        # Plot current frame landmarks (in red)
        self.ax.scatter(
            current_landmarks[:, 0],
            current_landmarks[:, 1],
            current_landmarks[:, 2],
            c='red',
            marker='^',
            s=50,
            alpha=0.8,
            label=f'Current Frame {self.current_frame}'
        )
        
        # Plot vectors if enabled
        if self.show_vectors:
            for i in range(len(base_landmarks)):
                self.ax.plot(
                    [base_landmarks[i, 0], current_landmarks[i, 0]],
                    [base_landmarks[i, 1], current_landmarks[i, 1]],
                    [base_landmarks[i, 2], current_landmarks[i, 2]],
                    'g-',
                    alpha=0.3,
                    linewidth=1
                )
        
        # Set labels and title
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f'Face Landmarks - Frame {self.current_frame}')
        self.ax.legend()
        
        # Auto-scale to fit all data
        all_points = np.vstack([base_landmarks, current_landmarks])
        x_range = [all_points[:, 0].min(), all_points[:, 0].max()]
        y_range = [all_points[:, 1].min(), all_points[:, 1].max()]
        z_range = [all_points[:, 2].min(), all_points[:, 2].max()]
        
        # Add some padding
        padding = 0.1
        x_pad = (x_range[1] - x_range[0]) * padding
        y_pad = (y_range[1] - y_range[0]) * padding
        z_pad = (z_range[1] - z_range[0]) * padding
        
        self.ax.set_xlim(x_range[0] - x_pad, x_range[1] + x_pad)
        self.ax.set_ylim(y_range[0] - y_pad, y_range[1] + y_pad)
        self.ax.set_zlim(z_range[0] - z_pad, z_range[1] + z_pad)
        
        self.canvas.draw()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    viewer = FaceLandmarkViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
