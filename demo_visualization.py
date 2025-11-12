#!/usr/bin/env python3
"""
Demonstration script to generate visualization examples
"""

from pathlib import Path
import numpy as np
import numpy.typing as npt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_demo_visualization() -> None:
    """Create a sample visualization to demonstrate the application's output"""
    
    # Load sample data
    data_path = Path('sample_landmarks.npy')
    data: npt.NDArray[np.float64] = np.load(data_path)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Base frame only
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    base_frame = 0
    base_landmarks = data[base_frame]
    ax1.scatter(base_landmarks[:, 0], base_landmarks[:, 1], base_landmarks[:, 2],
                c='blue', marker='o', s=30, alpha=0.6)
    ax1.set_title(f'Base Frame {base_frame}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Plot 2: Current frame only
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    current_frame = 25
    current_landmarks = data[current_frame]
    ax2.scatter(current_landmarks[:, 0], current_landmarks[:, 1], current_landmarks[:, 2],
                c='red', marker='^', s=50, alpha=0.8)
    ax2.set_title(f'Current Frame {current_frame}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Plot 3: Both frames together
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    ax3.scatter(base_landmarks[:, 0], base_landmarks[:, 1], base_landmarks[:, 2],
                c='blue', marker='o', s=30, alpha=0.6, label=f'Base Frame {base_frame}')
    ax3.scatter(current_landmarks[:, 0], current_landmarks[:, 1], current_landmarks[:, 2],
                c='red', marker='^', s=50, alpha=0.8, label=f'Current Frame {current_frame}')
    ax3.set_title('Both Frames')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend()
    
    # Plot 4: With vectors
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax4.scatter(base_landmarks[:, 0], base_landmarks[:, 1], base_landmarks[:, 2],
                c='blue', marker='o', s=30, alpha=0.6, label=f'Base Frame {base_frame}')
    ax4.scatter(current_landmarks[:, 0], current_landmarks[:, 1], current_landmarks[:, 2],
                c='red', marker='^', s=50, alpha=0.8, label=f'Current Frame {current_frame}')
    
    # Draw vectors
    for i in range(len(base_landmarks)):
        ax4.plot([base_landmarks[i, 0], current_landmarks[i, 0]],
                [base_landmarks[i, 1], current_landmarks[i, 1]],
                [base_landmarks[i, 2], current_landmarks[i, 2]],
                'g-', alpha=0.3, linewidth=1)
    
    ax4.set_title('With Vectors')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.legend()
    
    # Plot 5: Different frame
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    current_frame2 = 45
    current_landmarks2 = data[current_frame2]
    ax5.scatter(base_landmarks[:, 0], base_landmarks[:, 1], base_landmarks[:, 2],
                c='blue', marker='o', s=30, alpha=0.6, label=f'Base Frame {base_frame}')
    ax5.scatter(current_landmarks2[:, 0], current_landmarks2[:, 1], current_landmarks2[:, 2],
                c='red', marker='^', s=50, alpha=0.8, label=f'Current Frame {current_frame2}')
    ax5.set_title(f'Frame {current_frame2}')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('Z')
    ax5.legend()
    
    # Plot 6: Animation sequence preview
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    frames_to_show = [0, 10, 20, 30, 40]
    colors = ['blue', 'cyan', 'green', 'yellow', 'red']
    for frame, color in zip(frames_to_show, colors):
        landmarks = data[frame]
        ax6.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2],
                   c=color, marker='o', s=20, alpha=0.4, label=f'Frame {frame}')
    ax6.set_title('Animation Sequence Preview')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.set_zlabel('Z')
    ax6.legend()
    
    plt.tight_layout()
    demo_path = Path('demo_visualization.png')
    plt.savefig(demo_path, dpi=150, bbox_inches='tight')
    print(f"✓ Demo visualization saved to {demo_path}")
    
    # Create a summary figure
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})
    
    # Summary plot without vectors
    axes[0].scatter(base_landmarks[:, 0], base_landmarks[:, 1], base_landmarks[:, 2],
                    c='blue', marker='o', s=30, alpha=0.6, label='Base Frame')
    axes[0].scatter(current_landmarks[:, 0], current_landmarks[:, 1], current_landmarks[:, 2],
                    c='red', marker='^', s=50, alpha=0.8, label='Current Frame')
    axes[0].set_title('Face Landmarks Visualization')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_zlabel('Z')
    axes[0].legend()
    
    # Summary plot with vectors
    axes[1].scatter(base_landmarks[:, 0], base_landmarks[:, 1], base_landmarks[:, 2],
                    c='blue', marker='o', s=30, alpha=0.6, label='Base Frame')
    axes[1].scatter(current_landmarks[:, 0], current_landmarks[:, 1], current_landmarks[:, 2],
                    c='red', marker='^', s=50, alpha=0.8, label='Current Frame')
    for i in range(len(base_landmarks)):
        axes[1].plot([base_landmarks[i, 0], current_landmarks[i, 0]],
                     [base_landmarks[i, 1], current_landmarks[i, 1]],
                     [base_landmarks[i, 2], current_landmarks[i, 2]],
                     'g-', alpha=0.3, linewidth=1)
    axes[1].set_title('With Vectors Enabled')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_zlabel('Z')
    axes[1].legend()
    
    plt.tight_layout()
    summary_path = Path('summary_visualization.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"✓ Summary visualization saved to {summary_path}")


if __name__ == '__main__':
    create_demo_visualization()
