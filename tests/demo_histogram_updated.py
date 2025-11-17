#!/usr/bin/env python3
"""
Demonstrate the updated histogram visualization with y-axis tick and scaled x-axis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Set random seed for reproducibility
np.random.seed(42)

def create_updated_histogram_demo(output_file="histogram_demo_updated.png"):
    """Create a demo histogram showing the updated features"""
    
    # Simulate landmark distances with some outliers
    # Most distances are small (0-0.5), with a few large outliers
    normal_distances = np.random.uniform(0, 0.5, 190)
    outlier_distances = np.random.uniform(1.0, 2.0, 10)
    all_distances = np.concatenate([normal_distances, outlier_distances])
    
    # Calculate 95th percentile
    percentile_95 = np.percentile(all_distances, 95)
    
    # Separate outliers
    outlier_mask = all_distances > percentile_95
    non_outlier_distances = all_distances[~outlier_mask]
    outlier_count = outlier_mask.sum()
    
    # Create histogram
    hist_values, bin_edges = np.histogram(
        non_outlier_distances,
        bins=20,
        range=(0, percentile_95)
    )
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Plot histogram bars
    bar_width = (bin_edges[1] - bin_edges[0])
    bar_positions = bin_edges[:-1] + bar_width / 2
    
    ax.bar(bar_positions, hist_values, width=bar_width * 0.9,
           color='gray', edgecolor='black', linewidth=0.5)
    
    # Add outlier bar at the end (in red)
    if outlier_count > 0:
        outlier_position = bin_edges[-1] + bar_width
        ax.bar(outlier_position, outlier_count, width=bar_width * 0.9,
               color='red', edgecolor='darkred', linewidth=0.5)
    
    # Set labels and title
    ax.set_xlabel('Distance from Base Frame (×100)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('Updated Histogram with Y-axis Tick and Scaled X-axis', fontsize=11, fontweight='bold')
    
    # Customize x-axis - multiply by 100
    ax.set_xlim(0, outlier_position + bar_width if outlier_count > 0 else bin_edges[-1])
    
    # Add x-axis labels (multiplied by 100)
    ax.set_xticks([0, percentile_95, outlier_position if outlier_count > 0 else bin_edges[-1]])
    ax.set_xticklabels(['0', f'{percentile_95 * 100:.1f}', '>95%' if outlier_count > 0 else ''], fontsize=9)
    
    # Add y-axis tick at the top showing max value
    max_count = max(hist_values.max(), outlier_count)
    ax.set_ylim(0, max_count * 1.1)  # Add 10% headroom
    
    # Add custom y-axis tick at top
    ax.axhline(y=max_count, color='black', linewidth=1, linestyle='-', alpha=0.3)
    ax.text(-0.02, max_count, str(max_count), 
            transform=ax.get_yaxis_transform(),
            ha='right', va='center', fontsize=8, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add annotations
    ax.text(0.98, 0.98, f'Total points: {len(all_distances)}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if outlier_count > 0:
        ax.text(0.98, 0.88, f'Outliers (>95%): {outlier_count}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9, color='red',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    # Highlight the changes
    ax.text(0.02, 0.98, '✓ Y-axis tick at top\n✓ X-axis values ×100',
            transform=ax.transAxes, ha='left', va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Updated histogram demo saved to: {output_file}")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Total landmarks: {len(all_distances)}")
    print(f"  Distance range: [{all_distances.min():.4f}, {all_distances.max():.4f}]")
    print(f"  95th percentile: {percentile_95:.4f} (displayed as {percentile_95 * 100:.1f})")
    print(f"  Max count: {max_count}")
    print(f"  Non-outliers: {len(non_outlier_distances)}")
    print(f"  Outliers: {outlier_count}")


def main():
    print("=" * 60)
    print("Creating Updated Histogram Demonstration")
    print("=" * 60)
    print()
    
    create_updated_histogram_demo()
    
    print()
    print("=" * 60)
    print("Updated histogram demo created successfully! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
