#!/usr/bin/env python3
"""
Demonstrate the histogram visualization concept using matplotlib.
This shows what the histogram widget will display.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

# Set random seed for reproducibility
np.random.seed(42)


def create_demo_histogram(output_file="histogram_demo.png"):
    """Create a demo histogram showing the concept"""

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
        non_outlier_distances, bins=20, range=(0, percentile_95)
    )

    # Create the plot
    fig, ax = plt.subplots(figsize=(4, 3))

    # Plot histogram bars
    bar_width = bin_edges[1] - bin_edges[0]
    bar_positions = bin_edges[:-1] + bar_width / 2

    ax.bar(
        bar_positions,
        hist_values,
        width=bar_width * 0.9,
        color="gray",
        edgecolor="black",
        linewidth=0.5,
    )

    # Add outlier bar at the end (in red)
    if outlier_count > 0:
        outlier_position = bin_edges[-1] + bar_width
        ax.bar(
            outlier_position,
            outlier_count,
            width=bar_width * 0.9,
            color="red",
            edgecolor="darkred",
            linewidth=0.5,
        )

    # Set labels and title
    ax.set_xlabel("Distance from Base Frame", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title("Histogram of Landmark Distances", fontsize=10, fontweight="bold")

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--")

    # Customize x-axis
    ax.set_xlim(0, outlier_position + bar_width if outlier_count > 0 else bin_edges[-1])

    # Add x-axis labels
    ax.set_xticks(
        [0, percentile_95, outlier_position if outlier_count > 0 else bin_edges[-1]]
    )
    ax.set_xticklabels(
        ["0", f"{percentile_95:.2f}", ">95%" if outlier_count > 0 else ""], fontsize=8
    )

    # Add text annotations
    ax.text(
        0.98,
        0.98,
        f"Total points: {len(all_distances)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    if outlier_count > 0:
        ax.text(
            0.98,
            0.88,
            f"Outliers (>95%): {outlier_count}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="red",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
        )

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Demo histogram saved to: {output_file}")

    # Print statistics
    print("\nStatistics:")
    print(f"  Total landmarks: {len(all_distances)}")
    print(f"  Distance range: [{all_distances.min():.4f}, {all_distances.max():.4f}]")
    print(f"  95th percentile: {percentile_95:.4f}")
    print(f"  Non-outliers: {len(non_outlier_distances)}")
    print(f"  Outliers: {outlier_count}")
    print(f"  Mean distance: {all_distances.mean():.4f}")
    print(f"  Median distance: {np.median(all_distances):.4f}")


def create_comparison_demo(output_file="histogram_comparison_demo.png"):
    """Create a demo showing histograms before and after alignment"""

    # Simulate distances before and after alignment
    np.random.seed(42)

    # Before alignment: larger distances
    before_normal = np.random.uniform(0.5, 1.5, 180)
    before_outliers = np.random.uniform(3.0, 5.0, 20)
    before_distances = np.concatenate([before_normal, before_outliers])

    # After alignment: smaller distances (alignment reduces movement)
    after_normal = np.random.uniform(0, 0.5, 190)
    after_outliers = np.random.uniform(1.0, 2.0, 10)
    after_distances = np.concatenate([after_normal, after_outliers])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    def plot_histogram(ax, distances, title):
        percentile_95 = np.percentile(distances, 95)
        outlier_mask = distances > percentile_95
        non_outlier_distances = distances[~outlier_mask]
        outlier_count = outlier_mask.sum()

        hist_values, bin_edges = np.histogram(
            non_outlier_distances, bins=20, range=(0, percentile_95)
        )

        bar_width = bin_edges[1] - bin_edges[0]
        bar_positions = bin_edges[:-1] + bar_width / 2

        ax.bar(
            bar_positions,
            hist_values,
            width=bar_width * 0.9,
            color="gray",
            edgecolor="black",
            linewidth=0.5,
        )

        if outlier_count > 0:
            outlier_position = bin_edges[-1] + bar_width
            ax.bar(
                outlier_position,
                outlier_count,
                width=bar_width * 0.9,
                color="red",
                edgecolor="darkred",
                linewidth=0.5,
            )

        ax.set_xlabel("Distance", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")

        ax.text(
            0.98,
            0.98,
            f"Mean: {distances.mean():.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plot_histogram(ax1, before_distances, "Before Alignment")
    plot_histogram(ax2, after_distances, "After Alignment")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nComparison demo saved to: {output_file}")

    print("\nComparison Statistics:")
    print(
        f"  Before alignment - Mean: {before_distances.mean():.4f}, Median: {np.median(before_distances):.4f}"
    )
    print(
        f"  After alignment  - Mean: {after_distances.mean():.4f}, Median: {np.median(after_distances):.4f}"
    )
    print(
        f"  Reduction: {(1 - after_distances.mean() / before_distances.mean()) * 100:.1f}%"
    )


def main():
    print("=" * 60)
    print("Creating Histogram Demonstrations")
    print("=" * 60)
    print()

    create_demo_histogram()
    print()
    create_comparison_demo()

    print()
    print("=" * 60)
    print("Demo histograms created successfully! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
