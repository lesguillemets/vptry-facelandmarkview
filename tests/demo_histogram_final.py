#!/usr/bin/env python3
"""
Demonstrate the final histogram visualization with all requested features.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

# Set random seed for reproducibility
np.random.seed(42)


def create_final_histogram_demo(output_file="histogram_demo_final.png"):
    """Create a demo histogram showing all features including mean/variance"""

    # Simulate landmark distances with some outliers
    # Most distances are small (0-0.5), with a few large outliers
    normal_distances = np.random.uniform(0, 0.5, 190)
    outlier_distances = np.random.uniform(1.0, 2.0, 10)
    all_distances = np.concatenate([normal_distances, outlier_distances])

    # Calculate statistics
    mean_dist = all_distances.mean()
    var_dist = all_distances.var()

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

    # Create the plot with more space at bottom
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

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
    ax.set_xlabel("Distance from Base Frame (×100)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Final Histogram: All Features", fontsize=11, fontweight="bold")

    # Customize x-axis - multiply by 100
    ax.set_xlim(0, outlier_position + bar_width if outlier_count > 0 else bin_edges[-1])

    # Add x-axis labels (multiplied by 100)
    ax.set_xticks(
        [0, percentile_95, outlier_position if outlier_count > 0 else bin_edges[-1]]
    )
    ax.set_xticklabels(
        ["0", f"{percentile_95 * 100:.1f}", ">95%" if outlier_count > 0 else ""],
        fontsize=9,
    )

    # Add y-axis tick at the top showing max value
    max_count = max(hist_values.max(), outlier_count)
    ax.set_ylim(0, max_count * 1.1)  # Add 10% headroom

    # Add custom y-axis tick at top with more space on left
    ax.axhline(y=max_count, color="black", linewidth=1, linestyle="-", alpha=0.3)
    ax.text(
        -0.05,
        max_count,
        str(max_count),
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="center",
        fontsize=9,
        fontweight="bold",
    )

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    # Add mean and variance text below the histogram
    # Note: Show values BEFORE ×100 scaling
    stats_text = f"Mean: {mean_dist:.4f}  Var: {var_dist:.6f}"
    ax.text(
        0.5,
        -0.20,
        stats_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.6),
    )

    # Add annotations
    ax.text(
        0.98,
        0.98,
        f"Total points: {len(all_distances)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
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
            fontsize=9,
            color="red",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
        )

    # Highlight the changes
    ax.text(
        0.02,
        0.98,
        "✓ Y-tick not cut off\n✓ Mean & Var shown",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Final histogram demo saved to: {output_file}")

    # Print statistics
    print("\nStatistics:")
    print(f"  Total landmarks: {len(all_distances)}")
    print(f"  Mean distance: {mean_dist:.4f}")
    print(f"  Variance: {var_dist:.6f}")
    print(f"  Distance range: [{all_distances.min():.4f}, {all_distances.max():.4f}]")
    print(
        f"  95th percentile: {percentile_95:.4f} (displayed as {percentile_95 * 100:.1f})"
    )
    print(f"  Max count: {max_count}")


def main():
    print("=" * 60)
    print("Creating Final Histogram Demonstration")
    print("=" * 60)
    print()

    create_final_histogram_demo()

    print()
    print("=" * 60)
    print("Final histogram demo created successfully! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
