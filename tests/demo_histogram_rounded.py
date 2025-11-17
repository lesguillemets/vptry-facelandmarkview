#!/usr/bin/env python3
"""
Demonstrate the histogram with rounded x-max and y-max values.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

# Set random seed for reproducibility
np.random.seed(42)


def create_rounded_histogram_demo(output_file="histogram_demo_rounded.png"):
    """Create a demo histogram showing rounded x-max and y-max"""

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

    # Round x-max to nearest 0.5 (shown as 50 when ×100)
    percentile_95_rounded = np.ceil(percentile_95 / 0.5) * 0.5

    # Separate outliers
    outlier_mask = all_distances > percentile_95
    non_outlier_distances = all_distances[~outlier_mask]
    outlier_count = outlier_mask.sum()

    # Create histogram
    hist_values, bin_edges = np.histogram(
        non_outlier_distances,
        bins=20,
        range=(0, percentile_95_rounded),  # Using rounded value
    )

    # Round y-max to nearest 50
    max_count_raw = max(hist_values.max(), outlier_count)
    max_count_rounded = int(np.ceil(max_count_raw / 50) * 50)

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 5))

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
    ax.set_title(
        "Histogram with Rounded Axes for Frame Comparison",
        fontsize=11,
        fontweight="bold",
    )

    # Customize x-axis - multiply by 100
    ax.set_xlim(0, outlier_position + bar_width if outlier_count > 0 else bin_edges[-1])

    # Add x-axis labels (multiplied by 100)
    ax.set_xticks(
        [
            0,
            percentile_95_rounded,
            outlier_position if outlier_count > 0 else bin_edges[-1],
        ]
    )
    ax.set_xticklabels(
        [
            "0",
            f"{percentile_95_rounded * 100:.0f}",
            ">95%" if outlier_count > 0 else "",
        ],
        fontsize=9,
    )

    # Set y-axis limits with rounded max
    ax.set_ylim(0, max_count_rounded)

    # Add custom y-axis tick at top with rounded value
    ax.axhline(
        y=max_count_rounded, color="black", linewidth=1, linestyle="-", alpha=0.3
    )
    ax.text(
        -0.05,
        max_count_rounded,
        str(max_count_rounded),
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="blue",
    )

    # Also show the actual max value for comparison
    ax.axhline(y=max_count_raw, color="red", linewidth=1, linestyle="--", alpha=0.5)
    ax.text(
        -0.05,
        max_count_raw,
        f"{max_count_raw}*",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="center",
        fontsize=8,
        color="red",
    )

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    # Add mean and variance text below the histogram
    stats_text = f"Mean: {mean_dist:.4f}  Var: {var_dist:.6f}"
    ax.text(
        0.5,
        -0.18,
        stats_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.6),
    )

    # Add annotations showing the rounding
    rounding_info = (
        f"X-axis rounded: {percentile_95:.3f} → {percentile_95_rounded:.1f}\n"
        f"Y-axis rounded: {max_count_raw} → {max_count_rounded}"
    )
    ax.text(
        0.98,
        0.98,
        rounding_info,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
    )

    if outlier_count > 0:
        ax.text(
            0.98,
            0.78,
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
        "✓ X-max rounded to 0.5\n✓ Y-max rounded to 50",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Rounded histogram demo saved to: {output_file}")

    # Print statistics
    print(f"\nStatistics:")
    print(f"  Total landmarks: {len(all_distances)}")
    print(f"  Mean distance: {mean_dist:.4f}")
    print(f"  Variance: {var_dist:.6f}")
    print(
        f"  95th percentile: {percentile_95:.4f} → rounded to {percentile_95_rounded:.1f}"
    )
    print(
        f"  95th percentile (×100): {percentile_95 * 100:.1f} → {percentile_95_rounded * 100:.0f}"
    )
    print(f"  Max count: {max_count_raw} → rounded to {max_count_rounded}")


def main():
    print("=" * 60)
    print("Creating Rounded Histogram Demonstration")
    print("=" * 60)
    print()

    create_rounded_histogram_demo()

    print()
    print("=" * 60)
    print("Rounded histogram demo created successfully! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
