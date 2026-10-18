#!/usr/bin/env python3

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_PATH = SCRIPT_PATH.parents[3]
OUTPUT_PATH = SCRIPT_PATH.parents[1] / "initial_positions.pdf"

sys.path.insert(0, str(PROJECT_PATH / "py"))

from plummer import Plummer


# Standard model parameters.
PLUMMER_RADIUS = 1.0
NUMBER_OF_STARS = 20_000
SEED = 42

# Radial range shown in the cumulative mass plot, in units of a.
MAX_NORMALIZED_RADIUS = 5.0
NUMBER_OF_RADIAL_POINTS = 1000


# Scientific plotting style.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "mathtext.fontset": "dejavuserif",
})


def calculate_enclosed_mass_fraction(normalized_radius):
    """Return the analytical Plummer enclosed mass fraction."""
    numerator = normalized_radius**3
    denominator = (normalized_radius**2 + 1.0)**1.5

    return numerator / denominator


def generate_initial_positions():
    """Generate stellar positions using the project Plummer class."""
    cluster = Plummer(
        plummer_radius=PLUMMER_RADIUS,
        number_of_stars=NUMBER_OF_STARS,
        seed=SEED,
    )

    positions, _ = cluster.generate_plummer_cluster()

    return positions


def plot_initial_positions():
    """Plot projected stellar positions and the enclosed mass profile."""
    positions = generate_initial_positions()

    # Normalize positions by the Plummer scale radius.
    normalized_positions = positions / PLUMMER_RADIUS

    projected_radius = np.hypot(
        normalized_positions[:, 0],
        normalized_positions[:, 1],
    )

    three_dimensional_radius = np.linalg.norm(
        normalized_positions,
        axis=1,
    )

    number_of_stars = len(positions)

    # Calculate the empirical cumulative mass fraction.
    sorted_radius = np.sort(three_dimensional_radius)
    empirical_mass_fraction = (
        np.arange(1, number_of_stars + 1) / number_of_stars
    )

    # Calculate the theoretical Plummer profile.
    theoretical_radius = np.linspace(
        0.0,
        MAX_NORMALIZED_RADIUS,
        NUMBER_OF_RADIAL_POINTS,
    )

    theoretical_mass_fraction = calculate_enclosed_mass_fraction(
        theoretical_radius,
    )

    # Half-mass radius calculation: M(< r_h) = 0.5 -> r_h = a / sqrt(2^(2/3) - 1)
    half_mass_radius = 1.0 / np.sqrt(2.0**(2.0/3.0) - 1.0)
    half_mass_fraction = 0.5

    fig = plt.figure(figsize=(11, 5.2))

    # Layout: two subplots side by side with room at the bottom for the legend
    positions_ax = fig.add_axes([0.08, 0.28, 0.42, 0.62])
    mass_ax = fig.add_axes([0.55, 0.28, 0.40, 0.62])

    # Left panel: projection of the three-dimensional positions.
    scatter = positions_ax.scatter(
        normalized_positions[:, 0],
        normalized_positions[:, 1],
        c=three_dimensional_radius,
        cmap="turbo",
        s=6,
        alpha=0.85,
        edgecolors="none",
        rasterized=True,
    )

    colorbar = fig.colorbar(
        scatter,
        ax=positions_ax,
        pad=0.025,
    )
    colorbar.set_label(r"3D radius, $r/a$")
    colorbar.ax.tick_params(direction="in", labelsize=8)

    # Magenta solid circle for half-mass radius
    half_mass_circle = plt.Circle(
        (0.0, 0.0),
        half_mass_radius,
        color="magenta",
        fill=False,
        linestyle="-",
        linewidth=1.8,
        zorder=5,
    )
    positions_ax.add_patch(half_mass_circle)

    positions_ax.set_xlabel(r"$x/a$")
    positions_ax.set_ylabel(r"$y/a$")
    positions_ax.set_title("Projected stellar positions")

    positions_ax.set_aspect("equal", adjustable="box")
    positions_ax.grid(
        True,
        linestyle=":",
        linewidth=0.5,
        color="0.75",
        alpha=0.6,
    )
    positions_ax.set_axisbelow(True)

    # Right panel: empirical and analytical enclosed mass fractions.
    line_analytic, = mass_ax.plot(
        theoretical_radius,
        theoretical_mass_fraction,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Analytical Plummer profile",
    )

    line_empirical, = mass_ax.step(
        sorted_radius,
        empirical_mass_fraction,
        where="post",
        color="#2b5c8f",
        linewidth=1.5,
        label="Generated stars",
    )

    # Projections and marker for half-mass radius (M(<r) = 0.5)
    mass_ax.axvline(
        half_mass_radius,
        color="black",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )
    mass_ax.axhline(
        half_mass_fraction,
        color="black",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )
    mass_ax.scatter(
        [half_mass_radius],
        [half_mass_fraction],
        color="black",
        s=25,
        zorder=3,
    )

    # Scientific information box for half-mass radius (lowered to bottom-right corner)
    half_mass_text = (
        "Half-mass radius:\n"
        f"$M(<r) = 0.5\\,M$\n"
        f"$r_{{1/2}} \\approx {half_mass_radius:.3f}\\,a$"
    )

    mass_ax.text(
        0.95, 0.05, half_mass_text,
        transform=mass_ax.transAxes,
        ha="right", va="bottom",
        fontsize=8.5,
        color="black",
        bbox=dict(
            boxstyle="square,pad=0.5",
            facecolor="white",
            edgecolor="black",
            alpha=0.9,
            linewidth=0.8,
        ),
        zorder=5,
    )

    mass_ax.set_xlim(0.0, MAX_NORMALIZED_RADIUS)
    mass_ax.set_ylim(0.0, 1.02)
    mass_ax.set_xlabel(r"Three-dimensional radius, $r/a$")
    mass_ax.set_ylabel(r"Enclosed mass fraction, $M(<r)/M$")
    mass_ax.set_title("Enclosed mass profile")

    mass_ax.grid(
        True,
        linestyle=":",
        linewidth=0.5,
        color="0.75",
        alpha=0.6,
    )
    mass_ax.set_axisbelow(True)

    # Custom proxy handle for the half-mass radius line in legend (showing just a line instead of a full circle)
    legend_circle_proxy = Line2D(
        [0], [0],
        color="magenta",
        linestyle="-",
        linewidth=1.8,
        label=r"Half-mass radius ($r_{1/2}$)",
    )

    combined_handles = [line_analytic, line_empirical, legend_circle_proxy]
    combined_labels = [
        "Analytical Plummer profile",
        "Generated stars",
        r"Half-mass radius ($r_{1/2}$)",
    ]

    # Perfectly centered legend below both plots
    fig.legend(
        combined_handles,
        combined_labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.08),
        frameon=True,
        fancybox=False,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
        fontsize=8.5,
    )

    fig.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight")
    plt.close(fig)

    fraction_within_range = np.mean(
        three_dimensional_radius <= MAX_NORMALIZED_RADIUS
    )

    print(f"Generated {number_of_stars:,} stars.")
    print(f"Plummer radius: {PLUMMER_RADIUS}")
    print(f"Random seed: {SEED}")
    print(
        f"Fraction of stars within r/a <= {MAX_NORMALIZED_RADIUS}: "
        f"{fraction_within_range:.4f}"
    )
    print(f"Plot saved to: {OUTPUT_PATH}")


def main():
    plot_initial_positions()


if __name__ == "__main__":
    main()
