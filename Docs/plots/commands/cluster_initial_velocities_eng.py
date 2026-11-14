#!/usr/bin/env python3

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLOTS_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PLOTS_DIR / "cluster_initial_velocities_eng.pdf"

sys.path.insert(0, str(PROJECT_ROOT / "py"))

from plummer import Plummer


NUMBER_OF_STARS = 2_000
PLUMMER_RADIUS = 1.0
RANDOM_SEED = 26
NUMBER_OF_BINS = 80


def theoretical_plummer_pdf(normalized_speed):
    valid = (normalized_speed >= 0.0) & (normalized_speed <= 1.0)
    density = np.zeros_like(normalized_speed)

    x = normalized_speed[valid]
    density[valid] = 512.0 / (7.0 * np.pi) * x**2 * (1.0 - x**2)**3.5

    return density


def sample_unscaled_velocities(plummer):
    positions, velocities = plummer.generate_particles()

    radii = np.linalg.norm(positions, axis=1)
    speeds = np.linalg.norm(velocities, axis=1)
    escape_speeds = np.array([
        plummer._escape_velocity(radius) for radius in radii
    ])

    return speeds / escape_speeds


def plot_velocity_distribution(normalized_speeds):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 12,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "mathtext.fontset": "dejavuserif",
    })

    bin_edges = np.linspace(0.0, 1.0, NUMBER_OF_BINS + 1)
    theoretical_x = np.linspace(0.0, 1.0, 600)
    theoretical_y = theoretical_plummer_pdf(theoretical_x)

    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_axes([0.11, 0.28, 0.85, 0.64])

    ax.hist(
        normalized_speeds,
        bins=bin_edges,
        density=True,
        color="steelblue",
        alpha=0.40,
        edgecolor="steelblue",
        linewidth=0.8,
        label="Sampled data (empirical)",
    )

    ax.plot(
        theoretical_x,
        theoretical_y,
        color="darkgreen",
        linestyle="--",
        linewidth=2.2,
        label="Theoretical Plummer model",
    )

    ax.set_title("Initial velocity distribution")
    ax.set_xlabel(r"Normalized speed, $v/v_{\mathrm{esc}}(r)$")
    ax.set_ylabel("Probability density")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.05),
        frameon=True,
        fancybox=False,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
        fontsize=12,
        handlelength=2.5,
        borderpad=0.8,
        labelspacing=0.5,
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved to: {OUTPUT_PATH}")


def main():
    plummer = Plummer(
        number_of_particles=NUMBER_OF_STARS,
        cluster_radius=PLUMMER_RADIUS,
        cluster_seed=RANDOM_SEED,
    )

    normalized_speeds = sample_unscaled_velocities(plummer)
    plot_velocity_distribution(normalized_speeds)


if __name__ == "__main__":
    main()
