#!/usr/bin/env python3

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLOTS_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PLOTS_DIR / "velocity_distribution_comparison.pdf"

sys.path.insert(0, str(PROJECT_ROOT / "py"))

from plummer import Plummer


NUMBER_OF_STARS = 2_000
PLUMMER_RADIUS = 1.0
MAX_RADIUS_FACTOR = 10.0
RANDOM_SEED = 26
NUMBER_OF_BINS = 80


def theoretical_plummer_pdf(normalized_speed):
    valid = (normalized_speed >= 0.0) & (normalized_speed <= 1.0)
    density = np.zeros_like(normalized_speed)

    x = normalized_speed[valid]
    density[valid] = (
        512.0 / (7.0 * np.pi)
        * x**2
        * (1.0 - x**2)**3.5
    )

    return density


def sample_unscaled_velocities(plummer, number_of_stars):
    normalized_speeds = np.empty(number_of_stars)
    maximum_radius = MAX_RADIUS_FACTOR * plummer.plummer_radius

    for i in range(number_of_stars):
        radius = plummer._sample_radius()

        while radius > maximum_radius:
            radius = plummer._sample_radius()

        velocity = plummer._sample_velocity(radius)
        speed = np.linalg.norm(velocity)
        escape_speed = np.sqrt(-2.0 * plummer._potential_phi(radius))

        normalized_speeds[i] = speed / escape_speed

    return normalized_speeds


def configure_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 13,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "mathtext.fontset": "dejavuserif",
    })


def plot_velocity_distribution(normalized_speeds):
    configure_plot_style()

    bin_edges = np.linspace(0.0, 1.0, NUMBER_OF_BINS + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    empirical_density, _ = np.histogram(
        normalized_speeds,
        bins=bin_edges,
        density=True,
    )

    theoretical_x = np.linspace(0.0, 1.0, 600)
    theoretical_y = theoretical_plummer_pdf(theoretical_x)

    fig = plt.figure(figsize=(9, 6.5))

    ax = fig.add_axes([0.11, 0.28, 0.85, 0.64])

    ax.hist(
        normalized_speeds,
        bins=bin_edges,
        density=True,
        color="steelblue",
        alpha=0.30,
        edgecolor="steelblue",
        linewidth=0.6,
        label="Velocità campionate",
    )

    ax.plot(
        bin_centers,
        empirical_density,
        color="steelblue",
        linewidth=1.3,
        drawstyle="steps-mid",
        label="Distribuzione empirica",
    )

    ax.plot(
        theoretical_x,
        theoretical_y,
        color="darkgreen",
        linestyle="--",
        linewidth=2.2,
        label="Distribuzione teorica di Plummer",
    )

    ax.set_title("Distribuzione delle velocità iniziali")
    ax.set_xlabel(r"Velocità normalizzata, $v/v_{\mathrm{esc}}(r)$")
    ax.set_ylabel("Densità di probabilità")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.05),
        frameon=True,
        fancybox=False,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
        fontsize=13,
        handlelength=2.5,
        borderpad=0.8,
        labelspacing=0.5,
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Grafico salvato in: {OUTPUT_PATH}")


def main():
    plummer = Plummer(
        plummer_radius=PLUMMER_RADIUS,
        number_of_stars=NUMBER_OF_STARS,
        seed=RANDOM_SEED,
    )

    normalized_speeds = sample_unscaled_velocities(
        plummer,
        NUMBER_OF_STARS,
    )

    plot_velocity_distribution(normalized_speeds)


if __name__ == "__main__":
    main()

