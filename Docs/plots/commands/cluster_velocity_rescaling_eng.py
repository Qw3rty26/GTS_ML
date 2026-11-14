#!/usr/bin/env python3

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLOTS_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PLOTS_DIR / "cluster_velocity_rescaling_eng.pdf"

sys.path.insert(0, str(PROJECT_ROOT / "py"))

from plummer import Plummer, CLUSTER_MASS, GRAVITATIONAL_CONSTANT


NUMBER_OF_STARS = 2_000
PLUMMER_RADIUS = 1.0
MAX_RADIUS_FACTOR = 10.0
RANDOM_SEED = 26


def theoretical_plummer_pdf(normalized_speed):
    density = np.zeros_like(normalized_speed)
    valid = (normalized_speed >= 0.0) & (normalized_speed <= 1.0)
    x = normalized_speed[valid]

    density[valid] = 512.0 / (7.0 * np.pi) * x**2 * (1.0 - x**2)**3.5

    return density


def sample_initial_conditions(plummer):
    positions, velocities = plummer.generate_particles()
    radii = np.linalg.norm(positions, axis=1)

    return positions, velocities, radii


def calculate_virial_ratio(plummer, positions, velocities):
    star_mass = CLUSTER_MASS / plummer.number_of_particles
    softening_eps = 0.01 * plummer.cluster_radius

    kinetic_energy = 0.5 * star_mass * np.sum(velocities**2)
    potential_energy = 0.0

    for i in range(plummer.number_of_particles):
        for j in range(i + 1, plummer.number_of_particles):
            distance_vector = positions[j] - positions[i]
            distance = np.sqrt(np.sum(distance_vector**2) + softening_eps**2)
            potential_energy -= GRAVITATIONAL_CONSTANT * star_mass**2 / distance

    return 2.0 * kinetic_energy / abs(potential_energy)


def estimate_initial_density(samples, evaluation_points):
    estimator = gaussian_kde(samples)
    density = estimator(evaluation_points) + estimator(-evaluation_points) + estimator(2.0 - evaluation_points)
    density[evaluation_points > 1.0] = 0.0
    return density


def estimate_rescaled_density(samples, evaluation_points):
    estimator = gaussian_kde(samples)
    return estimator(evaluation_points) + estimator(-evaluation_points)


def plot_velocity_rescaling(
    initial_speeds,
    rescaled_speeds,
    rescaling_factor,
    q_before,
    q_after,
):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "mathtext.fontset": "dejavuserif",
    })

    maximum_speed = max(1.05, np.max(rescaled_speeds) * 1.05)
    evaluation_points = np.linspace(0.0, maximum_speed, 800)

    initial_density = estimate_initial_density(initial_speeds, evaluation_points)
    rescaled_density = estimate_rescaled_density(rescaled_speeds, evaluation_points)
    theoretical_density = theoretical_plummer_pdf(evaluation_points)
    density_difference = rescaled_density - initial_density

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(9, 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.0], "hspace": 0.10},
    )

    ax_top.plot(
        evaluation_points,
        initial_density,
        color="steelblue",
        linewidth=1.5,
        label="Density before rescaling",
    )
    ax_top.plot(
        evaluation_points,
        rescaled_density,
        color="darkorange",
        linewidth=1.8,
        label="Density after rescaling",
    )
    ax_top.plot(
        evaluation_points,
        theoretical_density,
        color="darkgreen",
        linestyle="--",
        linewidth=1.8,
        label="Theoretical Plummer distribution",
    )

    ax_top.set_title("Effect of virial rescaling on velocities")
    ax_top.set_ylabel(r"Probability density, $p$")
    ax_top.set_xlim(0.0, maximum_speed)
    ax_top.set_ylim(bottom=0.0)
    ax_top.grid(True, alpha=0.3)
    ax_top.yaxis.set_label_coords(-0.11, 0.5)

    statistics = (
        f"$\\lambda = {rescaling_factor:.4f}$\n"
        f"$Q_{{\\mathrm{{before}}}} = {q_before:.4f}$\n"
        f"$Q_{{\\mathrm{{after}}}} = {q_after:.4f}$"
    )

    ax_top.text(
        0.97, 0.92, statistics,
        transform=ax_top.transAxes,
        ha="right", va="top",
        fontsize=10,
        color="black",
        bbox=dict(
            boxstyle="square,pad=0.4",
            facecolor="white",
            edgecolor="black",
            alpha=0.9,
            linewidth=0.8,
        ),
        zorder=5,
    )

    ax_bot.plot(
        evaluation_points,
        density_difference,
        color="purple",
        linewidth=1.5,
        label=r"$\Delta p = p_{\mathrm{after}} - p_{\mathrm{before}}$",
    )
    ax_bot.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)

    ax_bot.set_xlabel(r"Normalized speed, $v/v_{\mathrm{esc}}(r)$")
    ax_bot.set_ylabel(r"$\Delta p$")
    ax_bot.set_xlim(0.0, maximum_speed)
    ax_bot.grid(True, alpha=0.3)
    ax_bot.yaxis.set_label_coords(-0.11, 0.5)

    handles_top, labels_top = ax_top.get_legend_handles_labels()
    handles_bot, labels_bot = ax_bot.get_legend_handles_labels()

    fig.legend(
        handles_top + handles_bot,
        labels_top + labels_bot,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.02),
        frameon=True,
        fancybox=False,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
        fontsize=11,
        handlelength=2.5,
        borderpad=0.8,
        labelspacing=0.5,
    )

    fig.subplots_adjust(left=0.15, right=0.97, top=0.92, bottom=0.22)

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

    positions, initial_velocities, radii = sample_initial_conditions(plummer)

    initial_speeds = np.linalg.norm(initial_velocities, axis=1)
    escape_speeds = np.array([
        plummer._escape_velocity(radius)
        for radius in radii
    ])

    normalized_initial_speeds = initial_speeds / escape_speeds

    q_before = calculate_virial_ratio(plummer, positions, initial_velocities)

    rescaling_factor = 1.0 / np.sqrt(q_before)
    rescaled_velocities = initial_velocities * rescaling_factor

    q_after = calculate_virial_ratio(plummer, positions, rescaled_velocities)

    rescaled_speeds = np.linalg.norm(rescaled_velocities, axis=1)
    normalized_rescaled_speeds = rescaled_speeds / escape_speeds

    plot_velocity_rescaling(
        normalized_initial_speeds,
        normalized_rescaled_speeds,
        rescaling_factor,
        q_before,
        q_after,
    )


if __name__ == "__main__":
    main()
