#!/usr/bin/env python3

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLOTS_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PLOTS_DIR / "velocity_distribution_comparison.pdf"

sys.path.insert(0, str(PROJECT_ROOT / "py"))

from plummer import Plummer


NUMBER_OF_STARS = 20_000
PLUMMER_RADIUS = 10.0
MAX_RADIUS_FACTOR = 10.0
RANDOM_SEED = 42


def sample_jeans_velocities(plummer, number_of_stars):
    """Sample Jeans velocities before applying the virial correction."""
    radius_vector, sigma_squared_vector = plummer._solve_jeans_equation()
    maximum_radius = MAX_RADIUS_FACTOR * plummer.plummer_radius
    normalized_speeds = np.empty(number_of_stars)

    for i in range(number_of_stars):
        radius = plummer._sample_radius()

        while radius > maximum_radius:
            radius = plummer._sample_radius()

        radial_versor = plummer._random_versor()

        velocity = plummer._sample_velocity(
            radius,
            radius_vector,
            sigma_squared_vector,
            radial_versor,
        )

        speed = np.linalg.norm(velocity)
        escape_speed = np.sqrt(-2.0 * plummer._potential_phi(radius))
        normalized_speeds[i] = speed / escape_speed

    return normalized_speeds


def sample_theoretical_plummer_velocities(number_of_stars, rng):
    """Sample the exact Plummer distribution for x = v / v_escape."""
    samples = np.empty(number_of_stars)

    maximum_x = np.sqrt(2.0 / 9.0)
    maximum_pdf = maximum_x**2 * (1.0 - maximum_x**2) ** 3.5

    filled = 0

    while filled < number_of_stars:
        remaining = number_of_stars - filled
        candidates = rng.random(max(256, 2 * remaining))
        acceptance = rng.random(candidates.size)

        relative_pdf = candidates**2 * (1.0 - candidates**2) ** 3.5
        accepted = candidates[acceptance * maximum_pdf < relative_pdf]

        count = min(accepted.size, remaining)
        samples[filled:filled + count] = accepted[:count]
        filled += count

    return samples


def theoretical_plummer_pdf(x):
    """Normalized analytical PDF for x = v / v_escape."""
    return (
        512.0 / (7.0 * np.pi)
        * x**2
        * (1.0 - x**2) ** 3.5
    )


def bounded_kde(samples, evaluation_points):
    """Estimate a density on [0, 1] using reflection at both boundaries."""
    estimator = gaussian_kde(samples)

    return (
        estimator(evaluation_points)
        + estimator(-evaluation_points)
        + estimator(2.0 - evaluation_points)
    )


def plot_velocity_distributions(jeans_speeds, direct_speeds):
    evaluation_points = np.linspace(0.0, 1.0, 600)

    jeans_density = bounded_kde(jeans_speeds, evaluation_points)
    direct_density = bounded_kde(direct_speeds, evaluation_points)
    theoretical_density = theoretical_plummer_pdf(evaluation_points)

    plt.rcParams.update({
        "font.family": "serif",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })

    fig = plt.figure(figsize=(9, 6.5))

    # Keep the plot wide and reserve space underneath for the legend
    ax = fig.add_axes([0.11, 0.28, 0.85, 0.64])

    ax.plot(
        evaluation_points,
        jeans_density,
        linewidth=2.0,
        label="Jeans sampling",
    )

    ax.plot(
        evaluation_points,
        direct_density,
        linewidth=2.0,
        label="Direct Plummer sampling",
    )

    ax.plot(
        evaluation_points,
        theoretical_density,
        color="green",
        linestyle="--",
        linewidth=2.0,
        label="Theoretical Plummer PDF",
    )

    ax.set_title("Initial velocity distributions")
    ax.set_xlabel(r"Normalized speed, $v / v_{\mathrm{esc}}(r)$")
    ax.set_ylabel("Probability density")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)

    # Legend centered outside, below the plot
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.08),
        frameon=True,
        fancybox=False,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
        fontsize=9,
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to: {OUTPUT_PATH}")


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    plummer = Plummer(
        plummer_radius=PLUMMER_RADIUS,
        number_of_stars=NUMBER_OF_STARS,
        seed=RANDOM_SEED,
    )

    jeans_speeds = sample_jeans_velocities(
        plummer,
        NUMBER_OF_STARS,
    )

    direct_speeds = sample_theoretical_plummer_velocities(
        NUMBER_OF_STARS,
        rng,
    )

    plot_velocity_distributions(jeans_speeds, direct_speeds)


if __name__ == "__main__":
    main()
