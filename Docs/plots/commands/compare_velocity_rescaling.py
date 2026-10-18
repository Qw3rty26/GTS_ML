#!/usr/bin/env python3

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_PATH = SCRIPT_PATH.parents[3]
OUTPUT_PATH = SCRIPT_PATH.parents[1] / "velocity_rescaling.pdf"

sys.path.insert(0, str(PROJECT_PATH / "py"))

from plummer import (
    CLUSTER_MASS,
    GRAVITATIONAL_CONSTANT,
    Plummer,
)


# Simulation parameters (standardized)
NUMBER_OF_STARS = 20_000
PLUMMER_RADIUS = 10.0
RANDOM_SEED = 42


def sample_jeans_cluster(plummer):
    """Generate a cluster using Jeans velocity sampling."""
    maximum_radius = 10.0 * plummer.plummer_radius
    radius_vector, sigma_squared_vector = plummer._solve_jeans_equation()

    positions = np.empty((plummer.number_of_stars, 3))
    velocities = np.empty_like(positions)

    for index in range(plummer.number_of_stars):
        radius = plummer._sample_radius()

        while radius > maximum_radius:
            radius = plummer._sample_radius()

        radial_versor = plummer._random_versor()
        positions[index] = radius * radial_versor

        velocities[index] = plummer._sample_velocity(
            radius,
            radius_vector,
            sigma_squared_vector,
            radial_versor,
        )

    return positions, velocities


def sample_direct_plummer_cluster(plummer):
    """Generate a cluster using direct sampling of Plummer speeds."""
    maximum_radius = 10.0 * plummer.plummer_radius

    positions = np.empty((plummer.number_of_stars, 3))
    velocities = np.empty_like(positions)

    maximum_x = np.sqrt(2.0 / 9.0)
    maximum_pdf = maximum_x**2 * (1.0 - maximum_x**2) ** 3.5

    for index in range(plummer.number_of_stars):
        radius = plummer._sample_radius()

        while radius > maximum_radius:
            radius = plummer._sample_radius()

        positions[index] = radius * plummer._random_versor()

        # Rejection sampling of x = v / v_esc.
        while True:
            x = plummer.rng.random()
            acceptance = plummer.rng.random()
            relative_pdf = x**2 * (1.0 - x**2) ** 3.5

            if acceptance * maximum_pdf < relative_pdf:
                break

        escape_speed = np.sqrt(
            -2.0 * plummer._potential_phi(radius)
        )

        speed = x * escape_speed
        velocities[index] = speed * plummer._random_versor()

    return positions, velocities


def calculate_energies(plummer, positions, velocities):
    """Calculate kinetic and softened gravitational potential energies."""
    star_mass = CLUSTER_MASS / plummer.number_of_stars
    softening = 0.01 * plummer.plummer_radius

    kinetic_energy = 0.5 * star_mass * np.sum(velocities**2)
    potential_energy = 0.0

    for i in range(plummer.number_of_stars):
        for j in range(i + 1, plummer.number_of_stars):
            distance_vector = positions[j] - positions[i]
            distance = np.sqrt(
                np.sum(distance_vector**2) + softening**2
            )

            potential_energy -= (
                GRAVITATIONAL_CONSTANT * star_mass**2 / distance
            )

    return kinetic_energy, potential_energy


def calculate_virial_ratio(kinetic_energy, potential_energy):
    """Calculate the virial ratio Q = 2K / |U|."""
    return 2.0 * kinetic_energy / abs(potential_energy)


def normalize_speeds(plummer, positions, velocities):
    """Normalize each speed by its local escape speed."""
    radii = np.linalg.norm(positions, axis=1)
    speeds = np.linalg.norm(velocities, axis=1)
    escape_speeds = np.sqrt(-2.0 * plummer._potential_phi(radii))

    return speeds / escape_speeds


def bounded_kde(samples, evaluation_points):
    """Estimate a density on [0, 1] using reflection at both boundaries."""
    estimator = gaussian_kde(samples)

    return (
        estimator(evaluation_points)
        + estimator(-evaluation_points)
        + estimator(2.0 - evaluation_points)
    )


def plot_velocity_rescaling(
    initial_speeds,
    rescaled_speeds,
    direct_speeds,
    virial_scale_factor,
    initial_virial_ratio,
    final_virial_ratio,
    direct_virial_ratio,
):
    """Plot normalized speed distributions and simulation diagnostics."""
    evaluation_points = np.linspace(0.0, 1.0, 600)

    initial_density = bounded_kde(initial_speeds, evaluation_points)
    rescaled_density = bounded_kde(rescaled_speeds, evaluation_points)
    direct_density = bounded_kde(direct_speeds, evaluation_points)

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
        initial_density,
        linewidth=2.0,
        label="Jeans sampling, before rescaling",
    )

    ax.plot(
        evaluation_points,
        rescaled_density,
        linewidth=2.0,
        label="Jeans sampling, after rescaling",
    )

    ax.plot(
        evaluation_points,
        direct_density,
        color="green",
        linestyle="--",
        linewidth=2.0,
        label="Direct Plummer sampling",
    )

    ax.set_title("Initial velocity rescaling")
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

    # --- Diagnostics text in the top right ---
    diagnostic_text = (
        f"$\\lambda = {virial_scale_factor:.4f}$\n"
        f"$Q_{{\\mathrm{{before}}}} = {initial_virial_ratio:.4f}$\n"
        f"$Q_{{\\mathrm{{after}}}} = {final_virial_ratio:.4f}$\n"
        f"$Q_{{\\mathrm{{direct\\ sampling}}}} = {direct_virial_ratio:.4f}$"
    )

    ax.text(
        0.97, 0.95, diagnostic_text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9.5,
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

    fig.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    jeans_plummer = Plummer(
        plummer_radius=PLUMMER_RADIUS,
        number_of_stars=NUMBER_OF_STARS,
        seed=RANDOM_SEED,
    )

    direct_plummer = Plummer(
        plummer_radius=PLUMMER_RADIUS,
        number_of_stars=NUMBER_OF_STARS,
        seed=RANDOM_SEED,
    )

    # Generate the Jeans-sampled cluster.
    positions, initial_velocities = sample_jeans_cluster(jeans_plummer)

    initial_kinetic_energy, potential_energy = calculate_energies(
        jeans_plummer,
        positions,
        initial_velocities,
    )

    initial_virial_ratio = calculate_virial_ratio(
        initial_kinetic_energy,
        potential_energy,
    )

    # Calculate and apply the global virial rescaling factor.
    virial_scale_factor = np.sqrt(
        abs(potential_energy) / (2.0 * initial_kinetic_energy)
    )

    rescaled_velocities = initial_velocities * virial_scale_factor

    final_kinetic_energy = (
        initial_kinetic_energy * virial_scale_factor**2
    )

    final_virial_ratio = calculate_virial_ratio(
        final_kinetic_energy,
        potential_energy,
    )

    # Generate an independent cluster using direct Plummer sampling.
    direct_positions, direct_velocities = sample_direct_plummer_cluster(
        direct_plummer
    )

    direct_kinetic_energy, direct_potential_energy = calculate_energies(
        direct_plummer,
        direct_positions,
        direct_velocities,
    )

    direct_virial_ratio = calculate_virial_ratio(
        direct_kinetic_energy,
        direct_potential_energy,
    )

    # Normalize all speeds by their respective local escape speeds.
    initial_speeds = normalize_speeds(
        jeans_plummer,
        positions,
        initial_velocities,
    )

    rescaled_speeds = normalize_speeds(
        jeans_plummer,
        positions,
        rescaled_velocities,
    )

    direct_speeds = normalize_speeds(
        direct_plummer,
        direct_positions,
        direct_velocities,
    )

    print("Simulation parameters:")
    print(f"  Number of stars: {NUMBER_OF_STARS}")
    print(f"  Plummer radius: {PLUMMER_RADIUS}")
    print(f"  Random seed: {RANDOM_SEED}")
    print()
    print(f"Velocity rescaling factor lambda: {virial_scale_factor:.8f}")
    print(f"Virial ratio before rescaling: {initial_virial_ratio:.8f}")
    print(f"Virial ratio after rescaling: {final_virial_ratio:.8f}")
    print(f"Virial ratio, direct Plummer sampling: {direct_virial_ratio:.8f}")

    plot_velocity_rescaling(
        initial_speeds,
        rescaled_speeds,
        direct_speeds,
        virial_scale_factor,
        initial_virial_ratio,
        final_virial_ratio,
        direct_virial_ratio,
    )

    print(f"\nPlot saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
