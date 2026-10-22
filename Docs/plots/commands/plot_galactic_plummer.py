#!/usr/bin/env python3

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_PATH = SCRIPT_PATH.parents[3]
OUTPUT_PATH = SCRIPT_PATH.parents[1] / "galactic_plummer_potential.pdf"

sys.path.insert(0, str(PROJECT_PATH / "py"))
from galactic_potential import GalacticPotential, _compute_plummer_accelerations


GRAVITATIONAL_CONSTANT = 1.0
GALAXY_MASS = 2000.0
GALAXY_RADIUS = 10.0

MIN_NORMALIZED_RADIUS = 0.0
MAX_NORMALIZED_RADIUS = 5.0
NUMBER_OF_POINTS = 1_000


def calculate_normalized_acceleration(potential, radius):
    coordinates = np.zeros((len(radius), 3), dtype=np.float64)
    coordinates[:, 0] = radius

    acceleration_x, acceleration_y, acceleration_z = (
        _compute_plummer_accelerations(
            coordinates,
            potential.get_galaxy_mass(),
            potential.get_galaxy_radius(),
        )
    )

    acceleration_magnitude = np.sqrt(
        acceleration_x**2
        + acceleration_y**2
        + acceleration_z**2
    )

    acceleration_scale = (
        GRAVITATIONAL_CONSTANT
        * potential.get_galaxy_mass()
        / potential.get_galaxy_radius()**2
    )

    return acceleration_magnitude / acceleration_scale


def plot_galactic_plummer():
    potential = GalacticPotential(
        plummer_radius=GALAXY_RADIUS,
        plummer_mass=GALAXY_MASS,
    )

    normalized_radius = np.linspace(
        MIN_NORMALIZED_RADIUS,
        MAX_NORMALIZED_RADIUS,
        NUMBER_OF_POINTS,
    )
    radius = normalized_radius * GALAXY_RADIUS

    potential_scale = (
        GRAVITATIONAL_CONSTANT * GALAXY_MASS / GALAXY_RADIUS
    )

    normalized_potential = np.array([
        potential.potential_phi(r) / potential_scale
        for r in radius
    ])

    normalized_acceleration = calculate_normalized_acceleration(
        potential,
        radius,
    )

    peak_radius = 1.0 / np.sqrt(2.0)
    peak_acceleration = 2.0 / (3.0 * np.sqrt(3.0))
    potential_at_peak = -1.0 / np.sqrt(peak_radius**2 + 1.0)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "mathtext.fontset": "dejavuserif",
    })

    figure, (ax_pot, ax_acc) = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
    )

    ax_pot.plot(
        normalized_radius,
        normalized_potential,
        linewidth=2.0,
        color="#1f77b4",
    )
    ax_pot.axhline(
        0.0,
        color="black",
        linewidth=0.8,
        alpha=0.6,
    )

    ax_pot.axvline(
        peak_radius,
        color="black",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )
    ax_pot.axhline(
        potential_at_peak,
        color="black",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )

    ax_pot.scatter(
        [peak_radius],
        [potential_at_peak],
        color="black",
        s=25,
        zorder=3,
    )

    pot_info_text = (
        "Potenziale all'accelerazione di picco:\n"
        f"$R / a_{{\\mathrm{{gal}}}} = 1/\\sqrt{{2}} \\approx {peak_radius:.3f}$\n"
        f"$\\tilde{{\\Phi}}_{{\\mathrm{{gal}}}} \\approx {potential_at_peak:.3f}$"
    )

    ax_pot.text(
        0.95, 0.05, pot_info_text,
        transform=ax_pot.transAxes,
        ha="right", va="bottom",
        fontsize=11,
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

    ax_pot.set_title("Potenziale galattico di Plummer")
    ax_pot.set_xlabel(r"Raggio normalizzato, $R / a_{\mathrm{gal}}$")
    ax_pot.set_ylabel(
        r"Potenziale normalizzato, $\Phi_{\mathrm{gal}} / (G M_{\mathrm{gal}} / a_{\mathrm{gal}})$"
    )
    ax_pot.set_xlim(MIN_NORMALIZED_RADIUS, MAX_NORMALIZED_RADIUS)
    ax_pot.set_ylim(-1.05, 0.05)
    ax_pot.grid(True, alpha=0.3)

    ax_acc.plot(
        normalized_radius,
        normalized_acceleration,
        linewidth=2.0,
        color="#ff7f0e",
    )

    ax_acc.axvline(
        peak_radius,
        color="black",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )
    ax_acc.axhline(
        peak_acceleration,
        color="black",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )

    ax_acc.scatter(
        [peak_radius],
        [peak_acceleration],
        color="black",
        s=25,
        zorder=3,
    )

    info_text = (
        "Accelerazione di picco:\n"
        f"$R / a_{{\\mathrm{{gal}}}} = 1/\\sqrt{{2}} \\approx {peak_radius:.3f}$\n"
        f"$|\\tilde{{a}}|_{{\\max}} \\approx {peak_acceleration:.3f}$"
    )

    ax_acc.text(
        0.95, 0.92, info_text,
        transform=ax_acc.transAxes,
        ha="right", va="top",
        fontsize=11,
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

    ax_acc.set_title("Intensità dell'accelerazione galattica")
    ax_acc.set_xlabel(r"Raggio normalizzato, $R / a_{\mathrm{gal}}$")
    ax_acc.set_ylabel(
        r"Accelerazione normalizzata, $|\mathbf{a}_{\mathrm{gal}}| / (G M_{\mathrm{gal}} / a_{\mathrm{gal}}^2)$"
    )
    ax_acc.set_xlim(MIN_NORMALIZED_RADIUS, MAX_NORMALIZED_RADIUS)
    ax_acc.set_ylim(bottom=0.0, top=0.45)
    ax_acc.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight")
    plt.close(figure)

    print(f"Massa della galassia: {GALAXY_MASS}")
    print(f"Raggio della galassia: {GALAXY_RADIUS}")
    print(f"Intervallo radiale: 0 - {MAX_NORMALIZED_RADIUS * GALAXY_RADIUS}")
    print(f"Grafico salvato in: {OUTPUT_PATH}")


def main():
    plot_galactic_plummer()


if __name__ == "__main__":
    main()
