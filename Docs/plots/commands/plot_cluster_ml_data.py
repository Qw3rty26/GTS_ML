#!/usr/bin/env python3

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_PATH = SCRIPT_PATH.parents[3]
OUTPUT_PATH = SCRIPT_PATH.parents[1] / "cluster_evolution.pdf"

RUNS = [
    {
        "path": PROJECT_PATH / "clusters" / "run_001" / "GTS" / "JSON" / "90418662_a300.0_e0.3.json",
        "label": r"$e = 0.3$",
        "color": "tab:blue",
    },
    {
        "path": PROJECT_PATH / "clusters" / "run_002" / "GTS" / "JSON" / "90418662_a300.0_e0.6.json",
        "label": r"$e = 0.6$",
        "color": "tab:orange",
    },
]


def configure_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 13,         # Aumentato da 11 a 13
        "axes.titlesize": 15,    # Aumentato da 13 a 15
        "axes.labelsize": 14,    # Aumentato da 12 a 14
        "xtick.labelsize": 12,   # Aumentato da 10 a 12
        "ytick.labelsize": 12,   # Aumentato da 10 a 12
        "legend.fontsize": 13,   # Aumentato da 11 a 13
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "mathtext.fontset": "dejavuserif",
    })


def load_snapshots(path):
    if not path.is_file():
        raise FileNotFoundError(f"File non trovato: {path}")

    with path.open("r", encoding="utf-8") as file:
        lines = [json.loads(line) for line in file if line.strip()]

    if not lines:
        raise ValueError(f"File JSON vuoto: {path}")

    metadata = lines[0]["metadata"]
    snapshots = lines[1:]

    return metadata, snapshots


def calculate_diagnostics(snapshot, gravitational_constant, softening):
    entities = snapshot["entities"]

    if not entities:
        return 0.0, 0.0

    positions = np.array([
        [entity["x"], entity["y"], entity["z"]]
        for entity in entities
    ], dtype=np.float64)

    velocities = np.array([
        [entity["vx"], entity["vy"], entity["vz"]]
        for entity in entities
    ], dtype=np.float64)

    masses = np.array([
        entity["mass"] for entity in entities
    ], dtype=np.float64)

    center_of_mass = snapshot["center_of_mass"]
    center_position = np.array([
        center_of_mass["x"],
        center_of_mass["y"],
        center_of_mass["z"],
    ])

    center_velocity = np.array([
        center_of_mass["vx"],
        center_of_mass["vy"],
        center_of_mass["vz"],
    ])

    # Half-mass radius
    distances = np.linalg.norm(positions - center_position, axis=1)
    sorted_indices = np.argsort(distances)
    cumulative_mass = np.cumsum(masses[sorted_indices])
    half_mass_index = np.searchsorted(cumulative_mass, masses.sum() / 2)
    half_mass_radius = distances[sorted_indices[half_mass_index]]

    # Bound mass fraction
    relative_velocities = velocities - center_velocity
    kinetic_energies = (
        0.5 * masses * np.sum(relative_velocities**2, axis=1)
    )

    position_differences = positions[:, None, :] - positions[None, :, :]
    distances_squared = (
        np.sum(position_differences**2, axis=2) + softening**2
    )
    distances_matrix = np.sqrt(distances_squared)
    np.fill_diagonal(distances_matrix, np.inf)

    potential_energies = (
        -gravitational_constant
        * masses
        * np.sum(masses[None, :] / distances_matrix, axis=1)
    )

    bound_mask = (kinetic_energies + potential_energies) < 0.0
    bound_mass_fraction = masses[bound_mask].sum() / masses.sum()

    return bound_mass_fraction, half_mass_radius


def smooth_signal(values, window_length=21, polyorder=3):
    """Filtro di Savitzky-Golay per rimuovere lo zig-zag preservando il trend."""
    if len(values) < window_length:
        return values
    if window_length % 2 == 0:
        window_length += 1
    return savgol_filter(values, window_length=window_length, polyorder=polyorder)


def main():
    configure_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    max_time = 0.0

    for run in RUNS:
        print(f"\nInput: {run['path']}")

        metadata, snapshots = load_snapshots(run["path"])

        times = []
        bound_fractions = []
        half_mass_radii = []

        for snapshot in snapshots:
            f_bound, r_half = calculate_diagnostics(
                snapshot,
                gravitational_constant=metadata["G"],
                softening=metadata["softening"],
            )

            times.append(snapshot["t"])
            bound_fractions.append(f_bound)
            half_mass_radii.append(r_half)

        times = np.array(times)
        bound_fractions = np.array(bound_fractions)
        half_mass_radii = np.array(half_mass_radii)

        max_time = max(max_time, np.max(times))

        # Smoothing della frazione legata per eliminare lo zig-zag
        smoothed_bound_fractions = smooth_signal(
            bound_fractions, window_length=21, polyorder=3
        )

        axes[0].plot(
            times,
            smoothed_bound_fractions,
            color=run["color"],
            linewidth=1.6,
            label=run["label"],
        )
        axes[1].plot(
            times,
            half_mass_radii,
            color=run["color"],
            linewidth=1.6,
            label=run["label"],
        )

    # --- Pannello 1: Frazione di massa legata ---
    axes[0].set_title("Evoluzione della frazione di massa legata")
    axes[0].set_xlabel(r"Tempo, $t$")
    axes[0].set_ylabel(r"Frazione di massa legata, $f_{\mathrm{bound}}$")
    axes[0].set_xlim(left=0.0, right=max_time * 1.02)
    axes[0].set_ylim(-0.02, 1.05)
    axes[0].grid(True, alpha=0.3)

    # --- Pannello 2: Raggio di metà massa ---
    axes[1].set_title("Evoluzione del raggio di metà massa")
    axes[1].set_xlabel(r"Tempo, $t$")
    axes[1].set_ylabel(r"Raggio di metà massa, $r_{1/2}$")
    axes[1].set_xlim(left=0.0, right=max_time * 1.02)
    axes[1].set_ylim(bottom=0.0)
    axes[1].grid(True, alpha=0.3)

    # Legenda unica centrata in basso
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.02),
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

    fig.subplots_adjust(
        left=0.08,
        right=0.96,
        top=0.88,
        bottom=0.22,
        wspace=0.22,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"\nPlot saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
