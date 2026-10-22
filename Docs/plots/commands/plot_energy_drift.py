#!/usr/bin/env python3

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLOTS_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PLOTS_DIR / "energy_drift.pdf"

INPUT_FILES = {
    0.01: PROJECT_ROOT / "clusters" / "run_003" / "GTS" / "JSON" / "90418662_a300.0_e0.6.json",
    0.005: PROJECT_ROOT / "clusters" / "run_004" / "GTS" / "JSON" / "90418662_a300.0_e0.6.json",
}


def find_value(data, names):
    """Find a metadata value by key, including nested dictionaries."""
    if isinstance(data, dict):
        for name in names:
            if name in data:
                return data[name]

        for value in data.values():
            result = find_value(value, names)
            if result is not None:
                return result

    return None


def entity_array(entities, field, components):
    """Extract positions, velocities or masses from a snapshot."""
    values = []

    for entity in entities:
        value = entity.get(field)

        if value is not None:
            if isinstance(value, dict):
                values.append([value[key] for key in components])
            elif isinstance(value, (list, tuple)):
                values.append(value)
            else:
                values.append(value)
            continue

        values.append([entity[key] for key in components])

    return np.asarray(values, dtype=np.float64)


def percentile_center_of_mass_radius(snapshot):
    """Calculate the percentile COM distance from the galactic center."""
    entities = snapshot["entities"]

    if not entities:
        return np.nan

    positions = entity_array(
        entities, "position", ("x", "y", "z")
    )
    masses = entity_array(entities, "mass", ())

    center_position = np.average(
        positions, axis=0, weights=masses
    )

    for percentile in (80, 65, 50, 35, 20):
        distances = np.linalg.norm(
            positions - center_position,
            axis=1
        )

        percentile_index = int(
            len(distances) * percentile / 100
        )

        maximum_distance = np.partition(
            distances, percentile_index
        )[percentile_index]

        inside = distances <= maximum_distance

        if np.sum(inside) < 5:
            break

        positions = positions[inside]
        masses = masses[inside]

        center_position = np.average(
            positions, axis=0, weights=masses
        )

    return np.linalg.norm(center_position)


def total_energy(snapshot, gravitational_constant, softening,
                 galaxy_mass, galaxy_radius):
    """Reproduce the total-energy calculation in verify_physics.py."""
    entities = snapshot["entities"]

    if not entities:
        return 0.0

    positions = entity_array(
        entities, "position", ("x", "y", "z")
    )
    velocities = entity_array(
        entities, "velocity", ("vx", "vy", "vz")
    )
    masses = entity_array(entities, "mass", ())

    total_mass = np.sum(masses)

    if total_mass <= 0:
        raise ValueError("Snapshot has no positive total mass.")

    center_velocity = np.average(
        velocities, axis=0, weights=masses
    )

    relative_velocities = velocities - center_velocity

    kinetic_internal = 0.5 * np.sum(
        masses * np.sum(relative_velocities**2, axis=1)
    )

    kinetic_com = (
        0.5 * total_mass * np.sum(center_velocity**2)
    )

    differences = positions[:, None, :] - positions[None, :, :]

    distances = np.sqrt(
        np.sum(differences**2, axis=2) + softening**2
    )

    np.fill_diagonal(distances, np.inf)

    mass_products = masses[:, None] * masses[None, :]
    upper_indices = np.triu_indices(len(masses), k=1)

    potential_internal = (
        -gravitational_constant
        * np.sum(mass_products[upper_indices] / distances[upper_indices])
    )

    radii = np.linalg.norm(positions, axis=1)

    potential_galaxy = (
        -gravitational_constant
        * galaxy_mass
        * np.sum(
            masses / np.sqrt(radii**2 + galaxy_radius**2)
        )
    )

    return (
        kinetic_internal
        + potential_internal
        + kinetic_com
        + potential_galaxy
    )


def load_history(path):
    """Read metadata and calculate energy and percentile COM per snapshot."""
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        first_line = file.readline()

        if not first_line.strip():
            raise ValueError(f"Empty JSON file: {path}")

        metadata = json.loads(first_line)

        gravitational_constant = find_value(
            metadata, ("G", "gravitational_constant")
        )
        softening = find_value(
            metadata, ("softening", "softening_eps")
        )
        galaxy_mass = find_value(
            metadata, ("galaxy_mass",)
        )
        galaxy_radius = find_value(
            metadata, ("galaxy_radius",)
        )
        orbit_period = find_value(
            metadata, ("orbit_period", "period")
        )

        parameters = {
            "G": gravitational_constant,
            "softening": softening,
            "galaxy_mass": galaxy_mass,
            "galaxy_radius": galaxy_radius,
        }

        missing = [
            name for name, value in parameters.items()
            if value is None
        ]

        if missing:
            raise ValueError(
                f"Missing parameters in {path}: {', '.join(missing)}. "
                "Check the metadata in the first JSON line."
            )

        times = []
        energies = []
        percentile_com_radii = []

        for line in file:
            if not line.strip():
                continue

            snapshot = json.loads(line)

            times.append(float(snapshot["t"]))

            energies.append(
                total_energy(
                    snapshot,
                    float(gravitational_constant),
                    float(softening),
                    float(galaxy_mass),
                    float(galaxy_radius),
                )
            )

            percentile_com_radii.append(
                percentile_center_of_mass_radius(snapshot)
            )

    if not times:
        raise ValueError(f"No snapshots found in {path}")

    times = np.asarray(times)
    energies = np.asarray(energies)
    percentile_com_radii = np.asarray(percentile_com_radii)

    order = np.argsort(times)
    times = times[order]
    energies = energies[order]
    percentile_com_radii = percentile_com_radii[order]

    initial_energy = energies[0]

    if initial_energy == 0:
        raise ValueError(f"Initial energy is zero in {path}")

    drift_percent = (
        (energies - initial_energy)
        / abs(initial_energy)
        * 100.0
    )

    return (
        times,
        energies,
        drift_percent,
        percentile_com_radii,
        orbit_period,
    )


def configure_plot_style():
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


def main():
    configure_plot_style()

    fig, (ax_energy, ax_radius) = plt.subplots(
        2,
        1,
        figsize=(9, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1]},
    )

    fig.subplots_adjust(
        left=0.13,
        right=0.95,
        top=0.95,
        bottom=0.16,
        hspace=0.12,
    )

    max_time = 0.0
    max_drift = 0.0
    orbit_period = None

    for dt_val, path in INPUT_FILES.items():
        (
            times,
            energies,
            drift,
            percentile_com_radii,
            period,
        ) = load_history(path)

        if period is not None:
            orbit_period = period

        maximum_drift = np.max(np.abs(drift))
        final_drift = drift[-1]

        max_time = max(max_time, np.max(times))
        max_drift = max(max_drift, np.max(np.abs(drift)))

        print(f"\ndt = {dt_val}")
        print(f"Input: {path}")
        print(f"Snapshots: {len(times)}")
        print(f"Initial energy: {energies[0]:.6e}")
        print(f"Final energy: {energies[-1]:.6e}")
        print(f"Maximum absolute energy drift: {maximum_drift:.6f}%")
        print(f"Final energy drift: {final_drift:+.6f}%")

        label = rf"$\Delta t = {dt_val}$"

        energy_line, = ax_energy.plot(
            times,
            drift,
            linewidth=1.4,
            label=label,
        )

        ax_radius.plot(
            times,
            percentile_com_radii,
            linewidth=1.4,
            color=energy_line.get_color(),
            label=label,
        )

    # Mark completed orbital periods on both plots.
    if orbit_period is not None and orbit_period > 0:
        t_current = orbit_period
        first_line = True

        while t_current < max_time:
            ax_energy.axvline(
                t_current,
                color="gray",
                linestyle=":",
                linewidth=0.9,
                alpha=0.6,
                label="Orbita completata" if first_line else "",
            )

            ax_radius.axvline(
                t_current,
                color="gray",
                linestyle=":",
                linewidth=0.9,
                alpha=0.6,
            )

            first_line = False
            t_current += orbit_period

    ax_energy.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )

    ax_energy.set_ylabel(
        r"Deriva energetica, $\Delta E / |E_0|$ (\%)"
    )
    ax_energy.set_title(
        r"Deriva energetica e distanza dal centro galattico"
    )
    ax_energy.set_ylim(bottom=0.0, top=max_drift * 1.1)
    ax_energy.grid(True, alpha=0.3)

    ax_radius.set_xlabel(r"Tempo, $t$")
    ax_radius.set_ylabel(
        r"Distanza del centro di massa, $R_{\mathrm{COM}}$"
    )
    ax_radius.set_xlim(left=0.0, right=max_time * 1.02)
    ax_radius.grid(True, alpha=0.3)

    handles, labels = ax_energy.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.025),
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

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"\nPlot saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
