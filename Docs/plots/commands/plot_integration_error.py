#!/usr/bin/env python3
"""Confronta l'errore relativo dell'energia per Leapfrog e IAS15."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Parametri fisici della simulazione.
G = 1.0
GALAXY_MASS = 2000.0
GALAXY_RADIUS = 10.0
SOFTENING = 0.01

# Soglia di errore di riferimento.
ERROR_THRESHOLD = 1e-3

# Numero massimo di snapshot utilizzati per ogni simulazione.
MAX_SNAPSHOTS = 200

OUTPUT = Path("Docs/plots/energy_error.pdf")


def json_file(path):
    path = Path(path)

    if path.is_file():
        return path

    if path.is_dir():
        files = sorted(path.glob("*.json"))

        if len(files) == 1:
            return files[0]

        raise ValueError(
            f"Atteso un solo file .json in {path}; trovati {len(files)}."
        )

    raise FileNotFoundError(path)


def read_sampled_snapshots(path):
    """Legge gli snapshot da un file JSON Lines, campionandoli uniformemente."""
    path = json_file(path)

    with path.open(encoding="utf-8") as file:
        first = next((line for line in file if line.strip()), None)

        if first is None or "metadata" not in json.loads(first):
            raise ValueError(f"Formato JSON Lines inatteso: {path}")

        count = sum(1 for line in file if line.strip())

    if count == 0:
        raise ValueError(f"Nessuno snapshot trovato in {path}.")

    selected = set(
        np.linspace(
            0,
            count - 1,
            min(count, MAX_SNAPSHOTS),
            dtype=int,
        )
    )

    snapshots = []

    with path.open(encoding="utf-8") as file:
        next(file)  # Salta la riga dei metadati.

        index = 0

        for line in file:
            if not line.strip():
                continue

            if index in selected:
                snapshots.append(json.loads(line))

            index += 1

    return path, snapshots


def state_arrays(snapshot):
    """Estrae posizioni, velocità e masse di tutte le stelle."""
    stars = snapshot["entities"]

    positions = np.array(
        [[star["x"], star["y"], star["z"]] for star in stars],
        dtype=float,
    )

    velocities = np.array(
        [[star["vx"], star["vy"], star["vz"]] for star in stars],
        dtype=float,
    )

    masses = np.array(
        [star["mass"] for star in stars],
        dtype=float,
    )

    return float(snapshot["t"]), positions, velocities, masses


def total_energy(positions, velocities, masses):
    """Calcola l'energia cinetica e potenziale del sistema."""
    kinetic = 0.5 * np.sum(
        masses * np.sum(velocities**2, axis=1)
    )

    # Energia potenziale gravitazionale tra le stelle.
    potential_stars = 0.0
    number_of_stars = len(masses)

    for start in range(0, number_of_stars - 1, 64):
        stop = min(start + 64, number_of_stars - 1)

        rows = np.arange(start, stop)[:, None]
        columns = np.arange(number_of_stars)[None, :]

        displacement = (
            positions[start:stop, None, :]
            - positions[None, :, :]
        )

        distances = np.sqrt(
            np.sum(displacement**2, axis=2) + SOFTENING**2
        )

        pair_masses = (
            masses[start:stop, None]
            * masses[None, :]
        )

        # Ogni coppia di stelle viene contata una sola volta.
        mask = columns > rows

        potential_stars -= G * np.sum(
            np.where(mask, pair_masses / distances, 0.0)
        )

    # Energia potenziale dovuta alla galassia, modellata
    # mediante un potenziale di Plummer.
    radius_squared = np.sum(positions**2, axis=1)

    potential_galaxy = -G * GALAXY_MASS * np.sum(
        masses
        / np.sqrt(radius_squared + GALAXY_RADIUS**2)
    )

    return float(kinetic + potential_stars + potential_galaxy)


def load_curve(source):
    """Carica una simulazione e calcola l'energia a ogni snapshot selezionato."""
    path, snapshots = read_sampled_snapshots(source)

    times = []
    energies = []
    initial_state = None

    for index, snapshot in enumerate(snapshots):
        time, positions, velocities, masses = state_arrays(snapshot)

        if index == 0:
            initial_state = np.column_stack(
                (positions, velocities, masses)
            )

        energy = total_energy(positions, velocities, masses)

        times.append(time)
        energies.append(energy)

    times = np.asarray(times)
    energies = np.asarray(energies)

    order = np.argsort(times)

    return (
        path,
        times[order],
        energies[order],
        initial_state,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "leapfrog_001",
        type=Path,
        help="File o cartella JSON: Leapfrog con dt=0.01",
    )

    parser.add_argument(
        "leapfrog_0001",
        type=Path,
        help="File o cartella JSON: Leapfrog con dt=0.001",
    )

    parser.add_argument(
        "ias15",
        type=Path,
        help="File o cartella JSON: simulazione IAS15",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
        help="Percorso del PDF da generare",
    )

    args = parser.parse_args()

    runs = [
        (r"Leapfrog ($\Delta t=0.01$)", args.leapfrog_001),
        (r"Leapfrog ($\Delta t=0.001$)", args.leapfrog_0001),
        ("IAS15", args.ias15),
    ]

    curves = []
    reference_state = None

    for label, source in runs:
        path, times, energies, initial_state = load_curve(source)

        if reference_state is None:
            reference_state = initial_state
        elif (
            initial_state.shape != reference_state.shape
            or not np.allclose(
                initial_state,
                reference_state,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError(
                "Le tre simulazioni non partono dallo stesso stato iniziale. "
                "Usa le stesse posizioni, velocità e masse."
            )

        if energies[0] == 0:
            raise ValueError(
                f"Energia iniziale nulla per la simulazione {label}."
            )

        relative_error = np.abs(
            (energies - energies[0]) / energies[0]
        )

        curves.append(
            (label, times, relative_error, path)
        )

    figure, axis = plt.subplots(figsize=(8.5, 5.2))

    for label, times, error, _ in curves:
        axis.semilogy(
            times,
            np.maximum(error, np.finfo(float).eps),
            label=label,
            linewidth=1.5,
        )

    axis.axhline(
        ERROR_THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=fr"Soglia di riferimento ($\epsilon={ERROR_THRESHOLD:.0e}$)",
    )

    axis.set_xlabel("Tempo di simulazione")
    axis.set_ylabel(
        r"Errore relativo dell'energia "
        r"$|E(t)-E(0)|/|E(0)|$"
    )
    axis.set_title("Conservazione dell'energia nei metodi di integrazione")

    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False)

    figure.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(
        args.output,
        format="pdf",
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Grafico salvato in: {args.output}")
    print(
        f"Parametri: G={G}, M_gal={GALAXY_MASS}, "
        f"R_gal={GALAXY_RADIUS}, softening={SOFTENING}"
    )

    for label, _, error, path in curves:
        print(
            f"{label}: {path} "
            f"(errore massimo sugli snapshot campionati: "
            f"{np.max(error):.3e})"
        )


if __name__ == "__main__":
    main()
