#!/usr/bin/env python3
"""Mostra la deriva energetica relativa per Leapfrog con dt=0.005."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


G = 1.0
GALAXY_MASS = 2000.0
GALAXY_RADIUS = 10.0
SOFTENING = 0.01
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

    selected = set(np.linspace(0, count - 1, min(count, MAX_SNAPSHOTS), dtype=int))
    snapshots = []

    with path.open(encoding="utf-8") as file:
        next(file)

        for index, line in enumerate(line for line in file if line.strip()):
            if index in selected:
                snapshots.append(json.loads(line))

    return path, snapshots


def state_arrays(snapshot):
    """Estrae posizioni, velocità e masse di tutte le stelle."""
    stars = snapshot["entities"]
    positions = np.array([[star["x"], star["y"], star["z"]] for star in stars], dtype=float)
    velocities = np.array([[star["vx"], star["vy"], star["vz"]] for star in stars], dtype=float)
    masses = np.array([star["mass"] for star in stars], dtype=float)

    return float(snapshot["t"]), positions, velocities, masses


def total_energy(positions, velocities, masses):
    """Calcola l'energia cinetica e potenziale del sistema."""
    kinetic = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
    potential_stars = 0.0
    number_of_stars = len(masses)

    for start in range(0, number_of_stars - 1, 64):
        stop = min(start + 64, number_of_stars - 1)
        rows = np.arange(start, stop)[:, None]
        columns = np.arange(number_of_stars)[None, :]
        displacement = positions[start:stop, None, :] - positions[None, :, :]
        distances = np.sqrt(np.sum(displacement**2, axis=2) + SOFTENING**2)
        pair_masses = masses[start:stop, None] * masses[None, :]
        mask = columns > rows
        potential_stars -= G * np.sum(np.where(mask, pair_masses / distances, 0.0))

    radius_squared = np.sum(positions**2, axis=1)
    potential_galaxy = -G * GALAXY_MASS * np.sum(
        masses / np.sqrt(radius_squared + GALAXY_RADIUS**2)
    )

    return float(kinetic + potential_stars + potential_galaxy)


def load_curve(source):
    """Carica una simulazione e calcola l'energia agli snapshot selezionati."""
    path, snapshots = read_sampled_snapshots(source)
    times = []
    energies = []

    for snapshot in snapshots:
        time, positions, velocities, masses = state_arrays(snapshot)
        times.append(time)
        energies.append(total_energy(positions, velocities, masses))

    times = np.asarray(times)
    energies = np.asarray(energies)
    order = np.argsort(times)

    return path, times[order], energies[order]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("simulation", type=Path, help="File o cartella JSON della simulazione con dt=0.005")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="Percorso del PDF da generare")
    args = parser.parse_args()

    path, times, energies = load_curve(args.simulation)

    if len(energies) == 0 or energies[0] == 0:
        raise ValueError("Energia iniziale nulla o nessuno snapshot disponibile.")

    relative_error = np.abs((energies - energies[0]) / energies[0])

    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.semilogy(times, np.maximum(relative_error, np.finfo(float).eps), linewidth=1.5)
    axis.set_xlabel("Tempo di simulazione")
    axis.set_ylabel(r"Errore relativo dell'energia $|E(t)-E(0)|/|E(0)|$")
    axis.set_title(r"Deriva energetica con Leapfrog ($\Delta t=0.005$)")
    axis.grid(True, which="both", alpha=0.25)
    figure.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, format="pdf", bbox_inches="tight")
    plt.close(figure)

    print(f"Grafico salvato in: {args.output}")
    print(f"Simulazione: {path}")
    print(f"Errore massimo sugli snapshot campionati: {np.max(relative_error):.3e}")


if __name__ == "__main__":
    main()
