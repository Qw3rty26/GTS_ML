#!/usr/bin/env python3

from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Percorsi del progetto
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_PATH = SCRIPT_PATH.parents[3]
CLUSTERS_PATH = PROJECT_PATH / "clusters"
OUTPUT_PATH = SCRIPT_PATH.parents[1] / "orbit_comparison.pdf"

# Colori standard per il confronto delle ultime due simulazioni
COLORS = ["tab:blue", "tab:orange"]


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


def find_latest_runs(clusters_path, n=2):
    """Individua automaticamente le ultime n run presenti nella cartella clusters."""
    if not clusters_path.exists():
        raise FileNotFoundError(f"Cartella clusters non trovata: {clusters_path}")

    run_dirs = sorted([
        d.name for d in clusters_path.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ])

    if len(run_dirs) < n:
        raise ValueError(f"Trovate solo {len(run_dirs)} run, ne servono almeno {n}.")

    return run_dirs[-n:]


def find_gts_json(run_directory):
    """Individua il file JSON GTS della simulazione."""
    json_directory = CLUSTERS_PATH / run_directory / "GTS" / "JSON"
    json_files = list(json_directory.glob("*.json"))

    if len(json_files) != 1:
        raise FileNotFoundError(
            f"Atteso un solo file JSON in {json_directory}, "
            f"trovati {len(json_files)}."
        )

    return json_files[0]


def load_gts_trajectory(json_path):
    """Legge i metadati e le coordinate senza caricare tutti gli snapshot."""
    times = []
    positions_x = []
    positions_y = []

    with json_path.open("r", encoding="utf-8") as json_file:
        first_line = json_file.readline()

        if not first_line:
            raise ValueError(f"File JSON vuoto: {json_path}")

        metadata = json.loads(first_line)["metadata"]

        for line in json_file:
            if not line.strip():
                continue

            snapshot = json.loads(line)
            center = snapshot["center_of_mass"]

            times.append(snapshot["t"])
            positions_x.append(center["x"])
            positions_y.append(center["y"])

    if not times:
        raise ValueError(f"Nessuno snapshot trovato in {json_path}")

    return metadata, times, positions_x, positions_y


def plot_orbits(simulations):
    """Confronta le traiettorie orbitali nel piano x-y."""
    configure_plot_style()

    fig = plt.figure(figsize=(9, 8.5))
    ax = fig.add_axes([0.13, 0.22, 0.82, 0.70])

    for i, simulation in enumerate(simulations):
        x = simulation["x"]
        y = simulation["y"]
        color = simulation["colore"]
        eccentricity = simulation["eccentricita"]
        model_label = simulation["model_label"]

        ax.plot(
            x,
            y,
            color=color,
            linewidth=1.5,
            label=f"{model_label} ($e = {eccentricity}$)",
        )

        # Posizione iniziale
        ax.scatter(
            x[0],
            y[0],
            marker="*",
            s=140,
            color=color,
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )

        # Posizione finale
        ax.scatter(
            x[-1],
            y[-1],
            marker="o",
            s=40,
            color=color,
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )

    # Centro della galassia
    ax.scatter(
        0,
        0,
        marker="*",
        s=200,
        color="tab:green",
        edgecolors="black",
        linewidths=0.6,
        zorder=6,
        label="Centro galattico",
    )

    # Elementi aggiuntivi per la legenda (posizioni iniziale e finale)
    marker_initial = Line2D(
        [0], [0], marker='*', color='w', markerfacecolor='gray',
        markeredgecolor='black', markersize=10, label="Posizione iniziale"
    )
    marker_final = Line2D(
        [0], [0], marker='o', color='w', markerfacecolor='gray',
        markeredgecolor='black', markersize=7, label="Posizione finale"
    )

    ax.set_title("Confronto delle traiettorie orbitali")
    ax.set_xlabel(r"Posizione $x$")
    ax.set_ylabel(r"Posizione $y$")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    handles.extend([marker_initial, marker_final])
    labels.extend(["Posizione iniziale", "Posizione finale"])

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.03),
        frameon=True,
        fancybox=False,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
        fontsize=11,
        handlelength=2.0,
        borderpad=0.8,
        labelspacing=0.5,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"\nGrafico salvato in: {OUTPUT_PATH}")


def main():
    latest_runs = find_latest_runs(CLUSTERS_PATH, n=2)
    simulations = []

    for i, run_dir in enumerate(latest_runs):
        json_path = find_gts_json(run_dir)
        metadata, times, positions_x, positions_y = load_gts_trajectory(json_path)

        eccentricity = metadata.get("orbit_eccentricity", "?")
        semi_major_axis = metadata.get("orbit_semi_major_axis", "?")
        model_label = f"Modello {i + 1}"

        simulations.append(
            {
                "model_label": model_label,
                "colore": COLORS[i % len(COLORS)],
                "eccentricita": eccentricity,
                "x": positions_x,
                "y": positions_y,
            }
        )

        print(f"\nSimulazione caricata ({run_dir}):")
        print(f"File JSON: {json_path}")
        print(f"Semiasse maggiore: {semi_major_axis}")
        print(f"Eccentricità: {eccentricity}")
        print(f"Snapshot letti: {len(times)}")
        print(
            f"Intervallo temporale: {times[0]:.3f} - {times[-1]:.3f}"
        )

    plot_orbits(simulations)


if __name__ == "__main__":
    main()
