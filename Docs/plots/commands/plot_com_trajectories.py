#!/usr/bin/env python3

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_PATH = SCRIPT_PATH.parents[3]
CLUSTERS_PATH = PROJECT_PATH / "clusters"
OUTPUT_PATH = SCRIPT_PATH.parents[1] / "com_trajectories.pdf"

RUNS = [
    {
        "directory": "run_001",
        "label": "Modello 1",
        "color": "tab:blue",
    },
    {
        "directory": "run_002",
        "label": "Modello 2",
        "color": "tab:orange",
    },
]


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


def load_trajectory(json_path):
    """Legge i metadati e le coordinate del centro di massa percentile."""
    times = []
    positions_x = []
    positions_y = []

    with json_path.open("r", encoding="utf-8") as json_file:
        first_line = json_file.readline()

        if not first_line.strip():
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

    return metadata, np.array(times), np.array(positions_x), np.array(positions_y)


def main():
    configure_plot_style()

    fig = plt.figure(figsize=(9, 8.5))
    ax = fig.add_axes([0.13, 0.22, 0.82, 0.70])

    for run in RUNS:
        json_path = find_gts_json(run["directory"])
        metadata, times, x, y = load_trajectory(json_path)

        eccentricity = metadata.get("orbit_eccentricity", "?")

        ax.plot(
            x,
            y,
            color=run["color"],
            linewidth=1.5,
            label=f"{run['label']} ($e = {eccentricity}$)",
        )

        # Posizione iniziale
        ax.scatter(
            x[0],
            y[0],
            marker="*",
            s=140,
            color=run["color"],
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
            color=run["color"],
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )

    # Centro galattico (pallino rosso)
    ax.scatter(
        0,
        0,
        marker="o",
        s=80,
        color="tab:red",
        edgecolors="black",
        linewidths=0.5,
        zorder=6,
        label="Centro galattico",
    )

    ax.set_title("Traiettorie del centro di massa percentile")
    ax.set_xlabel(r"Posizione $x$")
    ax.set_ylabel(r"Posizione $y$")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
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

    print(f"\nPlot saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
