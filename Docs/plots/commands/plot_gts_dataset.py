
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_DIR / "Docs" / "plots" / "ml_dataset.csv"
CONFIG_PATH = PROJECT_DIR / "configuration.json"
OUTPUT_DIR = PROJECT_DIR / "Docs" / "plots"

RADIUS_MAX = 500

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "savefig.dpi": 300,
})


def main():
    if not DATASET_PATH.exists():
        sys.exit(f"Dataset not found: {DATASET_PATH}")

    if not CONFIG_PATH.exists():
        sys.exit(f"Configuration not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, encoding="utf-8") as file:
        config = json.load(file)

    number_of_stars = config["cluster_generation"]["number_of_stars"]

    data = np.genfromtxt(
        DATASET_PATH,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    data = np.atleast_1d(data)

    if data.size == 0:
        sys.exit("Dataset is empty.")

    semi_major_axis = data["orbit_semi_major_axis"]
    eccentricity = data["orbit_eccentricity"]
    bound_fraction = data["bound_mass_fraction"]
    half_mass_radius = data["half_mass_radius"]

    valid = (
        np.isfinite(semi_major_axis)
        & np.isfinite(eccentricity)
        & np.isfinite(bound_fraction)
        & np.isfinite(half_mass_radius)
    )

    semi_major_axis = semi_major_axis[valid]
    eccentricity = eccentricity[valid]
    bound_fraction = bound_fraction[valid]
    half_mass_radius = half_mass_radius[valid]

    if len(semi_major_axis) == 0:
        sys.exit("No valid entries found in the dataset.")

    axes_a = np.unique(semi_major_axis)
    axes_e = np.unique(eccentricity)

    bound_grid = np.full((len(axes_e), len(axes_a)), np.nan)
    radius_grid = np.full((len(axes_e), len(axes_a)), np.nan)

    for i, e in enumerate(axes_e):
        for j, a in enumerate(axes_a):
            mask = (semi_major_axis == a) & (eccentricity == e)

            if np.any(mask):
                bound_grid[i, j] = np.mean(bound_fraction[mask])
                radius_grid[i, j] = np.mean(half_mass_radius[mask])

    def save_heatmap(values, title, colorbar_label, filename, vmin, vmax):
        fig, ax = plt.subplots(figsize=(9, 6))

        image = ax.imshow(
            values,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            extent=(
                axes_a[0],
                axes_a[-1],
                axes_e[0],
                axes_e[-1],
            ),
        )

        ax.set_xlabel("Semi-major axis $A$")
        ax.set_ylabel("Eccentricity $e$")
        ax.set_title(title)

        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label(colorbar_label)

        fig.text(
            0.98,
            0.98,
            rf"$N = {number_of_stars}$",
            ha="right",
            va="top",
        )

        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
        plt.close(fig)

    save_heatmap(
        bound_grid,
        "Bound mass fraction",
        "Bound mass fraction",
        "bound_mass_fraction.pdf",
        vmin=0,
        vmax=1,
    )

    save_heatmap(
        radius_grid,
        "Half-mass radius",
        "Half-mass radius",
        "half_mass_radius.pdf",
        vmin=0,
        vmax=RADIUS_MAX,
    )

    print(f"Heatmaps saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
