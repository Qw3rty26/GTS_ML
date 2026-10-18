import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


JSON_PATH = Path(
    "./clusters/run_001/GTS/JSON/1163923016_a294.0_e0.7.json"
)
OUTPUT_PATH = Path("./Docs/plots/com_trajectories.pdf")


def classical_com(entities):
    positions = np.array([
        [entity["x"], entity["y"], entity["z"]]
        for entity in entities
    ])
    masses = np.array([entity["mass"] for entity in entities])

    if len(masses) == 0 or masses.sum() <= 0:
        raise ValueError("Cannot calculate the classical COM.")

    return np.average(positions, axis=0, weights=masses)


def percentile_com(snapshot):
    center = snapshot["center_of_mass"]
    return np.array([center["x"], center["y"], center["z"]])


def main():
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"File not found: {JSON_PATH}")

    data_records = []

    # Stream line-by-line to prevent RAM exhaustion
    with JSON_PATH.open("r", encoding="utf-8") as file:
        first_line = file.readline()
        if not first_line.strip():
            raise ValueError(f"Empty or invalid metadata in {JSON_PATH}")

        for line in file:
            if not line.strip():
                continue
            snapshot = json.loads(line)
            t = snapshot["t"]
            c_pos = classical_com(snapshot["entities"])
            p_pos = percentile_com(snapshot)
            data_records.append((t, c_pos, p_pos))

    if not data_records:
        raise ValueError(f"No snapshots found in {JSON_PATH}")

    # Sort by time to ensure chronological order
    data_records.sort(key=lambda rec: rec[0])

    times = np.array([rec[0] for rec in data_records])
    classical_positions = np.array([rec[1] for rec in data_records])
    percentile_positions = np.array([rec[2] for rec in data_records])

    # Calculate angles to cleanly split orbits (2*pi per orbit)
    angles = np.arctan2(percentile_positions[:, 1], percentile_positions[:, 0])
    unwrapped_angles = np.unwrap(angles)
    delta_angles = unwrapped_angles - unwrapped_angles[0]

    # Find indices for end of 1st orbit and end of 2nd orbit
    orbit1_indices = np.where(delta_angles < 2.0 * np.pi)[0]
    orbit2_indices = np.where((delta_angles >= 2.0 * np.pi) & (delta_angles <= 4.0 * np.pi))[0]

    # Fallback if dataset is slightly shorter/longer
    if len(orbit1_indices) == 0:
        raise ValueError("Not enough data for the first orbit.")
    
    idx_end_orbit1 = orbit1_indices[-1] + 1
    
    # Slice data for orbit 1 and orbit 2
    t1, c1, p1 = times[:idx_end_orbit1], classical_positions[:idx_end_orbit1], percentile_positions[:idx_end_orbit1]
    
    if len(orbit2_indices) > 0:
        idx_start_orbit2 = orbit1_indices[-1]
        idx_end_orbit2 = orbit2_indices[-1] + 1
        t2, c2, p2 = times[idx_start_orbit2:idx_end_orbit2], classical_positions[idx_start_orbit2:idx_end_orbit2], percentile_positions[idx_start_orbit2:idx_end_orbit2]
    else:
        # If 2nd orbit is incomplete, take whatever is left
        t2, c2, p2 = times[idx_end_orbit1:], classical_positions[idx_end_orbit1:], percentile_positions[idx_end_orbit1:]

    # Setup 1x2 subplots
    figure, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Common initial position
    initial_position = percentile_positions[0]

    # --- Panel 1: First Orbit ---
    axes[0].plot(c1[:, 0], c1[:, 1], color="tab:blue", linewidth=1.5, label="Classical COM")
    axes[0].plot(p1[:, 0], p1[:, 1], color="tab:orange", linewidth=1.5, label="Percentile COM")
    axes[0].scatter(0, 0, color="green", marker="*", s=200, label="Galactic center", zorder=5)
    axes[0].scatter(initial_position[0], initial_position[1], color="red", marker="*", s=200, label="Common initial position", zorder=6)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("First Orbit (Agreement)")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].grid(True, alpha=0.25)

    # --- Panel 2: Second Orbit (Divergence) ---
    axes[1].plot(c2[:, 0], c2[:, 1], color="tab:blue", linewidth=1.5, label="Classical COM")
    axes[1].plot(p2[:, 0], p2[:, 1], color="tab:orange", linewidth=1.5, label="Percentile COM")
    axes[1].scatter(0, 0, color="green", marker="*", s=200, label="Galactic center", zorder=5)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("Second Orbit (Divergence)")
    axes[1].set_aspect("equal", adjustable="datalim")
    axes[1].grid(True, alpha=0.25)

    # Shared legend below
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.02))

    figure.tight_layout(rect=[0, 0.08, 1, 1])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight")
    plt.close(figure)

    print(f"Input: {JSON_PATH}")
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
