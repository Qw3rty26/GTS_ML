
import os
import csv
import argparse
from file_handle import File_Handle

SELECTED_COLUMNS = [
    "cluster_seed",
    "orbit_semi_major_axis",
    "orbit_eccentricity",
    "time_final",
    "cluster_kinetic_energy_initial",
    "cluster_kinetic_energy_final",
    "cluster_potential_energy_final",
    "cluster_softened_potential_energy_initial",
    "cluster_softened_potential_energy_final",
    "galactic_potential_energy_initial",
    "galactic_potential_energy_final",
    "cluster_total_energy_initial",
    "cluster_total_energy_final",
    "cluster_relative_energy_error_final",
    "cluster_kinetic_internal_energy_final",
    "cluster_virial_ratio_initial",
    "cluster_virial_ratio_final",
    "cluster_bound_mass_fraction_final",
    "cluster_half_mass_radius_initial",
    "cluster_half_mass_radius_final",
    "cluster_half_mass_radius_ratio_final",
    "cluster_root_mean_square_radius_final",
    "cluster_angular_momentum_final",
    "cluster_number_of_orbits_done_final"
]

INITIAL_FINAL_COLUMNS = [
    "cluster_kinetic_energy",
    "cluster_softened_potential_energy",
    "galactic_potential_energy",
    "cluster_total_energy",
    "cluster_virial_ratio",
    "cluster_half_mass_radius"
]

FINAL_COLUMNS = [
    "time",
    "cluster_potential_energy",
    "cluster_relative_energy_error",
    "cluster_kinetic_internal_energy",
    "cluster_bound_mass_fraction",
    "cluster_half_mass_radius_ratio",
    "cluster_root_mean_square_radius",
    "cluster_angular_momentum",
    "cluster_number_of_orbits_done"
]

class Generate_Machine_Learning_Dataset:

    def __init__(self, working_directory):
        self.working_directory = working_directory

    def _get_simulations_file_paths(self):
        simulations_file_paths = []

        for root, _, files in os.walk(self.working_directory):
            if "GTS" not in root.split(os.sep):
                continue

            for filename in files:
                if filename.endswith(".tsv") and not filename.endswith("_isolated.tsv"):
                    simulations_file_paths.append(os.path.join(root, filename))

        return simulations_file_paths

    def _save_dataset(self, machine_learning_dataset):
        dataset_file_path = os.path.join(self.working_directory, "machine_learning_dataset.tsv")

        with open(dataset_file_path, "w", newline="", encoding="utf-8") as dataset_file:
            writer = csv.DictWriter(dataset_file, fieldnames=SELECTED_COLUMNS, delimiter="\t")
            writer.writeheader()
            writer.writerows(machine_learning_dataset)

    def run(self):
        simulations_file_paths = self._get_simulations_file_paths()
        dataset = []

        for tsv_file_path in simulations_file_paths:
            initial_snapshot = File_Handle.load_tsv_snapshot(tsv_file_path, 0)
            final_snapshot = File_Handle.load_tsv_snapshot(tsv_file_path, -1)

            row = {
                "cluster_seed": final_snapshot["cluster_seed"],
                "orbit_semi_major_axis": final_snapshot["orbit_semi_major_axis"],
                "orbit_eccentricity": final_snapshot["orbit_eccentricity"]
            }

            for column in INITIAL_FINAL_COLUMNS:
                row[f"{column}_initial"] = initial_snapshot[column]
                row[f"{column}_final"] = final_snapshot[column]

            for column in FINAL_COLUMNS:
                row[f"{column}_final"] = final_snapshot[column]

            dataset.append(row)

        if dataset:
            self._save_dataset(dataset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to the working directory")
    args = parser.parse_args()

    Generate_Machine_Learning_Dataset(os.path.abspath(args.path)).run()


if __name__ == "__main__":
    main()
