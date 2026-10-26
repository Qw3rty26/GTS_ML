import os
import csv
from file_handle import File_Handle

SELECTED_COLUMNS = ["cluster_seed", "orbit_semi_major_axis", "orbit_eccentricity", "cluster_half_mass_radius", "cluster_bound_mass_fraction"]

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
            final_snapshot = File_Handle.load_tsv_snapshot(tsv_file_path, -1)
            dataset.append({column: final_snapshot[column] for column in SELECTED_COLUMNS})

        if dataset:
            self._save_dataset(dataset)
