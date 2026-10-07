import csv
import logging
import os

import io_manager

from cluster_diagnostics import ClusterDiagnostics
from simulation import Simulation


logger = logging.getLogger(__name__)


class ML_Dataset_Generation:

    def __init__(
        self,
        gts_json_dir_path
    ):
        self.gts_json_dir_path = gts_json_dir_path

    def _create_simulation(
        self,
        metadata,
        entities
    ):
        simulation = Simulation(metadata.simulation_config)

        for entity in entities:
            simulation.add_entity(entity)

        return simulation

    def _get_dataset_row(
        self,
        gts_json_file_path
    ):
        metadata, entities, _ = io_manager.load_json_snapshot(
            gts_json_file_path
        )

        simulation = self._create_simulation(
            metadata,
            entities
        )

        diagnostics = ClusterDiagnostics(simulation)

        total_mass = sum(
            entity.mass
            for entity in entities
        )

        bound_mass = diagnostics.get_bound_mass()

        bound_mass_fraction = (
            bound_mass / total_mass
            if total_mass != 0.0
            else 0.0
        )

        half_mass_radius = (
            diagnostics.get_half_mass_radius()
        )

        return (
            metadata.cluster_seed,
            metadata.orbit_semi_major_axis,
            metadata.orbit_eccentricity,
            bound_mass_fraction,
            half_mass_radius
        )

    def run(self):

        gts_json_files_path = sorted(
            os.path.join(
                self.gts_json_dir_path,
                filename
            )
            for filename in os.listdir(
                self.gts_json_dir_path
            )
            if filename.endswith(".json")
        )

        if not gts_json_files_path:
            logger.error(
                f"No GTS JSON files were found "
                f"in {self.gts_json_dir_path}!"
            )
            return

        csv_path = io_manager.get_csv_path()

        with open(
            csv_path,
            "w",
            newline=""
        ) as csv_file:

            writer = csv.writer(csv_file)

            writer.writerow([
                "cluster_seed",
                "orbit_semi_major_axis",
                "orbit_eccentricity",
                "bound_mass_fraction",
                "half_mass_radius"
            ])

            for gts_json_file_path in gts_json_files_path:
                writer.writerow(
                    self._get_dataset_row(
                        gts_json_file_path
                    )
                )

        logger.info(
            f"ML dataset generated: {csv_path}"
        )
