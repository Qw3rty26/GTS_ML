import logging
import numpy as np
import io_manager
from simulation import Simulation
from galactic_potential import GalacticPotential
from io_manager import Metadata


logger = logging.getLogger(__name__)


class Galactic_Tidal_Stripping:

    def __init__(
        self,
        cluster_json_file,
        metadata: Metadata
    ):
        self.metadata = metadata
        self._create_cluster(cluster_json_file)

    def _create_cluster(self, cluster_json_file):
        file_metadata, entities, center_of_mass = (
            io_manager.load_json_snapshot(cluster_json_file)
        )

        for key, value in vars(file_metadata).items():
            if value is not None:
                setattr(self.metadata, key, value)

        self.simulation = Simulation(
            self.metadata.simulation_config
        )

        for entity in entities:
            self.simulation.add_entity(entity)

        galactic_potential = GalacticPotential(
            self.metadata.galaxy_radius,
            self.metadata.galaxy_mass
        )

        self.simulation.add_galactic_potential(
            galactic_potential
        )

        r_apo = (
            self.metadata.orbit_semi_major_axis
            * (1.0 + self.metadata.orbit_eccentricity)
        )

        v_apo = galactic_potential.get_elliptical_apocenter_velocity(
            self.metadata.orbit_semi_major_axis,
            self.metadata.orbit_eccentricity
        )

        percentile_center_of_mass = (
            self.simulation.get_percentile_center_of_mass()
        )

        self.simulation.move_cluster(
            -percentile_center_of_mass.position.x,
            -percentile_center_of_mass.position.y,
            -percentile_center_of_mass.position.z
        )

        self.simulation.speed_cluster(
            -percentile_center_of_mass.velocity.vx,
            -percentile_center_of_mass.velocity.vy,
            -percentile_center_of_mass.velocity.vz
        )

        self.simulation.move_cluster(
            r_apo,
            0,
            0
        )

        self.simulation.speed_cluster(
            0,
            v_apo,
            0
        )

    def run(self):

        dt_snapshot = 0.1
        orbit_check_margin = 1.0

        orbits = 0
        last_orbit_time = self.simulation.get_time()

        # T = 2 * pi * sqrt(a^3 / (G * M))
        orbit_period = (
            2.0 * np.pi
            * np.sqrt(
                self.metadata.orbit_semi_major_axis**3
                / (
                    self.metadata.simulation_config.G
                    * self.metadata.galaxy_mass
                )
            )
        )

        next_orbit_check = (
            self.simulation.get_time()
            + orbit_period
            - orbit_check_margin
        )

        next_snapshot = (
            self.simulation.get_time()
            + dt_snapshot
        )

        io_manager.init_json_gts(self.metadata)

        self.metadata.simulation_config.t = (
            self.simulation.get_time()
        )

        entities = self.simulation.get_entities()
        percentile_center_of_mass = (
            self.simulation.get_percentile_center_of_mass()
        )

        io_manager.save_xyzv_snapshot_gts(
            self.metadata,
            entities,
            percentile_center_of_mass
        )

        io_manager.save_json_snapshot_gts(
            self.metadata,
            entities,
            percentile_center_of_mass
        )

        while orbits < self.metadata.number_of_orbits:

            self.simulation.integrate()

            current_time = self.simulation.get_time()

            if current_time >= next_snapshot:

                next_snapshot += dt_snapshot

                self.metadata.simulation_config.t = current_time

                entities = self.simulation.get_entities()
                percentile_center_of_mass = (
                    self.simulation.get_percentile_center_of_mass()
                )

                io_manager.save_xyzv_snapshot_gts(
                    self.metadata,
                    entities,
                    percentile_center_of_mass
                )

                io_manager.save_json_snapshot_gts(
                    self.metadata,
                    entities,
                    percentile_center_of_mass
                )

            if current_time >= next_orbit_check:

                new_orbits = (
                    self.simulation.get_number_of_orbits()
                )

                if new_orbits != orbits:

                    orbits = new_orbits

                    orbit_period = (
                        current_time - last_orbit_time
                    )

                    last_orbit_time = current_time

                    next_orbit_check = (
                        current_time
                        + orbit_period
                        - orbit_check_margin
                    )

                    logger.info(
                        f"\033[94mCluster "
                        f"{self.metadata.cluster_seed}_a"
                        f"{self.metadata.orbit_semi_major_axis}_e"
                        f"{self.metadata.orbit_eccentricity}: "
                        f"{orbits}/"
                        f"{self.metadata.number_of_orbits} "
                        f"orbits done.\033[0m"
                    )

        logger.info(
            f"Cluster "
            f"{self.metadata.cluster_seed}_a"
            f"{self.metadata.orbit_semi_major_axis}_e"
            f"{self.metadata.orbit_eccentricity}: "
            f"Completed."
        )
