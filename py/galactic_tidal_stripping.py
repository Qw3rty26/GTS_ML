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

        logger.info(
            f"Cluster "
            f"{self.metadata.cluster_seed}_a"
            f"{self.metadata.orbit_semi_major_axis}_e"
            f"{self.metadata.orbit_eccentricity}: "
            f"Creating..."
        )

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

        logger.info(
            f"Cluster "
            f"{self.metadata.cluster_seed}_a"
            f"{self.metadata.orbit_semi_major_axis}_e"
            f"{self.metadata.orbit_eccentricity}: "
            f"Running..."
        )

        dt_snapshot = 0.1

        theoretical_orbit_period = (
            2.0 * np.pi
            * np.sqrt(
                self.metadata.orbit_semi_major_axis**3
                / (
                    self.metadata.simulation_config.G
                    * self.metadata.galaxy_mass
                )
            )
        )

        logger.debug(
            f"Cluster "
            f"{self.metadata.cluster_seed}_a"
            f"{self.metadata.orbit_semi_major_axis}_e"
            f"{self.metadata.orbit_eccentricity}: "
            f"Theoretical orbital period: "
            f"{theoretical_orbit_period:.4f}"
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

        last_logged_orbits = 0

        ORBIT_TRACKING_START = 0.8

        next_orbit_completion_time = (
            self.simulation.get_time()
            + theoretical_orbit_period
        )

        next_orbit_tracking_start = (
            self.simulation.get_time()
            + (
                ORBIT_TRACKING_START
                * theoretical_orbit_period
            )
        )

        tracking_orbit = False

        while True:
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

            if (
                not tracking_orbit
                and current_time >= next_orbit_tracking_start
            ):
                tracking_orbit = True

            if tracking_orbit:
                current_orbits = (
                    self.simulation.get_number_of_orbits()
                )

                if current_orbits > last_logged_orbits:
                    last_logged_orbits = current_orbits
                    tracking_orbit = False

                    next_orbit_completion_time = (
                        current_time
                        + theoretical_orbit_period
                    )

                    logger.info(
                        f"Cluster "
                        f"{self.metadata.cluster_seed}_a"
                        f"{self.metadata.orbit_semi_major_axis}_e"
                        f"{self.metadata.orbit_eccentricity}: "
                        f"Orbit {current_orbits}/"
                        f"{self.metadata.number_of_orbits} completed "
                    )

                    if current_orbits >= self.metadata.number_of_orbits:
                        break

                    logger.debug(
                        f"Cluster "
                        f"{self.metadata.cluster_seed}_a"
                        f"{self.metadata.orbit_semi_major_axis}_e"
                        f"{self.metadata.orbit_eccentricity}: "
                        f"next_orbit_completion_time = "
                        f"{next_orbit_completion_time:.2f}, "
                        f"current_t = {current_time:.2f}"
                    )

                    next_orbit_tracking_start = (
                        current_time
                        + (
                            ORBIT_TRACKING_START
                            * theoretical_orbit_period
                        )
                    )

        logger.info(
            f"\033[94mCluster "
            f"{self.metadata.cluster_seed}_a"
            f"{self.metadata.orbit_semi_major_axis}_e"
            f"{self.metadata.orbit_eccentricity}: "
            f"Done.\033[0m"
        )
