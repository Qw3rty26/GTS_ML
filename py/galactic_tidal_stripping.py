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
        file_metadata, entities, center_of_mass = io_manager.load_json_snapshot(cluster_json_file)
        for key, value in vars(file_metadata).items():
            if value is not None:
                setattr(self.metadata, key, value)

        self.simulation = Simulation(self.metadata.simulation_config)

        for entity in entities:
            self.simulation.add_entity(entity)

        galactic_potential = GalacticPotential(self.metadata.galaxy_radius, self.metadata.galaxy_mass)
        self.simulation.add_galactic_potential(galactic_potential)


        r_apo = self.metadata.orbit_semi_major_axis * (1.0 + self.metadata.orbit_eccentricity)
        v_apo = galactic_potential.get_elliptical_apocenter_velocity(
            self.metadata.orbit_semi_major_axis, self.metadata.orbit_eccentricity
        )


        self.simulation.move_cluster(-center_of_mass.position.x, -center_of_mass.position.y, -center_of_mass.position.z)
        self.simulation.speed_cluster(-center_of_mass.velocity.vx, -center_of_mass.velocity.vy, -center_of_mass.velocity.vz)
        self.simulation.move_cluster(r_apo, 0, 0)
        self.simulation.speed_cluster(0, v_apo, 0)

    def run(self):

        DT_CLEANUP = 1.0
        DT_SNAPSHOT = 0.1

        next_cleanup = self.simulation.get_time() + DT_CLEANUP
        next_snapshot = self.simulation.get_time() + DT_SNAPSHOT

        orbits = 0

        io_manager.init_json_gts(self.metadata)

        self.metadata.simulation_config.t = self.simulation.get_time()
        _entities = self.simulation.get_entities()
        _percentile_center_of_mass = self.simulation.get_percentile_center_of_mass()
        io_manager.save_xyzv_snapshot_gts(self.metadata, _entities, _percentile_center_of_mass)
        io_manager.save_json_snapshot_gts(self.metadata, _entities, _percentile_center_of_mass)

        while self.simulation.cluster_diagnostics.get_cluster_orbits() < self.metadata.number_of_orbits:

            self.simulation.integrate()
            self.simulation.cluster_diagnostics.update_orbital_angle()

            if self.simulation.get_time() >= next_snapshot:
                next_snapshot += DT_SNAPSHOT
                self.metadata.simulation_config.t = self.simulation.get_time()
                _entities = self.simulation.get_entities()
                _percentile_center_of_mass = self.simulation.get_percentile_center_of_mass()
                io_manager.save_xyzv_snapshot_gts(self.metadata, _entities, _percentile_center_of_mass)
                io_manager.save_json_snapshot_gts(self.metadata, _entities, _percentile_center_of_mass)

#                if self.simulation.get_time() >= next_cleanup:
#                    next_cleanup += DT_CLEANUP
#                    number_of_entities_cleaned = self.simulation.clean_escaped_entities()
#                    if number_of_entities_cleaned > 0:
#                        logger.debug(
#                            f"Cluster {self.seed}: Cleaning {number_of_entities_cleaned} star/s..."
#                        )
            new_orbits = self.simulation.cluster_diagnostics.get_cluster_orbits()

            if orbits != new_orbits:
                orbits = new_orbits
                logger.info(f"\033[94mCluster {self.metadata.cluster_seed}_a{self.metadata.orbit_semi_major_axis}_e{self.metadata.orbit_eccentricity}: {orbits}/{self.metadata.number_of_orbits} orbits done.\033[0m")

        logger.info(f"Cluster {self.metadata.cluster_seed}_a{self.metadata.orbit_semi_major_axis}_e{self.metadata.orbit_eccentricity}: Completed.")
