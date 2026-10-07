import logging
import numpy as np
import io_manager
from simulation import Simulation
from galactic_potential import GalacticPotential

logger = logging.getLogger(__name__)

class Galactic_Tidal_Stripping:

    def __init__(
        self,
        cluster_json_file,
        galaxy_mass = 1000.0,
        galaxy_radius = 10.0,
        orbit_semi_major_axis = 100.0,
        orbit_eccentricity = 0.0,
        number_of_orbits = 1,
    ):
        self.galaxy_mass = galaxy_mass
        self.galaxy_radius = galaxy_radius
        self.orbit_semi_major_axis = orbit_semi_major_axis
        self.orbit_eccentricity = orbit_eccentricity
        self.number_of_orbits = number_of_orbits
        self._create_cluster(cluster_json_file)

    def _create_cluster(self, cluster_json_file):
        self.metadata, entities, center_of_mass = io_manager.load_json_snapshot(cluster_json_file)

        self.simulation = Simulation(self.metadata.simulation_config)

        for entity in entities:
            self.simulation.add_entity(entity)

        galactic_potential = GalacticPotential(self.galaxy_radius, self.galaxy_mass)
        self.simulation.add_galactic_potential(galactic_potential)


        r_apo = self.orbit_semi_major_axis * (1.0 + self.orbit_eccentricity)
        v_apo = galactic_potential.get_elliptical_apocenter_velocity(
            self.orbit_semi_major_axis, self.orbit_eccentricity
        )

        self.simulation.move_cluster(r_apo, 0, 0)
        self.simulation.speed_cluster(0, v_apo, 0)

    def run(self):

        DT_CLEANUP = 1.0
        DT_SNAPSHOT = 0.1

        next_cleanup = self.simulation.get_time() + DT_CLEANUP
        next_snapshot = self.simulation.get_time() + DT_SNAPSHOT

        orbits = 0

        self.metadata.galaxy_mass = self.galaxy_mass
        self.metadata.galaxy_radius = self.galaxy_radius
        self.metadata.orbit_semi_major_axis = self.orbit_semi_major_axis
        self.metadata.orbit_eccentricity = self.orbit_eccentricity
        self.metadata.number_of_orbits = self.number_of_orbits

        io_manager.init_json_gts(self.metadata)

        while self.simulation.cluster_diagnostics.get_cluster_orbits() < self.number_of_orbits:

            self.simulation.integrate()

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
                logger.info(f"Cluster {self.metadata.cluster_seed}: {orbits}/{self.number_of_orbits} orbits done.")

