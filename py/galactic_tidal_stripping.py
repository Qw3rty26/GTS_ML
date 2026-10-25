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
        metadata: Metadata,
        json_and_xyzv_snapshots_dt = 1
    ):
        self.metadata = metadata
        self.json_and_xyzv_snapshots_dt = json_and_xyzv_snapshots_dt
        self._create_cluster(cluster_json_file)

    def _save_to_file(self):

        self.metadata.simulation_config.t = (
            self.simulation.get_time()
        )

        entities = self.simulation.get_entities()

        bound_center_of_mass = (
            self.simulation.get_bound_center_of_mass()
        )

        io_manager.save_xyzv_snapshot_gts(
            self.metadata,
            entities,
            bound_center_of_mass
        )

        io_manager.save_json_snapshot_gts(
            self.metadata,
            entities,
            bound_center_of_mass
        )

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

        center_of_mass = self.simulation.get_center_of_mass()

        self.simulation.move_cluster(
            -center_of_mass.x,
            -center_of_mass.y,
            -center_of_mass.z
        )

        self.simulation.speed_cluster(
            -center_of_mass.vx,
            -center_of_mass.vy,
            -center_of_mass.vz
        )

        self.simulation.move_cluster(r_apo,0,0)
        self.simulation.cluster_diagnostics.apply_virial_correction()
        self.simulation.speed_cluster(0,v_apo,0)

    def run(self):

        logger.info(
            f"Cluster "
            f"{self.metadata.cluster_seed}_a"
            f"{self.metadata.orbit_semi_major_axis}_e"
            f"{self.metadata.orbit_eccentricity}: "
            f"Running..."
        )
        theoretical_orbit_period = (
            2.0 * np.pi
            * np.sqrt(
                (
                    self.metadata.galaxy_radius**2
                    + self.metadata.orbit_semi_major_axis**2
                )**1.5
                / (
                    self.metadata.simulation_config.G
                    * self.metadata.galaxy_mass
                )
            )
        )

        print(f"T: {self.simulation.simulation.t}")
        print(f"E_tot: {self.simulation.cluster_diagnostics.get_system_total_energy()}")
        print(f"U_gal: {self.simulation.cluster_diagnostics.get_system_galactic_potential_energy()}")
        print(f"U_cluster: {self.simulation.cluster_diagnostics.get_system_potential_energy()}")
        print(f"K_cluster: {self.simulation.cluster_diagnostics.get_system_kinetic_energy()}")
        #print(f"E_tot rebound: {self.simulation.simulation.energy()}")
        print(f"bmf: {self.simulation.cluster_diagnostics.get_bound_mass()}")
        print(f"hmr: {self.simulation.cluster_diagnostics.get_half_mass_radius():.3f}\n")
        print(f"rms: {self.simulation.cluster_diagnostics.get_root_mean_square_radius()}")
        #bound_com = self.simulation.cluster_diagnostics.get_bound_center_of_mass()
        #print(f"bound_com: x:{bound_com.position.x} y:{bound_com.position.y} z:{bound_com.position.z}")

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
            + self.json_and_xyzv_snapshots_dt
        )

        io_manager.init_json_gts(self.metadata)

        self._save_to_file()

        last_logged_orbits = 0

        ORBIT_TRACKING_START = 0.8
        ORBIT_TIMEOUT_FACTOR = 2.0

        current_time = self.simulation.get_time()
        orbit_timeout = (
            current_time
            + ORBIT_TIMEOUT_FACTOR * theoretical_orbit_period
        )

        next_orbit_completion_time = (
            current_time
            + theoretical_orbit_period
        )

        next_orbit_tracking_start = (
            current_time
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
                #print(f"T: {self.simulation.simulation.t}")
                #print(f"E_tot: {self.simulation.cluster_diagnostics.get_system_total_energy()}")
                #print(f"Q: {self.simulation.cluster_diagnostics.get_virial_ratio()}")
                #print(f"E_tot rebound: {self.simulation.simulation.energy()}")
                #print(f"bmf: {self.simulation.cluster_diagnostics.get_bound_mass()}")
                #print(f"hmr: {self.simulation.cluster_diagnostics.get_half_mass_radius():.3f}")
                #print(f"rms: {self.simulation.cluster_diagnostics.get_root_mean_square_radius()}\n")
                #bound_com = self.simulation.cluster_diagnostics.get_bound_center_of_mass()
                #print(f"bound_com: x:{bound_com.position.x} y:{bound_com.position.y} z:{bound_com.position.z}")
                next_snapshot += self.json_and_xyzv_snapshots_dt
                self._save_to_file()

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

                    orbit_timeout = (
                        current_time
                        + ORBIT_TIMEOUT_FACTOR
                        * theoretical_orbit_period
                    )

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
                        logger.debug(
                            f"Cluster "
                            f"{self.metadata.cluster_seed}_a"
                            f"{self.metadata.orbit_semi_major_axis}_e"
                            f"{self.metadata.orbit_eccentricity}: "
                            f"current_t = {current_time:.2f}"
                        )
                        self._save_to_file()
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

            if current_time >= orbit_timeout:
                logger.warning(
                    f"Cluster "
                    f"{self.metadata.cluster_seed}_a"
                    f"{self.metadata.orbit_semi_major_axis}_e"
                    f"{self.metadata.orbit_eccentricity}: "
                    f"Orbit timeout after "
                    f"{ORBIT_TIMEOUT_FACTOR * theoretical_orbit_period:.2f} "
                    f"time units. "
                    f"Completed {last_logged_orbits}/"
                    f"{self.metadata.number_of_orbits} orbits."
                )
                self._save_to_file()
                break
        print(f"T: {self.simulation.simulation.t}")
        print(f"E_tot: {self.simulation.cluster_diagnostics.get_system_total_energy()}")
        print(f"U_gal: {self.simulation.cluster_diagnostics.get_system_galactic_potential_energy()}")
        print(f"U_cluster: {self.simulation.cluster_diagnostics.get_system_potential_energy()}")
        print(f"K_cluster: {self.simulation.cluster_diagnostics.get_system_kinetic_energy()}")
        #print(f"E_tot rebound: {self.simulation.simulation.energy()}")
        print(f"bmf: {self.simulation.cluster_diagnostics.get_bound_mass()}")
        print(f"hmr: {self.simulation.cluster_diagnostics.get_half_mass_radius():.3f}\n")
        print(f"rms: {self.simulation.cluster_diagnostics.get_root_mean_square_radius()}")
        #bound_com = self.simulation.cluster_diagnostics.get_bound_center_of_mass()
        #print(f"bound_com: x:{bound_com.position.x} y:{bound_com.position.y} z:{bound_com.position.z}")

        logger.info(
            f"\033[94mCluster "
            f"{self.metadata.cluster_seed}_a"
            f"{self.metadata.orbit_semi_major_axis}_e"
            f"{self.metadata.orbit_eccentricity}: "
            f"Done.\033[0m"
        )
