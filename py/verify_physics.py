import logging
import os

import io_manager
import numpy as np

from simulation import Simulation
from cluster_diagnostics import ClusterDiagnostics
from galactic_potential import GalacticPotential


logger = logging.getLogger(__name__)


class Verify_Physics:

    def __init__(
        self,
        cluster_json_dir_path
    ):
        self.cluster_json_dir_path = cluster_json_dir_path

    def _create_simulation(
        self,
        metadata,
        entities
    ):
        simulation = Simulation(metadata.simulation_config)

        for entity in entities:
            simulation.add_entity(entity)

        if metadata.galaxy_mass is not None:
            galactic_potential = GalacticPotential(
                metadata.galaxy_radius,
                metadata.galaxy_mass
            )

            simulation.add_galactic_potential(
                galactic_potential
            )

        return simulation

    def _verify_simulation(
        self,
        cluster_json_file_path
    ):
        filename = os.path.basename(cluster_json_file_path)

        metadata_start, entities_start, _ = io_manager.load_json_snapshot(
            cluster_json_file_path,
            snapshot_index=0
        )

        metadata_end, entities_end, _ = io_manager.load_json_snapshot(
            cluster_json_file_path,
            snapshot_index=-1
        )

        simulation_start = self._create_simulation(
            metadata_start,
            entities_start
        )

        simulation_end = self._create_simulation(
            metadata_end,
            entities_end
        )

        diagnostics_start = ClusterDiagnostics(simulation_start)
        diagnostics_end = ClusterDiagnostics(simulation_end)

        if metadata_start.galaxy_mass is None:
            self._verify_cluster(
                filename,
                diagnostics_start,
                diagnostics_end
            )
        else:
            self._verify_gts(
                filename,
                diagnostics_start,
                diagnostics_end
            )

    def _verify_cluster(
        self,
        filename,
        diagnostics_start,
        diagnostics_end
    ):
        energy_start = diagnostics_start.get_total_energy()
        energy_end = diagnostics_end.get_total_energy()

        if energy_start != 0.0:
            energy_drift = (
                abs(energy_end - energy_start)
                / abs(energy_start)
            )
        else:
            energy_drift = 0.0

        angular_momentum_start = (
            diagnostics_start.get_angular_momentum()
        )

        angular_momentum_end = (
            diagnostics_end.get_angular_momentum()
        )

        angular_momentum_drift = np.linalg.norm(
            angular_momentum_end - angular_momentum_start
        )

        virial_ratio_start = (
            diagnostics_start.get_virial_ratio()
        )

        virial_ratio_end = (
            diagnostics_end.get_virial_ratio()
        )

        energy_status = (
            "[OK]"
            if energy_drift < 0.05
            else "[WARN]"
        )

        angular_momentum_status = (
            "[OK]"
            if angular_momentum_drift < 0.05
            else "[WARN]"
        )

        virial_status = (
            "[OK]"
            if abs(virial_ratio_start - 1.0) < 0.1
            else "[WARN]"
        )

        logger.info("")
        logger.info(f"Simulation: {filename}")
        logger.info("")

        logger.info("   ENERGY")
        logger.info(
            f"      E0             = {energy_start:.6e}"
        )
        logger.info(
            f"      Ef             = {energy_end:.6e}"
        )
        logger.info(
            f"      |dE/E0|        = {energy_drift:.4e} "
            f"{energy_status}"
        )

        logger.info("")
        logger.info("   ANGULAR MOMENTUM")
        logger.info(
            f"      L0             = {angular_momentum_start}"
        )
        logger.info(
            f"      Lf             = {angular_momentum_end}"
        )
        logger.info(
            f"      |dL|           = {angular_momentum_drift:.4e} "
            f"{angular_momentum_status}"
        )

        logger.info("")
        logger.info("   VIRIAL RATIO")
        logger.info(
            f"      Q_start        = {virial_ratio_start:.3f} "
            f"{virial_status}"
        )
        logger.info(
            f"      Q_end          = {virial_ratio_end:.3f}"
        )

    def _verify_gts(
        self,
        filename,
        diagnostics_start,
        diagnostics_end
    ):
        simulation_start = diagnostics_start.simulation
        simulation_end = diagnostics_end.simulation

        positions_start, masses_start = (
            simulation_start.get_entity_arrays()
        )

        positions_end, masses_end = (
            simulation_end.get_entity_arrays()
        )

        total_mass_start = np.sum(masses_start)
        total_mass_end = np.sum(masses_end)

        energy_internal_start = (
            diagnostics_start.get_total_energy()
        )

        energy_internal_end = (
            diagnostics_end.get_total_energy()
        )

        galactic_potential_energy_start = (
            diagnostics_start.get_galactic_potential_energy()
        )

        galactic_potential_energy_end = (
            diagnostics_end.get_galactic_potential_energy()
        )

        center_of_mass_start = (
            simulation_start.get_center_of_mass()
        )

        center_of_mass_end = (
            simulation_end.get_center_of_mass()
        )

        center_of_mass_velocity_start = np.array([
            center_of_mass_start.vx,
            center_of_mass_start.vy,
            center_of_mass_start.vz
        ])

        center_of_mass_velocity_end = np.array([
            center_of_mass_end.vx,
            center_of_mass_end.vy,
            center_of_mass_end.vz
        ])

        center_of_mass_kinetic_energy_start = (
            0.5
            * total_mass_start
            * np.sum(center_of_mass_velocity_start ** 2)
        )

        center_of_mass_kinetic_energy_end = (
            0.5
            * total_mass_end
            * np.sum(center_of_mass_velocity_end ** 2)
        )

        total_energy_start = (
            energy_internal_start
            + center_of_mass_kinetic_energy_start
            + galactic_potential_energy_start
        )

        total_energy_end = (
            energy_internal_end
            + center_of_mass_kinetic_energy_end
            + galactic_potential_energy_end
        )

        if total_energy_start != 0.0:
            energy_drift = (
                abs(total_energy_end - total_energy_start)
                / abs(total_energy_start)
            )
        else:
            energy_drift = 0.0

        angular_momentum_start = (
            diagnostics_start.get_total_angular_momentum()
        )

        angular_momentum_end = (
            diagnostics_end.get_total_angular_momentum()
        )

        angular_momentum_drift = np.linalg.norm(
            angular_momentum_end - angular_momentum_start
        )

        virial_ratio_start = (
            diagnostics_start.get_virial_ratio()
        )

        virial_ratio_end = (
            diagnostics_end.get_virial_ratio()
        )

        half_mass_radius_start = (
            diagnostics_start.get_half_mass_radius()
        )

        half_mass_radius_end = (
            diagnostics_end.get_half_mass_radius()
        )

        bound_mass_start = self._get_bound_mass(
            simulation_start,
            positions_start,
            masses_start
        )

        bound_mass_end = self._get_bound_mass(
            simulation_end,
            positions_end,
            masses_end
        )

        bound_mass_fraction_start = (
            bound_mass_start / total_mass_start
            if total_mass_start != 0.0
            else 0.0
        )

        bound_mass_fraction_end = (
            bound_mass_end / total_mass_end
            if total_mass_end != 0.0
            else 0.0
        )

        energy_status = (
            "[OK]"
            if energy_drift < 0.05
            else "[WARN]"
        )

        angular_momentum_status = (
            "[OK]"
            if angular_momentum_drift < 0.05
            else "[WARN]"
        )

        logger.info("")
        logger.info(f"Simulation: {filename}")
        logger.info("")

        logger.info("   TOTAL ENERGY")
        logger.info(
            f"      E0             = {total_energy_start:.6e}"
        )
        logger.info(
            f"      Ef             = {total_energy_end:.6e}"
        )
        logger.info(
            f"      |dE/E0|        = {energy_drift:.4e} "
            f"{energy_status}"
        )

        logger.info("")
        logger.info("   TOTAL ANGULAR MOMENTUM")
        logger.info(
            f"      L0             = {angular_momentum_start}"
        )
        logger.info(
            f"      Lf             = {angular_momentum_end}"
        )
        logger.info(
            f"      |dL|           = {angular_momentum_drift:.4e} "
            f"{angular_momentum_status}"
        )

        logger.info("")
        logger.info("   VIRIAL RATIO")
        logger.info(
            f"      Q_start        = {virial_ratio_start:.3f}"
        )
        logger.info(
            f"      Q_end          = {virial_ratio_end:.3f}"
        )

        logger.info("")
        logger.info("   BOUND MASS")
        logger.info(
            f"      f_bound_start  = {bound_mass_fraction_start:.3f}"
        )
        logger.info(
            f"      f_bound_end    = {bound_mass_fraction_end:.3f}"
        )

        logger.info("")
        logger.info("   HALF-MASS RADIUS")
        logger.info(
            f"      r50_start      = {half_mass_radius_start:.3f}"
        )
        logger.info(
            f"      r50_end        = {half_mass_radius_end:.3f}"
        )

    def _get_bound_mass(
        self,
        simulation,
        positions,
        masses
    ):
        number_of_entities = len(masses)

        if number_of_entities == 0:
            return 0.0

        center_of_mass = simulation.get_center_of_mass()

        center_of_mass_velocity = np.array([
            center_of_mass.vx,
            center_of_mass.vy,
            center_of_mass.vz
        ])

        velocities = np.empty(
            (number_of_entities, 3),
            dtype=np.float64
        )

        particles = simulation.simulation.particles

        for i, particle in enumerate(particles):
            velocities[i] = (
                particle.vx,
                particle.vy,
                particle.vz
            )

        relative_velocities = (
            velocities - center_of_mass_velocity
        )

        kinetic_energies = (
            0.5
            * masses
            * np.sum(
                relative_velocities ** 2,
                axis=1
            )
        )

        position_differences = (
            positions[:, np.newaxis, :]
            - positions[np.newaxis, :, :]
        )

        distance_squared = (
            np.sum(
                position_differences ** 2,
                axis=2
            )
            + simulation.get_softening() ** 2
        )

        distances = np.sqrt(distance_squared)

        np.fill_diagonal(distances, np.inf)

        potential_energies = (
            -simulation.get_gravitational_constant()
            * masses
            * np.sum(
                masses[np.newaxis, :] / distances,
                axis=1
            )
        )

        total_energies = (
            kinetic_energies
            + potential_energies
        )

        return np.sum(
            masses[total_energies < 0.0]
        )

    def run(self):

        cluster_json_files_path = sorted(
            os.path.join(
                self.cluster_json_dir_path,
                filename
            )
            for filename in os.listdir(
                self.cluster_json_dir_path
            )
            if filename.endswith(".json")
        )

        if not cluster_json_files_path:
            logger.error(
                f"No cluster JSON files were found "
                f"in {self.cluster_json_dir_path}!"
            )
            return

        logger.info("PHYSICS VERIFICATION STARTED")

        for cluster_json_file_path in cluster_json_files_path:
            self._verify_simulation(
                cluster_json_file_path
            )

        logger.info("")
        logger.info("PHYSICS VERIFICATION COMPLETE")
