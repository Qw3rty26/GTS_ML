
import logging
import os

import numpy as np

import io_manager
from simulation import Simulation
from cluster_diagnostics import ClusterDiagnostics
from galactic_potential import GalacticPotential


logger = logging.getLogger(__name__)


class Verify_Physics:

    def __init__(self, cluster_json_dir_path):
        self.cluster_json_dir_path = cluster_json_dir_path

    def _create_simulation(self, metadata, entities):
        simulation = Simulation(metadata.simulation_config)

        for entity in entities:
            simulation.add_entity(entity)

        if metadata.galaxy_mass is not None:
            galactic_potential = GalacticPotential(
                metadata.galaxy_radius,
                metadata.galaxy_mass
            )
            simulation.add_galactic_potential(galactic_potential)

        return simulation

    def _get_angular_momentum(self, simulation):
        entities = simulation.get_entities()

        if not entities:
            return np.zeros(3)

        positions = np.array([
            [entity.position.x, entity.position.y, entity.position.z]
            for entity in entities
        ], dtype=np.float64)

        velocities = np.array([
            [entity.velocity.vx, entity.velocity.vy, entity.velocity.vz]
            for entity in entities
        ], dtype=np.float64)

        masses = np.array([
            entity.mass for entity in entities
        ], dtype=np.float64)

        return np.sum(
            masses[:, np.newaxis] * np.cross(positions, velocities),
            axis=0
        )

    def _get_total_energy(self, diagnostics):
        simulation = diagnostics.simulation

        internal_energy = diagnostics.get_system_total_energy()

        center_of_mass = simulation.get_center_of_mass()

        total_mass = sum(
            entity.mass
            for entity in simulation.get_entities()
        )

        center_of_mass_velocity = np.array([
            center_of_mass.vx,
            center_of_mass.vy,
            center_of_mass.vz
        ])

        center_of_mass_kinetic_energy = (
            0.5
            * total_mass
            * np.sum(center_of_mass_velocity**2)
        )

        galactic_potential_energy = (
            diagnostics.get_system_galactic_potential_energy()
        )

        return (
            internal_energy
            + center_of_mass_kinetic_energy
            + galactic_potential_energy
        )

    def _get_relative_drift(self, initial, final):
        if abs(initial) < 1e-15:
            return abs(final - initial)

        return abs(final - initial) / abs(initial)

    def _verify_simulation(self, cluster_json_file_path):
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

    def _verify_cluster(self, filename, diagnostics_start, diagnostics_end):
        energy_start = diagnostics_start.get_system_total_energy()
        energy_end = diagnostics_end.get_system_total_energy()

        energy_drift = self._get_relative_drift(
            energy_start,
            energy_end
        )

        angular_momentum_start = self._get_angular_momentum(
            diagnostics_start.simulation
        )

        angular_momentum_end = self._get_angular_momentum(
            diagnostics_end.simulation
        )

        angular_momentum_drift = np.linalg.norm(
            angular_momentum_end - angular_momentum_start
        )

        virial_ratio_start = diagnostics_start.get_virial_ratio()
        virial_ratio_end = diagnostics_end.get_virial_ratio()

        energy_status = "[OK]" if energy_drift < 0.05 else "[WARN]"
        angular_momentum_status = (
            "[OK]" if angular_momentum_drift < 0.05 else "[WARN]"
        )

        logger.info("")
        logger.info(f"Simulation: {filename}")
        logger.info("")
        logger.info("   ENERGY")
        logger.info(f"      E0             = {energy_start:.6e}")
        logger.info(f"      Ef             = {energy_end:.6e}")
        logger.info(
            f"      |dE/E0|        = {energy_drift:.4e} "
            f"{energy_status}"
        )

        logger.info("")
        logger.info("   ANGULAR MOMENTUM")
        logger.info(f"      L0             = {angular_momentum_start}")
        logger.info(f"      Lf             = {angular_momentum_end}")
        logger.info(
            f"      |dL|           = {angular_momentum_drift:.4e} "
            f"{angular_momentum_status}"
        )

        logger.info("")
        logger.info("   VIRIAL RATIO")
        logger.info(f"      Q_start        = {virial_ratio_start:.3f}")
        logger.info(f"      Q_end          = {virial_ratio_end:.3f}")

    def _verify_gts(self, filename, diagnostics_start, diagnostics_end):
        energy_start = self._get_total_energy(diagnostics_start)
        energy_end = self._get_total_energy(diagnostics_end)

        energy_drift = self._get_relative_drift(
            energy_start,
            energy_end
        )

        angular_momentum_start = self._get_angular_momentum(
            diagnostics_start.simulation
        )

        angular_momentum_end = self._get_angular_momentum(
            diagnostics_end.simulation
        )

        angular_momentum_drift = np.linalg.norm(
            angular_momentum_end - angular_momentum_start
        )

        virial_ratio_start = diagnostics_start.get_virial_ratio()
        virial_ratio_end = diagnostics_end.get_virial_ratio()

        half_mass_radius_start = diagnostics_start.get_half_mass_radius()
        half_mass_radius_end = diagnostics_end.get_half_mass_radius()

        total_mass_start = sum(
            entity.mass
            for entity in diagnostics_start.simulation.get_entities()
        )

        total_mass_end = sum(
            entity.mass
            for entity in diagnostics_end.simulation.get_entities()
        )

        bound_mass_fraction_start = (
            diagnostics_start.get_bound_mass() / total_mass_start
            if total_mass_start > 0.0 else 0.0
        )

        bound_mass_fraction_end = (
            diagnostics_end.get_bound_mass() / total_mass_end
            if total_mass_end > 0.0 else 0.0
        )

        energy_status = "[OK]" if energy_drift < 0.05 else "[WARN]"
        angular_momentum_status = (
            "[OK]" if angular_momentum_drift < 0.05 else "[WARN]"
        )

        logger.info("")
        logger.info(f"Simulation: {filename}")
        logger.info("")
        logger.info("   TOTAL ENERGY")
        logger.info(f"      E0             = {energy_start:.6e}")
        logger.info(f"      Ef             = {energy_end:.6e}")
        logger.info(
            f"      |dE/E0|        = {energy_drift:.4e} "
            f"{energy_status}"
        )

        logger.info("")
        logger.info("   TOTAL ANGULAR MOMENTUM")
        logger.info(f"      L0             = {angular_momentum_start}")
        logger.info(f"      Lf             = {angular_momentum_end}")
        logger.info(
            f"      |dL|           = {angular_momentum_drift:.4e} "
            f"{angular_momentum_status}"
        )

        logger.info("")
        logger.info("   VIRIAL RATIO")
        logger.info(f"      Q_start        = {virial_ratio_start:.3f}")
        logger.info(f"      Q_end          = {virial_ratio_end:.3f}")

        logger.info("")
        logger.info("   BOUND MASS")
        logger.info(f"      f_bound_start  = {bound_mass_fraction_start:.3f}")
        logger.info(f"      f_bound_end    = {bound_mass_fraction_end:.3f}")

        logger.info("")
        logger.info("   HALF-MASS RADIUS")
        logger.info(f"      r50_start      = {half_mass_radius_start:.3f}")
        logger.info(f"      r50_end        = {half_mass_radius_end:.3f}")

    def run(self):
        cluster_json_files_path = sorted(
            os.path.join(self.cluster_json_dir_path, filename)
            for filename in os.listdir(self.cluster_json_dir_path)
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
            self._verify_simulation(cluster_json_file_path)

        logger.info("")
        logger.info("PHYSICS VERIFICATION COMPLETE")
