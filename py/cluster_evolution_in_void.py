
import logging
import numpy as np
import io_manager
from simulation import Simulation
from io_manager import Metadata


logger = logging.getLogger(__name__)


class Cluster_Evolution_In_Void:

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
            f"{self.metadata.cluster_seed}"
            f"Creating..."
        )

        self.simulation = Simulation(
            self.metadata.simulation_config
        )

        for entity in entities:
            self.simulation.add_entity(entity)

    def _get_values(self):

        return {
            "t": self.simulation.get_time(),
            "E_tot": self.simulation.cluster_diagnostics.get_system_total_energy(),
            "U_cluster": self.simulation.cluster_diagnostics.get_system_potential_energy(),
            "K_cluster": self.simulation.cluster_diagnostics.get_system_kinetic_energy(),
            "bmf": self.simulation.cluster_diagnostics.get_bound_mass(),
            "hmr": self.simulation.cluster_diagnostics.get_half_mass_radius(),
            "rms": self.simulation.cluster_diagnostics.get_root_mean_square_radius(),
            "Q": self.simulation.cluster_diagnostics.get_virial_ratio(),
            "N": len(self.simulation.get_entities())
        }

    def _print_values(self, label, values):

        print(f"{label}")
        print(f"T: {values['t']:.6f}")
        print(f"N: {values['N']}")
        print(f"E_tot: {values['E_tot']:.8e}")
        print(f"U_cluster: {values['U_cluster']:.8e}")
        print(f"K_cluster: {values['K_cluster']:.8e}")
        print(f"Q: {values['Q']:.6f}")
        print(f"bmf: {values['bmf']}")
        print(f"hmr: {values['hmr']:.6f}")
        print(f"rms: {values['rms']:.6f}")
        print()

    def run(self):

        logger.info(
            f"Cluster "
            f"{self.metadata.cluster_seed}"
            f"Running..."
        )

        initial = self._get_values()

        print("\n\nINITIAL VALUES BEFORE CLUSTER IS IN THE VOID")
        self._print_values("", initial)

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

        while current_time < self.metadata.end_time:
            self.simulation.integrate()

            current_time = self.simulation.get_time()

            if current_time >= next_snapshot:
                next_snapshot += self.json_and_xyzv_snapshots_dt
                self._save_to_file()

        self._save_to_file()

        final = self._get_values()

        print("FINAL VALUES AFTER CLUSTER WAS IN THE VOID")
        self._print_values("", final)

        print("CHANGES")
        print(f"T iniziale: {initial['t']:.6f}")
        print(f"T finale: {final['t']:.6f}")
        print(f"T richiesto: {self.metadata.end_time:.6f}")
        print(f"Tempo evoluto: {final['t'] - initial['t']:.6f}")
        print(f"N iniziale: {initial['N']}")
        print(f"N finale: {final['N']}")
        print(f"Stelle perse: {initial['N'] - final['N']}")

        print("\nENERGY")
        delta_energy = final["E_tot"] - initial["E_tot"]
        print(f"E iniziale: {initial['E_tot']:.8e}")
        print(f"E finale: {final['E_tot']:.8e}")
        print(f"Delta E: {delta_energy:.8e}")

        if initial["E_tot"] != 0:
            energy_drift = delta_energy / abs(initial["E_tot"])
            print(f"Deriva energetica: {energy_drift:.8e}")
            print(f"Deriva energetica (%): {energy_drift * 100:.6f}%")
            print(f"Errore energetico relativo: {abs(energy_drift):.8e}")
            print(f"Errore energetico relativo (%): {abs(energy_drift) * 100:.6f}%")
        else:
            print("Deriva energetica: non definita (E iniziale = 0)")
            print("Errore energetico relativo: non definito (E iniziale = 0)")

        print("\nHMR")
        delta_hmr = final["hmr"] - initial["hmr"]
        print(f"HMR iniziale: {initial['hmr']:.6f}")
        print(f"HMR finale: {final['hmr']:.6f}")
        print(f"Delta HMR: {delta_hmr:.6f}")

        if initial["hmr"] != 0:
            hmr_ratio = final["hmr"] / initial["hmr"]
            hmr_change = (hmr_ratio - 1) * 100
            print(f"HMR finale / HMR iniziale: {hmr_ratio:.6f}")
            print(f"Variazione HMR: {hmr_change:+.6f}%")
        else:
            print("HMR finale / HMR iniziale: non definito (HMR iniziale = 0)")

        print("\nENERGIE")
        print(f"Delta U: {final['U_cluster'] - initial['U_cluster']:.8e}")
        print(f"Delta K: {final['K_cluster'] - initial['K_cluster']:.8e}")
        print(f"Q iniziale: {initial['Q']:.6f}")
        print(f"Q finale: {final['Q']:.6f}")
        print(f"Delta Q: {final['Q'] - initial['Q']:+.6f}")
        print("\n\n")
