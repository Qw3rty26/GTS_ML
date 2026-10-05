import logging
import time
import io_manager
import os
import numpy as np

logger = logging.getLogger(__name__)


def verify_simulation(run_path, configuration):
    logger.debug("loading JSON GTS files...")
    json_path = os.path.join(run_path, "GTS", "JSON")

    json_files = [
        os.path.join(json_path, file)
        for file in os.listdir(json_path)
        if file.endswith(".json")
    ]

    logger.info("------------------------------------------")
    logger.info("            VERIFYING PHYSICS             ")
    logger.info("")
    logger.info(f"    NO. SIMULATIONS: {len(json_files)}")
    logger.info("")
    logger.info("------------------------------------------")

    for json_file in json_files:
        filename = os.path.basename(json_file)

        with open(json_file, "r") as f:
            sim_start = io_manager.load_json_snapshot(f, snapshot_index=0)
        with open(json_file, "r") as f:
            sim_end = io_manager.load_json_snapshot(f, snapshot_index=-1)

        # --------------------------------------------------
        # Total Energy Relative Error (Internal Cluster Energy)
        # --------------------------------------------------
        e_start = sim_start.cluster_diagnostics.get_total_energy()
        e_end = sim_end.cluster_diagnostics.get_total_energy()
        delta_e_rel = abs(e_end - e_start) / abs(e_start) if e_start != 0 else 0.0

        # --------------------------------------------------
        # Angular Momentum (Informational in GTS due to tides)
        # --------------------------------------------------
        l_start = np.array(sim_start.cluster_diagnostics.get_total_angular_momentum_bound())
        l_end = np.array(sim_end.cluster_diagnostics.get_total_angular_momentum_bound())
        l_start_norm = np.linalg.norm(l_start)

        if l_start_norm != 0.0:
            delta_l_rel = np.linalg.norm(l_end - l_start) / l_start_norm
        else:
            delta_l_rel = 0.0

        # --------------------------------------------------
        # Virial Ratio
        # --------------------------------------------------
        q_start = sim_start.cluster_diagnostics.get_virial_ratio_bound()
        q_end = sim_end.cluster_diagnostics.get_virial_ratio_bound()

        # --------------------------------------------------
        # ML Targets Extraction (Bound Mass Fraction & Half-Mass Radius)
        # --------------------------------------------------
        initial_bound_mass = sim_start.cluster_diagnostics.get_bound_mass()
        final_bound_mass = sim_end.cluster_diagnostics.get_bound_mass()
        
        # Avoid division by zero
        bound_mass_fraction = (
            final_bound_mass / initial_bound_mass 
            if initial_bound_mass > 0 else 0.0
        )
        
        final_half_mass_radius = sim_end.cluster_diagnostics.get_half_mass_radius()

        # --------------------------------------------------
        # Status checks & thresholds
        # --------------------------------------------------
        e_status = "[OK]" if delta_e_rel < 0.05 else "[WARN]"
        # Angular momentum changes due to galactic tides, so we mark it as [INFO/TIDES] instead of a strict warning
        l_status = "[TIDES]" 
        q_init_status = "[OK]" if abs(q_start - 1.0) < 0.1 else "[WARN]"

        logger.info(f"Simulation: {filename}")
        logger.info(f"   [Energy Drift]      |dE/E0| = {delta_e_rel:.4e} {e_status}")
        logger.info(f"   [Ang. Momentum]     |dL/L0| = {delta_l_rel:.4e} {l_status}")
        logger.info(f"   [Virial Init]       Q_start = {q_start:.3f} {q_init_status}")
        logger.info(f"   [Virial Final]      Q_end   = {q_end:.3f}")
        logger.info(f"   --- ML Targets ---")
        logger.info(f"   [Bound Mass Frac]   f_bound = {bound_mass_fraction:.3f}")
        logger.info(f"   [Half-Mass Radius]  R_half  = {final_half_mass_radius:.3f}")
        logger.info("------------------------------------------")
