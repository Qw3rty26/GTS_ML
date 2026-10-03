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
    logger.info("            VERIFYING PHYSICS")
    logger.info("")
    logger.info(f"  NO. SIMULATIONS: {len(json_files)}")
    logger.info("")
    logger.info("------------------------------------------")

    for json_file in json_files:
        filename = os.path.basename(json_file)

        with open(json_file, "r") as f:
            sim_start = io_manager.load_json_snapshot(f, snapshot_index=0)
        with open(json_file, "r") as f:
            sim_end = io_manager.load_json_snapshot(f, snapshot_index=-1)

        e_start = sim_start.cluster_diagnostics.get_total_energy()
        e_end = sim_end.cluster_diagnostics.get_total_energy()
        delta_e_rel = abs(e_end - e_start) / abs(e_start) if e_start != 0 else 0

        l_start = np.array(sim_start.cluster_diagnostics.get_total_angular_momentum())
        l_end = np.array(sim_end.cluster_diagnostics.get_total_angular_momentum())
        l_start_norm = np.linalg.norm(l_start)
        l_end_norm = np.linalg.norm(l_end)
        delta_l_rel = abs(l_end_norm - l_start_norm) / l_start_norm if l_start_norm != 0 else 0

        q_start = sim_start.cluster_diagnostics.get_virial_ratio()
        q_end = sim_end.cluster_diagnostics.get_virial_ratio()

        e_status = "[OK]" if delta_e_rel < 0.05 else "[WARN]"
        l_status = "[OK]" if delta_l_rel < 0.05 else "[WARN]"
        q_init_status = "[OK]" if abs(q_start - 1.0) < 0.1 else "[WARN]"

        logger.info(f"Simulation: {filename}")
        logger.info(f"  [Energy Drift]      |dE/E0| = {delta_e_rel:.4e} {e_status}")
        logger.info(f"  [Ang. Momentum]     |dL/L0| = {delta_l_rel:.4e} {l_status}")
        logger.info(f"  [Virial Init]       Q_start = {q_start:.3f} {q_init_status}")
        logger.info(f"  [Virial Final]      Q_end   = {q_end:.3f}")
        logger.info("------------------------------------------")
