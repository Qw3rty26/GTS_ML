import argparse
import logging
import time
import os
import itertools
import io_manager
import multiprocessing as mp
from multiprocessing import Pool

from simulation_batch import run_cluster_generation as run_gen
from simulation_batch import run_galaxy_tidal_stripping as run_gts
from logger_settings import configure_logging
from verify_physics import verify_simulation

def init_worker(verbose, debug):
    configure_logging(verbose, debug)

def run_gen_wrapper(args):
    return run_gen(*args)


def run_gts_wrapper(args):
    return run_gts(*args)

def _generate_clusters(logger, configuration, simulation_args, verbose, debug):

    simulation = configuration["simulation"]
    cluster = configuration["cluster generation"]
    output = configuration["output directory"]

    logger.info("------------------------------------------")
    logger.info("            GENERATING CLUSTERS")
    logger.info("")
    logger.info(f"  NO. SIMULATIONS: {len(simulation_args)}")
    logger.info(f"  STARS: {cluster['number_of_stars']}")
    logger.info(f"  DT: {simulation['dt']}")
    logger.info(f"  G: {simulation['G']}")
    logger.info(f"  SOFTENING: {simulation['softening']}")
    logger.info(f"  TIME WARP: {simulation['time_warp']}")
    logger.info(f"  INTEGRATOR: {simulation['integrator']}")
    logger.info(f"  OUTPUT: {output}/")
    logger.info("")
    logger.info("------------------------------------------")

    computing_time = time.perf_counter()
    chunk_size = max(1, len(simulation_args) // (os.cpu_count() * 2))

    with Pool(initializer=init_worker, initargs=(verbose, debug)) as pool:
        for completed, _ in enumerate(
            pool.imap_unordered(run_gen_wrapper, simulation_args, chunksize=chunk_size),
            1
        ):
            logger.info(
                f"\033[32mProgress: "
                f"{completed}/{len(simulation_args)} "
                f"Clusters generated.\033[0m"
            )

    computing_time = time.perf_counter() - computing_time

    logger.info("------------------------------------------")
    logger.info("        CLUSTER GENERATION COMPLETED")
    logger.info("")
    logger.info(f"  EXECUTION TIME: {computing_time:.3f} seconds")
    logger.info("")
    logger.info("------------------------------------------")


def _simulate_gts(logger, configuration, simulation_args, verbose, debug):

    simulation = configuration["simulation"]
    gts = configuration["galactic environment"]
    output = configuration["output directory"]

    logger.info("------------------------------------------")
    logger.info("               EVOLVING GALAXY")
    logger.info("")
    logger.info(f"  NO. SIMULATIONS: {len(simulation_args)}")
    logger.info(f"  INTEGRATOR: {simulation['integrator']}")
    logger.info(f"  DT: {simulation['dt']}")
    logger.info(f"  GALAXY MASS: {gts['galaxy_mass']}")
    logger.info(f"  GALAXY RADIUS: {gts['galaxy_radius']}")
    logger.info(f"  OUTPUT: {output}/")
    logger.info("")
    logger.info("------------------------------------------")

    computing_time = time.perf_counter()
    chunk_size = max(1, len(simulation_args) // (os.cpu_count() * 2))

    with Pool(initializer=init_worker, initargs=(verbose, debug)) as pool:
        for completed, _ in enumerate(
            pool.imap_unordered(run_gts_wrapper, simulation_args, chunksize=chunk_size),
            1
        ):
            logger.info(
                f"\033[32mProgress: "
                f"{completed}/{len(simulation_args)} "
                f"Simulations completed.\033[0m"
            )

    computing_time = time.perf_counter() - computing_time

    logger.info("------------------------------------------")
    logger.info("         GALAXY EVOLUTION COMPLETED")
    logger.info("")
    logger.info(f"  EXECUTION TIME: {computing_time:.3f} seconds")
    logger.info("")
    logger.info("------------------------------------------")


def main():

    parser = argparse.ArgumentParser(
        description="Generate Plummer clusters and simulate galactic tidal strippings"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="../configuration.json",
        help="path to configuration file"
    )

    parser.add_argument(
        "--load-cluster-dir",
        type=str,
        default=None,
        help="path to an existing GEN/JSON directory to reuse clusters"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="enable verbose output"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable debug output"
    )

    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    configuration = io_manager.load_json_file(config_path)

    configure_logging(args.verbose, args.debug)

    logger = logging.getLogger(__name__)

    simulation = configuration["simulation"]
    cluster = configuration["cluster generation"]
    gts = configuration["galactic environment"]
    ml_inputs = configuration["ml_dataset_input"]

    simulation_args = []
    run_path = None
    cluster_files = None
    cluster_json = None
    cluster_xyzv = None
    gts_json = None
    gts_xyzv = None

    if args.load_cluster_dir:
        cluster_json = os.path.abspath(args.load_cluster_dir)
        if not os.path.exists(cluster_json):
            logger.error(f"Provided cluster directory does not exist: {cluster_json}")
            exit(1)
        logger.info(f"Reusing existing clusters from: {cluster_json}")
        run_path = io_manager.create_run_directory(configuration["output directory"])
        gts_json = os.path.join(run_path, "GTS", "JSON")
        gts_xyzv = os.path.join(run_path, "GTS", "XYZV")
    else:
        run_path = io_manager.create_run_directory(configuration["output directory"])
        cluster_json = os.path.join(run_path, "GEN", "JSON")
        cluster_xyzv = os.path.join(run_path, "GEN", "XYZV")
        gts_json = os.path.join(run_path, "GTS", "JSON")
        gts_xyzv = os.path.join(run_path, "GTS", "XYZV")

        for cluster_id in range(cluster["number_of_clusters"]):
            simulation_args.append(
                (
                    cluster_id,
                    cluster["cluster_radius"],
                    cluster["number_of_stars"],
                    simulation["dt"],
                    simulation["G"],
                    simulation["softening"],
                    simulation["time_warp"],
                    simulation["integrator"],
                    cluster_json,
                    cluster_xyzv
                )
            )

        _generate_clusters(logger, configuration, simulation_args, args.verbose, args.debug)

    cluster_files = [
        os.path.join(cluster_json, file)
        for file in os.listdir(cluster_json)
        if file.endswith(".json")
    ]

    if not cluster_files:
        logger.error("No cluster JSON files found to process!")
        exit(1)

    if run_path is None:
        run_path = io_manager.create_run_directory(configuration["output directory"])

    semi_axes = ml_inputs["semi_major_axes"]
    eccentricities = ml_inputs["eccentricities"]

    orbit_grid = list(itertools.product(semi_axes, eccentricities))
    gts_args = []
    sim_counter = 0

    for cluster_file in cluster_files:
        for a, e in orbit_grid:
            gts_args.append(
                (
                    cluster_file,
                    gts["galaxy_mass"],
                    gts["galaxy_radius"],
                    a,
                    e,
                    gts_json,
                    gts_xyzv,
                    sim_counter
                )
            )
            sim_counter += 1

    _simulate_gts(logger, configuration, gts_args, args.verbose, args.debug)

    if run_path and os.path.exists(run_path):
        verify_simulation(run_path, configuration)
    else:
        logger.warning(f"No valid directory {run_path} was found to be verified.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
