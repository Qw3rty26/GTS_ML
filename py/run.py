import os
#limit the number of threads that numpy can use to 1
MAX_THREADS = "1"
os.environ["OMP_NUM_THREADS"] = MAX_THREADS
os.environ["MKL_NUM_THREADS"] = MAX_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = MAX_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = MAX_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = MAX_THREADS

import argparse
import logging
import time
import itertools
import io_manager
import multiprocessing as mp
import numpy as np
from multiprocessing import Pool
from cluster_generation import Cluster_Generation
from cluster_evolution_in_void import Cluster_Evolution_In_Void
from verify_physics import Verify_Physics

from io_manager import IOPaths, Metadata
from simulation import SimulationConfig
from logger_settings import configure_logging

LOGGING_VERBOSE = True
LOGGING_DEBUG = False

logger = logging.getLogger(__name__)

io_paths = None

def initialise_logging_worker(io_paths, logging_verbose, logging_debug):
    configure_logging(logging_verbose, logging_debug)
    io_manager.set_io_paths(io_paths)

def _run_cluster_generation_wrapper(args):
    simulation_config, metadata, json_and_xyzv_snapshots_dt = args
    cluster_generation = Cluster_Generation(
        simulation_config = simulation_config,
        metadata = metadata,
        json_and_xyzv_snapshots_dt = json_and_xyzv_snapshots_dt
    )
    cluster_generation.run()


def _run_cluster_generation(configuration):
    max_workers = configuration.get("cores_used", 1)

    logger.info(f"------------------------------------------")
    logger.info(f"        CLUSTER GENERATION STARTED")
    logger.info(f"")
    logger.info(f" Master Seed: {configuration["cluster_generation"]["master_seed"]}")
    logger.info(f" Number of Clusters: {configuration["cluster_generation"]["number_of_clusters"]}")
    logger.info(f" Number of Stars: {configuration["cluster_generation"]["number_of_stars"]}")
    logger.info(f" Cluster Radius: {configuration["cluster_generation"]["cluster_radius"]}")
    logger.info(f"------------------------------------------")
    logger.info(f" dt: {configuration["simulation_config"]["dt"]}")
    logger.info(f" t: {configuration["simulation_config"]["t"]}")
    logger.info(f" G: {configuration["simulation_config"]["G"]}")
    logger.info(f" Softening: {configuration["simulation_config"]["softening"]}")
    logger.info(f" Time Warp: {configuration["simulation_config"]["time_warp"]}")
    logger.info(f" Integrator: {configuration["simulation_config"]["integrator"]}")
    logger.info(f"------------------------------------------")
    logger.info(f" Saved Snapshots dt: {configuration["json_and_xyzv_snapshots_dt"]}")
    logger.info(f" Output Directory: {io_paths.gen_dir}")
    logger.info(f" Cores in parallel: {max_workers}")
    logger.info(f"------------------------------------------")
    logger.info(f"")
    logger.info(f"")

    computing_time = time.perf_counter()

    cluster_generation = configuration["cluster_generation"]

    simulation_config = SimulationConfig(
        dt = configuration["simulation_config"]["dt"],
        t = configuration["simulation_config"]["t"],
        G = configuration["simulation_config"]["G"],
        softening = configuration["simulation_config"]["softening"],
        time_warp = configuration["simulation_config"]["time_warp"],
        integrator = configuration["simulation_config"]["integrator"]
    )

    master_seed = configuration["cluster_generation"]["master_seed"]
    number_of_clusters = configuration["cluster_generation"]["number_of_clusters"]

    master_rng = np.random.Generator(np.random.MT19937(master_seed))
    cluster_seeds = master_rng.integers(low=0, high=2**31 - 1, size = number_of_clusters, dtype=int)
    gen_args = []

    for cluster_seed in cluster_seeds:
        metadata = Metadata (
            master_seed = configuration["cluster_generation"]["master_seed"],
            cluster_seed = int(cluster_seed),
            number_of_clusters = configuration["cluster_generation"]["number_of_clusters"],
            cluster_radius = configuration["cluster_generation"]["cluster_radius"],
            initial_number_of_entities = configuration["cluster_generation"]["number_of_stars"],
            simulation_config = simulation_config
        )
        gen_args.append((simulation_config, metadata, configuration["json_and_xyzv_snapshots_dt"]))

    with Pool(processes=max_workers, initializer=initialise_logging_worker, initargs=(io_paths, LOGGING_VERBOSE, LOGGING_DEBUG,)) as pool:
        for i, _ in enumerate(pool.imap_unordered(_run_cluster_generation_wrapper, gen_args), 1):
            if i % 5 == 0 or i == number_of_clusters:
                logger.info(f"\033[32mProgress: {i}/{number_of_clusters} clusters generated.\033[0m")

    computing_time = time.perf_counter() - computing_time

    logger.info(f"------------------------------------------")
    logger.info(f"        CLUSTER GENERATION COMPLETE")
    logger.info(f"")
    logger.info(f" Execution time: {computing_time:.3f} seconds")
    logger.info(f"------------------------------------------")
    logger.info(f"")
    logger.info(f"")

def _run_cluster_evolution_in_void_wrapper(args):
    cluster_json_file, metadata, json_and_xyzv_snapshots_dt= args
    cluster_evolution_in_void = Cluster_Evolution_In_Void(
        cluster_json_file = cluster_json_file,
        metadata = metadata,
        json_and_xyzv_snapshots_dt = json_and_xyzv_snapshots_dt
    )
    cluster_evolution_in_void.run()

def _run_cluster_evolution_in_void(configuration, cluster_json_dir_path):

    cluster_json_files_path = [
        os.path.join(cluster_json_dir_path, f)
        for f in os.listdir(cluster_json_dir_path)
        if f.endswith(".json")
    ]

    if not cluster_json_files_path:
        logger.error(f"No cluster JSON files were found in {cluster_json_dir_path}!")
        return


    total_tasks = (
        len(cluster_json_files_path)
    )


    max_workers = configuration.get("cores_used", 1)

    logger.info(f"------------------------------------------")
    logger.info(f"    EVOLVING CLUSTER IN THE VOID")
    logger.info(f"")
    logger.info(f" Number of Simulations: {total_tasks}")
    logger.info(f" Evolving until T: {configuration["cluster_evolution_in_void"]["end_time"]}")
    logger.info(f"------------------------------------------")
    logger.info(f" dt: {configuration["simulation_config"]["dt"]}")
    logger.info(f" G: {configuration["simulation_config"]["G"]}")
    logger.info(f" Softening: {configuration["simulation_config"]["softening"]}")
    logger.info(f" Time Warp: {configuration["simulation_config"]["time_warp"]}")
    logger.info(f" Integrator: {configuration["simulation_config"]["integrator"]}")
    logger.info(f"------------------------------------------")
    logger.info(f" Saved Snapshots dt: {configuration["json_and_xyzv_snapshots_dt"]}")
    logger.info(f" Output Directory: {io_paths.gts_dir}")
    logger.info(f" Cores in parallel: {max_workers}")
    logger.info(f"------------------------------------------")
    logger.info(f"")
    logger.info(f"")
    computing_time = time.perf_counter()

    gts_args = []
    for cluster_json_file_path in cluster_json_files_path:
        metadata = Metadata (
            end_time = configuration["cluster_evolution_in_void"]["end_time"]
        )
        gts_args.append((cluster_json_file_path, metadata, configuration["json_and_xyzv_snapshots_dt"]))

    with Pool(processes=max_workers, initializer=initialise_logging_worker, initargs=(io_paths, LOGGING_VERBOSE, LOGGING_DEBUG)) as pool:
        for i, _ in enumerate(pool.imap_unordered(_run_cluster_evolution_in_void_wrapper, gts_args), 1):
            if i % 5 == 0 or i == total_tasks:
                logger.info(f"\033[32mProgress: {i}/{total_tasks} clusters simulated.\033[0m")

    computing_time = time.perf_counter() - computing_time

    logger.info(f"------------------------------------------")
    logger.info(f"    CLUSTER EVOLUTION COMPLETE")
    logger.info(f"")
    logger.info(f" Execution time: {computing_time:.3f} seconds")
    logger.info(f"------------------------------------------")
    logger.info(f"")
    logger.info(f"")

def main():
    global io_paths
    global LOGGING_VERBOSE
    global LOGGING_DEBUG

    parser = argparse.ArgumentParser(
        description="Generate Plummer clusters and simulate it in the void"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../configuration.json",
        help="path to configuration.json file"
    )
    parser.add_argument(
        "--load-cluster-dir",
        type=str,
        default=None,
        help="path to an existing directory containing JSON cluster files"
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
    LOGGING_VERBOSE = args.verbose
    LOGGING_DEBUG = args.debug
    configure_logging(args.verbose, args.debug)
    logger = logging.getLogger(__name__)

    io_paths = IOPaths(configuration["output_directory"])

    io_manager.set_io_paths(io_paths)

    if not args.load_cluster_dir:
        _run_cluster_generation(configuration)
        cluster_json_dir_path = io_paths.gen_json_dir
    else:
        cluster_json_dir_path = args.load_cluster_dir

    _run_cluster_evolution_in_void(configuration, cluster_json_dir_path)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
