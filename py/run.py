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
from galactic_tidal_stripping import Galactic_Tidal_Stripping
from verify_physics import Verify_Physics
from ml_dataset_generation import ML_Dataset_Generation

from io_manager import IOPaths, Metadata
from simulation import SimulationConfig
from logger_settings import configure_logging

LOGGING_VERBOSE = True
LOGGING_DEBUG = False
logger = logging.getLogger(__name__)

io_paths = None

def initialise_logging_worker(io_paths):
    configure_logging(LOGGING_VERBOSE, LOGGING_DEBUG)
    io_manager.set_io_paths(io_paths)

def _run_cluster_generation_wrapper(args):
    simulation_config, metadata = args
    cluster_generation = Cluster_Generation(
        simulation_config = simulation_config,
        metadata = metadata
    )
    cluster_generation.run()


def _run_cluster_generation(configuration):

    logger.info(f"CLUSTER GENERATION STARTED")
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
            initial_number_of_entities = configuration["cluster_generation"]["number_of_stars"]
        )
        gen_args.append((simulation_config, metadata))

    max_workers = configuration.get("cores_used", os.cpu_count())
    with Pool(processes=max_workers, initializer=initialise_logging_worker, initargs=(io_paths,)) as pool:
        for i, _ in enumerate(pool.imap_unordered(_run_cluster_generation_wrapper, gen_args), 1):
            if i % 5 == 0 or i == number_of_clusters:
                logger.info(f"\033[32mProgress: {i}/{number_of_clusters} clusters generated.\033[0m")

    computing_time = time.perf_counter() - computing_time
    logger.debug(f"Execution time: {computing_time:.3f} seconds")

    logger.info(f"CLUSTER GENERATION COMPLETE")

def _run_galactic_tidal_stripping_wrapper(args):
    cluster_json_file, metadata = args
    galactic_tidal_stripping = Galactic_Tidal_Stripping(
        cluster_json_file = cluster_json_file,
        metadata = metadata
    )
    galactic_tidal_stripping.run()

def _run_galactic_tidal_stripping(configuration, cluster_json_dir_path):

    logger.info(f"GALACTIC TIDAL STRIPPING STARTED")
    computing_time = time.perf_counter()

    cluster_json_files_path = [
        os.path.join(cluster_json_dir_path, f)
        for f in os.listdir(cluster_json_dir_path)
        if f.endswith(".json")
    ]

    if not cluster_json_files_path:
        logger.error(f"No cluster JSON files were found in {cluster_json_dir_path}!")
        return

    orbit_semi_major_axes = configuration["ml_dataset_input"]["semi_major_axes"]
    orbit_eccentricities = configuration["ml_dataset_input"]["eccentricities"]

    ml_grid = list(itertools.product(orbit_semi_major_axes, orbit_eccentricities))
    gts_args = []
    for cluster_json_file_path in cluster_json_files_path:
        for orbit_semi_major_axis, orbit_eccentricity in ml_grid:
            metadata = Metadata (
                galaxy_mass = configuration["galactic_environment"]["galaxy_mass"],
                galaxy_radius = configuration["galactic_environment"]["galaxy_radius"],
                orbit_semi_major_axis = orbit_semi_major_axis,
                orbit_eccentricity = orbit_eccentricity,
                number_of_orbits = configuration["galactic_environment"]["number_of_orbits"]
            )
            gts_args.append((cluster_json_file_path, metadata))

    max_workers = configuration.get("cores_used", os.cpu_count())
    total_tasks = len(gts_args)
    with Pool(processes=max_workers, initializer=initialise_logging_worker, initargs=(io_paths,)) as pool:
        for i, _ in enumerate(pool.imap_unordered(_run_galactic_tidal_stripping_wrapper, gts_args), 1):
            if i % 5 == 0 or i == total_tasks:
                logger.info(f"\033[32mProgress: {i}/{total_tasks} galaxies simulated.\033[0m")

    computing_time = time.perf_counter() - computing_time
    logger.debug(f"Execution time: {computing_time:.3f} seconds")

    logger.info(f"GALACTIC TIDAL STRIPPING COMPLETE")

def _run_verify_physics(cluster_json_dir_path):
    verify_physics = Verify_Physics(cluster_json_dir_path)
    verify_physics.run()

def _run_ml_dataset_generation(gts_json_dir_path):
    ml_dataset_generation = ML_Dataset_Generation(
        gts_json_dir_path
    )
    ml_dataset_generation.run()

def main():
    global io_paths

    parser = argparse.ArgumentParser(
        description="Generate Plummer clusters and simulate galactic tidal strippings"
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
    configure_logging(args.verbose, args.debug)
    logger = logging.getLogger(__name__)

    io_paths = IOPaths(configuration["output_directory"])

    io_manager.set_io_paths(io_paths)

    if not args.load_cluster_dir:
        _run_cluster_generation(configuration)
        cluster_json_dir_path = io_paths.gen_json_dir
        _run_galactic_tidal_stripping(configuration, cluster_json_dir_path)
    else:
        _run_galactic_tidal_stripping(configuration, args.load_cluster_dir)

    _run_verify_physics(io_paths.gen_json_dir)
    _run_verify_physics(io_paths.gts_json_dir)

    _run_ml_dataset_generation(io_paths.gts_json_dir)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
