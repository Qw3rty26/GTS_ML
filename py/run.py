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
import multiprocessing as mp
import numpy as np
from multiprocessing import Pool
from cluster_generation import Cluster_Generation, Cluster_Configuration
from simulation import Simulation_Configuration
from file_handle import File_Handle

def _run_cluster_generation_wrapper(args):
    cluster_configuration = args
    cluster_generation = Cluster_Generation(cluster_configuration)
    cluster_generation.run()

def _run_cluster_generation(configuration, working_directory_path):
    max_workers = configuration["cores_used"]

    number_of_clusters = configuration["cluster_generation"]["number_of_clusters"]
    master_seed = configuration["cluster_generation"]["master_seed"]
    master_rng = np.random.Generator(np.random.MT19937(master_seed))
    cluster_seeds = master_rng.integers(low=0, high=2**31 - 1, size = number_of_clusters, dtype=int)

    simulation_configuration = Simulation_Configuration(
        integration_time = configuration["simulation_config"]["dt"],
        starting_time = configuration["simulation_config"]["t"],
        gravitational_constant = configuration["simulation_config"]["G"],
        softening = configuration["simulation_config"]["softening"],
        integrator = configuration["simulation_config"]["integrator"]
    )

    cluster_generation_arguments = []
    for cluster_seed in cluster_seeds:
        cluster_configuration = Cluster_Configuration (
            simulation_configuration,
            number_of_particles = configuration["cluster_generation"]["number_of_particles"],
            master_seed = configuration["cluster_generation"]["master_seed"],
            cluster_seed = int(cluster_seed),
            cluster_radius = configuration["cluster_generation"]["cluster_radius"],
            snapshot_interval = configuration["cluster_generation"]["snapshot_interval"],
            snapshot_file_path = os.path.join(working_directory_path, "GEN", f"{cluster_seed}")
        )
        os.makedirs(cluster_configuration.snapshot_file_path, exist_ok=True)
        cluster_generation_arguments.append(cluster_configuration)

    with Pool(processes = max_workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_run_cluster_generation_wrapper, cluster_generation_arguments), 1):
            if i % 5 == 0 or i == number_of_clusters:
                print(f"{i}/{number_of_clusters} clusters generated")


def _get_working_directory_path(output_directory):
    os.makedirs(output_directory, exist_ok=True)

    existing_runs = []

    for name in os.listdir(output_directory):
        run_name = name.removeprefix("run_")

        if (name.startswith("run_") and run_name.isdigit() and os.path.isdir(os.path.join(output_directory, name))):
            existing_runs.append(int(run_name))

    run_number = max(existing_runs, default=0) + 1

    working_directory_path = os.path.join(output_directory, f"run_{run_number:03d}")
    os.makedirs(os.path.join(working_directory_path, "GEN"), exist_ok=True)
    return working_directory_path


def main():
    parser = argparse.ArgumentParser(description = "GTS")
    parser.add_argument("--config", type = str, default = "./configuration.json", help = "path to configuration.json file")
    parser.add_argument("--debug", action="store_true", help="enable debug output")

    command_arguments = parser.parse_args()
    configuration_file_path = os.path.abspath(command_arguments.config)
    configuration = File_Handle.load_json_file(configuration_file_path)

    working_directory_path = _get_working_directory_path(configuration["output_directory"])

    _run_cluster_generation(configuration, working_directory_path)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
