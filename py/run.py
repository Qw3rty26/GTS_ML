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
import multiprocessing as mp
import numpy as np
from itertools import product
from multiprocessing import Pool
from cluster_generation import Cluster_Generation, Cluster_Configuration, Simulation_Configuration
from galactic_tidal_stripping import Galactic_Tidal_Stripping, Galactic_Tidal_Stripping_Configuration, Galaxy_Configuration
from generate_machine_learning_dataset import Generate_Machine_Learning_Dataset
from file_handle import File_Handle

class Debug_Formatter(logging.Formatter):
    PURPLE = "\033[35m"
    RESET = "\033[0m"

    def format(self, record):
        message = super().format(record)
        if record.levelno == logging.DEBUG:
            return f"{self.PURPLE}{message}{self.RESET}"
        return message


def _configure_logging(debug):
    handler = logging.StreamHandler()
    handler.setFormatter(Debug_Formatter("%(levelname)s: %(message)s"))

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        handlers=[handler]
    )

    logging.getLogger("numba").setLevel(logging.WARNING) # numba needs to shut up man

def _run_cluster_generation_wrapper(args):
    cluster_configuration = args
    cluster_generation = Cluster_Generation(cluster_configuration)
    cluster_generation.run()

def _run_cluster_generation(configuration, cluster_seeds, working_directory_path):
    max_workers = configuration["cores_used"]

    number_of_clusters = configuration["cluster_generation"]["number_of_clusters"]

    simulation_configuration = Simulation_Configuration(
        integration_time = configuration["simulation_config"]["dt"],
        starting_time = configuration["simulation_config"]["t"],
        gravitational_constant = configuration["simulation_config"]["G"],
        softening = configuration["simulation_config"]["softening"],
        integrator = configuration["simulation_config"]["integrator"]
    )

    logging.debug(
        "------- CLUSTER GENERATION -------\n"
        "Clusters: %d | Particles: %d | Radius: %s\n"
        "Workers: %d | Snapshot interval: %s\n"
        "Integrator: %s | dt: %s | G: %s | Softening: %s",
        number_of_clusters,
        configuration["cluster_generation"]["number_of_particles"],
        configuration["cluster_generation"]["cluster_radius"],
        max_workers,
        configuration["cluster_generation"]["snapshot_interval"],
        configuration["simulation_config"]["integrator"],
        configuration["simulation_config"]["dt"],
        configuration["simulation_config"]["G"],
        configuration["simulation_config"]["softening"]
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

    logging.debug("Prepared %d cluster generation tasks", len(cluster_generation_arguments))
    with Pool(
        processes = max_workers,
        initializer = _configure_logging,
        initargs = (logging.getLogger().isEnabledFor(logging.DEBUG),)
    ) as pool:
        for i, _ in enumerate(pool.imap_unordered(_run_cluster_generation_wrapper, cluster_generation_arguments), 1):
            if i % 5 == 0 or i == number_of_clusters:
                logging.info("%d/%d clusters generated", i, number_of_clusters)

def _run_galactic_tidal_stripping_wrapper(args):
    galactic_tidal_stripping_configuration, cluster_file_path = args
    galactic_tidal_stripping = Galactic_Tidal_Stripping(galactic_tidal_stripping_configuration, cluster_file_path)
    galactic_tidal_stripping.run()


def _run_galactic_tidal_stripping(configuration, cluster_seeds, working_directory_path):
    max_workers = configuration["cores_used"]

    simulation_configuration = Simulation_Configuration(
        integration_time = configuration["simulation_config"]["dt"],
        starting_time = configuration["simulation_config"]["t"],
        gravitational_constant = configuration["simulation_config"]["G"],
        softening = configuration["simulation_config"]["softening"],
        integrator = configuration["simulation_config"]["integrator"]
    )

    logging.debug(
        "------- GALACTIC TIDAL STRIPPING -------\n"
        "Cluster seeds: %d | Workers: %d\n"
        "Galaxy mass: %s | Galaxy radius: %s\n"
        "Semimajor axes: %s | Eccentricities: %s\n"
        "Number of orbits: %s | End time: %s\n"
        "Simulate isolated cluster: %s\n"
        "Integrator: %s | dt: %s | G: %s | Softening: %s",
        len(cluster_seeds),
        max_workers,
        configuration["galactic_tidal_stripping"]["galaxy_mass"],
        configuration["galactic_tidal_stripping"]["galaxy_radius"],
        configuration["galactic_tidal_stripping"]["orbit_semi_major_axis"],
        configuration["galactic_tidal_stripping"]["orbit_eccentricity"],
        configuration["galactic_tidal_stripping"]["number_of_orbits"],
        configuration["galactic_tidal_stripping"]["end_time"],
        configuration["galactic_tidal_stripping"]["simulate_isolated_cluster"],
        configuration["simulation_config"]["integrator"],
        configuration["simulation_config"]["dt"],
        configuration["simulation_config"]["G"],
        configuration["simulation_config"]["softening"]
    )

    galactic_tidal_stripping_arguments = []
    for cluster_seed, (orbit_semi_major_axis, orbit_eccentricity) in product(
        cluster_seeds,
        product(configuration["galactic_tidal_stripping"]["orbit_semi_major_axis"], configuration["galactic_tidal_stripping"]["orbit_eccentricity"])
    ):
        snapshot_file_path = os.path.join(working_directory_path, "GTS", f"{cluster_seed}_a{orbit_semi_major_axis}_e{orbit_eccentricity}")
        cluster_file_path = os.path.join(working_directory_path, "GEN", f"{cluster_seed}", f"{cluster_seed}.json")

        galactic_tidal_stripping_configuration = Galactic_Tidal_Stripping_Configuration(
            simulation_configuration,
            galaxy_configuration = Galaxy_Configuration(
                galaxy_mass = configuration["galactic_tidal_stripping"]["galaxy_mass"],
                galaxy_radius = configuration["galactic_tidal_stripping"]["galaxy_radius"],
                orbit_semi_major_axis = orbit_semi_major_axis,
                orbit_eccentricity = orbit_eccentricity
            ),
            number_of_orbits = configuration["galactic_tidal_stripping"]["number_of_orbits"],
            simulate_isolated_cluster = configuration["galactic_tidal_stripping"]["simulate_isolated_cluster"],
            end_time = configuration["galactic_tidal_stripping"]["end_time"],
            cluster_seed = cluster_seed,
            snapshot_interval = configuration["galactic_tidal_stripping"]["snapshot_interval"],
            snapshot_file_path = snapshot_file_path
        )

        os.makedirs(snapshot_file_path, exist_ok=True)
        galactic_tidal_stripping_arguments.append((galactic_tidal_stripping_configuration, cluster_file_path))

    logging.debug("Prepared %d galactic tidal stripping tasks", len(galactic_tidal_stripping_arguments))

    with Pool(
        processes = max_workers,
        initializer = _configure_logging,
        initargs = (logging.getLogger().isEnabledFor(logging.DEBUG),)
    ) as pool:
        for i, _ in enumerate(pool.imap_unordered(_run_galactic_tidal_stripping_wrapper, galactic_tidal_stripping_arguments), 1):
            logging.info("%d/%d galactic tidal stripping simulations completed.", i, len(galactic_tidal_stripping_arguments))

def _run_machine_learning_dataset_generation(working_directory):

    logging.debug("Dataset source directory: %s", working_directory)

    generate_machine_learning_dataset = Generate_Machine_Learning_Dataset(working_directory)
    generate_machine_learning_dataset.run()

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
    _configure_logging(command_arguments.debug)

    configuration_file_path = os.path.abspath(command_arguments.config)
    logging.debug("Configuration file: %s", configuration_file_path)

    configuration = File_Handle.load_json_file(configuration_file_path)

    working_directory_path = _get_working_directory_path(configuration["output_directory"])
    number_of_clusters = configuration["cluster_generation"]["number_of_clusters"]
    master_seed = configuration["cluster_generation"]["master_seed"]
    master_rng = np.random.Generator(np.random.MT19937(master_seed))
    cluster_seeds = master_rng.integers(low=0, high=2**31 - 1, size = number_of_clusters, dtype=int)

    logging.info("Output directory: %s", working_directory_path)
    logging.debug("Master seed: %d", master_seed)

    logging.info("Starting cluster generation (output: %s)", os.path.join(working_directory_path, "GEN"))
    _run_cluster_generation(configuration, cluster_seeds, working_directory_path)
    logging.info("Cluster generation completed")

    logging.info("Starting galactic tidal stripping simulations (output: %s)", os.path.join(working_directory_path, "GTS"))
    _run_galactic_tidal_stripping(configuration, cluster_seeds, working_directory_path)
    logging.info("Galactic tidal stripping simulations completed")

    logging.info("Starting dataset generation (output: %s)", working_directory_path)
    _run_machine_learning_dataset_generation(working_directory_path)
    logging.info("Dataset generation completed")

    logging.info("Execution completed")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
