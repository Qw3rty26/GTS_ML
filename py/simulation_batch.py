from simulation import Simulation
from plummer import Plummer
from galactic_potential import GalacticPotential
import logging
import io_manager
import numpy as np
import os

logger = logging.getLogger(__name__)

def try_to_clean_stars(cluster_name, simulation):
    escaped_entity_ids = (
        simulation.cluster_diagnostics.get_escaped_entity_ids()
    )

    if escaped_entity_ids:

        relative_error = simulation.cluster_diagnostics.get_total_energy_relative_error_percentage()

        simulation.clean_escaped_stars(escaped_entity_ids)

        simulation.cluster_diagnostics.set_initial_total_energy()
        new_total_energy = simulation.cluster_diagnostics.get_total_energy()

        logger.debug(f"Simulation no. {cluster_name}: Cleaning {len(escaped_entity_ids)} star/s Relative error before cleaning: {relative_error:.8f}% New total energy: {new_total_energy:.8f}")



def generate_cluster(cluster_id, cluster_radius, number_of_stars, dt, G, softening, time_warp, integrator):

    np.random.seed(cluster_id)
    STAR_MASS = 1.0 / number_of_stars

    logger.debug(f"Cluster no. {cluster_id}: Generating Plummer Cluster...")

    plummer = Plummer(cluster_radius, number_of_stars)
    positions, velocities = plummer.generate_plummer_cluster()

    simulation = Simulation(dt, G, softening, time_warp, integrator)

    for id, (pos, vel) in enumerate(zip(positions, velocities)):
        simulation.add_entity(pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], STAR_MASS, id)

    simulation.simulation.move_to_com()
    simulation.cluster_diagnostics.set_initial_total_energy()
    simulation.cluster_diagnostics.set_initial_angular_momentum()
    return simulation


def clean_cluster(cluster_id, simulation, end_time, xyzv_file, json_file):

    next_cleanup = simulation.simulation.t + 1.0
    next_snapshot = simulation.simulation.t + 0.01
    io_manager.init_json_snapshot(simulation, json_file)

    while simulation.simulation.t < end_time:
        simulation.update()

        if simulation.simulation.t >= next_snapshot:
            io_manager.save_xyzv_snapshot(simulation, xyzv_file)
            io_manager.record_json_snapshot(simulation, json_file)
            next_snapshot += 0.01

        if simulation.simulation.t >= next_cleanup:
            next_cleanup += 1.0
            try_to_clean_stars(cluster_id, simulation)


def evolve_cluster(cluster_name, simulation, number_of_orbits, xyzv_file, json_file):

    next_cleanup = simulation.simulation.t + 1.0
    next_snapshot = simulation.simulation.t + 1.0
    io_manager.init_json_snapshot(simulation, json_file)
    orbits = 0

    while simulation.cluster_diagnostics.get_cluster_orbits() < number_of_orbits:

        simulation.update()

        if simulation.simulation.t >= next_snapshot:
            io_manager.save_xyzv_snapshot(simulation, xyzv_file)
            io_manager.record_json_snapshot(simulation, json_file)
            next_snapshot += 0.1

#        if simulation.simulation.t >= next_cleanup:
#            next_cleanup += 1.0
#            try_to_clean_stars(cluster_name, simulation)

        simulation.cluster_diagnostics.update_orbital_angle()

        new_orbits = simulation.cluster_diagnostics.get_cluster_orbits()

        if orbits != new_orbits:
            orbits = new_orbits
            logger.info(f"Cluster no. {cluster_name}: {orbits}/{number_of_orbits} done")


def run_cluster_generation(cluster_id, cluster_radius, number_of_stars, dt, G, softening, time_warp, integrator, json_output_path, xyzv_output_path):

    logger.info(f"Cluster no. {cluster_id}: Generating...")

    simulation = generate_cluster(cluster_id, cluster_radius, number_of_stars, dt, G, softening, time_warp, integrator)
    initial_total_energy = simulation.cluster_diagnostics.get_initial_total_energy()
    initial_angular_momentum = simulation.cluster_diagnostics.get_initial_angular_momentum()
    logger.debug(f"Cluster no. {cluster_id}:\n"
                 f"Initial total energy: {initial_total_energy:.7f}\n"
                 f"Initial angular momentum: {initial_angular_momentum}\n"
                )

    END_TIME = (
        20 * cluster_radius ** (3 / 2)
        / np.sqrt(number_of_stars)
    )

    os.makedirs(json_output_path, exist_ok=True)
    os.makedirs(xyzv_output_path, exist_ok=True)

    xyzv_path = os.path.join(xyzv_output_path, f"cluster_{cluster_id}.xyzv")
    json_path = os.path.join(json_output_path, f"cluster_{cluster_id}.json")

    with open(xyzv_path, "w") as xyzv_file:
        logger.debug(f"Cluster no. {cluster_id}: Cleaning...")
        clean_cluster(cluster_id, simulation, END_TIME, xyzv_file, json_path)

    final_total_energy = simulation.cluster_diagnostics.get_total_energy()
    energy_relative_error = simulation.cluster_diagnostics.get_total_energy_relative_error_percentage()
    final_angular_momentum = simulation.cluster_diagnostics.get_total_angular_momentum()
    angular_momentum_relative_error = simulation.cluster_diagnostics.get_total_angular_momentum_error_percentage()
    logger.debug(f"Cluster no. {cluster_id}:\n"
                 f"Final cluster total energy: {final_total_energy:.7f} ({energy_relative_error:.7f}% relative error)\n"
                 f"Final cluster angular momentum: {final_angular_momentum} ({angular_momentum_relative_error:.4f}% relative error)\n"
                )


    logger.debug(f"Cluster no. {cluster_id}: Done.")


def run_galaxy_tidal_stripping(cluster_path, galaxy_mass, galaxy_radius, number_of_orbits, json_output_path, xyzv_output_path):
    CLUSTER_DISTANCE_FROM_GALAXY_FACTOR = 10

    cluster_file = open(cluster_path, "r")

    cluster_name = os.path.splitext(os.path.basename(cluster_path))[0]

    logger.debug(f"Simulation {cluster_name}: Running...")

    simulation = io_manager.load_json_snapshot(cluster_file)

    galactic_potential = GalacticPotential(galaxy_radius, galaxy_mass)

    orbital_radius = CLUSTER_DISTANCE_FROM_GALAXY_FACTOR * galaxy_radius
    orbital_velocity = galactic_potential.get_cluster_initial_velocity(orbital_radius)

    simulation.add_galactic_potential(galactic_potential)

    simulation.move_cluster(orbital_radius, 0, 0)
    simulation.speed_cluster(0, orbital_velocity, 0)
    simulation.cluster_diagnostics.set_initial_total_energy()

    initial_total_energy = simulation.cluster_diagnostics.get_initial_total_energy()
    logger.debug(f"Cluster no. {cluster_name}:\n"
                 f"Initial total energy: {initial_total_energy:.7f}\n"
                )

    os.makedirs(xyzv_output_path, exist_ok=True)
    os.makedirs(json_output_path, exist_ok=True)

    xyzv_path = os.path.join(xyzv_output_path, f"{cluster_name}.xyzv")
    json_path = os.path.join(json_output_path, f"{cluster_name}.json")

    with open(xyzv_path, "w") as xyzv_file:
        logger.debug(f"Simulation {cluster_name}: Evolving...")
        evolve_cluster(cluster_name, simulation, number_of_orbits, xyzv_file, json_path)

    final_total_energy = simulation.cluster_diagnostics.get_total_energy()
    energy_relative_error = simulation.cluster_diagnostics.get_total_energy_relative_error_percentage()
    logger.debug(f"Cluster no. {cluster_name}:\n"
                 f"Final cluster total energy: {final_total_energy:.7f} ({energy_relative_error:.7f}% relative error)\n"
                )

    logger.debug(f"Simulation {cluster_name}: Done.")
