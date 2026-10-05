import logging
import os
import numpy as np
import io_manager
from simulation import Simulation
from plummer import Plummer
from galactic_potential import GalacticPotential

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

        logger.debug(
            f"Simulation no. {cluster_name}: Cleaning {len(escaped_entity_ids)} star/s "
            f"Relative error before cleaning: {relative_error:.8f}% "
            f"New total energy: {new_total_energy:.8f}"
        )


def generate_cluster(seed, cluster_radius, number_of_stars, dt, G, softening, time_warp, integrator):
#TODO aggiungere un seed di base
    STAR_MASS = 1.0 / number_of_stars

    logger.debug(f"Cluster no. {seed}: Generating Plummer Cluster...")

    plummer = Plummer(cluster_radius, number_of_stars, seed=seed)
    positions, velocities = plummer.generate_plummer_cluster()

    simulation = Simulation(dt, G, softening, time_warp, integrator)

    for id, (pos, vel) in enumerate(zip(positions, velocities)):
        simulation.add_entity(pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], STAR_MASS, id)

    simulation.simulation.move_to_com()
    simulation.cluster_diagnostics.set_initial_total_energy()
    simulation.cluster_diagnostics.set_initial_angular_momentum()
    return simulation


def clean_cluster(cluster_id, simulation, end_time, xyzv_file, json_file):

    dt_cleanup = 1.0
    dt_snapshot = 0.01

    next_cleanup = simulation.simulation.t + dt_cleanup
    next_snapshot = simulation.simulation.t + dt_snapshot
    io_manager.init_json_snapshot(simulation, json_file)

    while simulation.simulation.t < end_time:
        simulation.update()

        if simulation.simulation.t >= next_snapshot:
            io_manager.save_xyzv_snapshot(simulation, xyzv_file)
            io_manager.record_json_snapshot(simulation, json_file)
            next_snapshot += dt_snapshot

        if simulation.simulation.t >= next_cleanup:
            next_cleanup += dt_cleanup
            try_to_clean_stars(cluster_id, simulation)


def evolve_cluster(cluster_name, simulation, number_of_orbits, xyzv_file, json_file):

    dt_cleanup = 1.0
    dt_snapshot = 0.1

    next_cleanup = simulation.simulation.t + dt_cleanup
    next_snapshot = simulation.simulation.t + dt_snapshot
    io_manager.init_json_snapshot(simulation, json_file)
    orbits = 0

    while simulation.cluster_diagnostics.get_cluster_orbits() < number_of_orbits:

        simulation.update()

        if simulation.simulation.t >= next_snapshot:
            io_manager.save_xyzv_snapshot(simulation, xyzv_file)
            io_manager.record_json_snapshot(simulation, json_file)
            next_snapshot += dt_snapshot

#       if simulation.simulation.t >= next_cleanup:
#           next_cleanup += dt_cleanup
#           try_to_clean_stars(cluster_name, simulation)

        simulation.cluster_diagnostics.update_orbital_angle()

        new_orbits = simulation.cluster_diagnostics.get_cluster_orbits()

        if orbits != new_orbits:
            orbits = new_orbits
            logger.info(f"Simulation no. {cluster_name}: {orbits}/{number_of_orbits} orbits done")


def run_cluster_generation(seed, cluster_radius, number_of_stars, dt, G, softening, time_warp, integrator, json_output_path, xyzv_output_path):

    logger.info(f"Cluster no. {seed}: Generating...")

    simulation = generate_cluster(seed, cluster_radius, number_of_stars, dt, G, softening, time_warp, integrator)
    initial_total_energy = simulation.cluster_diagnostics.get_initial_total_energy()
    initial_angular_momentum = simulation.cluster_diagnostics.get_initial_angular_momentum()
    logger.debug(
        f"Cluster no. {seed}:\n"
        f"Initial total energy: {initial_total_energy:.7f}\n"
        f"Initial angular momentum: {initial_angular_momentum}\n"
    )

    END_TIME = (
        20 * cluster_radius ** (3 / 2)
        / np.sqrt(number_of_stars)
    )

    os.makedirs(json_output_path, exist_ok=True)
    os.makedirs(xyzv_output_path, exist_ok=True)

    xyzv_path = os.path.join(xyzv_output_path, f"cluster_{seed}.xyzv")
    json_path = os.path.join(json_output_path, f"cluster_{seed}.json")

    with open(xyzv_path, "w") as xyzv_file:
        logger.debug(f"Cluster no. {seed}: Cleaning...")
        clean_cluster(seed, simulation, END_TIME, xyzv_file, json_path)

    final_total_energy = simulation.cluster_diagnostics.get_total_energy()
    energy_relative_error = simulation.cluster_diagnostics.get_total_energy_relative_error_percentage()
    final_angular_momentum = simulation.cluster_diagnostics.get_total_angular_momentum()
    angular_momentum_relative_error = simulation.cluster_diagnostics.get_total_angular_momentum_error_percentage()
    logger.debug(
        f"Cluster no. {seed}:\n"
        f"Final cluster total energy: {final_total_energy:.7f} ({energy_relative_error:.7f}% relative error)\n"
        f"Final cluster angular momentum: {final_angular_momentum} ({angular_momentum_relative_error:.4f}% relative error)\n"
    )

    logger.debug(f"Cluster no. {seed}: Done.")


def run_galaxy_tidal_stripping(cluster_path, galaxy_mass, galaxy_radius, a, e, json_output_path, xyzv_output_path, sim_id):
    NUMBER_OF_ORBITS = 2
    cluster_name = os.path.splitext(os.path.basename(cluster_path))[0]

    logger.debug(f"Simulation {sim_id} ({cluster_name}, a={a}, e={e}): Running...")

    with open(cluster_path, "r") as cluster_file:
        simulation = io_manager.load_json_snapshot(cluster_file)

    galactic_potential = GalacticPotential(galaxy_radius, galaxy_mass)
    simulation.add_galactic_potential(galactic_potential)

    r_apo = a * (1.0 + e)
    v_apo = galactic_potential.get_elliptical_apocenter_velocity(a, e)

    simulation.move_cluster(r_apo, 0, 0)
    simulation.speed_cluster(0, v_apo, 0)
    simulation.cluster_diagnostics.set_initial_total_energy()

    initial_total_energy = simulation.cluster_diagnostics.get_initial_total_energy()
    logger.debug(
        f"Simulation {sim_id}:\n"
        f"Initial total energy: {initial_total_energy:.7f}\n"
    )

    os.makedirs(xyzv_output_path, exist_ok=True)
    os.makedirs(json_output_path, exist_ok=True)

    xyzv_path = os.path.join(xyzv_output_path, f"sim_{sim_id}.xyzv")
    json_path = os.path.join(json_output_path, f"sim_{sim_id}.json")

    with open(xyzv_path, "w") as xyzv_file:
        logger.debug(f"Simulation {sim_id}: Evolving...")
        evolve_cluster(sim_id, simulation, NUMBER_OF_ORBITS, xyzv_file, json_path)

    final_total_energy = simulation.cluster_diagnostics.get_total_energy()
    energy_relative_error = simulation.cluster_diagnostics.get_total_energy_relative_error_percentage()
    logger.debug(
        f"Simulation {sim_id}:\n"
        f"Final cluster total energy: {final_total_energy:.7f} ({energy_relative_error:.7f}% relative error)\n"
    )

    logger.debug(f"Simulation {sim_id}: Done.")
