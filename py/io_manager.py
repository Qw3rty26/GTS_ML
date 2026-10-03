from simulation import Simulation
from galactic_potential import GalacticPotential

import json
import os

#TODO save json and xyzv paths in here, and give them to simulation_batch under a io_manager object
#TODO remove the opening of files from other .py

def create_run_directory(output_path):
    run_id = 0

    while os.path.exists(os.path.join(output_path, f"run_{run_id:03d}")):
        run_id += 1

    run_path = os.path.join(output_path, f"run_{run_id:03d}")
    os.makedirs(os.path.join(run_path, "GEN", "JSON"))
    os.makedirs(os.path.join(run_path, "GEN", "XYZV"))
    os.makedirs(os.path.join(run_path, "GTS", "JSON"))
    os.makedirs(os.path.join(run_path, "GTS", "XYZV"))

    return run_path

def get_latest_run_directory(output_path):
    run_id = 1
    latest_path = None
    while os.path.exists(os.path.join(output_path, f"run_{run_id:03d}")):
        latest_path = os.path.join(output_path, f"run_{run_id:03d}")
        run_id += 1
    return latest_path

def save_xyzv_snapshot(simulation, file_handle):
    has_galaxy = simulation.galactic_potential is not None
    num_particles = len(simulation.simulation.particles)
    extra_lines = 2 if has_galaxy else 1

    file_handle.write(f"{num_particles + extra_lines}\n")

    if has_galaxy:
        gp = simulation.galactic_potential
        file_handle.write(
            f"Galactic Tidal Stripped Cluster "
            f"Properties=species:S:1:id:I:1:pos:R:3:vel:R:3 "
            f"t={simulation.simulation.t} dt={simulation.simulation.dt} "
            f"M={gp.get_galaxy_mass()} a={gp.get_galaxy_radius()}\n"
        )
        file_handle.write("O 10000 0 0 0 0 0 0\n")
    else:
        file_handle.write(
            f"Plummer Star Cluster "
            f"Properties=species:S:1:id:I:1:pos:R:3:vel:R:3 "
            f"t={simulation.simulation.t} dt={simulation.simulation.dt}\n"
        )

    com = simulation.cluster_diagnostics.get_center_of_mass()
    file_handle.write(f"C 9999 {com.x} {com.y} {com.z} {com.vx} {com.vy} {com.vz}\n")

    for p in simulation.simulation.particles:
        file_handle.write(f"H {p.name} {p.x} {p.y} {p.z} {p.vx} {p.vy} {p.vz}\n")


def init_json_snapshot(simulation, file_path):

    galaxy_data = None
    if simulation.galactic_potential is not None:
        galaxy_data = {
            "mass": simulation.galactic_potential.get_galaxy_mass(),
            "radius": simulation.galactic_potential.get_galaxy_radius(),
        }

    header = {
        "dt": simulation.simulation.dt,
        "G": simulation.simulation.G,
        "softening": simulation.simulation.softening,
        "time_warp": simulation.time_warp,
        "integrator": str(simulation.simulation.integrator),
        "galaxy": galaxy_data,
    }

    with open(file_path, "w") as f:
        f.write(json.dumps({"metadata": header}) + "\n")


def record_json_snapshot(simulation, file_path):

    snapshot = {
        "time": simulation.simulation.t,
        "entities": [
            {
                "name": p.name,
                "mass": p.m,
                "x": p.x,
                "y": p.y,
                "z": p.z,
                "vx": p.vx,
                "vy": p.vy,
                "vz": p.vz,
            }
            for p in simulation.simulation.particles
        ],
    }

    with open(file_path, "a") as f:
        f.write(json.dumps(snapshot) + "\n")

def load_json_snapshot(file_handle, snapshot_index=-1):

    lines = [
        line.strip() for line in file_handle.readlines() if line.strip()
    ]

    data = json.loads(lines[0])["metadata"]

    simulation = Simulation(
        dt=data["dt"],
        G=data["G"],
        softening=data["softening"],
        time_warp=data["time_warp"],
        integrator=data["integrator"],
    )

    if data["galaxy"]:
        galactic_potential = GalacticPotential(
            data["galaxy"]["radius"], data["galaxy"]["mass"]
        )
        simulation.add_galactic_potential(galactic_potential)

    line_index = (
        snapshot_index + 1 if snapshot_index >= 0 else snapshot_index
    )
    chosen_snapshot = json.loads(lines[line_index])

    simulation.simulation.t = chosen_snapshot["time"]

    for entity in chosen_snapshot["entities"]:
        simulation.add_entity(
            x=entity["x"],
            y=entity["y"],
            z=entity["z"],
            vx=entity["vx"],
            vy=entity["vy"],
            vz=entity["vz"],
            mass=entity["mass"],
            name=entity["name"],
        )

    return simulation

def load_json_file(file_handle):
    with open(file_handle, "r") as file:
        return json.load(file)
