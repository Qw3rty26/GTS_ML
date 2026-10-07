import os
import json

from functools import cached_property
from dataclasses import dataclass, field
from entity import Entity, Position, Velocity
from simulation import SimulationConfig


@dataclass
class IOPaths:

    output_path: str = "./clusters"

    @cached_property
    def get_run_dir(self):
        run_id = 1
        while os.path.exists(os.path.join(self.output_path, f"run_{run_id:03d}")):
            run_id += 1
        dir = os.path.join(self.output_path, f"run_{run_id:03d}")
        os.makedirs(dir, exist_ok=True)
        return dir

    @property
    def gen_json_dir(self):
        dir = os.path.join(self.get_run_dir, "GEN", "JSON")
        os.makedirs(dir, exist_ok=True)
        return dir

    @property
    def gen_xyzv_dir(self):
        dir = os.path.join(self.get_run_dir, "GEN", "XYZV")
        os.makedirs(dir, exist_ok=True)
        return dir

    @property
    def gts_json_dir(self):
        dir = os.path.join(self.get_run_dir, "GTS", "JSON")
        os.makedirs(dir, exist_ok=True)
        return dir

    @property
    def gts_xyzv_dir(self):
        dir = os.path.join(self.get_run_dir, "GTS", "XYZV")
        os.makedirs(dir, exist_ok=True)
        return dir

    def get_gen_xyzv_path(self, cluster_seed):
        xyzv_path = os.path.join(self.gen_xyzv_dir, f"{cluster_seed}.xyzv")
        return xyzv_path

    def get_gen_json_path(self, cluster_seed):
        json_path = os.path.join(self.gen_json_dir, f"{cluster_seed}.json")
        return json_path

    def get_gts_xyzv_path(self, cluster_seed, orbit_semi_major_axis, orbit_eccentricity):
        xyzv_path = os.path.join(self.gts_xyzv_dir, f"{cluster_seed}_a{orbit_semi_major_axis}_e{orbit_eccentricity}.xyzv")
        return xyzv_path

    def get_gts_json_path(self, cluster_seed, orbit_semi_major_axis, orbit_eccentricity):
        json_path = os.path.join(self.gts_json_dir, f"{cluster_seed}_a{orbit_semi_major_axis}_e{orbit_eccentricity}.json")
        return json_path

    def get_csv_path(self):
        return os.path.join(self.get_run_dir, "ml_dataset.csv")

_paths = IOPaths()

def set_io_paths(io_paths: IOPaths):
    _ = io_paths.get_run_dir
    global _paths
    _paths = io_paths

def get_csv_path():
    return _paths.get_csv_path()

@dataclass
class Metadata:
    master_seed: int = 1
    number_of_clusters: int = 1
    cluster_seed: int = 1
    cluster_radius: float = 1.0
    initial_number_of_entities: int = 10
    simulation_config: SimulationConfig = field(default_factory=SimulationConfig)
    galaxy_mass: float = None
    galaxy_radius: float = None
    orbit_semi_major_axis: float = None
    orbit_eccentricity: float = None
    number_of_orbits: int = 1
    half_mass_radius: float = None
    bound_mass_fraction: float = None
    #need to add energy and stuff here, like angular momentum

def save_xyzv_snapshot_gen(metadata: Metadata, entities, center_of_mass: Entity):

    xyzv_path = _paths.get_gen_xyzv_path(metadata.cluster_seed)

    with open(xyzv_path, "a") as xyzv_file:

        xyzv_file.write(f"{len(entities) + 1}\n")

        xyzv_file.write(
            f"seed={metadata.cluster_seed} "
            f"t={metadata.simulation_config.t:.5f} "
            f"r={metadata.cluster_radius} "
            f"Properties=species:S:1:id:I:1:pos:R:3:vel:R:3 "
            f"\n"
        )

        xyzv_file.write(
            f"Fe " # center of mass is represented as an iron element in OVITO
            f"9999 "
            f"{center_of_mass.position.x} "
            f"{center_of_mass.position.y} "
            f"{center_of_mass.position.z} "
            f"{center_of_mass.velocity.vx} "
            f"{center_of_mass.velocity.vy} "
            f"{center_of_mass.velocity.vz}"
            f"\n"
        )

        for entity in entities:
            xyzv_file.write(
                f"H " # every entity is represented as a hydrogen element in OVITO
                f"{entity.name} "
                f"{entity.position.x} "
                f"{entity.position.y} "
                f"{entity.position.z} "
                f"{entity.velocity.vx} "
                f"{entity.velocity.vy} "
                f"{entity.velocity.vz}"
                f"\n"
            )


def save_xyzv_snapshot_gts(metadata: Metadata, entities, center_of_mass: Entity):

    xyzv_path = _paths.get_gts_xyzv_path(metadata.cluster_seed, metadata.orbit_semi_major_axis, metadata.orbit_eccentricity)

    with open(xyzv_path, "a") as xyzv_file:

        xyzv_file.write(f"{len(entities) + 2}\n")

        xyzv_file.write(
            f"seed={metadata.cluster_seed} "
            f"t={metadata.simulation_config.t:.5f} "
            f"a={metadata.orbit_semi_major_axis} "
            f"e={metadata.orbit_eccentricity} "
            f"Gmass={metadata.galaxy_mass} "
            f"Grad={metadata.galaxy_radius} "
            f"Properties=species:S:1:id:I:1:pos:R:3:vel:R:3 "
            f"\n"
        )

        xyzv_file.write("O -1 0 0 0 0 0 0\n") # center of the galaxy is represented as an oxygen element in OVITO

        xyzv_file.write(
            f"Fe " # center of mass is represented as an iron element in OVITO
            f"9999 "
            f"{center_of_mass.position.x} "
            f"{center_of_mass.position.y} "
            f"{center_of_mass.position.z} "
            f"{center_of_mass.velocity.vx} "
            f"{center_of_mass.velocity.vy} "
            f"{center_of_mass.velocity.vz}"
            f"\n"
        )

        for entity in entities:
            xyzv_file.write(
                f"H " # every entity is represented as a hydrogen element in OVITO
                f"{entity.name} "
                f"{entity.position.x} "
                f"{entity.position.y} "
                f"{entity.position.z} "
                f"{entity.velocity.vx} "
                f"{entity.velocity.vy} "
                f"{entity.velocity.vz}"
                f"\n"
            )


def init_json_gen(metadata: Metadata):

    json_path = _paths.get_gen_json_path(metadata.cluster_seed)

    header = {
        "master_seed": metadata.master_seed,
        "cluster_seed": metadata.cluster_seed,
        "cluster_radius": metadata.cluster_radius,
        "initial_number_of_entities": metadata.initial_number_of_entities,
        "dt": metadata.simulation_config.dt,
        "G": metadata.simulation_config.G,
        "softening": metadata.simulation_config.softening,
        "time_warp": metadata.simulation_config.time_warp,
        "integrator": metadata.simulation_config.integrator
        #need to add energy and values like that here
    }

    with open(json_path, "w") as json_file:
        json_file.write(json.dumps({"metadata": header}) + "\n")

def save_json_snapshot_gen(metadata: Metadata, entities, center_of_mass: Entity):

    json_path = _paths.get_gen_json_path(metadata.cluster_seed)

    snapshot = {
        "t": metadata.simulation_config.t,
        "center_of_mass": {
            "x": center_of_mass.position.x,
            "y": center_of_mass.position.y,
            "z": center_of_mass.position.z,
            "vx": center_of_mass.velocity.vx,
            "vy": center_of_mass.velocity.vy,
            "vz": center_of_mass.velocity.vz
        },
        "entities": [
            {
                "name": entity.name,
                "x": entity.position.x,
                "y": entity.position.y,
                "z": entity.position.z,
                "vx": entity.velocity.vx,
                "vy": entity.velocity.vy,
                "vz": entity.velocity.vz,
                "mass": entity.mass
            } for entity in entities
        ]
    }

    with open(json_path, "a") as json_file:
        json_file.write(json.dumps(snapshot) + "\n")


def init_json_gts(metadata: Metadata):

    json_path = _paths.get_gts_json_path(metadata.cluster_seed, metadata.orbit_semi_major_axis, metadata.orbit_eccentricity)

    header = {
        "master_seed": metadata.master_seed,
        "cluster_seed": metadata.cluster_seed,
        "cluster_radius": metadata.cluster_radius,
        "initial_number_of_entities": metadata.initial_number_of_entities,
        "dt": metadata.simulation_config.dt,
        "G": metadata.simulation_config.G,
        "softening": metadata.simulation_config.softening,
        "time_warp": metadata.simulation_config.time_warp,
        "integrator": metadata.simulation_config.integrator,
        "galaxy_mass": metadata.galaxy_mass,
        "galaxy_radius": metadata.galaxy_radius,
        "orbit_semi_major_axis": metadata.orbit_semi_major_axis,
        "orbit_eccentricity": metadata.orbit_eccentricity,
        "number_of_orbits": metadata.number_of_orbits,
        "half_mass_radius": metadata.half_mass_radius,
        "bound_mass_fraction": metadata.bound_mass_fraction
        #need to add energy and values like that here
    }

    with open(json_path, "w") as json_file:
        json_file.write(json.dumps({"metadata": header}) + "\n")

def save_json_snapshot_gts(metadata: Metadata, entities, center_of_mass: Entity):

    json_path = _paths.get_gts_json_path(metadata.cluster_seed, metadata.orbit_semi_major_axis, metadata.orbit_eccentricity)

    snapshot = {
        "t": metadata.simulation_config.t,
        "center_of_mass": {
            "x": center_of_mass.position.x,
            "y": center_of_mass.position.y,
            "z": center_of_mass.position.z,
            "vx": center_of_mass.velocity.vx,
            "vy": center_of_mass.velocity.vy,
            "vz": center_of_mass.velocity.vz
        },
        "entities": [
            {
                "name": entity.name,
                "x": entity.position.x,
                "y": entity.position.y,
                "z": entity.position.z,
                "vx": entity.velocity.vx,
                "vy": entity.velocity.vy,
                "vz": entity.velocity.vz,
                "mass": entity.mass
            } for entity in entities
        ]
    }

    with open(json_path, "a") as json_file:
        json_file.write(json.dumps(snapshot) + "\n")

def load_json_snapshot(json_path, snapshot_index = -1):

    with open(json_path, "r") as json_file:
        lines = json_file.readlines()

    if not lines:
        raise ValueError(f"The file {json_path} is empty.")

    json_metadata = json.loads(lines[0])["metadata"]

    line_index = snapshot_index + 1 if snapshot_index >= 0 else snapshot_index
    json_data = json.loads(lines[line_index])

    metadata = Metadata (
        master_seed = json_metadata["master_seed"],
        cluster_seed = json_metadata["cluster_seed"],
        simulation_config = SimulationConfig (
            dt = json_metadata["dt"],
            t = json_data["t"],
            G = json_metadata["G"],
            softening = json_metadata["softening"],
            time_warp = json_metadata["time_warp"],
            integrator = json_metadata["integrator"]
        ),
        galaxy_mass = json_metadata.get("galaxy_mass"),
        galaxy_radius = json_metadata.get("galaxy_radius"),
        orbit_semi_major_axis = json_metadata.get("orbit_semi_major_axis"),
        orbit_eccentricity = json_metadata.get("orbit_eccentricity"),
        number_of_orbits = json_metadata.get("number_of_orbits"),
    )

    entities = []
    for entity in json_data["entities"]:
        new_entity = Entity (
            name = entity["name"],
            position = Position (
                x = entity["x"],
                y = entity["y"],
                z = entity["z"]
            ),
            velocity = Velocity (
                vx = entity["vx"],
                vy = entity["vy"],
                vz = entity["vz"]
            ),
            mass = entity["mass"]
        )

        entities.append(new_entity)

    center_of_mass = Entity (
        name = "center_of_mass",
        position = Position (
            x = json_data["center_of_mass"]["x"],
            y = json_data["center_of_mass"]["y"],
            z = json_data["center_of_mass"]["z"]
        ),
        velocity = Velocity (
            vx = json_data["center_of_mass"]["vx"],
            vy = json_data["center_of_mass"]["vy"],
            vz = json_data["center_of_mass"]["vz"]
        ),
        mass = -1.0
    )

    return metadata, entities, center_of_mass

def load_json_file(json_path):
    with open(json_path, "r") as json_file:
        return json.load(json_file)
