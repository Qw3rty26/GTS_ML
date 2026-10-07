import rebound
from dataclasses import dataclass, field

from cluster_diagnostics import ClusterDiagnostics

@dataclass
class SimulationConfig:
    dt: float = 0.01
    t: float = 0.0
    G: float = 1.0
    softening: float = 0.1
    time_warp: int = 1
    integrator: str = "leapfrog"

@dataclass
class Position:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

@dataclass
class Velocity:
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

@dataclass
class Entity:
    name: str = "0"
    position: Position = field(default_factory = Position)
    velocity: Velocity = field(default_factory = Velocity)
    mass: float = 0.0

class Simulation:

    def __init__(self, simulation_config: SimulationConfig = None):
        self.simulation = rebound.Simulation()

        if simulation_config is not None:
            self.simulation_config = simulation_config
        else:
            self.simulation_config = SimulationConfig()

        self._set_simulation_configs()

        self.galactic_potential = None
        self.cluster_diagnostics = ClusterDiagnostics(self)

    def _set_simulation_configs(self):
        self.simulation.dt = self.simulation_config.dt
        self.simulation.t = self.simulation_config.t
        self.simulation.G = self.simulation_config.G
        self.simulation.softening = self.simulation_config.softening
        self.simulation.integrator = self.simulation_config.integrator
        self.time_warp = self.simulation_config.time_warp

    def integrate(self):
        for _ in range(self.time_warp):
            self.simulation.integrate(
                self.simulation.t + self.simulation.dt
            )

    def get_time(self):
        return self.simulation.t

    def get_entities(self):
        entities = []
        for entity in self.simulation.particles:
            entity = Entity(
                name = entity.name,
                position = Position(
                    x = entity.x,
                    y = entity.y,
                    z = entity.z
                ),
                velocity = Velocity(
                    vx = entity.vx,
                    vy = entity.vy,
                    vz = entity.vz
                ),
                mass = entity.m
            )
            entities.append(entity)
        return entities

    def get_center_of_mass(self):
        return self.simulation.com()

    def get_percentile_center_of_mass(self):
        return self.cluster_diagnostics.get_percentile_center_of_mass()

    def move_cluster(self, moved_x, moved_y, moved_z):
        for particle in self.simulation.particles:
            particle.x += moved_x
            particle.y += moved_y
            particle.z += moved_z

    def move_cluster_to_center_of_mass(self):
        self.simulation.move_to_com()

    def speed_cluster(self, speed_x, speed_y, speed_z):
        for particle in self.simulation.particles:
            particle.vx += speed_x
            particle.vy += speed_y
            particle.vz += speed_z

    def _apply_galactic_forces(self, sim_pointer):
        if self.galactic_potential:
            self.galactic_potential.add_galaxy_forces(self.simulation.particles)

    def add_galactic_potential(self, galactic_potential):
        self.galactic_potential = galactic_potential
        self.simulation.additional_forces = self._apply_galactic_forces

    def add_entity(self, entity):
        self.simulation.add(
            name = entity.name,
            x = entity.position.x,
            y = entity.position.y,
            z = entity.position.z,
            vx = entity.velocity.vx,
            vy = entity.velocity.vy,
            vz = entity.velocity.vz,
            m = entity.mass
        )

    def remove_entity(self, entity_id):
        self.simulation.remove(entity_id)

    def clean_escaped_entities(self):
        escaped_entity_ids = self.cluster_diagnostics.get_escaped_entity_ids()

        for entity_id in sorted(escaped_entity_ids, reverse=True):
            self.simulation.remove(entity_id)
        return len(escaped_entity_ids)
