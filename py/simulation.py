import rebound
import numpy as np
from dataclasses import dataclass, field

from entity import Entity, Position, Velocity
from cluster_diagnostics import ClusterDiagnostics

@dataclass
class SimulationConfig:
    dt: float = 0.01
    t: float = 0.0
    G: float = 1.0
    softening: float = 0.1
    time_warp: int = 1
    integrator: str = "leapfrog"

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

    def integrate(self):
        for _ in range(self.simulation_config.time_warp):
            self.simulation.integrate(
                self.simulation.t + self.simulation.dt
            )

    def get_gravitational_constant(self):
        return self.simulation_config.G

    def get_softening(self):
        return self.simulation_config.softening

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

    # used for heavy optimisations
    def get_entity_arrays(self):
        particles = self.simulation.particles
        number_of_entities = len(particles)

        positions = np.empty(
            (number_of_entities, 3),
            dtype=np.float64
        )

        masses = np.empty(
            number_of_entities,
            dtype=np.float64
        )

        for i, particle in enumerate(particles):
            positions[i] = (
                particle.x,
                particle.y,
                particle.z
            )
            masses[i] = particle.m

        return positions, masses


    def get_center_of_mass(self):
        return self.simulation.com()

    def get_percentile_center_of_mass(self):
        return self.cluster_diagnostics.get_percentile_center_of_mass()

    def get_number_of_orbits(self):
        return self.cluster_diagnostics.get_number_of_orbits()

    def move_cluster(self, moved_x, moved_y, moved_z):
        for entity in self.simulation.particles:
            entity.x += moved_x
            entity.y += moved_y
            entity.z += moved_z

    def move_cluster_to_center_of_mass(self):
        self.simulation.move_to_com()

    def speed_cluster(self, speed_x, speed_y, speed_z):
        for entity in self.simulation.particles:
            entity.vx += speed_x
            entity.vy += speed_y
            entity.vz += speed_z

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

    def remove_entity(self, entity_index):
        self.simulation.remove(entity_index)

    def clean_unbound_entities(self):
        unbound_entity_indices = (
            self.cluster_diagnostics.get_unbound_entity_indices()
        )

        for entity_index in sorted(unbound_entity_indices, reverse=True):
            self.simulation.remove(entity_index)

        return len(unbound_entity_indices)
