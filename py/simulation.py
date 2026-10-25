import rebound
import numpy as np
from dataclasses import dataclass
from cluster_diagnostics import Cluster_Diagnostics

@dataclass
class Simulation_Configuration:
    integration_time: float = None
    starting_time: float = None
    gravitational_constant: float = None
    softening: float = None
    integrator: str = None

class Simulation:

    def __init__(self, simulation_configuration: Simulation_Configuration):
        self.rebound_handle = rebound.Simulation() # create rebound object

        self.simulation_configuration = simulation_configuration
        self._initialise_rebound()
        self.diagnostics = Cluster_Diagnostics(self)

    def _initialise_rebound(self):
        self.rebound_handle.dt = self.simulation_configuration.integration_time
        self.rebound_handle.t = self.simulation_configuration.starting_time
        self.rebound_handle.G = self.simulation_configuration.gravitational_constant
        self.rebound_handle.softening = self.simulation_configuration.softening
        self.rebound_handle.integrator = self.simulation_configuration.integrator

    def integrate(self):
        self.rebound_handle.steps(1) #integrate one step

    def get_particle_positions(self):
        particles = self.rebound_handle.particles
        number_of_particles = len(particles)

        particle_positions = np.empty(
            (number_of_particles, 3),
            dtype = np.float64
        )

        for i, particle in enumerate(particles):
            particle_positions[i] = (
                particle.x,
                particle.y,
                particle.z
            )
        return particle_positions

    def get_particle_velocities(self):
        particles = self.rebound_handle.particles
        number_of_particles = len(particles)

        particle_velocities = np.empty(
            (number_of_particles, 3),
            dtype = np.float64
        )

        for i, particle in enumerate(particles):
            particle_velocities[i] = (
                particle.vx,
                particle.vy,
                particle.vz
            )
        return particle_velocities

    def get_particle_masses(self):
        particles = self.rebound_handle.particles
        number_of_particles = len(particles)

        particle_masses = np.empty(
            number_of_particles,
            dtype = np.float64
        )

        for i, particle in enumerate(particles):
            particle_masses[i] = particle.m
        return particle_masses


    def move_cluster(self, moved_x, moved_y, moved_z):
        for particle in self.rebound_handle.particles:
            particle.x += moved_x
            particle.y += moved_y
            particle.z += moved_z

    def move_cluster_to_center_of_mass(self):
        self.simulation.move_to_com()

    def speed_cluster(self, speed_x, speed_y, speed_z):
        for particle in self.rebound_handle.particles:
            particle.vx += speed_x
            particle.vy += speed_y
            particle.vz += speed_z

    def add_particle(self, particle):
        self.rebound_handle.add(
            name = particle.name,
            x = particle.x,
            y = particle.y,
            z = particle.z,
            vx = particle.vx,
            vy = particle.vy,
            vz = particle.vz,
            m = particle.m
        )

    def remove_entity(self, particle_name):
        self.rebound_handle.remove(particle_name)

