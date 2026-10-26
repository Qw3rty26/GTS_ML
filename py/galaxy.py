from ctypes import sizeof
from rebound import Particle
import numpy as np
from numba import njit
from dataclasses import dataclass

GRAVITATIONAL_CONSTANT = 1.0

@dataclass
class Galaxy_Configuration:
    galaxy_mass: float = None
    galaxy_radius: float = None
    orbit_semi_major_axis: float = None
    orbit_eccentricity: float = None

class Galaxy:

    def __init__(self, galaxy_configuration: Galaxy_Configuration = None):
        self.galaxy_configuration = galaxy_configuration

    def get_potential_phi(self, radius):
        #                G * M_gal
        # Phi(r) = - -----------------
        #             sqrt(r^2 + b^2)

        numerator = -1.0 * GRAVITATIONAL_CONSTANT * self.galaxy_configuration.galaxy_mass

        denominator = (radius**2 + self.galaxy_configuration.galaxy_radius**2)
        denominator = np.sqrt(denominator)

        potential_phi = numerator / denominator
        return potential_phi

    def get_circular_velocity(self, radius):
        #                G * M_gal * r^2
        # v_c = sqrt ( ------------------- )
        #               (r^2 + b^2)^(3/2)

        numerator = GRAVITATIONAL_CONSTANT * self.galaxy_configuration.galaxy_mass * (radius**2.0)

        denominator = (radius**2 + self.galaxy_configuration.galaxy_radius**2)**(3.0/2.0)
        circular_velocity = numerator / denominator

        circular_velocity = np.sqrt(circular_velocity)
        return circular_velocity

    def get_apoapsis_distance(self):
        # r_apo = a * (1 + e)
        r_apo = self.galaxy_configuration.orbit_semi_major_axis * (1 + self.galaxy_configuration.orbit_eccentricity)
        return r_apo

    def get_periapsis_distance(self):
        # r_peri = a * (1 - e)
        r_peri = self.galaxy_configuration.orbit_semi_major_axis * (1 - self.galaxy_configuration.orbit_eccentricity)
        return r_peri

    def get_initial_orbital_position(self):
        initial_orbital_position = np.array([self.get_apoapsis_distance(), 0.0, 0.0])
        return initial_orbital_position

    def get_angular_momentum(self):
        #        2 * (Phi(r_apo) - Phi(r_peri))
        # h^2 = --------------------------------
        #        1/r_peri^2 - 1/r_apo^2

        phi_apo = self.get_potential_phi(self.get_apoapsis_distance())
        phi_peri = self.get_potential_phi(self.get_periapsis_distance())

        numerator = 2.0 * (phi_apo - phi_peri)

        denominator = 1.0 / (self.get_periapsis_distance())**2
        denominator = denominator - (1.0 / (self.get_apoapsis_distance())**2)

        angular_momentum = numerator / denominator
        angular_momentum = np.sqrt(angular_momentum)
        return angular_momentum


    def get_initial_orbital_velocity(self):
        if self.galaxy_configuration.orbit_eccentricity == 0:
            # Circular orbit: v = v_c(r_apo)
            return np.array([0.0, self.get_circular_velocity(self.get_apoapsis_distance()), 0.0])

        #            h
        # v_apo = -------
        #          r_apo
        initial_orbital_velocity = self.get_angular_momentum() / self.get_apoapsis_distance()
        return np.array([0.0, initial_orbital_velocity, 0.0])

    def get_acceleration(self, position):
        #           -G * M_gal * r
        # a(r) = -------------------
        #         (r^2 + b^2)^(3/2)

        x, y, z = position

        squared_distance = x * x + y * y + z * z
        galaxy_radius = self.galaxy_configuration.galaxy_radius

        factor = -GRAVITATIONAL_CONSTANT * self.galaxy_configuration.galaxy_mass / (
            squared_distance + galaxy_radius * galaxy_radius
        )**1.5

        return np.array((factor * x, factor * y, factor * z))

    @staticmethod
    @njit(fastmath=True)
    def _add_plummer_accelerations(particle_data, galaxy_mass, galaxy_radius_squared):
        numerator = -GRAVITATIONAL_CONSTANT * galaxy_mass

        for i in range(len(particle_data)):
            x = particle_data["x"][i]
            y = particle_data["y"][i]
            z = particle_data["z"][i]

            distance_squared = x * x + y * y + z * z + galaxy_radius_squared
            factor = numerator / (distance_squared * np.sqrt(distance_squared))

            particle_data["ax"][i] += factor * x
            particle_data["ay"][i] += factor * y
            particle_data["az"][i] += factor * z

    def add_galaxy_forces(self, rebound_simulation):
        particles = rebound_simulation.particles
        particle_array = particles._ps

        particle_dtype = np.dtype({
            "names": ["x", "y", "z", "ax", "ay", "az"],
            "formats": [np.float64] * 6,
            "offsets": [Particle.x.offset, Particle.y.offset, Particle.z.offset, Particle.ax.offset, Particle.ay.offset, Particle.az.offset],
            "itemsize": sizeof(Particle),
        })

        particle_data = np.ndarray(len(particle_array), dtype=particle_dtype, buffer=particle_array)

        self._add_plummer_accelerations(particle_data, self.galaxy_configuration.galaxy_mass, self.galaxy_configuration.galaxy_radius**2)
