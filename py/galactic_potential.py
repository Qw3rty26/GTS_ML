import numpy as np
from numba import njit
from ctypes import sizeof
from rebound.particle import Particle

GRAVITATIONAL_CONSTANT = 1

# Numba is used to make this faster by not calling Python interpreter at each call
@njit(fastmath=True)
def _compute_plummer_accelerations(coords, plummer_mass, plummer_radius):
    n = coords.shape[0]
    ax = np.empty(n, dtype=np.float64)
    ay = np.empty(n, dtype=np.float64)
    az = np.empty(n, dtype=np.float64)

    numerator = -GRAVITATIONAL_CONSTANT * plummer_mass
    plummer_radius_squared = plummer_radius**2

    for i in range(n):
        x = coords[i, 0]
        y = coords[i, 1]
        z = coords[i, 2]

        #                                   - G * M
        # acceleration_factor  = -------------------------------
        #                         (a^2 + x^2 + y^2 + z^2)^(3/2)

        radius_squared = x**2 + y**2 + z**2
        denominator = (plummer_radius_squared + radius_squared)**(1.5)

        acceleration_factor = numerator / denominator

        ax[i] = acceleration_factor * x
        ay[i] = acceleration_factor * y
        az[i] = acceleration_factor * z

    return ax, ay, az


class GalacticPotential:

    def __init__(self, plummer_radius=1, plummer_mass=1):
        if plummer_radius <= 0:
            raise ValueError("plummer_radius must be greater than 0")

        if plummer_mass <= 0:
            raise ValueError("plummer_mass must be greater than 0")

        self.plummer_radius = plummer_radius  # a
        self.plummer_mass = plummer_mass      # M

    def get_galaxy_radius(self):
        return self.plummer_radius

    def get_galaxy_mass(self):
        return self.plummer_mass

    def potential_phi(self, radius_from_center):
        #             - G * M
        # phi(r) = -----------------
        #           sqrt(a^2 + r^2)

        numerator = -1 * GRAVITATIONAL_CONSTANT * self.plummer_mass
        denominator = np.sqrt(self.plummer_radius**2 + radius_from_center**2)

        phi = numerator / denominator
        return phi

    def get_cluster_circular_velocity(self, radius):
        #                    G * M * r^2
        # v_circ = sqrt( -------------------- )
        #                 (a^2 + r^2)^(3/2)

        numerator = (GRAVITATIONAL_CONSTANT * self.plummer_mass * radius**2)
        denominator = (self.plummer_radius**2 + radius**2)**(1.5)

        velocity = np.sqrt(numerator / denominator)
        return velocity

    def get_elliptical_apocenter_velocity(self, a, e):
        #         2 * (Phi(r_apo) - Phi(r_peri))
        # L^2 = -------------------------------
        #        (1 / r_peri^2) - (1 / r_apo^2)

        if(e == 0.0):
            return self.get_cluster_circular_velocity(a)

        r_apo = a * (1.0 + e)
        r_peri = a * (1.0 - e)

        phi_apo = self.potential_phi(r_apo)
        phi_peri = self.potential_phi(r_peri)

        numerator = 2.0 * (phi_apo - phi_peri)
        denominator = (1.0 / (r_peri**2)) - (1.0 / (r_apo**2))

        L_squared = numerator / denominator
        if L_squared < 0:
            raise ValueError("Invalid values for 'a' and 'e' for this potential.")

        L = np.sqrt(L_squared)
        v_apo = L / r_apo
        return v_apo

    # some obscure magic is happening here, all you need to know is that
    # we are calculating the galactic pull for each entity and adding the acceleration to it
    def add_galaxy_forces(self, particles):
        particle_array = particles._ps

        particle_dtype = np.dtype({
            "names": ["x", "y", "z", "ax", "ay", "az"],
            "formats": [
                np.float64,
                np.float64,
                np.float64,
                np.float64,
                np.float64,
                np.float64
            ],
            "offsets": [
                Particle.x.offset,
                Particle.y.offset,
                Particle.z.offset,
                Particle.ax.offset,
                Particle.ay.offset,
                Particle.az.offset
            ],
            "itemsize": sizeof(Particle)
        })

        particle_data = np.ndarray(
            len(particle_array),
            dtype=particle_dtype,
            buffer=particle_array
        )

        coords = np.empty(
            (len(particle_array), 3),
            dtype=np.float64
        )

        coords[:, 0] = particle_data["x"]
        coords[:, 1] = particle_data["y"]
        coords[:, 2] = particle_data["z"]

        ax, ay, az = _compute_plummer_accelerations(
            coords,
            self.plummer_mass,
            self.plummer_radius
        )

        particle_data["ax"] += ax
        particle_data["ay"] += ay
        particle_data["az"] += az
