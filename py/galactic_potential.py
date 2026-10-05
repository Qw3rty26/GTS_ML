import numpy as np
from numba import njit

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

        #                         - G * M
        # a = -----------------------------------------------
        #       (a^2 + x^2 + y^2 + z^2)^(3/2) * (vector)

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

    def _potential_phi(self, radius_from_center):
        #                        - G * M
        # phi(r) = ---------------------------------
        #           sqrt(a^2 + r^2)

        numerator = -1 * GRAVITATIONAL_CONSTANT * self.plummer_mass
        denominator = np.sqrt(self.plummer_radius**2 + radius_from_center**2)

        phi = numerator / denominator
        return phi

    def get_cluster_initial_velocity(self, radius):
        #                           G * M * r^2
        # v_circ = sqrt( --------------------------------- )
        #                 (a^2 + r^2)^(3/2)

        numerator = (GRAVITATIONAL_CONSTANT * self.plummer_mass * radius**2)
        denominator = (self.plummer_radius**2 + radius**2)**(1.5)

        velocity = np.sqrt(numerator / denominator)
        return velocity

    def get_elliptical_apocenter_velocity(self, a, e):
        #                        2 * (Phi(r_apo) - Phi(r_peri))
        # L^2 = -------------------------------------------------------------
        #        (1 / r_peri^2) - (1 / r_apo^2)

        r_apo = a * (1.0 + e)
        r_peri = a * (1.0 - e)

        if abs(e) < 1e-6:
            return self.get_cluster_initial_velocity(r_apo)

        phi_apo = self._potential_phi(r_apo)
        phi_peri = self._potential_phi(r_peri)

        numerator = 2.0 * (phi_apo - phi_peri)
        denominator = (1.0 / (r_peri**2)) - (1.0 / (r_apo**2))

        L_squared = numerator / denominator
        if L_squared < 0:
            raise ValueError("Invalid values for 'a' and 'e' for this potential.")

        L = np.sqrt(L_squared)
        v_apo = L / r_apo
        return v_apo

    def add_galaxy_forces(self, particles):
        n = len(particles)
        coords = np.empty((n, 3), dtype=np.float64)

        for i, p in enumerate(particles):
            coords[i, 0] = p.x
            coords[i, 1] = p.y
            coords[i, 2] = p.z

        ax, ay, az = _compute_plummer_accelerations(
            coords, self.plummer_mass, self.plummer_radius
        )

        for i, particle in enumerate(particles):
            particle.ax += ax[i]
            particle.ay += ay[i]
            particle.az += az[i]
