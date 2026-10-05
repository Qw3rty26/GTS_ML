import numpy as np
import scipy.optimize as so
import scipy.integrate as si

GRAVITATIONAL_CONSTANT = 1 # G
CLUSTER_MASS = 1           # M

class Plummer:

    def __init__(self, plummer_radius = 1, number_of_stars = 1, seed = None):

        if plummer_radius <= 0:
            raise ValueError("plummer_radius must be greater than 0")

        if number_of_stars < 1:
            raise ValueError("number_of_stars must be at least 1")

        self.plummer_radius = plummer_radius   # a
        self.number_of_stars = number_of_stars # N

        if seed is not None:
            self.rng = np.random.Generator(np.random.MT19937(seed))
        else:
            self.rng = np.random.default_rng()


    def _density_rho(self, radius_from_center):

        #                   3 * M * a^2
        # rho(r) = ----------------------------
        #           4 * pi * (a^2 + r^2)^(5/2)

        numerator = 3 * CLUSTER_MASS * (self.plummer_radius ** 2)

        denominator = 4 * np.pi * ((self.plummer_radius ** 2 + radius_from_center**2)**(5/2))

        rho = numerator / denominator
        return rho

    def _derivative_density_rho(self, radius_from_center):

        #      r
        # y = ---
        #      a

        #  d_rho     -15 * M * y * (1 + y^2)^(-7/2)
        # ------- = ----------------------------
        #   d_r           4 * pi * a^4

        y = radius_from_center / self.plummer_radius

        numerator = - 15 * CLUSTER_MASS * y * (1 + y**2)**(-7/2)

        denominator = 4 * np.pi * self.plummer_radius**4

        d_rho = numerator / denominator
        return d_rho

    def _potential_phi(self, radius_from_center):

        #               - G * M
        # phi(r) = -----------------
        #           sqrt(a^2 + r^2)

        numerator = -1 * GRAVITATIONAL_CONSTANT * CLUSTER_MASS

        denominator = np.sqrt(self.plummer_radius**2 + radius_from_center**2)

        phi = numerator / denominator
        return phi

    def _derivative_potential_phi(self, radius_from_center):

        #      r
        # y = ---
        #      a

        #  d_phi     G * M * y * (1 + y^2)^(-3/2)
        # ------- = ------------------------------
        #   d_r                 a^2

        y = radius_from_center / self.plummer_radius

        numerator = GRAVITATIONAL_CONSTANT * CLUSTER_MASS * y * (1 + y**2)**(-3/2)

        denominator = self.plummer_radius**2

        d_phi = numerator / denominator
        return d_phi

    def _enclosed_mass(self, radius_from_center):

        #                 M * r^3
        # mass(<r) = -------------------
        #             (a^2 + r^2)^(3/2)

        numerator = CLUSTER_MASS * radius_from_center**3

        denominator = (self.plummer_radius**2 + radius_from_center**2)**(3/2)

        mass = numerator / denominator
        return mass

    def _sample_radius(self):

        # given u, solve:
        # M(r) - u = 0
        # for r

        mass_fraction_u = self.rng.random()

        def mass_equation(radius):
            return self._enclosed_mass(radius) - mass_fraction_u

        radius_guess = self.plummer_radius # good starting guess

        sampled_radius = so.fsolve(mass_equation, radius_guess)[0]

        return sampled_radius

    def _random_versor(self):
        versor = self.rng.normal(size = 3) # generate a random point in 3D space
        norm = np.linalg.norm(versor) # normalise the versor so that its length is 1
        if norm == 0:
            return self._random_versor()
        else:
            return versor / norm

    def _anisotropy_beta(self, radius_from_center):

        # beta(r) = 0.0 assuming the plummer model is isotropic

        beta = 0.0
        return beta

    def _jeans_equation(self, radius_from_center, sigma_squared):

        #  d_sigma^2_r(r)       d_phi(r)     2 beta(r) * sigma(r)^2     sigma(r)^2     d_rho(r)
        # ---------------- = - ---------- - ------------------------ - ------------ * ----------
        #       d_r               d_r                  r                   rho(r)        d_r

        derivative = - self._derivative_potential_phi(radius_from_center)
        derivative -= (2 * self._anisotropy_beta(radius_from_center) * sigma_squared) / radius_from_center
        derivative -= (sigma_squared / self._density_rho(radius_from_center)) * self._derivative_density_rho(radius_from_center)

        return derivative

    def _solve_jeans_equation(self):

        MAX_RADIUS = 10 * self.plummer_radius # approaching infinity
        MIN_RADIUS = 1e-5                     # approaching 0 but never 0 so that we don't have undefined divisions

        radius_vector = np.linspace(MAX_RADIUS, MIN_RADIUS, 2000)

        sigma_squared_vector = si.odeint(
            self._jeans_equation,
            0.0,
            radius_vector,
            tfirst = True
        )

        radius_vector = np.flip(radius_vector)
        sigma_squared_vector = np.flip(sigma_squared_vector).flatten()

        return radius_vector, sigma_squared_vector

    def _sample_velocity(self, radius_from_center, radius_vector, sigma_squared_vector, radial_versor):

        while True:

            sigma_squared = np.interp(
                radius_from_center,
                radius_vector,
                sigma_squared_vector
            )

            radial_velocity = self.rng.normal(
                scale = np.sqrt(sigma_squared)
            )

            theta_velocity = self.rng.normal(
                scale = np.sqrt((1.0 - self._anisotropy_beta(radius_from_center)) * sigma_squared)
            )

            phi_velocity = self.rng.normal(
                scale = np.sqrt((1.0 - self._anisotropy_beta(radius_from_center)) * sigma_squared)
            )

            theta = np.arccos(radial_versor[2])
            phi = np.arctan2(radial_versor[1], radial_versor[0])

            theta_versor = np.array([
                np.cos(theta) * np.cos(phi),
                np.cos(theta) * np.sin(phi),
                -np.sin(theta)
            ])

            phi_versor = np.array([
                -np.sin(phi),
                np.cos(phi),
                0.0
            ])

            velocity = (
                radial_velocity * radial_versor +
                theta_velocity * theta_versor +
                phi_velocity * phi_versor
            )

            kinetic_energy = 0.5 * np.dot(velocity, velocity)

            potential_energy = self._potential_phi(radius_from_center)

            total_energy = kinetic_energy + potential_energy

            if total_energy < 0:
                return velocity

    def _apply_virial_correction(self, positions, velocities):

        #                              | U |
        # virial_scale_factor = sqrt( ------- )
        #                               2K

        star_mass = CLUSTER_MASS / self.number_of_stars

        #              G * m^2
        # U = - sum ( --------- )
        #       i<j      r_ij

        softening_eps = 0.01 * self.plummer_radius
        total_potential_energy = 0.0

        for i in range (self.number_of_stars):
            for j in range (i + 1, self.number_of_stars):

                distance_vector = positions[j] - positions[i]
                distance = np.sqrt(np.sum(distance_vector**2) + softening_eps**2)

                total_potential_energy -= GRAVITATIONAL_CONSTANT * (star_mass**2) / distance

        #      1
        # K = --- * m * v^2
        #      2

        total_kinetic_energy = 0.5 * star_mass * np.sum(velocities**2)

        numerator = np.abs(total_potential_energy)
        denominator = 2.0 * total_kinetic_energy

        virial_scale_factor = np.sqrt(numerator / denominator)

        velocities = velocities * virial_scale_factor

        return velocities


    def _debug(self, positions, velocities, radius_vector, sigma_squared_vector):
        for r in [0, 1, 2, 5]:
            sigma_num = np.interp(r, radius_vector, sigma_squared_vector)
            sigma_exact = GRAVITATIONAL_CONSTANT * CLUSTER_MASS / (
                6*np.sqrt(r*r + self.plummer_radius**2)
            )
            print("\n", r, "\nsigma theoretical: ", sigma_num, "\nsigma exact:       ", sigma_exact, "\ndifference:        ", sigma_exact - sigma_num)
        print("\n\nstar positions:\n", positions, "\n\nstar velocities:\n", velocities)

    def generate_plummer_cluster(self):
        MAX_RADIUS = 10 * self.plummer_radius # approaching infinity

        positions = []
        velocities = []

        radius_vector, sigma_squared_vector = self._solve_jeans_equation()

        for i in range(self.number_of_stars):

            radius = self._sample_radius()
            while radius > MAX_RADIUS:
                radius = self._sample_radius()

            radial_versor = self._random_versor()
            xyz = radius * radial_versor
            positions.append(xyz)

            v = self._sample_velocity(radius, radius_vector, sigma_squared_vector, radial_versor)
            velocities.append(v)

        positions = np.array(positions)
        velocities = np.array(velocities)

        #self._debug(positions, velocities, radius_vector, sigma_squared_vector)

        velocities = self._apply_virial_correction(positions, velocities)

        return positions, velocities
