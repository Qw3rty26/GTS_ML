import numpy as np

GRAVITATIONAL_CONSTANT = 1 # G
CLUSTER_MASS = 1           # M

class Plummer:

    def __init__(self, plummer_radius = 1, number_of_stars = 1, seed = None, softening = 0.1):

        if plummer_radius <= 0:
            raise ValueError("plummer_radius must be greater than 0")

        if number_of_stars < 1:
            raise ValueError("number_of_stars must be at least 1")

        if softening < 0:
            raise ValueError("softening must be non-negative")

        self.plummer_radius = plummer_radius   # a
        self.number_of_stars = number_of_stars # N
        self.softening = softening

        if seed is not None:
            self.rng = np.random.Generator(np.random.MT19937(seed))
        else:
            self.rng = np.random.default_rng()


    def _potential_phi(self, radius_from_center):

        #               - G * M
        # phi(r) = -----------------
        #           sqrt(a^2 + r^2)

        numerator = -1 * GRAVITATIONAL_CONSTANT * CLUSTER_MASS

        denominator = np.sqrt(self.plummer_radius**2 + radius_from_center**2)

        phi = numerator / denominator
        return phi

    def _enclosed_mass(self, radius_from_center):

        #                 M * r^3
        # mass(<r) = -------------------
        #             (a^2 + r^2)^(3/2)

        numerator = CLUSTER_MASS * radius_from_center**3

        denominator = (self.plummer_radius**2 + radius_from_center**2)**(3/2)

        mass = numerator / denominator
        return mass

    def _sample_radius(self):

        #              a
        # r = --------------------
        #      sqrt(u^(-2/3) - 1)

        #
        # u ~ U(0,1) <-- random number between 0 and 1
        #

        numerator = self.plummer_radius

        u = self.rng.random()

        denominator = u**(-2.0 / 3.0) - 1

        denominator = np.sqrt(denominator)

        sampled_radius = numerator / denominator

        return sampled_radius

    def _random_versor(self):
        versor = self.rng.normal(size = 3) # generate a random point in 3D space
        norm = np.linalg.norm(versor) # normalise the versor so that its length is 1
        if norm == 0:
            return self._random_versor()
        else:
            return versor / norm

    def _sample_velocity(self, radius_from_center):

        #
        # v = sqrt(x) * v_esc
        #

        #
        # x ~ Beta(3/2, 9/2)
        #

        x = self.rng.beta(1.5, 4.5)

        escape_velocity = np.sqrt( -2.0 * self._potential_phi(radius_from_center))

        speed = escape_velocity * np.sqrt(x)

        return speed * self._random_versor()


    def _apply_virial_correction(self, positions, velocities):

        #                              | W |
        # virial_scale_factor = sqrt( ------- )
        #                               2K

        star_mass = CLUSTER_MASS / self.number_of_stars

        #              G * m^2 * r_ij^2
        # W = - sum ( ------------------------- )
        #       i<j   (r_ij^2 + epsilon^2)^(3/2)

        total_virial_term = 0.0

        for i in range (self.number_of_stars):
            for j in range (i + 1, self.number_of_stars):

                distance_vector = positions[j] - positions[i]
                distance_squared = np.sum(distance_vector**2)

                softened_distance_squared = distance_squared + self.softening**2

                total_virial_term -= (
                    GRAVITATIONAL_CONSTANT
                    * star_mass**2
                    * distance_squared
                    / softened_distance_squared**1.5
                )

        #      1
        # K = --- * m * (v - v_COM)^2
        #      2

        center_of_mass_velocity = np.mean(velocities, axis=0)
        relative_velocities = velocities - center_of_mass_velocity

        total_kinetic_energy = (
            0.5 * star_mass * np.sum(relative_velocities**2)
        )

        numerator = np.abs(total_virial_term)
        denominator = 2.0 * total_kinetic_energy

        virial_scale_factor = np.sqrt(numerator / denominator)

        relative_velocities = relative_velocities * virial_scale_factor

        return relative_velocities

    def generate_plummer_cluster(self):
        positions = []
        velocities = []

        for i in range(self.number_of_stars):

            radius = self._sample_radius()
            radial_versor = self._random_versor()

            position = radius * radial_versor
            positions.append(position)

            velocity = self._sample_velocity(radius)
            velocities.append(velocity)

        positions = np.array(positions)
        velocities = np.array(velocities)

        velocities = self._apply_virial_correction(positions, velocities)

        return positions, velocities
