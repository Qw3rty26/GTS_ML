import numpy as np

CLUSTER_MASS = 1.0
GRAVITATIONAL_CONSTANT = 1.0

class Plummer:

    def __init__(self, number_of_particles, cluster_radius = None, cluster_seed = None):
        self.number_of_particles = number_of_particles
        self.cluster_radius = cluster_radius
        self.rng = np.random.Generator(np.random.MT19937(cluster_seed))


    def _potential_phi(self, distance_from_center):
        #              - G * M
        # phi(r) = -----------------
        #           sqrt(a^2 + r^2)

        numerator = -1 * GRAVITATIONAL_CONSTANT * CLUSTER_MASS

        denominator = np.sqrt(self.cluster_radius**2 + distance_from_center**2)

        potential_phi = numerator / denominator
        return potential_phi

    def _enclosed_mass(self, distance_from_center):
        #                 M * r^3
        # mass(<r) = -------------------
        #             (a^2 + r^2)^(3/2)

        numerator = CLUSTER_MASS * distance_from_center**3
        denominator = (self.cluster_radius**2 + distance_from_center**2)**(3.0/2.0)

        enclosed_mass = numerator / denominator
        return enclosed_mass

    def _normalised_random_versor(self):
        random_versor = self.rng.normal(size = 3) # generate a random point in 3D space
        versor_length = np.linalg.norm(random_versor)

        if versor_length == 0:
            return self._normalised_random_versor()

        normalised_random_versor = random_versor / versor_length # versor length is now 1
        return normalised_random_versor

    def _sample_radius(self):
        #                 r_max^3
        # F_max = -----------------------
        #          (r_max^2 + a^2)^(3/2)

        MAX_RADIUS = 10 * self.cluster_radius

        numerator = MAX_RADIUS**3
        denominator = (MAX_RADIUS**2 + self.cluster_radius**2)**(3.0/2.0)
        F_max = numerator / denominator

        #
        # u ~ U (0,1) <-- random number between 0 and 1
        #
        u = max(self.rng.random(), np.finfo(float).tiny)

        #              a
        # r = --------------------
        #      sqrt((u * F_max)^(-2/3) - 1)

        numerator = self.cluster_radius

        denominator = u * F_max
        denominator = denominator**(-2.0/3.0)
        denominator = denominator - 1
        denominator = np.sqrt(denominator)

        sampled_radius = numerator / denominator
        return sampled_radius

    def _sample_position(self):
        sampled_radius = self._sample_radius()
        normalised_random_versor = self._normalised_random_versor()

        sampled_position = sampled_radius * normalised_random_versor
        return sampled_position

    def _escape_velocity(self, distance_from_center):
        #                        2 * G * M
        # v_esc(r) = sqrt( ------------------- )
        #                   sqrt ( r^2 + a^2 )

        numerator = 2 * GRAVITATIONAL_CONSTANT * CLUSTER_MASS

        denominator = distance_from_center**2 + self.cluster_radius**2
        denominator = np.sqrt(denominator)

        escape_velocity = np.sqrt(numerator / denominator)
        return escape_velocity

    def _sample_speed_ratio(self):
        #
        # X ~ Beta(3/2, 9/2)
        #

        X = self.rng.beta(3.0/2.0, 9.0/2.0)

        #
        # q = sqrt( X )
        #

        sampled_speed_ratio = np.sqrt(X)
        return sampled_speed_ratio


    def _sample_velocity(self, position):
        distance_from_center = np.linalg.norm(position)

        escape_velocity = self._escape_velocity(distance_from_center)

        speed_ratio = self._sample_speed_ratio()

        sampled_velocity = escape_velocity * speed_ratio

        normalised_random_versor = self._normalised_random_versor()

        sampled_velocity = sampled_velocity * normalised_random_versor
        return sampled_velocity


    def generate_particles(self):
        particle_positions = np.empty((self.number_of_particles, 3))
        particle_velocities = np.empty((self.number_of_particles, 3))

        for i in range(self.number_of_particles):
            particle_positions[i] = self._sample_position()
            particle_velocities[i] = self._sample_velocity(particle_positions[i])

        return particle_positions, particle_velocities

    def get_relaxation_time(self):
        RELAXATION_FACTOR = 5.0
        #
        # r_h = 1.305 * a
        #

        r_h = 1.305 * self.cluster_radius

        #
        # gamma = 0.4
        #

        GAMMA = 0.4

        #           0.138 * N             r_h^3
        # t_rh = --------------- * sqrt( ------- )
        #         ln(gamma * N)           G * M

        numerator = r_h**3.0
        denominator = GRAVITATIONAL_CONSTANT * CLUSTER_MASS
        numerator = np.sqrt(numerator / denominator)
        numerator = 0.138 * self.number_of_particles * numerator
        denominator = np.log(GAMMA * self.number_of_particles)

        relaxation_time = numerator / denominator

        relaxation_time = relaxation_time * RELAXATION_FACTOR

        return relaxation_time
