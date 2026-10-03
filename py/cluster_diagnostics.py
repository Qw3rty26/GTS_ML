import logging
import numpy as np

logger = logging.getLogger(__name__)


class ClusterDiagnostics:

    def __init__(self, simulation):

        if simulation is None:
            raise ValueError("object simulation is None")

        self._cached_half_mass_radius = None
        self._cached_center_of_mass = None

        self.simulation = simulation
        self.rebound_simulation = simulation.simulation

        self.initial_total_energy = None
        self.initial_angular_momentum = None
        self.orbital_angle = self.get_cluster_orbital_angle()

        self.total_rotation = 0.0
        self.orbits = 0

    def get_center_of_mass(self):

        particles = self.rebound_simulation.particles

        if len(particles) == 0:
            com = self.rebound_simulation.com()
            class BoundCoM:
                def __init__(self, p, v):
                    self.x, self.y, self.z = p[0], p[1], p[2]
                    self.vx, self.vy, self.vz = v[0], v[1], v[2]
            return BoundCoM([com.x, com.y, com.z], [com.vx, com.vy, com.vz])

        pos = np.array([[p.x, p.y, p.z] for p in particles])
        vel = np.array([[p.vx, p.vy, p.vz] for p in particles])
        masses = np.array([p.m for p in particles])

        total_mass = np.sum(masses)

        com_pos = np.sum(pos * masses[:, np.newaxis], axis=0) / total_mass
        com_vel = np.sum(vel * masses[:, np.newaxis], axis=0) / total_mass

        percentiles = [80, 65, 50, 35, 20]

        for perc in percentiles:

            distances = np.linalg.norm(pos - com_pos, axis=1)

            cutoff = np.percentile(distances, perc)

            mask = distances <= cutoff

            if np.sum(mask) < 5:
                break

            sub_pos = pos[mask]
            sub_vel = vel[mask]
            sub_masses = masses[mask]

            sub_total_mass = np.sum(sub_masses)

            com_pos = np.sum(sub_pos * sub_masses[:, np.newaxis], axis=0) / sub_total_mass
            com_vel = np.sum(sub_vel * sub_masses[:, np.newaxis], axis=0) / sub_total_mass

        class BoundCoM:
            def __init__(self, p, v):
                self.x = p[0]
                self.y = p[1]
                self.z = p[2]
                self.vx = v[0]
                self.vy = v[1]
                self.vz = v[2]

        return BoundCoM(com_pos, com_vel)

    def _cache_center_of_mass(self):
        self._cached_center_of_mass = self.get_center_of_mass()

    def _get_effective_center_of_mass(self):
        if self._cached_center_of_mass is None:
            return self.get_center_of_mass()
        return self._cached_center_of_mass

    def get_cluster_orbits(self):
        return self.orbits

    def get_cluster_orbital_angle(self):

        #                   180 * arctan(y_COM, x_COM)
        # orbital_angle = ---------------------------
        #                             pi

        particles = self.rebound_simulation.particles

        if len(particles) == 0:
            com = self.rebound_simulation.com()
            x_position, y_position = com.x, com.y
        else:
            positions = np.array([[p.x, p.y] for p in particles])
            x_position, y_position = np.mean(positions, axis=0)

        numerator = 180.0 * np.arctan2(y_position, x_position)

        denominator = np.pi

        orbital_angle = numerator / denominator

        return float(orbital_angle)

    def update_orbital_angle(self):

        # assuming center of galaxy is at (0, 0, 0)
        new_angle = self.get_cluster_orbital_angle()

        rotation = new_angle - self.orbital_angle

        if rotation < -180.0:
            rotation += 360.0
        elif rotation > 180.0:
            rotation -= 360.0

        self.total_rotation += rotation

        if abs(self.total_rotation) >= 360.0:
            self.orbits += 1

            if self.total_rotation >= 360.0:
                self.total_rotation -= 360.0
            else:
                self.total_rotation += 360.0

        self.orbital_angle = new_angle

    def get_galactic_potential_energy(self):

        #                   G * M_gal * m_i
        # U_gal = - sum_i ( --------------------- )
        #                   sqrt(a^2 + r_i^2)

        if not hasattr(self.simulation, "galactic_potential") or self.simulation.galactic_potential is None:
            return 0.0

        particles = self.rebound_simulation.particles

        if len(particles) == 0:
            return 0.0

        pos = np.array([[p.x, p.y, p.z] for p in particles])
        masses = np.array([p.m for p in particles])

        radius_i_squared = np.sum(pos**2, axis=1)

        G = self.rebound_simulation.G
        galaxy_mass = self.simulation.galactic_potential.get_galaxy_mass()
        galaxy_radius = self.simulation.galactic_potential.get_galaxy_radius()

        numerator = G * galaxy_mass * masses

        denominator = np.sqrt(galaxy_radius**2 + radius_i_squared)

        total_potential_energy = np.sum(numerator / denominator)

        total_potential_energy *= -1

        return float(total_potential_energy)

    def get_total_energy(self):

        #
        # E = K + U_cluster + U_gal
        #

        total_energy = self.rebound_simulation.energy()

        total_energy += self.get_galactic_potential_energy()

        return total_energy

    def set_initial_total_energy(self):
        self.initial_total_energy = self.get_total_energy()

    def get_initial_total_energy(self):
        return self.initial_total_energy

    def get_total_energy_relative_error(self):

        #                                   |E(t) - E_0|
        # total_energy_relative_error(t) = --------------
        #                                       |E_0|

        if self.initial_total_energy is None:
            raise ValueError("Initial total energy cannot be None")

        if self.initial_total_energy == 0:
            logger.warning("Initial total energy is 0")
            return 0.0

        current_total_energy = self.get_total_energy()

        numerator = abs(current_total_energy - self.initial_total_energy)

        denominator = abs(self.initial_total_energy)

        total_energy_relative_error = numerator / denominator

        return total_energy_relative_error

    def get_total_energy_relative_error_percentage(self):
        total_energy_relative_error_percentage = self.get_total_energy_relative_error() * 100.0
        return total_energy_relative_error_percentage

    def get_total_angular_momentum(self):

        # L_internal = sum((r_i - r_COM) x m_i * (v_i - v_COM))

        particles = self.rebound_simulation.particles

        if len(particles) == 0:
            return np.zeros(3)

        center_of_mass = self.get_center_of_mass()

        pos = np.array([[p.x - center_of_mass.x, p.y - center_of_mass.y, p.z - center_of_mass.z] for p in particles])
        vel = np.array([[p.vx - center_of_mass.vx, p.vy - center_of_mass.vy, p.vz - center_of_mass.vz] for p in particles])
        masses = np.array([p.m for p in particles])

        angular_momenta = masses[:, np.newaxis] * np.cross(pos, vel)

        total_angular_momentum = np.sum(angular_momenta, axis=0)

        return total_angular_momentum

    def get_total_angular_momentum_bound(self):

        # L_bound = sum_{i in bound} ((r_i - r_COM_bound) x m_i * (v_i - v_COM_bound))

        escaped_ids = set(self.get_escaped_entity_ids())
        particles = [p for i, p in enumerate(self.rebound_simulation.particles) if i not in escaped_ids]

        if len(particles) == 0:
            return np.zeros(3)

        pos = np.array([[p.x, p.y, p.z] for p in particles])
        vel = np.array([[p.vx, p.vy, p.vz] for p in particles])
        masses = np.array([p.m for p in particles])

        com_pos = np.average(pos, axis=0, weights=masses)
        com_vel = np.average(vel, axis=0, weights=masses)

        rel_pos = pos - com_pos
        rel_vel = vel - com_vel

        angular_momenta = masses[:, np.newaxis] * np.cross(rel_pos, rel_vel)

        return np.sum(angular_momenta, axis=0)

    def set_initial_angular_momentum(self):
        self.initial_angular_momentum = self.get_total_angular_momentum()

    def get_initial_angular_momentum(self):
        return self.initial_angular_momentum

    def get_total_angular_momentum_error(self):

        #                                             |L(t) - L_0|
        # total_angular_momentum_relative_error(t) = --------------
        #                                                |L_0|

        if self.initial_angular_momentum is None:
            raise ValueError("Initial angular momentum cannot be None")

        denominator = np.linalg.norm(self.initial_angular_momentum)

        if denominator == 0.0:
            return 0.0

        current_total_angular_momentum = self.get_total_angular_momentum()

        numerator = np.linalg.norm(
            current_total_angular_momentum - self.initial_angular_momentum
        )

        total_angular_momentum_error = numerator / denominator

        return float(total_angular_momentum_error)

    def get_total_angular_momentum_error_percentage(self):
        total_angular_momentum_error_percentage = self.get_total_angular_momentum_error() * 100.0
        return total_angular_momentum_error_percentage

    def get_half_mass_radius(self):

        particles = self.rebound_simulation.particles

        if len(particles) == 0:
            return 0.0

        center_of_mass = self.get_center_of_mass()

        pos = np.array([[p.x - center_of_mass.x, p.y - center_of_mass.y, p.z - center_of_mass.z] for p in particles])
        masses = np.array([p.m for p in particles])

        distances = np.linalg.norm(pos, axis=1)

        sort_indices = np.argsort(distances)
        sorted_distances = distances[sort_indices]
        sorted_masses = masses[sort_indices]

        cumulative_masses = np.cumsum(sorted_masses)
        half_mass = cumulative_masses[-1] / 2.0

        half_mass_index = np.searchsorted(cumulative_masses, half_mass)

        return float(sorted_distances[half_mass_index])

    def _cache_half_mass_radius(self):
        self._cached_half_mass_radius = self.get_half_mass_radius()

    def get_cluster_kinetic_energy(self):

        #                     1
        # kinetic_energy = --- * sum_i ( m_i * |v_i - v_COM|^2 )
        #                     2

        particles = self.rebound_simulation.particles

        if len(particles) == 0:
            return 0.0

        center_of_mass = self.get_center_of_mass()

        vel = np.array([[p.vx - center_of_mass.vx, p.vy - center_of_mass.vy, p.vz - center_of_mass.vz] for p in particles])
        masses = np.array([p.m for p in particles])

        velocity_squared = np.sum(vel**2, axis=1)

        total_kinetic_energy = 0.5 * np.sum(masses * velocity_squared)

        return float(total_kinetic_energy)

    def get_cluster_potential_energy(self):

        #                       G * m_i * m_j
        # U_cluster = - sum ( ----------------- )
        #               i<j        distance_ij

        particles = self.rebound_simulation.particles
        number_of_particles = len(particles)

        if number_of_particles < 2:
            return 0.0

        pos = np.array([[p.x, p.y, p.z] for p in particles])
        masses = np.array([p.m for p in particles])

        difference = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        distance_squared = np.sum(difference**2, axis=-1) + self.rebound_simulation.softening**2
        distances = np.sqrt(distance_squared)

        mass_matrix = masses[:, np.newaxis] * masses[np.newaxis, :]

        index_i_upper, index_j_upper = np.triu_indices(number_of_particles, k=1)

        potential_sum = np.sum(mass_matrix[index_i_upper, index_j_upper] / distances[index_i_upper, index_j_upper])

        total_potential_energy = -self.rebound_simulation.G * potential_sum

        return float(total_potential_energy)

    def get_virial_ratio(self):

        #       2 * K_internal
        # Q = --------------
        #       |U_internal|

        kinetic_energy = self.get_cluster_kinetic_energy()

        potential_energy = self.get_cluster_potential_energy()

        if potential_energy == 0.0:
            return 0.0

        numerator = 2.0 * kinetic_energy

        denominator = abs(potential_energy)

        virial_ratio = numerator / denominator

        return virial_ratio

    def get_virial_ratio_bound(self):

        #             2 * K_bound
        # Q_bound = ---------------
        #            |U_bound|

        escaped_ids = set(self.get_escaped_entity_ids())
        particles = [p for i, p in enumerate(self.rebound_simulation.particles) if i not in escaped_ids]
        number_of_particles = len(particles)

        if number_of_particles < 2:
            return 0.0

        pos = np.array([[p.x, p.y, p.z] for p in particles])
        vel = np.array([[p.vx, p.vy, p.vz] for p in particles])
        masses = np.array([p.m for p in particles])

        com_pos = np.average(pos, axis=0, weights=masses)
        com_vel = np.average(vel, axis=0, weights=masses)

        rel_vel = vel - com_vel
        kinetic_energy = 0.5 * np.sum(masses * np.sum(rel_vel**2, axis=1))

        difference = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        distance_squared = np.sum(difference**2, axis=-1) + self.rebound_simulation.softening**2
        distances = np.sqrt(distance_squared)

        mass_matrix = masses[:, np.newaxis] * masses[np.newaxis, :]
        index_i_upper, index_j_upper = np.triu_indices(number_of_particles, k=1)

        potential_sum = np.sum(mass_matrix[index_i_upper, index_j_upper] / distances[index_i_upper, index_j_upper])
        potential_energy = -self.rebound_simulation.G * potential_sum

        if potential_energy == 0.0:
            return 0.0

        return float(2.0 * kinetic_energy / abs(potential_energy))

    def get_entity_kinetic_energy(self, entity):

        #                     1
        # kinetic_energy_i = --- m_i v_i^2
        #                     2

        center_of_mass = self._get_effective_center_of_mass()

        relative_velocity_x = entity.vx - center_of_mass.vx
        relative_velocity_y = entity.vy - center_of_mass.vy
        relative_velocity_z = entity.vz - center_of_mass.vz

        relative_velocity_squared = (
            relative_velocity_x**2 +
            relative_velocity_y**2 +
            relative_velocity_z**2
        )

        entity_kinetic_energy = 0.5 * entity.m * relative_velocity_squared

        return float(entity_kinetic_energy)

    def get_entity_potential_energy(self, entity_i):

        #                                         m_i * m_j
        # potential_energy_i = - G * sum_j!=i( ------------- )
        #                                         distance_ij

        particles = self.rebound_simulation.particles

        if len(particles) < 2:
            return 0.0

        position_i = np.array([entity_i.x, entity_i.y, entity_i.z])
        other_positions = np.array([[p.x, p.y, p.z] for p in particles if p is not entity_i])
        other_masses = np.array([p.m for p in particles if p is not entity_i])

        difference = other_positions - position_i
        distances = np.sqrt(np.sum(difference**2, axis=1) + self.rebound_simulation.softening**2)

        entity_potential_energy = - self.rebound_simulation.G * np.sum(entity_i.m * other_masses / distances)

        return float(entity_potential_energy)

    def get_entity_total_energy(self, entity):

        #
        # total_energy_i = kinetic_energy_i + potential_energy_i
        #

        entity_kinetic_energy = self.get_entity_kinetic_energy(entity)
        entity_potential_energy = self.get_entity_potential_energy(entity)

        entity_total_energy = entity_kinetic_energy + entity_potential_energy

        return entity_total_energy

    def _is_entity_escaped(self, entity):
        return self.get_entity_total_energy(entity) >= 0.0

    def get_escaped_entity_ids(self):

        particles = self.rebound_simulation.particles
        number_of_particles = len(particles)

        if number_of_particles == 0:
            return []

        self._cache_center_of_mass()
        center_of_mass = self._cached_center_of_mass

        pos = np.array([[p.x - center_of_mass.x, p.y - center_of_mass.y, p.z - center_of_mass.z] for p in particles])
        vel = np.array([[p.vx - center_of_mass.vx, p.vy - center_of_mass.vy, p.vz - center_of_mass.vz] for p in particles])
        masses = np.array([p.m for p in particles])

        kinetic_energy_per_star = 0.5 * masses * np.sum(vel**2, axis=1)

        difference = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        distance_squared = np.sum(difference**2, axis=-1) + self.rebound_simulation.softening**2
        distances = np.sqrt(distance_squared)

        np.fill_diagonal(distances, np.inf)

        potential_energy_per_star = -self.rebound_simulation.G * masses * np.sum(masses[np.newaxis, :] / distances, axis=1)

        total_energy_per_star = kinetic_energy_per_star + potential_energy_per_star

        escaped_indices = np.where(total_energy_per_star >= 0.0)[0].tolist()

        return escaped_indices
