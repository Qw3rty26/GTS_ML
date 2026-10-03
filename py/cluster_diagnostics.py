import numpy as np
import logging

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

        self.total_rotation = 0
        self.orbits = 0

    def get_center_of_mass(self):

        particles = self.rebound_simulation.particles

        if len(particles) == 0:
            return self.rebound_simulation.com()

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

    def get_cluster_orbits(self):
        return self.orbits

    def get_cluster_orbital_angle(self):
        center_of_mass = self.get_center_of_mass()

        numerator = 180 * np.arctan2(center_of_mass.y, center_of_mass.x)

        denominator = np.pi

        orbital_angle = numerator / denominator

        return orbital_angle

    def update_orbital_angle(self):

        # assuming center of galaxy is at (0, 0, 0)
        new_angle = self.get_cluster_orbital_angle()

        rotation = new_angle - self.orbital_angle

        if rotation < -180:
            rotation += 360
        elif rotation > 180:
            rotation -= 360

        self.total_rotation += abs(rotation)

        if abs(self.total_rotation) >= 360:
            self.orbits += 1

            if self.total_rotation >= 360:
                self.total_rotation -= 360
            else:
                self.total_rotation += 360

        self.orbital_angle = new_angle

    def get_galactic_potential_energy(self):

        #                     G * M_gal * m_i
        # U_gal = - sum_i ( --------------------- )
        #                     sqrt(a^2 + r_i^2)

        if self.simulation.galactic_potential is None:
            return 0.0

        total_potential_energy = 0.0

        for entity in self.rebound_simulation.particles:

            radius_i_squared = (
                entity.x**2 +
                entity.y**2 +
                entity.z**2
            )

            numerator = self.rebound_simulation.G * self.simulation.galactic_potential.get_galaxy_mass() * entity.m

            denominator = np.sqrt( self.simulation.galactic_potential.get_galaxy_radius()**2 + radius_i_squared )

            total_potential_energy += numerator / denominator

        total_potential_energy *= -1

        return total_potential_energy


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
        #                                      |E_0|

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
        total_energy_relative_error_percentage = self.get_total_energy_relative_error() * 100
        return total_energy_relative_error_percentage

    def get_total_angular_momentum(self):

        # L_internal = sum((r_i - r_COM) x m_i * (v_i - v_COM))

        center_of_mass = self.get_center_of_mass()

        total_angular_momentum = np.zeros(3)

        for entity in self.rebound_simulation.particles:

            position = np.array([
                entity.x - center_of_mass.x,
                entity.y - center_of_mass.y,
                entity.z - center_of_mass.z
            ])

            velocity = np.array([
                entity.vx - center_of_mass.vx,
                entity.vy - center_of_mass.vy,
                entity.vz - center_of_mass.vz
            ])

            total_angular_momentum += entity.m * np.cross(position, velocity)

        return total_angular_momentum

    def set_initial_angular_momentum(self):
       self.initial_angular_momentum = self.get_total_angular_momentum()

    def get_initial_angular_momentum(self):
       return self.initial_angular_momentum

    def get_total_angular_momentum_error(self):

        #                                             |L(t) - L_0|
        # total_angular_momentum_relative_error(t) = --------------
        #                                                 |L_0|

        current_total_angular_momentum = self.get_total_angular_momentum()

        numerator = np.linalg.norm(
            current_total_angular_momentum - self.initial_angular_momentum
        )

        denominator = np.linalg.norm(self.initial_angular_momentum)

        if denominator == 0.0:
            return 0.0

        total_angular_momentum_error = numerator / denominator

        return total_angular_momentum_error

    def get_total_angular_momentum_error_percentage(self):
        total_angular_momentum_error_percentage = self.get_total_angular_momentum_error() * 100
        return total_angular_momentum_error_percentage

    def get_half_mass_radius(self):

       com = self.get_center_of_mass()

       distances = []

       for p in self.rebound_simulation.particles:
          dx = p.x - com.x
          dy = p.y - com.y
          dz = p.z - com.z

          r = np.sqrt(dx*dx + dy*dy + dz*dz)

          distances.append(r)

       distances.sort()
       half_index = len(distances) // 2

       return distances[half_index]

    def _cache_half_mass_radius(self):
        self._cached_half_mass_radius = self.get_half_mass_radius()

    def get_cluster_kinetic_energy(self):

        #                    1
        # kinetic_energy = --- * sum_i ( m_i * |v_i - v_COM|^2 )
        #                    2

        center_of_mass = self.get_center_of_mass()

        total_kinetic_energy = 0.0

        for entity in self.rebound_simulation.particles:

            relative_velocity_x = entity.vx - center_of_mass.vx
            relative_velocity_y = entity.vy - center_of_mass.vy
            relative_velocity_z = entity.vz - center_of_mass.vz

            relative_velocity_squared = (
                relative_velocity_x**2 +
                relative_velocity_y**2 +
                relative_velocity_z**2
            )

            total_kinetic_energy += 0.5 * entity.m * relative_velocity_squared

        return total_kinetic_energy

    def get_cluster_potential_energy(self):

        #                      G * m_i * m_j
        # U_cluster = - sum ( ----------------- )
        #               i<j      distance_ij

        total_potential_energy = 0.0

        particles = self.rebound_simulation.particles

        number_of_particles = len(particles)

        for index_i in range(number_of_particles):

            entity_i = particles[index_i]

            for index_j in range(index_i + 1, number_of_particles):

                entity_j = particles[index_j]

                distance_x = entity_i.x - entity_j.x
                distance_y = entity_i.y - entity_j.y
                distance_z = entity_i.z - entity_j.z

                softened_distance = np.sqrt(
                    distance_x**2 +
                    distance_y**2 +
                    distance_z**2 +
                    self.rebound_simulation.softening**2
                )

                total_potential_energy += (
                    entity_i.m * entity_j.m / softened_distance
                )

        total_potential_energy *= -self.rebound_simulation.G

        return total_potential_energy

    def get_virial_ratio(self):

#TODO mettere K_total e U_total

        #     2 * K_internal
        # Q = --------------
        #     |U_internal|

        kinetic_energy = self.get_cluster_kinetic_energy()

        potential_energy = self.get_cluster_potential_energy()

        if potential_energy == 0:
            return 0.0

        numerator = 2.0 * kinetic_energy

        denominator = abs(potential_energy)

        virial_ratio = numerator / denominator

        return virial_ratio

    def get_entity_kinetic_energy(self, entity):

        #                     1
        # kinetic_energy_i = --- m_i v_i^2
        #                     2

        v_rel_x = entity.vx - self._cached_center_of_mass.vx
        v_rel_y = entity.vy - self._cached_center_of_mass.vy
        v_rel_z = entity.vz - self._cached_center_of_mass.vz

        entity_kinetic_energy = 0.5 * entity.m * (
            v_rel_x**2 +
            v_rel_y**2 +
            v_rel_z**2
        )

        return entity_kinetic_energy

    def get_entity_potential_energy(self, entity_i):

        #                                        m_i * m_j
        # potential_energy_i = - G * sum_j!=i( ------------- )
        #                                       distance_ij

        entity_potential_energy = 0.0

        for entity_j in self.rebound_simulation.particles:
           if entity_j is entity_i: # j != i
               continue

           distance_x = entity_i.x - entity_j.x
           distance_y = entity_i.y - entity_j.y
           distance_z = entity_i.z - entity_j.z

           softened_distance = np.sqrt(
               distance_x**2 +
               distance_y**2 +
               distance_z**2 +
               self.rebound_simulation.softening**2
           )

           entity_potential_energy += (
               entity_i.m *
               entity_j.m /
               softened_distance
           )

        entity_potential_energy = - self.rebound_simulation.G * entity_potential_energy

        return entity_potential_energy

    def get_entity_total_energy(self, entity):

        #
        # total_energy_i = kinetic_energy_i + potential_energy_i
        #

        entity_kinetic_energy = self.get_entity_kinetic_energy(entity)
        entity_potential_energy = self.get_entity_potential_energy(entity)

        entity_total_energy = entity_kinetic_energy + entity_potential_energy

        return entity_total_energy

    def _is_entity_escaped(self, entity, distance_factor = 15.0):

        relative_x = entity.x - self._cached_center_of_mass.x
        relative_y = entity.y - self._cached_center_of_mass.y
        relative_z = entity.z - self._cached_center_of_mass.z

        distance_from_com_sq = relative_x**2 + relative_y**2 + relative_z **2

        max_distance_threshold_sq = (distance_factor * self._cached_half_mass_radius)**2

        if distance_from_com_sq < max_distance_threshold_sq:
            return False

        return self.get_entity_total_energy(entity) > 0

    def get_escaped_entity_ids(self):

        escaped_entity_indices = []
        self._cache_center_of_mass()
        self._cache_half_mass_radius()

        for i, entity in enumerate(self.rebound_simulation.particles):
            if self._is_entity_escaped(entity):
                escaped_entity_indices.append(i)

        return escaped_entity_indices

    def get_snapshot(self):

       com = self.get_center_of_mass()
       snapshot = { # returns a JSON object containing an array of cluster diagnostic data
          "initial_energy": self.get_initial_total_energy(),
          "error_energy": self.get_total_energy_relative_error(),
          "half_mass_radius": self.get_half_mass_radius(),
          "center_of_mass": {
             "x": com.x,
             "y": com.y,
             "z": com.z
          }
       }
       return snapshot

