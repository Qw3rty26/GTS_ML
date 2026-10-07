import logging
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

class ClusterDiagnostics:

    def __init__(self, simulation):

        if simulation is None:
            raise ValueError("object simulation is None")

        self._cached_center_of_mass = None
        self._cached_com_time = -1.0
        self._cached_com_state = None
        self._prev_y = None

        self.simulation = simulation
        self.rebound_simulation = simulation.simulation

        self.initial_total_energy = None
        self.initial_angular_momentum = None
        self.orbital_angle = self.get_cluster_orbital_angle()

        self.total_rotation = 0.0
        self.orbits = 0

    def get_percentile_center_of_mass(self) -> "Entity":
        from simulation import Entity, Position, Velocity

        sim = self.rebound_simulation

        if self._cached_com_state is not None and self._cached_com_time == sim.t:
            return self._cached_com_state

        n_particles = len(sim.particles)

        if n_particles == 0:
            com = sim.com()
            return Entity(
                m=0.0,
                position=Position(com.x, com.y, com.z),
                velocity=Velocity(com.vx, com.vy, com.vz)
            )

        pos = np.array([[p.x, p.y, p.z] for p in sim.particles], dtype=np.float64)
        vel = np.array([[p.vx, p.vy, p.vz] for p in sim.particles], dtype=np.float64)
        masses = np.array([p.m for p in sim.particles], dtype=np.float64)

        com_pos = np.average(pos, axis=0, weights=masses)
        com_vel = np.average(vel, axis=0, weights=masses)

        percentiles = [80, 65, 50, 35, 20]

        for perc in percentiles:

            distances = np.linalg.norm(pos - com_pos, axis=1)

            k = int(len(distances) * perc / 100)
            cutoff = np.partition(distances, k)[k]

            mask = distances <= cutoff

            if np.sum(mask) < 5:
                break

            sub_pos = pos[mask]
            sub_vel = vel[mask]
            sub_masses = masses[mask]

            com_pos = np.average(sub_pos, axis=0, weights=sub_masses)
            com_vel = np.average(sub_vel, axis=0, weights=sub_masses)

        total_mass = float(np.sum(masses))
        com_state = Entity(
            m=total_mass,
            position=Position(float(com_pos[0]), float(com_pos[1]), float(com_pos[2])),
            velocity=Velocity(float(com_vel[0]), float(com_vel[1]), float(com_vel[2]))
        )

        self._cached_com_time = sim.t
        self._cached_com_state = com_state

        return com_state

    def _cache_center_of_mass(self):
        self._cached_center_of_mass = self.get_percentile_center_of_mass()

    def _get_effective_center_of_mass(self) -> "Entity":
        if self._cached_center_of_mass is None:
            return self.get_percentile_center_of_mass()
        return self._cached_center_of_mass

    def get_center_of_mass(self) -> "Entity":
        return self.get_percentile_center_of_mass()

    def get_cluster_orbits(self):
        return self.orbits

    def get_cluster_orbital_angle(self):

        sim = self.rebound_simulation

        if len(sim.particles) == 0:
            x_position, y_position = 0.0, 0.0
        else:
            com = self.get_center_of_mass()
            x_position, y_position = com.position.x, com.position.y

        numerator = 180.0 * np.arctan2(y_position, x_position)
        denominator = np.pi
        orbital_angle = numerator / denominator

        return float(orbital_angle)

    def update_orbital_angle(self):

        new_angle = self.get_cluster_orbital_angle()
        com = self.get_center_of_mass()
        x, y = com.position.x, com.position.y

        if self._prev_y is not None:
            if self._prev_y * y < 0.0 and x > 0.0:
                self.orbits += 1

        self._prev_y = y
        self.orbital_angle = new_angle

    def get_galactic_potential_energy(self):

        if not hasattr(self.simulation, "galactic_potential") or self.simulation.galactic_potential is None:
            return 0.0

        sim = self.rebound_simulation

        if len(sim.particles) == 0:
            return 0.0

        pos = np.array([[p.x, p.y, p.z] for p in sim.particles], dtype=np.float64)
        masses = np.array([p.m for p in sim.particles], dtype=np.float64)

        radius_i_squared = np.sum(pos**2, axis=1)

        G = sim.G
        galaxy_mass = self.simulation.galactic_potential.get_galaxy_mass()
        galaxy_radius = self.simulation.galactic_potential.get_galaxy_radius()

        numerator = G * galaxy_mass * masses
        denominator = np.sqrt(galaxy_radius**2 + radius_i_squared)

        total_potential_energy = np.sum(numerator / denominator)
        total_potential_energy *= -1

        return float(total_potential_energy)

    def get_total_energy(self):

        total_energy = self.rebound_simulation.energy()
        total_energy += self.get_galactic_potential_energy()

        return total_energy

    def set_initial_total_energy(self):
        self.initial_total_energy = self.get_total_energy()

    def get_initial_total_energy(self):
        return self.initial_total_energy

    def get_total_energy_relative_error(self):

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

        sim = self.rebound_simulation

        if len(sim.particles) == 0:
            return np.zeros(3)

        center_of_mass = self.get_center_of_mass()

        pos = np.array([[p.x, p.y, p.z] for p in sim.particles], dtype=np.float64) - np.array([center_of_mass.position.x, center_of_mass.position.y, center_of_mass.position.z])
        vel = np.array([[p.vx, p.vy, p.vz] for p in sim.particles], dtype=np.float64) - np.array([center_of_mass.velocity.vx, center_of_mass.velocity.vy, center_of_mass.velocity.vz])
        masses = np.array([p.m for p in sim.particles], dtype=np.float64)

        cross_products = np.cross(pos, vel)
        angular_momenta = np.average(cross_products, axis=0, weights=masses) * np.sum(masses)

        return angular_momenta

    def get_total_angular_momentum_bound(self):

        escaped_ids = set(self.get_escaped_entity_ids())
        sim = self.rebound_simulation

        valid_particles = [p for i, p in enumerate(sim.particles) if i not in escaped_ids]

        if not valid_particles:
            return np.zeros(3)

        pos = np.array([[p.x, p.y, p.z] for p in valid_particles], dtype=np.float64)
        vel = np.array([[p.vx, p.vy, p.vz] for p in valid_particles], dtype=np.float64)
        masses = np.array([p.m for p in valid_particles], dtype=np.float64)

        com_pos = np.average(pos, axis=0, weights=masses)
        com_vel = np.average(vel, axis=0, weights=masses)

        rel_pos = pos - com_pos
        rel_vel = vel - com_vel

        cross_products = np.cross(rel_pos, rel_vel)
        return np.average(cross_products, axis=0, weights=masses) * np.sum(masses)

    def set_initial_angular_momentum(self):
        self.initial_angular_momentum = self.get_total_angular_momentum()

    def get_initial_angular_momentum(self):
        return self.initial_angular_momentum

    def get_total_angular_momentum_error(self):

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

        sim = self.rebound_simulation

        if len(sim.particles) == 0:
            return 0.0

        escaped_ids = set(self.get_escaped_entity_ids())
        valid_particles = [p for i, p in enumerate(sim.particles) if i not in escaped_ids]

        if not valid_particles:
            return 0.0

        center_of_mass = self.get_center_of_mass()

        pos = np.array([[p.x, p.y, p.z] for p in valid_particles], dtype=np.float64) - np.array([center_of_mass.position.x, center_of_mass.position.y, center_of_mass.position.z])
        masses = np.array([p.m for p in valid_particles], dtype=np.float64)

        distances = np.linalg.norm(pos, axis=1)

        sort_indices = np.argsort(distances)
        sorted_distances = distances[sort_indices]
        sorted_masses = masses[sort_indices]

        cumulative_masses = np.cumsum(sorted_masses)
        half_mass = cumulative_masses[-1] / 2.0

        half_mass_index = np.searchsorted(cumulative_masses, half_mass)

        return float(sorted_distances[half_mass_index])

    def get_bound_mass(self):

        escaped_ids = set(self.get_escaped_entity_ids())
        sim = self.rebound_simulation

        bound_mass = sum(p.m for i, p in enumerate(sim.particles) if i not in escaped_ids)

        return float(bound_mass)

    def get_cluster_kinetic_energy(self):

        sim = self.rebound_simulation

        if len(sim.particles) == 0:
            return 0.0

        center_of_mass = self.get_center_of_mass()

        vel = np.array([[p.vx, p.vy, p.vz] for p in sim.particles], dtype=np.float64) - np.array([center_of_mass.velocity.vx, center_of_mass.velocity.vy, center_of_mass.velocity.vz])
        masses = np.array([p.m for p in sim.particles], dtype=np.float64)

        velocity_squared = np.sum(vel**2, axis=1)
        total_kinetic_energy = 0.5 * np.sum(masses * velocity_squared)

        return float(total_kinetic_energy)

    def get_cluster_potential_energy(self):

        sim = self.rebound_simulation
        number_of_particles = len(sim.particles)

        if number_of_particles < 2:
            return 0.0

        pos = np.array([[p.x, p.y, p.z] for p in sim.particles], dtype=np.float64)
        masses = np.array([p.m for p in sim.particles], dtype=np.float64)

        difference = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        distance_squared = np.sum(difference**2, axis=-1) + sim.softening**2
        distances = np.sqrt(distance_squared)

        mass_matrix = masses[:, np.newaxis] * masses[np.newaxis, :]
        index_i_upper, index_j_upper = np.triu_indices(number_of_particles, k=1)

        potential_sum = np.sum(mass_matrix[index_i_upper, index_j_upper] / distances[index_i_upper, index_j_upper])
        total_potential_energy = -sim.G * potential_sum

        return float(total_potential_energy)

    def get_virial_ratio(self):

        kinetic_energy = self.get_cluster_kinetic_energy()
        potential_energy = self.get_cluster_potential_energy()

        if potential_energy == 0.0:
            return 0.0

        numerator = 2.0 * kinetic_energy
        denominator = abs(potential_energy)
        virial_ratio = numerator / denominator

        return virial_ratio

    def get_virial_ratio_bound(self):

        escaped_ids = set(self.get_escaped_entity_ids())
        sim = self.rebound_simulation

        valid_particles = [p for i, p in enumerate(sim.particles) if i not in escaped_ids]
        number_of_particles = len(valid_particles)

        if number_of_particles < 2:
            return 0.0

        pos = np.array([[p.x, p.y, p.z] for p in valid_particles], dtype=np.float64)
        vel = np.array([[p.vx, p.vy, p.vz] for p in valid_particles], dtype=np.float64)
        masses = np.array([p.m for p in valid_particles], dtype=np.float64)

        com_pos = np.average(pos, axis=0, weights=masses)
        com_vel = np.average(vel, axis=0, weights=masses)

        rel_vel = vel - com_vel
        kinetic_energy = 0.5 * np.sum(masses * np.sum(rel_vel**2, axis=1))

        difference = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        distance_squared = np.sum(difference**2, axis=-1) + sim.softening**2
        distances = np.sqrt(distance_squared)

        mass_matrix = masses[:, np.newaxis] * masses[np.newaxis, :]
        index_i_upper, index_j_upper = np.triu_indices(number_of_particles, k=1)

        potential_sum = np.sum(mass_matrix[index_i_upper, index_j_upper] / distances[index_i_upper, index_j_upper])
        potential_energy = -sim.G * potential_sum

        if potential_energy == 0.0:
            return 0.0

        return float(2.0 * kinetic_energy / abs(potential_energy))

    def get_entity_total_energy(self, index):

        sim = self.rebound_simulation
        p = sim.particles[index]
        com = self._get_effective_center_of_mass()

        vx = p.vx - com.velocity.vx
        vy = p.vy - com.velocity.vy
        vz = p.vz - com.velocity.vz
        kinetic_energy = 0.5 * p.m * (vx**2 + vy**2 + vz**2)

        pos = np.array([[pt.x, pt.y, pt.z] for pt in sim.particles], dtype=np.float64)
        masses = np.array([pt.m for pt in sim.particles], dtype=np.float64)

        pos_i = pos[index]
        diff = pos - pos_i
        dist_sq = np.sum(diff**2, axis=1) + sim.softening**2
        distances = np.sqrt(dist_sq)
        distances[index] = np.inf

        potential_energy = -sim.G * p.m * np.sum(masses / distances)

        galactic_potential = 0.0
        if hasattr(self.simulation, "galactic_potential") and self.simulation.galactic_potential is not None:
            galaxy_mass = self.simulation.galactic_potential.get_galaxy_mass()
            galaxy_radius = self.simulation.galactic_potential.get_galaxy_radius()
            r_i = np.linalg.norm(pos_i)
            galactic_potential = -sim.G * galaxy_mass * p.m / np.sqrt(galaxy_radius**2 + r_i**2)

        return float(kinetic_energy + potential_energy + galactic_potential)

    def _is_entity_escaped(self, index):
        return self.get_entity_total_energy(index) >= 0.0

    def get_escaped_entity_ids(self):
        sim = self.rebound_simulation
        number_of_particles = len(sim.particles)

        if number_of_particles == 0:
            return []

        self._cache_center_of_mass()

        escaped_indices = [i for i in range(number_of_particles) if self._is_entity_escaped(i)]

        return escaped_indices
