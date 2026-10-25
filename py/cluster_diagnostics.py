import numpy as np
from dataclasses import dataclass

@dataclass
class Diagnostics:
    time: float = None
    cluster_kinetic_energy: float = None
    cluster_potential_energy: float = None
    cluster_total_energy: float = None
    cluster_kinetic_internal_energy: float = None
    cluster_virial_ratio: float = None
    cluster_bound_mass_fraction: float = None


class Cluster_Diagnostics:

    def __init__(self, simulation):
        self.simulation = simulation

    def get_center_of_mass_position(self):
        #          N
        # r_COM = sum(m_i * r_i) / sum(m_i)
        #         i=1

        particle_masses = self.simulation.get_particle_masses()
        particle_positions = self.simulation.get_particle_positions()

        total_mass = np.sum(particle_masses)
        center_of_mass_position = np.sum(particle_masses[:, np.newaxis] * particle_positions, axis=0) / total_mass

        rebound_center_of_mass = self.simulation.rebound_handle.com
        rebound_center_of_mass_position = np.array([rebound_center_of_mass.x, rebound_center_of_mass.y, rebound_center_of_mass.z])

        if not np.allclose(center_of_mass_position, rebound_center_of_mass_position, rtol=1e-5, atol=1e-8):
            raise ValueError(f"Center of mass position mismatch: calculated={center_of_mass_position}, REBOUND={rebound_center_of_mass_position}")

        return center_of_mass_position

    def get_center_of_mass_velocity(self):
        #          N
        # v_COM = sum(m_i * v_i) / sum(m_i)
        #         i=1

        particle_masses = self.simulation.get_particle_masses()
        particle_velocities = self.simulation.get_particle_velocities()

        total_mass = np.sum(particle_masses)
        center_of_mass_velocity = np.sum(particle_masses[:, np.newaxis] * particle_velocities, axis=0) / total_mass

        rebound_center_of_mass = self.simulation.rebound_handle.com()
        rebound_center_of_mass_velocity = np.array([rebound_center_of_mass.vx, rebound_center_of_mass.vy, rebound_center_of_mass.vz])

        if not np.allclose(center_of_mass_velocity, rebound_center_of_mass_velocity, rtol=1e-5, atol=1e-8):
            raise ValueError(f"Center of mass velocity mismatch: calculated={center_of_mass_velocity}, REBOUND={rebound_center_of_mass_velocity}")

        return center_of_mass_velocity

    def get_cluster_kinetic_energy(self):
        #  1     N
        # ___ * sum ( m_i (v_x_i^2 + v_y_i^2 + v_z_i^2) )
        #  2    i=1

        particle_masses = self.simulation.get_particle_masses()
        particle_velocities = self.simulation.get_particle_velocities()

        squared_speeds = np.sum(particle_velocities**2, axis=1)
        cluster_kinetic_energy = 0.5 * np.sum(particle_masses * squared_speeds)
        return cluster_kinetic_energy

    def get_cluster_potential_energy(self):
        #          N-1    N      m_i * m_j
        # U = -G * sum   sum ( ------------ )
        #          i=1  j=i+1      r_ij

        gravitational_constant = self.simulation.simulation_configuration.gravitational_constant

        particle_masses = self.simulation.get_particle_masses()
        particle_positions = self.simulation.get_particle_positions()

        position_differences = particle_positions[:, np.newaxis, :] - particle_positions[np.newaxis, :, :]

        squared_distances = np.sum(position_differences**2, axis=2)
        distances = np.sqrt(squared_distances)
        np.fill_diagonal(distances, np.inf)
        mass_products = particle_masses[:, np.newaxis] * particle_masses[np.newaxis, :]

        potential_energy_matrix = -gravitational_constant * mass_products / distances

        cluster_potential_energy = np.sum(np.triu(potential_energy_matrix, k=1))
        return cluster_potential_energy

    def get_cluster_total_energy(self):
        #
        # E_tot = K + U
        #

        cluster_kinetic_energy = self.get_cluster_kinetic_energy()
        cluster_potential_energy = self.get_cluster_potential_energy()
        cluster_total_energy = cluster_kinetic_energy + cluster_potential_energy

        _rebound_total_energy = self.simulation.rebound_handle.energy()

        if not np.isclose(cluster_total_energy, _rebound_total_energy, rtol=1e-2):
            raise ValueError(f"TOTAL ENERGY MISMATCH:",
                f"calculated: E_tot={cluster_total_energy} K={cluster_kinetic_energy} U={cluster_potential_energy}",
                f"REBOUND={_rebound_total_energy}")

        return cluster_total_energy

    # used mainly for the virial ratio
    def get_cluster_internal_kinetic_energy(self):
        #  1     N
        # ___ * sum ( m_i * |v_i - v_COM|^2 )
        #  2    i=1

        particle_masses = self.simulation.get_particle_masses()
        particle_velocities = self.simulation.get_particle_velocities()

        center_of_mass_velocity = self.get_center_of_mass_velocity()
        relative_velocities = particle_velocities - center_of_mass_velocity

        squared_speeds = np.sum(relative_velocities**2, axis=1)
        cluster_kinetic_energy = 0.5 * np.sum(particle_masses * squared_speeds)
        return cluster_kinetic_energy

    def get_cluster_softened_potential_energy(self):
        #          N-1    N              m_i * m_j
        # U = -G * sum   sum ( ------------------------------ )
        #          i=1  j=i+1   sqrt( r_ij^2 + softening^2 )

        gravitational_constant = self.simulation.simulation_configuration.gravitational_constant
        softening = self.simulation.simulation_configuration.softening

        particle_masses = self.simulation.get_particle_masses()
        particle_positions = self.simulation.get_particle_positions()

        position_differences = (particle_positions[:, np.newaxis, :] - particle_positions[np.newaxis, :, :])

        squared_distances = np.sum(position_differences**2, axis=2)
        softened_distances = np.sqrt(squared_distances + softening**2)

        mass_products = particle_masses[:, np.newaxis] * particle_masses[np.newaxis, :]

        potential_energy_matrix = (-gravitational_constant * mass_products / softened_distances)

        cluster_potential_energy = np.sum(np.triu(potential_energy_matrix, k=1))
        return cluster_potential_energy

    def _get_virial_term(self):
        #          N-1    N           m_i * m_j * r_ij^2
        # W = -G * sum   sum ( ------------------------------ )
        #          i=1  j=i+1   (r_ij^2 + softening^2)^(3/2)

        gravitational_constant = self.simulation.simulation_configuration.gravitational_constant
        softening = self.simulation.simulation_configuration.softening

        particle_masses = self.simulation.get_particle_masses()
        particle_positions = self.simulation.get_particle_positions()

        position_differences = particle_positions[:, np.newaxis, :] - particle_positions[np.newaxis, :, :]

        squared_distances = np.sum(position_differences**2, axis=2)
        mass_products = particle_masses[:, np.newaxis] * particle_masses[np.newaxis, :]

        virial_term_matrix = -gravitational_constant * mass_products * squared_distances / (squared_distances + softening**2)**1.5

        _virial_term = np.sum(np.triu(virial_term_matrix, k=1))
        return _virial_term

    def get_cluster_virial_ratio(self):
        #      2 * K_internal
        # Q = ----------------
        #           |W|

        internal_kinetic_energy = self.get_cluster_internal_kinetic_energy()
        virial_term = self._get_virial_term()

        if virial_term == 0:
            raise ValueError("Cannot calculate virial ratio: virial term is zero.")

        virial_ratio = 2 * internal_kinetic_energy / abs(virial_term)
        return virial_ratio

    def _get_particle_gravitational_potential(self, particle_masses, squared_distances, source_indices):
        #             N
        # phi_i = -G * sum ( m_j / sqrt(r_ij^2 + softening^2) )
        #          j in S, j != i

        gravitational_constant = self.simulation.simulation_configuration.gravitational_constant
        softening = self.simulation.simulation_configuration.softening

        softened_distances = np.sqrt(
            squared_distances[:, source_indices] + softening**2
        )

        source_masses = particle_masses[source_indices]

        potential = -gravitational_constant * np.sum(
            source_masses[np.newaxis, :] / softened_distances,
            axis=1
        )

        source_mask = np.zeros(len(particle_masses), dtype=bool)
        source_mask[source_indices] = True

        if softening > 0:
            potential[source_mask] += (
                gravitational_constant * particle_masses[source_mask] / softening
            )

        return potential

    def _get_bound_particles(self):
        # e_i = 1/2 * |v_i - v_COM|^2 + phi_i
        # S' = {i : e_i < 0}

        particle_masses = self.simulation.get_particle_masses()
        particle_positions = self.simulation.get_particle_positions()
        particle_velocities = self.simulation.get_particle_velocities()

        number_of_particles = len(particle_masses)

        if number_of_particles <= 3:
            return np.zeros(number_of_particles, dtype=bool)

        position_differences = (
            particle_positions[:, np.newaxis, :]
            - particle_positions[np.newaxis, :, :]
        )

        squared_distances = np.sum(position_differences**2, axis=2)

        all_indices = np.arange(number_of_particles)

        initial_potential = self._get_particle_gravitational_potential(
            particle_masses,
            squared_distances,
            all_indices
        )

        center_index = np.argmin(initial_potential)
        nearest_indices = np.argsort(
            squared_distances[center_index]
        )[:min(11, number_of_particles)]

        center_masses = particle_masses[nearest_indices]
        center_of_mass_velocity = np.sum(
            center_masses[:, np.newaxis] * particle_velocities[nearest_indices],
            axis=0
        ) / np.sum(center_masses)

        bound_particles = np.ones(number_of_particles, dtype=bool)

        for _ in range(10):
            bound_indices = np.flatnonzero(bound_particles)

            if len(bound_indices) <= 3:
                return np.zeros(number_of_particles, dtype=bool)

            bound_masses = particle_masses[bound_indices]
            center_of_mass_velocity = np.sum(
                bound_masses[:, np.newaxis] * particle_velocities[bound_indices],
                axis=0
            ) / np.sum(bound_masses)

            relative_velocities = particle_velocities - center_of_mass_velocity
            squared_speeds = np.sum(relative_velocities**2, axis=1)

            gravitational_potential = self._get_particle_gravitational_potential(
                particle_masses,
                squared_distances,
                bound_indices
            )

            specific_energy = 0.5 * squared_speeds + gravitational_potential
            new_bound_particles = specific_energy < 0

            if np.array_equal(bound_particles, new_bound_particles):
                break

            bound_particles = new_bound_particles

        if np.count_nonzero(bound_particles) <= 3:
            return np.zeros(number_of_particles, dtype=bool)

        return bound_particles

    def get_bound_mass_fraction(self):
        #             S
        #            sum(m_i)
        #             i
        # f_bound = ----------
        #             N
        #            sum(m_i)
        #            i=1
        particle_masses = self.simulation.get_particle_masses()
        bound_particles = self._get_bound_particles()

        bound_mass = np.sum(particle_masses[bound_particles])
        total_mass = np.sum(particle_masses)

        bound_mass_fraction = bound_mass / total_mass
        return bound_mass_fraction


    def get_diagnostics(self):
        diagnostics = Diagnostics (
            time = self.simulation.rebound_handle.t,
            cluster_kinetic_energy = self.get_cluster_kinetic_energy(),
            cluster_potential_energy = self.get_cluster_potential_energy(),
            cluster_total_energy = self.get_cluster_total_energy(),
            cluster_kinetic_internal_energy = self.get_cluster_internal_kinetic_energy(),
            cluster_virial_ratio = self.get_cluster_virial_ratio(),
            cluster_bound_mass_fraction = self.get_bound_mass_fraction()
        )
        return diagnostics

    def print_diagnostics(self, diagnostics: Diagnostics = None):
        if diagnostics is None: diagnostics = self.get_diagnostics()
        print(f"Time: {diagnostics.time}")
        print(f"Kinetic energy: {diagnostics.cluster_kinetic_energy}")
        print(f"Potential energy: {diagnostics.cluster_potential_energy}")
        print(f"Total energy: {diagnostics.cluster_total_energy}")
        print(f"Internal kinetic energy: {diagnostics.cluster_kinetic_internal_energy}")
        print(f"Virial ratio: {diagnostics.cluster_virial_ratio}")
        print(f"Bound Mass Fraction: {diagnostics.cluster_bound_mass_fraction}")
