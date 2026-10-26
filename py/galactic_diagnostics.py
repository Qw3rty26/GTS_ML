import numpy as np
from dataclasses import dataclass

@dataclass
class Galactic_Diagnostics:
    time: float = None
    cluster_kinetic_energy: float = None
    cluster_potential_energy: float = None
    cluster_total_energy: float = None
    cluster_relative_energy_error: float = None
    cluster_kinetic_internal_energy: float = None
    cluster_virial_ratio: float = None
    cluster_bound_mass_fraction: float = None
    cluster_half_mass_radius: float = None
    cluster_half_mass_radius_ratio: float = None
    cluster_root_mean_square_radius: float = None
    cluster_angular_momentum: float = None
    cluster_number_of_orbits_done: int = None

class Diagnostics:

    def __init__(self, simulation):
        self.simulation = simulation

        self.initial_cluster_total_energy = None
        self.initial_cluster_half_mass_radius = None
        self.previous_bound_center_of_mass_position_y = None
        self.number_of_orbits_done = 0


    def get_center_of_mass_position(self):
        #          N
        # r_COM = sum(m_i * r_i) / sum(m_i)
        #         i=1

        particle_masses = self.simulation.get_particle_masses()
        particle_positions = self.simulation.get_particle_positions()

        total_mass = np.sum(particle_masses)
        center_of_mass_position = np.sum(particle_masses[:, np.newaxis] * particle_positions, axis=0) / total_mass

        rebound_center_of_mass = self.simulation.rebound_handle.com()
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

        if self.simulation.rebound_handle.t == 0.0:
            self.initial_cluster_total_energy = cluster_total_energy

        return cluster_total_energy

    def get_relative_energy_error(self, total_energy_t = None):
        #            E_tot(t) - E_tot(0)
        # delta_E = ---------------------
        #                |E_tot(0)|

        if total_energy_t is None: total_energy_t = self.get_cluster_total_energy()

        numerator = total_energy_t - self.initial_cluster_total_energy
        denominator = abs(self.initial_cluster_total_energy)

        relative_energy_error = numerator / denominator
        return relative_energy_error

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
        cluster_internal_kinetic_energy = 0.5 * np.sum(particle_masses * squared_speeds)
        return cluster_internal_kinetic_energy

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

        cluster_softened_potential_energy = np.sum(np.triu(potential_energy_matrix, k=1))
        return cluster_softened_potential_energy

    def _get_internal_virial_term(self):
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

        internal_virial_term = np.sum(np.triu(virial_term_matrix, k=1))
        return internal_virial_term

    def _get_galactic_virial_term(self):
        #                  N
        # W_gal = sum ( m_i * rho_i . (a_gal(r_i) - a_gal(R_COM)) )
        #                 i=1

        particle_masses = self.simulation.get_particle_masses()
        particle_positions = self.simulation.get_particle_positions()

        center_of_mass_position = self.get_center_of_mass_position()
        relative_positions = particle_positions - center_of_mass_position

        particle_accelerations = np.array([
            self.simulation.galaxy_handle.get_acceleration(position)
            for position in particle_positions
        ])

        center_of_mass_acceleration = self.simulation.galaxy_handle.get_acceleration(center_of_mass_position)
        relative_accelerations = particle_accelerations - center_of_mass_acceleration

        galactic_virial_term = np.sum(particle_masses * np.sum(relative_positions * relative_accelerations, axis=1))
        return galactic_virial_term

    def _get_virial_term(self):
        # W = W_internal + W_gal

        internal_virial_term = self._get_internal_virial_term()
        galactic_virial_term = self._get_galactic_virial_term()

        virial_term = internal_virial_term + galactic_virial_term
        return virial_term

    def get_cluster_virial_ratio(self):
        #      2 * K_internal
        # Q = ----------------
        #           |W|

        internal_kinetic_energy = self.get_cluster_internal_kinetic_energy()
        virial_term = self._get_virial_term()

        if np.isclose(virial_term, 0.0):
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
        #        1
        # e_i = --- * |v_i - v_COM|^2 + phi_i
        #        2
        # S' = {i : e_i < 0}

        particle_masses = self.simulation.get_particle_masses()
        particle_positions = self.simulation.get_particle_positions()
        particle_velocities = self.simulation.get_particle_velocities()

        number_of_particles = len(particle_masses)

        if number_of_particles <= 3:
            return np.zeros(number_of_particles, dtype=bool)

        position_differences = particle_positions[:, np.newaxis, :] - particle_positions[np.newaxis, :, :]
        squared_distances = np.sum(position_differences**2, axis=2)

        all_indices = np.arange(number_of_particles)
        initial_potential = self._get_particle_gravitational_potential(particle_masses, squared_distances, all_indices)
        center_index = np.argmin(initial_potential)
        nearest_indices = np.argsort(squared_distances[center_index])[:min(11, number_of_particles)]
        bound_center_of_mass_velocity = np.average(particle_velocities[nearest_indices], axis=0, weights=particle_masses[nearest_indices])

        bound_particles = np.ones(number_of_particles, dtype=bool)

        for _ in range(10):
            bound_indices = np.flatnonzero(bound_particles)

            if len(bound_indices) <= 3:
                return np.zeros(number_of_particles, dtype=bool)

            relative_velocities = particle_velocities - bound_center_of_mass_velocity
            squared_speeds = np.sum(relative_velocities**2, axis=1)

            gravitational_potential = self._get_particle_gravitational_potential(particle_masses, squared_distances, bound_indices)
            specific_energy = 0.5 * squared_speeds + gravitational_potential
            new_bound_particles = specific_energy < 0

            if np.array_equal(bound_particles, new_bound_particles):
                break

            bound_particles = new_bound_particles

            if np.count_nonzero(bound_particles) <= 3:
                return np.zeros(number_of_particles, dtype=bool)

            _, bound_center_of_mass_velocity = self.get_bound_center_of_mass(bound_particles)

        return bound_particles

    def _check_orbit_done(self, bound_center_of_mass_position_y):

        if self.previous_bound_center_of_mass_position_y is None:
            if bound_center_of_mass_position_y > 0:
                self.previous_bound_center_of_mass_position_y = bound_center_of_mass_position_y
            return

        if self.previous_bound_center_of_mass_position_y < 0 and bound_center_of_mass_position_y >= 0:
            self.number_of_orbits_done += 1

        self.previous_bound_center_of_mass_position_y = bound_center_of_mass_position_y

    def get_bound_center_of_mass(self, bound_particles = None):
        #              sum(m_i * r_i), i in S
        # r_COM_bound = ----------------------
        #                  sum(m_i), i in S
        #
        #              sum(m_i * v_i), i in S
        # v_COM_bound = ----------------------
        #                  sum(m_i), i in S

        if bound_particles is None:
            bound_particles = self._get_bound_particles()

        particle_masses = self.simulation.get_particle_masses()
        particle_positions = self.simulation.get_particle_positions()
        particle_velocities = self.simulation.get_particle_velocities()

        particle_masses = particle_masses[bound_particles]
        particle_positions = particle_positions[bound_particles]
        particle_velocities = particle_velocities[bound_particles]

        bound_masses = particle_masses
        bound_positions = particle_positions
        bound_velocities = particle_velocities

        total_bound_mass = np.sum(bound_masses)

        if total_bound_mass == 0:
            raise ValueError("Cannot calculate bound center of mass: no bound mass.")

        bound_center_of_mass_position = np.sum(bound_masses[:, np.newaxis] * bound_positions, axis=0) / total_bound_mass
        bound_center_of_mass_velocity = np.sum(bound_masses[:, np.newaxis] * bound_velocities, axis=0) / total_bound_mass

        return bound_center_of_mass_position, bound_center_of_mass_velocity

    def get_bound_mass_fraction(self, bound_particles = None):
        #            sum(m_i), i in S
        # f_bound = ------------------
        #                  N
        #                 sum(m_i)
        #                 i=1
        particle_masses = self.simulation.get_particle_masses()

        bound_mass = np.sum(particle_masses[bound_particles])
        total_mass = np.sum(particle_masses)

        bound_mass_fraction = bound_mass / total_mass
        return bound_mass_fraction

    def get_half_mass_radius(self, bound_particles = None):
        #                                    M_bound
        # r_half = r_k such that M(<r_k) >= ---------
        #                                       2

        particle_masses = self.simulation.get_particle_masses()
        particle_positions = self.simulation.get_particle_positions()

        if bound_particles is not None:
            particle_masses = particle_masses[bound_particles]
            particle_positions = particle_positions[bound_particles]

        bound_masses = particle_masses
        bound_positions = particle_positions

        if len(bound_masses) == 0:
            raise ValueError("Cannot calculate half-mass radius: no bound particles.")

        center_of_mass_position, _ = self.get_bound_center_of_mass(bound_particles)

        distances = np.linalg.norm(bound_positions - center_of_mass_position, axis=1)

        sorted_indices = np.argsort(distances)
        sorted_masses = bound_masses[sorted_indices]
        sorted_distances = distances[sorted_indices]

        half_mass = np.sum(sorted_masses) / 2
        half_mass_index = np.searchsorted(np.cumsum(sorted_masses), half_mass, side="left")

        if self.simulation.rebound_handle.t == 0:
            self.initial_cluster_half_mass_radius = sorted_distances[half_mass_index]

        return sorted_distances[half_mass_index]

    def get_half_mass_radius_ratio(self, half_mass_radius_t = None):
        #              r_hmr(t)
        # ratio_hmr = ----------
        #              r_hmr(0)

        if half_mass_radius_t is None: half_mass_radius_t = self.get_half_mass_radius()

        half_mass_radius_ratio = half_mass_radius_t / self.initial_cluster_half_mass_radius
        return half_mass_radius_ratio

    def get_root_mean_square_radius(self, bound_particles = None, bound_center_of_mass_position = None):
        #                  1      N_bound
        # r_RMS = sqrt(--------- * sum( |r_i - r_COM_bound|^2) )
        #               N_bound   i in S

        particle_positions = self.simulation.get_particle_positions()

        bound_positions = particle_positions[bound_particles]

        if len(bound_positions) == 0:
            raise ValueError("Cannot calculate RMS radius: no bound particles.")

        squared_distances = np.sum( (bound_positions - bound_center_of_mass_position) ** 2, axis=1 )

        root_mean_square_radius = np.sqrt( np.mean(squared_distances) )
        return root_mean_square_radius

    def get_cluster_angular_momentum(self):
        #          | sum(m_i * (r_i x v_i) ) |
        # L_tot =  ---------------------------
        #                    i

        particle_masses = self.simulation.get_particle_masses()
        particle_positions = self.simulation.get_particle_positions()
        particle_velocities = self.simulation.get_particle_velocities()

        cluster_angular_momentum = np.linalg.norm(np.sum(particle_masses[:, np.newaxis] * np.cross(particle_positions, particle_velocities), axis=0))
        return cluster_angular_momentum

    def get_galactic_diagnostics(self):
        bound_particles = self._get_bound_particles()
        bound_center_of_mass_position, bound_center_of_mass_velocity = self.get_bound_center_of_mass()
        self._check_orbit_done(bound_center_of_mass_position[1])
        total_energy_t = self.get_cluster_total_energy()
        half_mass_radius_t = self.get_half_mass_radius()

        galactic_diagnostics = Galactic_Diagnostics(
            time = self.simulation.rebound_handle.t,
            cluster_kinetic_energy = self.get_cluster_kinetic_energy(),
            cluster_potential_energy = self.get_cluster_potential_energy(),
            cluster_total_energy = total_energy_t,
            cluster_virial_ratio = self.get_cluster_virial_ratio(),
            cluster_relative_energy_error = self.get_relative_energy_error(total_energy_t),
            cluster_kinetic_internal_energy = self.get_cluster_internal_kinetic_energy(),
            cluster_bound_mass_fraction = self.get_bound_mass_fraction(bound_particles),
            cluster_half_mass_radius = half_mass_radius_t,
            cluster_half_mass_radius_ratio = self.get_half_mass_radius_ratio(half_mass_radius_t),
            cluster_root_mean_square_radius = self.get_root_mean_square_radius(bound_particles, bound_center_of_mass_position),
            cluster_angular_momentum = self.get_cluster_angular_momentum(),
            cluster_number_of_orbits_done = self.number_of_orbits_done
        )
        return galactic_diagnostics

    def print_diagnostics(self, galactic_diagnostics: Galactic_Diagnostics = None):
        if galactic_diagnostics is None:
            galactic_diagnostics = self.get_galactic_diagnostics()

        print("---------------------------DIAGNOSTICS---------------------------")
        print(f"Time: {galactic_diagnostics.time}")
        print(f"Kinetic energy: {galactic_diagnostics.cluster_kinetic_energy}")
        print(f"Potential energy: {galactic_diagnostics.cluster_potential_energy}")
        print(f"Total energy: {galactic_diagnostics.cluster_total_energy}")
        print(f"Internal kinetic energy: {galactic_diagnostics.cluster_kinetic_internal_energy}")
        print(f"Virial ratio: {cluster_diagnostics.cluster_virial_ratio}")
        print(f"Bound Mass Fraction: {galactic_diagnostics.cluster_bound_mass_fraction}")
        print(f"Half Mass Radius: {galactic_diagnostics.cluster_half_mass_radius}")
        print("-----------------------------------------------------------------\n")
