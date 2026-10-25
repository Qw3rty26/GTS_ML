import numpy as np

from entity import Entity, Position, Velocity


class ClusterDiagnostics:

    def __init__(self, simulation):
        self.simulation = simulation
        self.number_of_orbits_done = 0
        self.previous_center_of_mass_y = None
        self.bound_entity_indices = None
        self.tracked_entity_indices = None

    def _get_entity_energies(
        self,
        entity_positions,
        entity_velocities,
        entity_masses,
        source_indices,
        center_of_mass_velocity
    ):
        # E_i = 1/2 * |v_i - v_COM|^2 - G * sum(m_j / r_ij)

        gravitational_constant = (
            self.simulation.get_gravitational_constant()
        )
        softening = self.simulation.get_softening()

        relative_velocities = (
            entity_velocities - center_of_mass_velocity
        )

        kinetic_energies = (
            0.5 * np.sum(relative_velocities**2, axis=1)
        )

        source_positions = entity_positions[source_indices]
        source_masses = entity_masses[source_indices]

        position_differences = (
            entity_positions[:, np.newaxis, :]
            - source_positions[np.newaxis, :, :]
        )

        distances_squared = np.sum(
            position_differences**2,
            axis=2
        )

        distances = np.sqrt(distances_squared + softening**2)

        source_lookup = np.full(
            len(entity_positions),
            -1,
            dtype=np.int64
        )

        source_lookup[source_indices] = np.arange(len(source_indices))

        entity_indices = np.arange(len(entity_positions))
        source_entity_mask = source_lookup >= 0

        distances[
            entity_indices[source_entity_mask],
            source_lookup[source_entity_mask]
        ] = np.inf

        potential_energies = (
            -gravitational_constant
            * np.sum(
                source_masses[np.newaxis, :] / distances,
                axis=1
            )
        )

        return kinetic_energies + potential_energies

    def get_bound_entities(self):
        entities = self.simulation.get_entities()

        if not entities:
            self.bound_entity_indices = np.array([], dtype=np.int64)
            return []

        entity_positions = np.array([
            [entity.position.x, entity.position.y, entity.position.z]
            for entity in entities
        ], dtype=np.float64)

        entity_velocities = np.array([
            [entity.velocity.vx, entity.velocity.vy, entity.velocity.vz]
            for entity in entities
        ], dtype=np.float64)

        entity_masses = np.array([
            entity.mass for entity in entities
        ], dtype=np.float64)

        number_of_entities = len(entities)

        if self.bound_entity_indices is not None:
            source_indices = self.bound_entity_indices

            if len(source_indices) <= 3:
                self.bound_entity_indices = np.array([], dtype=np.int64)
                return []

            center_of_mass_velocity = np.average(
                entity_velocities[source_indices],
                axis=0,
                weights=entity_masses[source_indices]
            )

        else:
            if number_of_entities <= 3:
                self.bound_entity_indices = np.array([], dtype=np.int64)
                return []

            gravitational_constant = (
                self.simulation.get_gravitational_constant()
            )
            softening = self.simulation.get_softening()

            position_differences = (
                entity_positions[:, np.newaxis, :]
                - entity_positions[np.newaxis, :, :]
            )

            distances_squared = np.sum(
                position_differences**2,
                axis=2
            )

            distances = np.sqrt(distances_squared + softening**2)
            np.fill_diagonal(distances, np.inf)

            gravitational_potentials = (
                -gravitational_constant
                * np.sum(
                    entity_masses[np.newaxis, :] / distances,
                    axis=1
                )
            )

            center_index = int(np.argmin(gravitational_potentials))

            distances_from_center = np.linalg.norm(
                entity_positions - entity_positions[center_index],
                axis=1
            )

            initial_group_indices = np.argsort(
                distances_from_center
            )[:min(11, number_of_entities)]

            center_of_mass_velocity = np.average(
                entity_velocities[initial_group_indices],
                axis=0,
                weights=entity_masses[initial_group_indices]
            )

            source_indices = np.arange(number_of_entities)

        for _ in range(10):
            entity_energies = self._get_entity_energies(
                entity_positions,
                entity_velocities,
                entity_masses,
                source_indices,
                center_of_mass_velocity
            )

            new_source_indices = np.flatnonzero(
                entity_energies < 0.0
            )

            if len(new_source_indices) <= 3:
                self.bound_entity_indices = np.array([], dtype=np.int64)
                return []

            if np.array_equal(source_indices, new_source_indices):
                source_indices = new_source_indices
                break

            source_indices = new_source_indices

            center_of_mass_velocity = np.average(
                entity_velocities[source_indices],
                axis=0,
                weights=entity_masses[source_indices]
            )

        self.bound_entity_indices = source_indices.copy()

        # Keep a reference group for orbital tracking after dissolution.
        if self.tracked_entity_indices is None:
            self.tracked_entity_indices = source_indices.copy()

        return [
            entities[index]
            for index in self.bound_entity_indices
        ]

    def get_bound_mass(self):
        bound_entities = self.get_bound_entities()

        return sum(
            entity.mass
            for entity in bound_entities
        )

    def get_bound_center_of_mass(self) -> Entity:
        # Calculate the center of mass using bound entities only.

        bound_entities = self.get_bound_entities()

        if not bound_entities:
            raise ValueError(
                "Cannot calculate the center of mass: no bound entities."
            )

        entity_positions = np.array([
            [entity.position.x, entity.position.y, entity.position.z]
            for entity in bound_entities
        ], dtype=np.float64)

        entity_velocities = np.array([
            [entity.velocity.vx, entity.velocity.vy, entity.velocity.vz]
            for entity in bound_entities
        ], dtype=np.float64)

        entity_masses = np.array([
            entity.mass for entity in bound_entities
        ], dtype=np.float64)

        center_of_mass_position = np.average(
            entity_positions,
            axis=0,
            weights=entity_masses
        )

        center_of_mass_velocity = np.average(
            entity_velocities,
            axis=0,
            weights=entity_masses
        )

        return Entity(
            name="bound_center_of_mass",
            position=Position(
                center_of_mass_position[0],
                center_of_mass_position[1],
                center_of_mass_position[2]
            ),
            velocity=Velocity(
                center_of_mass_velocity[0],
                center_of_mass_velocity[1],
                center_of_mass_velocity[2]
            ),
            mass=np.sum(entity_masses)
        )

    def get_tracked_center_of_mass(self) -> Entity:
        # R_COM = sum(m_i * r_i) / sum(m_i)
        # V_COM = sum(m_i * v_i) / sum(m_i)

        entities = self.simulation.get_entities()

        if not entities:
            raise ValueError(
                "Cannot calculate the tracked center of mass: no entities."
            )

        if self.tracked_entity_indices is None:
            self.get_bound_entities()

        if (
            self.tracked_entity_indices is None
            or len(self.tracked_entity_indices) == 0
            or np.max(self.tracked_entity_indices) >= len(entities)
        ):
            raise ValueError(
                "Cannot calculate the tracked center of mass: "
                "tracked entities are no longer available."
            )

        tracked_entities = [
            entities[index]
            for index in self.tracked_entity_indices
        ]

        entity_positions = np.array([
            [entity.position.x, entity.position.y, entity.position.z]
            for entity in tracked_entities
        ], dtype=np.float64)

        entity_velocities = np.array([
            [entity.velocity.vx, entity.velocity.vy, entity.velocity.vz]
            for entity in tracked_entities
        ], dtype=np.float64)

        entity_masses = np.array([
            entity.mass for entity in tracked_entities
        ], dtype=np.float64)

        center_of_mass_position = np.average(
            entity_positions,
            axis=0,
            weights=entity_masses
        )

        center_of_mass_velocity = np.average(
            entity_velocities,
            axis=0,
            weights=entity_masses
        )

        return Entity(
            name="tracked_center_of_mass",
            position=Position(
                center_of_mass_position[0],
                center_of_mass_position[1],
                center_of_mass_position[2]
            ),
            velocity=Velocity(
                center_of_mass_velocity[0],
                center_of_mass_velocity[1],
                center_of_mass_velocity[2]
            ),
            mass=np.sum(entity_masses)
        )

    def get_number_of_orbits(self):
        center_of_mass = self.get_tracked_center_of_mass()
        current_center_of_mass_y = center_of_mass.position.y

        if self.previous_center_of_mass_y is not None:
            if (
                self.previous_center_of_mass_y < 0.0
                and current_center_of_mass_y >= 0.0
            ):
                self.number_of_orbits_done += 1

        self.previous_center_of_mass_y = current_center_of_mass_y

        return self.number_of_orbits_done

    def get_system_kinetic_energy(self):
        # K = 1/2 * sum(m_i * |v_i|^2)

        entities = self.simulation.get_entities()

        if not entities:
            return 0.0

        entity_masses = np.array([
            entity.mass for entity in entities
        ], dtype=np.float64)

        entity_velocities = np.array([
            [entity.velocity.vx, entity.velocity.vy, entity.velocity.vz]
            for entity in entities
        ], dtype=np.float64)

        return 0.5 * np.sum(
            entity_masses * np.sum(entity_velocities**2, axis=1)
        )

    def get_system_potential_energy(self):
        # U = -G * sumi<j (m_i * m_j / sqrt(r_ij^2 + softening_epsilon^2))

        entities = self.simulation.get_entities()

        epsilon = self.simulation.get_softening()

        if len(entities) < 2:
            return 0.0

        entity_positions = np.array([
            [entity.position.x, entity.position.y, entity.position.z]
            for entity in entities
        ], dtype=np.float64)

        entity_masses = np.array([
            entity.mass for entity in entities
        ], dtype=np.float64)

        position_differences = (
            entity_positions[:, np.newaxis, :]
            - entity_positions[np.newaxis, :, :]
        )

        distances_squared = np.sum(position_differences**2, axis=2)
        distances = np.sqrt(distances_squared + epsilon**2)

        mass_products = (
            entity_masses[:, np.newaxis]
            * entity_masses[np.newaxis, :]
        )

        upper_triangle_indices = np.triu_indices(
            len(entities),
            k=1
        )

        return (
            -self.simulation.get_gravitational_constant()
            * np.sum(
                mass_products[upper_triangle_indices]
                / distances[upper_triangle_indices]
            )
        )

    def get_system_galactic_potential_energy(self):
        # U_gal = sum(m_i * Phi_galaxy(r_i))

        galactic_potential = self.simulation.galactic_potential

        if galactic_potential is None:
            return 0.0

        entities = self.simulation.get_entities()
        galactic_potential_energy = 0.0

        for entity in entities:
            radius = np.sqrt(
                entity.position.x**2
                + entity.position.y**2
                + entity.position.z**2
            )

            galactic_potential_energy += (
                entity.mass
                * galactic_potential.potential_phi(radius)
            )

        return galactic_potential_energy

    def get_system_total_energy(self):
        # E_tot = K_cluster + U_cluster + U_galaxy
        return (
            self.get_system_kinetic_energy()
            + self.get_system_potential_energy()
            + self.get_system_galactic_potential_energy()
        )

    def get_entity_potential_energy(self, entity):
        # U_i = -G * m_i * sum(m_j / r_ij)

        entities = self.simulation.get_entities()
        entity_position = np.array([
            entity.position.x,
            entity.position.y,
            entity.position.z
        ])

        potential_energy = 0.0
        softening = self.simulation.get_softening()

        for other_entity in entities:
            if other_entity.name == entity.name:
                continue

            position_difference = (
                entity_position
                - np.array([
                    other_entity.position.x,
                    other_entity.position.y,
                    other_entity.position.z
                ])
            )

            distance = np.sqrt(
                np.sum(position_difference**2) + softening**2
            )

            potential_energy -= (
                self.simulation.get_gravitational_constant()
                * entity.mass
                * other_entity.mass
                / distance
            )

        return potential_energy

    def get_entity_kinetic_energy(self, entity):
        # K_i = 1/2 * m_i * |v_i - v_COM|^2

        center_of_mass = self.get_bound_center_of_mass()

        center_of_mass_velocity = np.array([
            center_of_mass.velocity.vx,
            center_of_mass.velocity.vy,
            center_of_mass.velocity.vz
        ])

        entity_velocity = np.array([
            entity.velocity.vx,
            entity.velocity.vy,
            entity.velocity.vz
        ])

        relative_velocity = entity_velocity - center_of_mass_velocity

        return (
            0.5
            * entity.mass
            * np.sum(relative_velocity**2)
        )

    def get_entity_total_energy(self, entity):
        return (
            self.get_entity_kinetic_energy(entity)
            + self.get_entity_potential_energy(entity)
        )

    def get_entity_total_energy_with_galactic_potential(self, entity):
        energy = self.get_entity_total_energy(entity)

        galactic_potential = self.simulation.galactic_potential

        if galactic_potential is not None:
            radius = np.sqrt(
                entity.position.x**2
                + entity.position.y**2
                + entity.position.z**2
            )

            energy += (
                entity.mass
                * galactic_potential.potential_phi(radius)
            )

        return energy

    def _get_cluster_virial_term(self):
        # W_cluster = -G * sum_{i<j} m_i * m_j * r_ij^2
        #             / (r_ij^2 + epsilon^2)^(3/2)

        entities = self.simulation.get_entities()

        if len(entities) < 2:
            return 0.0

        positions, masses = self.simulation.get_entity_arrays()

        gravitational_constant = (
            self.simulation.get_gravitational_constant()
        )
        softening_squared = self.simulation.get_softening() ** 2

        virial_term = 0.0

        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                distance_squared = np.sum(
                    (positions[j] - positions[i]) ** 2
                )

                virial_term -= (
                    gravitational_constant
                    * masses[i]
                    * masses[j]
                    * distance_squared
                    / (distance_squared + softening_squared) ** 1.5
                )

        return virial_term

    def get_virial_ratio(self):
        # Q = 2K / |W|

        kinetic_energy = self.get_system_kinetic_energy()
        virial_term = self._get_cluster_virial_term()

        if virial_term == 0.0:
            return 0.0

        return 2.0 * kinetic_energy / abs(virial_term)

    def apply_virial_correction(self):
        # 2K + W_cluster + W_tidal = 0
        # W_tidal = sum(m_i * x_i . (a_i - a_COM))
        # v_i,new = V_COM + lambda * (v_i - V_COM)
        # lambda = sqrt(-(W_cluster + W_tidal) / (2K))

        entities = self.simulation.get_entities()

        if not entities:
            raise ValueError(
                "Cannot apply virial correction: no entities."
            )

        positions, masses = self.simulation.get_entity_arrays()
        particles = self.simulation.simulation.particles

        velocities = np.array([
            [particle.vx, particle.vy, particle.vz]
            for particle in particles
        ], dtype=np.float64)

        center_of_mass_position = np.average(
            positions,
            axis=0,
            weights=masses
        )

        center_of_mass_velocity = np.average(
            velocities,
            axis=0,
            weights=masses
        )

        relative_positions = positions - center_of_mass_position
        relative_velocities = velocities - center_of_mass_velocity

        kinetic_energy = 0.5 * np.sum(
            masses
            * np.sum(relative_velocities**2, axis=1)
        )

        if kinetic_energy <= 0.0:
            raise ValueError(
                "Cannot apply virial correction: "
                "internal kinetic energy must be positive."
            )

        cluster_virial_term = self._get_cluster_virial_term()
        tidal_virial_term = 0.0

        galactic_potential = self.simulation.galactic_potential

        if galactic_potential is not None:
            gravitational_constant = (
                self.simulation.get_gravitational_constant()
            )
            galaxy_mass = galactic_potential.get_galaxy_mass()
            galaxy_radius_squared = (
                galactic_potential.get_galaxy_radius() ** 2
            )

            center_radius_squared = np.sum(
                center_of_mass_position**2
            )

            center_acceleration = (
                -gravitational_constant
                * galaxy_mass
                * center_of_mass_position
                / (galaxy_radius_squared + center_radius_squared) ** 1.5
            )

            for position, mass in zip(positions, masses):
                radius_squared = np.sum(position**2)

                acceleration = (
                    -gravitational_constant
                    * galaxy_mass
                    * position
                    / (galaxy_radius_squared + radius_squared) ** 1.5
                )

                relative_acceleration = (
                    acceleration - center_acceleration
                )

                tidal_virial_term += (
                    mass
                    * np.dot(
                        position - center_of_mass_position,
                        relative_acceleration
                    )
                )

        total_virial_term = (
            cluster_virial_term + tidal_virial_term
        )

        if total_virial_term >= 0.0:
            raise ValueError(
                "Cannot apply virial correction: "
                "the total virial term must be negative."
            )

        velocity_scale = np.sqrt(
            -total_virial_term / (2.0 * kinetic_energy)
        )

        print(f"velocity scale: {velocity_scale}")

        corrected_velocities = (
            center_of_mass_velocity
            + velocity_scale * relative_velocities
        )

        for particle, velocity in zip(particles, corrected_velocities):
            particle.vx = velocity[0]
            particle.vy = velocity[1]
            particle.vz = velocity[2]

    def get_half_mass_radius(self):
        bound_entities = self.get_bound_entities()

        if not bound_entities:
            return 0.0

        entity_positions = np.array([
            [entity.position.x, entity.position.y, entity.position.z]
            for entity in bound_entities
        ], dtype=np.float64)

        entity_masses = np.array([
            entity.mass for entity in bound_entities
        ], dtype=np.float64)

        # r_HMR encloses half of the total bound mass around the bound COM.
        center_of_mass_position = np.average(
            entity_positions,
            axis=0,
            weights=entity_masses
        )

        distances = np.linalg.norm(
            entity_positions - center_of_mass_position,
            axis=1
        )

        sorted_indices = np.argsort(distances)
        sorted_distances = distances[sorted_indices]
        sorted_masses = entity_masses[sorted_indices]

        half_mass = np.sum(sorted_masses) / 2.0
        cumulative_mass = np.cumsum(sorted_masses)

        half_mass_index = np.searchsorted(
            cumulative_mass,
            half_mass
        )

        return float(sorted_distances[half_mass_index])

    def get_unbound_entity_indices(self):
        # E_i >= 0 -> unbound in the galactic environment

        entities = self.simulation.get_entities()

        return [
            index
            for index, entity in enumerate(entities)
            if self.get_entity_total_energy_with_galactic_potential(entity) >= 0.0
        ]

    def get_root_mean_square_radius(self):
        entities = self.simulation.get_entities()

        if not entities:
            return 0.0

        center_of_mass = self.simulation.get_center_of_mass()

        positions = np.array([
            [entity.position.x, entity.position.y, entity.position.z]
            for entity in entities
        ])

        center = np.array([
            center_of_mass.x,
            center_of_mass.y,
            center_of_mass.z
        ])

        distances_squared = np.sum((positions - center)**2, axis=1)

        return float(np.sqrt(np.mean(distances_squared)))
