import numpy as np

from entity import Entity, Position, Velocity


class ClusterDiagnostics:

    def __init__(self, simulation):
        self.simulation = simulation
        self.number_of_orbits_done = 0
        self.previous_percentile_center_of_mass_y = None

    def get_percentile_center_of_mass(self) -> Entity:
        entities = self.simulation.get_entities()

        entity_positions = np.array([
            [entity.position.x, entity.position.y, entity.position.z]
            for entity in entities
        ], dtype=np.float64)

        entity_velocities = np.array([
            [entity.velocity.vx, entity.velocity.vy, entity.velocity.vz]
            for entity in entities
        ], dtype=np.float64)

        entity_masses = np.array([
            entity.mass
            for entity in entities
        ], dtype=np.float64)

        center_of_mass = self.simulation.get_center_of_mass()

        center_of_mass_position = np.array([
            center_of_mass.x,
            center_of_mass.y,
            center_of_mass.z
        ])

        center_of_mass_velocity = np.array([
            center_of_mass.vx,
            center_of_mass.vy,
            center_of_mass.vz
        ])

        percentiles = [80, 65, 50, 35, 20]

        for percentile in percentiles:
            distances_from_center = np.linalg.norm(
                entity_positions - center_of_mass_position,
                axis=1
            )

            percentile_index = int(
                len(distances_from_center) * percentile / 100
            )

            maximum_distance = np.partition(
                distances_from_center,
                percentile_index
            )[percentile_index]

            entities_inside_percentile = (
                distances_from_center <= maximum_distance
            )

            number_of_entities_inside_percentile = np.sum(
                entities_inside_percentile
            )

            if number_of_entities_inside_percentile < 5:
                break

            entity_positions = entity_positions[
                entities_inside_percentile
            ]

            entity_velocities = entity_velocities[
                entities_inside_percentile
            ]

            entity_masses = entity_masses[
                entities_inside_percentile
            ]

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
            name="percentile_center_of_mass",
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
            mass=-1
        )

    # Used to optimise get_number_of_orbits().
    def get_percentile_center_of_mass_y(self):
        entity_positions, entity_masses = (
            self.simulation.get_entity_arrays()
        )

        center_of_mass = self.simulation.get_center_of_mass()

        center_of_mass_position = np.array([
            center_of_mass.x,
            center_of_mass.y,
            center_of_mass.z
        ])

        percentiles = [80, 65, 50, 35, 20]

        for percentile in percentiles:
            position_differences = (
                entity_positions - center_of_mass_position
            )

            distance_squared = np.sum(
                position_differences ** 2,
                axis=1
            )

            percentile_index = int(
                len(distance_squared) * percentile / 100
            )

            maximum_distance_squared = np.partition(
                distance_squared,
                percentile_index
            )[percentile_index]

            entities_inside_percentile = (
                distance_squared <= maximum_distance_squared
            )

            if np.sum(entities_inside_percentile) < 5:
                break

            entity_positions = entity_positions[
                entities_inside_percentile
            ]

            entity_masses = entity_masses[
                entities_inside_percentile
            ]

            total_mass = np.sum(entity_masses)

            center_of_mass_position = (
                np.sum(
                    entity_positions * entity_masses[:, np.newaxis],
                    axis=0
                )
                / total_mass
            )

        return center_of_mass_position[1]

    def get_number_of_orbits(self):
        current_percentile_center_of_mass_y = (
            self.get_percentile_center_of_mass_y()
        )

        if self.previous_percentile_center_of_mass_y is not None:
            if (
                self.previous_percentile_center_of_mass_y < 0
                and current_percentile_center_of_mass_y >= 0
            ):
                self.number_of_orbits_done += 1
        else:
            self.previous_percentile_center_of_mass_y = 0
            return self.number_of_orbits_done

        self.previous_percentile_center_of_mass_y = (
            current_percentile_center_of_mass_y
        )

        return self.number_of_orbits_done

    def get_kinetic_energy(self):
        #     1
        # K = - * sum( m_i * |v_i - v_COM|^2 )
        #     2

        entities = self.simulation.get_entities()
        center_of_mass = self.simulation.get_center_of_mass()

        center_of_mass_velocity = np.array([
            center_of_mass.vx,
            center_of_mass.vy,
            center_of_mass.vz
        ])

        entity_masses = np.array([
            entity.mass
            for entity in entities
        ], dtype=np.float64)

        entity_velocities = np.array([
            [entity.velocity.vx, entity.velocity.vy, entity.velocity.vz]
            for entity in entities
        ], dtype=np.float64)

        relative_velocities = (
            entity_velocities - center_of_mass_velocity
        )

        relative_velocity_squared = np.sum(
            relative_velocities ** 2,
            axis=1
        )

        kinetic_energy = 0.5 * np.sum(
            entity_masses * relative_velocity_squared
        )

        return kinetic_energy

    def get_potential_energy(self):
        #              G * m_i * m_j
        # U = - sum ---------------------------
        #       i<j  sqrt(|r_i - r_j|^2 + eps^2)

        entities = self.simulation.get_entities()

        if len(entities) < 2:
            return 0.0

        entity_positions = np.array([
            [entity.position.x, entity.position.y, entity.position.z]
            for entity in entities
        ], dtype=np.float64)

        entity_masses = np.array([
            entity.mass
            for entity in entities
        ], dtype=np.float64)

        position_differences = (
            entity_positions[:, np.newaxis, :]
            - entity_positions[np.newaxis, :, :]
        )

        distance_squared = (
            np.sum(position_differences ** 2, axis=-1)
            + self.simulation.get_softening() ** 2
        )

        distances = np.sqrt(distance_squared)

        mass_products = (
            entity_masses[:, np.newaxis]
            * entity_masses[np.newaxis, :]
        )

        upper_triangle_indices = np.triu_indices(
            len(entities),
            k=1
        )

        potential_energy = (
            -self.simulation.get_gravitational_constant()
            * np.sum(
                mass_products[upper_triangle_indices]
                / distances[upper_triangle_indices]
            )
        )

        return potential_energy

    def get_total_energy(self):
        return (
            self.get_kinetic_energy()
            + self.get_potential_energy()
        )

    def get_entity_potential_energy(self, entity):
        #                   G * m_i * m_j
        # U_i = - sum ---------------------------
        #        j!=i  sqrt(|r_i - r_j|^2 + eps^2)

        entities = self.simulation.get_entities()

        entity_position = np.array([
            entity.position.x,
            entity.position.y,
            entity.position.z
        ])

        entity_potential_energy = 0.0

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

            distance_squared = (
                np.sum(position_difference ** 2)
                + self.simulation.get_softening() ** 2
            )

            distance = np.sqrt(distance_squared)

            entity_potential_energy -= (
                self.simulation.get_gravitational_constant()
                * entity.mass
                * other_entity.mass
                / distance
            )

        return entity_potential_energy

    def get_entity_kinetic_energy(self, entity):
        #       1
        # K_i = - * m_i * |v_i - v_COM|^2
        #       2

        center_of_mass = self.simulation.get_center_of_mass()

        center_of_mass_velocity = np.array([
            center_of_mass.vx,
            center_of_mass.vy,
            center_of_mass.vz
        ])

        entity_velocity = np.array([
            entity.velocity.vx,
            entity.velocity.vy,
            entity.velocity.vz
        ])

        relative_velocity = (
            entity_velocity - center_of_mass_velocity
        )

        entity_kinetic_energy = (
            0.5
            * entity.mass
            * np.sum(relative_velocity ** 2)
        )

        return entity_kinetic_energy

    def get_entity_total_energy(self, entity):
        # E_i = K_i + U_i

        return (
            self.get_entity_kinetic_energy(entity)
            + self.get_entity_potential_energy(entity)
        )

    def get_virial_ratio(self):
        #       2K
        # Q = -------
        #       |U|

        kinetic_energy = self.get_kinetic_energy()
        potential_energy = self.get_potential_energy()

        if potential_energy == 0:
            return 0.0

        return (
            2.0 * kinetic_energy
            / abs(potential_energy)
        )

    def get_angular_momentum(self):
        # L = sum( m_i * (r_i - r_COM) x (v_i - v_COM) )

        entities = self.simulation.get_entities()
        center_of_mass = self.simulation.get_center_of_mass()

        center_of_mass_position = np.array([
            center_of_mass.x,
            center_of_mass.y,
            center_of_mass.z
        ])

        center_of_mass_velocity = np.array([
            center_of_mass.vx,
            center_of_mass.vy,
            center_of_mass.vz
        ])

        entity_positions = np.array([
            [entity.position.x, entity.position.y, entity.position.z]
            for entity in entities
        ], dtype=np.float64)

        entity_velocities = np.array([
            [entity.velocity.vx, entity.velocity.vy, entity.velocity.vz]
            for entity in entities
        ], dtype=np.float64)

        entity_masses = np.array([
            entity.mass
            for entity in entities
        ], dtype=np.float64)

        relative_positions = (
            entity_positions - center_of_mass_position
        )

        relative_velocities = (
            entity_velocities - center_of_mass_velocity
        )

        angular_momenta = (
            entity_masses[:, np.newaxis]
            * np.cross(
                relative_positions,
                relative_velocities
            )
        )

        return np.sum(angular_momenta, axis=0)

    def get_half_mass_radius(self):
        # M(<r_50) = 1/2 * M_total

        entities = self.simulation.get_entities()

        if not entities:
            return 0.0

        center_of_mass = self.simulation.get_center_of_mass()

        center_of_mass_position = np.array([
            center_of_mass.x,
            center_of_mass.y,
            center_of_mass.z
        ])

        entity_positions = np.array([
            [entity.position.x, entity.position.y, entity.position.z]
            for entity in entities
        ], dtype=np.float64)

        entity_masses = np.array([
            entity.mass
            for entity in entities
        ], dtype=np.float64)

        distances_from_center = np.linalg.norm(
            entity_positions - center_of_mass_position,
            axis=1
        )

        sorted_indices = np.argsort(distances_from_center)

        sorted_distances = distances_from_center[sorted_indices]
        sorted_masses = entity_masses[sorted_indices]

        total_mass = np.sum(sorted_masses)
        half_mass = total_mass / 2

        cumulative_mass = np.cumsum(sorted_masses)

        half_mass_index = np.searchsorted(
            cumulative_mass,
            half_mass
        )

        return sorted_distances[half_mass_index]

    def get_bound_mass(self):
        # E_i < 0 -> bound to the cluster

        entities = self.simulation.get_entities()
        bound_mass = 0.0

        for entity in entities:
            if self.get_entity_total_energy(entity) < 0:
                bound_mass += entity.mass

        return bound_mass

    def get_entity_energy_for_cleanup(self, entity):
        # E_i = K_i + U_i + m_i * Phi_galaxy(r_i)

        energy = self.get_entity_total_energy(entity)

        galactic_potential = self.simulation.galactic_potential

        if galactic_potential is not None:
            radius_from_galaxy = np.sqrt(
                entity.position.x**2
                + entity.position.y**2
                + entity.position.z**2
            )

            energy += (
                entity.mass
                * galactic_potential.potential_phi(
                    radius_from_galaxy
                )
            )

        return energy

    def get_unbound_entity_indices(self):
        # E_i >= 0 -> unbound
        # E_i < 0  -> bound

        entities = self.simulation.get_entities()

        return [
            index
            for index, entity in enumerate(entities)
            if self.get_entity_energy_for_cleanup(entity) >= 0.0
        ]

    def get_galactic_potential_energy(self):
        # U_gal = sum( m_i * Phi_galaxy(r_i) )

        galactic_potential = self.simulation.galactic_potential

        if galactic_potential is None:
            return 0.0

        entities = self.simulation.get_entities()

        galactic_potential_energy = 0.0

        for entity in entities:
            radius_from_galaxy = np.sqrt(
                entity.position.x**2
                + entity.position.y**2
                + entity.position.z**2
            )

            galactic_potential_energy += (
                entity.mass
                * galactic_potential.potential_phi(radius_from_galaxy)
            )

        return galactic_potential_energy

    def get_total_angular_momentum(self):
        # L = sum( m_i * r_i x v_i )

        entities = self.simulation.get_entities()

        entity_positions = np.array([
            [entity.position.x, entity.position.y, entity.position.z]
            for entity in entities
        ], dtype=np.float64)

        entity_velocities = np.array([
            [entity.velocity.vx, entity.velocity.vy, entity.velocity.vz]
            for entity in entities
        ], dtype=np.float64)

        entity_masses = np.array([
            entity.mass
            for entity in entities
        ], dtype=np.float64)

        angular_momenta = (
            entity_masses[:, np.newaxis]
            * np.cross(
                entity_positions,
                entity_velocities
            )
        )

        return np.sum(angular_momenta, axis=0)
