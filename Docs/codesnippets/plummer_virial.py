def _apply_virial_correction(self, positions, velocities):

    star_mass = CLUSTER_MASS / self.number_of_stars

    softening_eps = 0.01 * self.plummer_radius
    total_potential_energy = 0.0

    for i in range (self.number_of_stars):
        for j in range (i + 1, self.number_of_stars):

            distance_vector = positions[j] - positions[i]
            distance = np.sqrt(np.sum(distance_vector**2) + softening_eps**2)

            total_potential_energy -= GRAVITATIONAL_CONSTANT * (star_mass**2) / distance

    total_kinetic_energy = 0.5 * star_mass * np.sum(velocities**2)

    numerator = np.abs(total_potential_energy)
    denominator = 2.0 * total_kinetic_energy

    virial_scale_factor = np.sqrt(numerator / denominator)

    velocities = velocities * virial_scale_factor

    return velocities
