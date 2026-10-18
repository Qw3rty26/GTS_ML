def _sample_velocity(self, radius_from_center, radius_vector, sigma_squared_vector, radial_versor):

    while True:

        sigma_squared = np.interp(
            radius_from_center,
            radius_vector,
            sigma_squared_vector
        )

        radial_velocity = self.rng.normal(
            scale = np.sqrt(sigma_squared)
        )

        theta_velocity = self.rng.normal(
            scale = np.sqrt((1.0 - self._anisotropy_beta(radius_from_center)) * sigma_squared)
        )

        phi_velocity = self.rng.normal(
            scale = np.sqrt((1.0 - self._anisotropy_beta(radius_from_center)) * sigma_squared)
        )

        theta = np.arccos(radial_versor[2])
        phi = np.arctan2(radial_versor[1], radial_versor[0])

        theta_versor = np.array([
            np.cos(theta) * np.cos(phi),
            np.cos(theta) * np.sin(phi),
            -np.sin(theta)
        ])

        phi_versor = np.array([
            -np.sin(phi),
            np.cos(phi),
            0.0
        ])

        velocity = (
            radial_velocity * radial_versor +
            theta_velocity * theta_versor +
            phi_velocity * phi_versor
        )

        kinetic_energy = 0.5 * np.dot(velocity, velocity)

        potential_energy = self._potential_phi(radius_from_center)

        total_energy = kinetic_energy + potential_energy

        if total_energy < 0:
            return velocity
