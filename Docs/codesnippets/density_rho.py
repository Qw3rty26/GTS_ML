def _density_rho(self, radius_from_center):

    #                   3 * M * a^2
    # rho(r) = ----------------------------
    #           4 * pi * (a^2 + r^2)^(5/2)

    numerator = 3 * CLUSTER_MASS * (self.plummer_radius ** 2)

    denominator = 4 * np.pi * ((self.plummer_radius ** 2 + radius_from_center**2)**(5/2))

    density_rho = numerator / denominator
    return density_rho
