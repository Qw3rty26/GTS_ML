def _sample_radius(self):
    mass_fraction_u = self.rng.random()
    def mass_equation(radius):
        return self._enclosed_mass(radius) - mass_fraction_u

    radius_guess = self.plummer_radius # good starting guess
    sampled_radius = so.fsolve(mass_equation, radius_guess)[0]
    return sampled_radius

def _random_versor(self):
    versor = self.rng.normal(size = 3) # generate a random point in 3D space
    norm = np.linalg.norm(versor)
    if norm == 0:
        return self._random_versor()
    else:
        return versor / norm
