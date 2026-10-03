import rebound

from cluster_diagnostics import ClusterDiagnostics

class Simulation:

    def __init__(self, dt, G, softening, time_warp, integrator):

        self.simulation = rebound.Simulation()
        self.galactic_potential = None
        self.cluster_diagnostics = ClusterDiagnostics(self)

        self.simulation.dt = dt
        self.simulation.t = 0
        self.simulation.G = G
        self.simulation.softening = softening
        self.simulation.integrator = integrator
        self.time_warp = time_warp

    def update(self):
        for _ in range(self.time_warp):
            self.simulation.integrate(
                self.simulation.t + self.simulation.dt
            )

    def move_cluster(self, moved_x, moved_y, moved_z):
        for particle in self.simulation.particles:
            particle.x += moved_x
            particle.y += moved_y
            particle.z += moved_z

    def speed_cluster(self, speed_x, speed_y, speed_z):
        for particle in self.simulation.particles:
            particle.vx += speed_x
            particle.vy += speed_y
            particle.vz += speed_z

    def _apply_galactic_forces(self, sim_pointer):
        self.galactic_potential.add_galaxy_forces(self.simulation.particles)

    def add_galactic_potential(self, galactic_potential):
        self.galactic_potential = galactic_potential
        self.simulation.additional_forces = self._apply_galactic_forces

    def add_entity(self, x=0, y=0, z=0, vx=0, vy=0, vz=0, mass=0.1, name=0):
        self.simulation.add(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, m=mass, name=str(name))


    def remove_entity(self, entity_id):
        self.simulation.remove(entity_id)

    def clean_escaped_stars(self, escaped_entity_indices):
        for index in sorted(escaped_entity_indices, reverse=True):
            self.simulation.remove(index)

        return len(escaped_entity_indices)
