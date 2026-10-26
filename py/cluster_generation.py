import os
import time
import numpy as np
import rebound
from dataclasses import dataclass
from simulation import Simulation, Simulation_Configuration
from plummer import Plummer
from file_handle import File_Handle
from cluster_diagnostics import Cluster_Diagnostics

@dataclass
class Cluster_Configuration:
    simulation_configuration: Simulation_Configuration = None
    number_of_particles: int = None
    master_seed: int = None
    cluster_seed: int = None
    cluster_radius: float = None
    snapshot_interval: float = None
    snapshot_file_path: str = None

class Cluster_Generation:

    def __init__(self, cluster_configuration: Cluster_Configuration):
        self.cluster_configuration = cluster_configuration

        json_file_path = os.path.join(self.cluster_configuration.snapshot_file_path, f"{self.cluster_configuration.cluster_seed}.json")
        self.json_file_descriptor = File_Handle.open_json_snapshot(json_file_path)

        self.xyzv_file_path = os.path.join(self.cluster_configuration.snapshot_file_path, f"{self.cluster_configuration.cluster_seed}.xyzv")

        self.tsv_file_path = os.path.join(self.cluster_configuration.snapshot_file_path, f"{self.cluster_configuration.cluster_seed}.tsv")

        self._create_simulation()

    def _save_snapshot(self):
        particle_positions = self.simulation.get_particle_positions()
        particle_velocities = self.simulation.get_particle_velocities()
        particle_masses = self.simulation.get_particle_masses()
        cluster_diagnostics = self.simulation.diagnostics.get_cluster_diagnostics()

        File_Handle.save_json_snapshot(
            self.json_file_descriptor,
            particle_positions,
            particle_velocities,
            particle_masses
        )
        File_Handle.save_xyzv_snapshot(
            self.xyzv_file_path,
            particle_positions,
            particle_velocities
        )
        File_Handle.save_tsv_snapshot(
            self.tsv_file_path,
            cluster_diagnostics
        )

    def _create_simulation(self):

        PARTICLE_MASS = 1.0 / self.cluster_configuration.number_of_particles

        print(f"Creating plummer_handle")
        self.plummer_handle = Plummer(self.cluster_configuration.number_of_particles, self.cluster_configuration.cluster_radius, self.cluster_configuration.cluster_seed)

        particle_positions, particle_velocities = self.plummer_handle.generate_particles()

        print(f"Adding the particles to rebound")
        self.simulation = Simulation(self.cluster_configuration.simulation_configuration)
        for particle_name, (particle_position, particle_velocity) in enumerate(zip(particle_positions, particle_velocities)):
            particle = rebound.Particle(
                name = str(particle_name),
                x = particle_position[0],
                y = particle_position[1],
                z = particle_position[2],
                vx = particle_velocity[0],
                vy = particle_velocity[1],
                vz = particle_velocity[2],
                m = PARTICLE_MASS
            )
            self.simulation.add_particle(particle)
            self.simulation.rebound_handle.move_to_com()
        #self.simulation.diagnostics.print_diagnostics()

    def run(self):
        try:
            END_TIME = self.plummer_handle.get_relaxation_time()
            print(f"Running generation in the void until {END_TIME}")
            _computing_time = time.perf_counter()
            _next_snapshot_time = self.simulation.rebound_handle.t + self.cluster_configuration.snapshot_interval
            self._save_snapshot()
            while self.simulation.rebound_handle.t < END_TIME:
                self.simulation.integrate()

                if self.simulation.rebound_handle.t >= _next_snapshot_time:
                    self._save_snapshot()
                    _next_snapshot_time += self.cluster_configuration.snapshot_interval
            _computing_time = time.perf_counter() - _computing_time
            print(f"Done simulating in {_computing_time} seconds.")
            #self.simulation.diagnostics.print_diagnostics()
        finally:
            File_Handle.close_json_snapshot(self.json_file_descriptor)



