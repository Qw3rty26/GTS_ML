
import os
import time
import numpy as np
import rebound
from dataclasses import dataclass
from simulation import Simulation, Simulation_Configuration
from galaxy import Galaxy, Galaxy_Configuration
from file_handle import File_Handle
from galactic_diagnostics import Galactic_Diagnostics

@dataclass
class Galactic_Tidal_Stripping_Configuration:
    simulation_configuration: Simulation_Configuration = None
    galaxy_configuration: Galaxy_Configuration = None
    number_of_orbits: int = None
    end_time: float = None
    cluster_seed: int = None
    snapshot_interval: float = None
    snapshot_file_path: str = None

class Galactic_Tidal_Stripping:

    def __init__(self, galactic_tidal_stripping_configuration: Galactic_Tidal_Stripping_Configuration = None, cluster_file_path=None):
        self.galactic_tidal_stripping_configuration = galactic_tidal_stripping_configuration
        # this is what being obsessed with OOP does to a mf
        self.simulation_name = f"{self.galactic_tidal_stripping_configuration.cluster_seed}_a{self.galactic_tidal_stripping_configuration.galaxy_configuration.orbit_semi_major_axis}_e{self.galactic_tidal_stripping_configuration.galaxy_configuration.orbit_eccentricity}"

        json_file_path = os.path.join(self.galactic_tidal_stripping_configuration.snapshot_file_path, f"{self.simulation_name}.json")
        self.json_file_descriptor = File_Handle.open_json_snapshot(json_file_path)

        isolated_json_file_path = os.path.join(self.galactic_tidal_stripping_configuration.snapshot_file_path, f"{self.simulation_name}_isolated.json")
        self.isolated_json_file_descriptor = File_Handle.open_json_snapshot(isolated_json_file_path)

        self.xyzv_file_path = os.path.join(self.galactic_tidal_stripping_configuration.snapshot_file_path, f"{self.simulation_name}.xyzv")
        self.tsv_file_path = os.path.join(self.galactic_tidal_stripping_configuration.snapshot_file_path, f"{self.simulation_name}.tsv")

        self.isolated_xyzv_file_path = os.path.join(self.galactic_tidal_stripping_configuration.snapshot_file_path, f"{self.simulation_name}_isolated.xyzv")
        self.isolated_tsv_file_path = os.path.join(self.galactic_tidal_stripping_configuration.snapshot_file_path, f"{self.simulation_name}_isolated.tsv")

        self._create_simulation(cluster_file_path)

    def _save_snapshot(self):
        particle_positions = self.simulation.get_particle_positions()
        particle_velocities = self.simulation.get_particle_velocities()
        particle_masses = self.simulation.get_particle_masses()
        bound_center_of_mass_position, bound_center_of_mass_velocity = self.simulation.diagnostics.get_bound_center_of_mass()
        galactic_diagnostics = self.simulation.diagnostics.get_galactic_diagnostics()

        File_Handle.save_json_snapshot(
            self.json_file_descriptor,
            particle_positions,
            particle_velocities,
            particle_masses
        )
        File_Handle.save_xyzv_snapshot_gts(
            self.xyzv_file_path,
            particle_positions,
            particle_velocities,
            bound_center_of_mass_position,
            bound_center_of_mass_velocity
        )
        File_Handle.save_tsv_snapshot_gts(
            self.tsv_file_path,
            self.galactic_tidal_stripping_configuration,
            galactic_diagnostics
        )

    def _save_isolated_snapshot(self):
        particle_positions = self.isolated_simulation.get_particle_positions()
        particle_velocities = self.isolated_simulation.get_particle_velocities()
        particle_masses = self.isolated_simulation.get_particle_masses()
        cluster_diagnostics = self.isolated_simulation.diagnostics.get_cluster_diagnostics()

        File_Handle.save_json_snapshot(
            self.isolated_json_file_descriptor,
            particle_positions,
            particle_velocities,
            particle_masses
        )
        File_Handle.save_xyzv_snapshot(
            self.isolated_xyzv_file_path,
            particle_positions,
            particle_velocities
        )
        File_Handle.save_tsv_snapshot(
            self.isolated_tsv_file_path,
            cluster_diagnostics
        )

    def _create_simulation(self, cluster_file_path):

        print(f"Simulation {self.simulation_name}: Creating galaxy_handle")
        self.galaxy_handle = Galaxy(self.galactic_tidal_stripping_configuration.galaxy_configuration)

        particle_positions, particle_velocities, particle_masses = File_Handle.load_json_cluster(cluster_file_path, -1)

        print(f"Simulation {self.simulation_name}: Adding the particles to rebound")
        self.simulation = Simulation(self.galactic_tidal_stripping_configuration.simulation_configuration)
        self.isolated_simulation = Simulation(self.galactic_tidal_stripping_configuration.simulation_configuration)

        for particle_name, (particle_position, particle_velocity, particle_mass) in enumerate(zip(particle_positions, particle_velocities, particle_masses)):
            particle = rebound.Particle(
                name = str(particle_name),
                x = particle_position[0],
                y = particle_position[1],
                z = particle_position[2],
                vx = particle_velocity[0],
                vy = particle_velocity[1],
                vz = particle_velocity[2],
                m = particle_mass
            )
            isolated_particle = rebound.Particle(
                name = str(particle_name),
                x = particle_position[0],
                y = particle_position[1],
                z = particle_position[2],
                vx = particle_velocity[0],
                vy = particle_velocity[1],
                vz = particle_velocity[2],
                m = particle_mass
            )
            self.simulation.add_particle(particle)
            self.isolated_simulation.add_particle(isolated_particle)

        self.simulation.rebound_handle.move_to_com()
        self.isolated_simulation.rebound_handle.move_to_com()

        bound_center_of_mass_position, _ = self.simulation.diagnostics.get_bound_center_of_mass()
        self.simulation.move_cluster(*(self.galaxy_handle.get_initial_orbital_position() - bound_center_of_mass_position))
        self.simulation.speed_cluster(*self.galaxy_handle.get_initial_orbital_velocity())

        self.isolated_simulation.move_cluster(*(self.galaxy_handle.get_initial_orbital_position() - bound_center_of_mass_position))
        self.isolated_simulation.speed_cluster(*self.galaxy_handle.get_initial_orbital_velocity())

        self.simulation.add_galaxy(self.galaxy_handle)
        #self.simulation.diagnostics.print_diagnostics()

    def run(self):
        try:
            print(f"Simulation {self.simulation_name}: Running Galactic Tidal Stripping Simulation")
            _computing_time = time.perf_counter()
            _next_snapshot_time = self.simulation.rebound_handle.t + self.galactic_tidal_stripping_configuration.snapshot_interval
            self._save_snapshot()
            self._save_isolated_snapshot()
            while self.simulation.rebound_handle.t < self.galactic_tidal_stripping_configuration.end_time:
            #while self.simulation.diagnostics.number_of_orbits_done < self.galactic_tidal_stripping_configuration.number_of_orbits:
                self.simulation.integrate()
                self.isolated_simulation.integrate()

                if self.simulation.rebound_handle.t >= _next_snapshot_time:
                    self._save_snapshot()
                    self._save_isolated_snapshot()
                    _next_snapshot_time += self.galactic_tidal_stripping_configuration.snapshot_interval
            self._save_snapshot()
            _computing_time = time.perf_counter() - _computing_time
            print(f"Simulation {self.simulation_name}: Done simulating in {_computing_time} seconds.")
            #self.simulation.diagnostics.print_diagnostics()
        finally:
            File_Handle.close_json_snapshot(self.json_file_descriptor)
            File_Handle.close_json_snapshot(self.isolated_json_file_descriptor)
