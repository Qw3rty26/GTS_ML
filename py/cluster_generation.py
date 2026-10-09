import logging
import numpy as np
import io_manager
from io_manager import Metadata
from simulation import Simulation, SimulationConfig, Entity, Position, Velocity
from plummer import Plummer

logger = logging.getLogger(__name__)

class Cluster_Generation:

    def __init__(
        self,
        simulation_config: SimulationConfig,
        metadata: Metadata,
        json_and_xyzv_snapshots_dt = 1,
    ):
        self.json_and_xyzv_snapshots_dt = json_and_xyzv_snapshots_dt
        self.metadata = metadata
        self.simulation_config = simulation_config
        self.plummer = None
        self.simulation = None
        self._create_cluster()

    def _save_to_file(self):
            self.metadata.simulation_config.t = self.simulation.get_time()
            _entities = self.simulation.get_entities()
            _percentile_center_of_mass = self.simulation.get_percentile_center_of_mass()
            io_manager.save_xyzv_snapshot_gen(self.metadata, _entities, _percentile_center_of_mass)
            io_manager.save_json_snapshot_gen(self.metadata, _entities, _percentile_center_of_mass)

    def _create_cluster(self):
        logger.info(f"Cluster {self.metadata.cluster_seed}: Creating...")
        ENTITY_MASS = 1 / self.metadata.initial_number_of_entities

        self.plummer = Plummer(self.metadata.cluster_radius, self.metadata.initial_number_of_entities, self.metadata.cluster_seed)
        self.simulation = Simulation(self.simulation_config)

        entity_positions, entity_velocities = self.plummer.generate_plummer_cluster()

        for entity_name, (entity_position, entity_velocity) in enumerate(zip(entity_positions, entity_velocities)):
            new_entity = Entity(
                    name = str(entity_name),
                    position = Position(
                        x = entity_position[0],
                        y = entity_position[1],
                        z = entity_position[2]
                    ),
                    velocity = Velocity(
                        vx = entity_velocity[0],
                        vy = entity_velocity[1],
                        vz = entity_velocity[2]
                    ),
                    mass = ENTITY_MASS
                )
            self.simulation.add_entity(new_entity)

        self.simulation.move_cluster_to_center_of_mass()

    def run(self):
        logger.info(f"Cluster {self.metadata.cluster_seed}: Running...")

        #calculated through kepler's third law
        END_TIME = (
            20 * self.metadata.cluster_radius ** (3/2)
            / np.sqrt(self.metadata.initial_number_of_entities)
        )

        logger.debug(f"Cluster {self.metadata.cluster_seed}: END_TIME = {END_TIME:.4f}")

        DT_CLEANUP = 1.0

        next_cleanup = self.simulation.get_time() + DT_CLEANUP
        next_snapshot = self.simulation.get_time() + self.json_and_xyzv_snapshots_dt 

        io_manager.init_json_gen(self.metadata)

        self._save_to_file()


        while self.simulation.get_time() < END_TIME:
            self.simulation.integrate()

            if self.simulation.get_time() >= next_snapshot:
                next_snapshot += self.json_and_xyzv_snapshots_dt
                self._save_to_file()

            if self.simulation.get_time() >= next_cleanup:
                next_cleanup += DT_CLEANUP
                number_of_entities_cleaned = self.simulation.clean_unbound_entities()
                if number_of_entities_cleaned > 0:
                    logger.debug(f"Cluster {self.metadata.cluster_seed}: Cleaning {number_of_entities_cleaned} star/s...")
        logger.info(f"\033[94mCluster {self.metadata.cluster_seed}: Generated.\033[0m")
