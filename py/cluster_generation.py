import logging
import numpy as np
import io_manager
from simulation import Simulation, SimulationConfig, Entity, Position, Velocity
from plummer import Plummer

logger = logging.getLogger(__name__)

class Cluster_Generation:

    def __init__(
        self,
        simulation_config: SimulationConfig,
        seed = 0,
        cluster_radius = 1.0,
        number_of_entities = 10,
    ):
        self.seed = seed
        self.cluster_radius = cluster_radius
        self.number_of_entities = number_of_entities
        self.simulation_config = simulation_config
        self.plummer = None
        self.simulation = None

        logger.debug(f"Cluster {self.seed}: Generating...")
        self._create_cluster()


    def _create_cluster(self):
        ENTITY_MASS = 1 / self.number_of_entities

        self.plummer = Plummer(self.cluster_radius, self.number_of_entities, self.seed)
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
        #calculated through kepler's third law
        END_TIME = (
            20 * self.cluster_radius ** (3 / 2)
            / np.sqrt(self.number_of_entities)
        )

        DT_CLEANUP = 1.0
        DT_SNAPSHOT = 0.01

        next_cleanup = self.simulation.get_time() + DT_CLEANUP
        next_snapshot = self.simulation.get_time() + DT_SNAPSHOT

        _metadata = io_manager.Metadata (
            cluster_seed = self.seed,
            simulation_config = self.simulation_config,
            cluster_radius = self.cluster_radius,
            initial_number_of_entities = self.number_of_entities
        )

        io_manager.init_json_gen(_metadata)

        while self.simulation.get_time() < END_TIME:
            self.simulation.integrate()

            if self.simulation.get_time() >= next_snapshot:
                next_snapshot += DT_SNAPSHOT
                _metadata.simulation_config.t = self.simulation.get_time()
                _entities = self.simulation.get_entities()
                _percentile_center_of_mass = self.simulation.get_percentile_center_of_mass()
                io_manager.save_xyzv_snapshot_gen(_metadata, _entities, _percentile_center_of_mass)
                io_manager.save_json_snapshot_gen(_metadata, _entities, _percentile_center_of_mass)

            if self.simulation.get_time() >= next_cleanup:
                next_cleanup += DT_CLEANUP
                number_of_entities_cleaned = self.simulation.clean_escaped_entities()
                if number_of_entities_cleaned > 0:
                    logger.debug(
                        f"Cluster {self.seed}: Cleaning {number_of_entities_cleaned} star/s..."
                    )

