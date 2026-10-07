from dataclasses import dataclass, field
@dataclass
class Position:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

@dataclass
class Velocity:
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

@dataclass
class Entity:
    name: str = "0"
    position: Position = field(default_factory = Position)
    velocity: Velocity = field(default_factory = Velocity)
    mass: float = 0.0
