import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class Ships:
    """Physics-based ship simulation with vectorized operations for multiple ships.

    Represents a collection of ships with both state variables and physics parameters.
    All calculations use PyTorch tensors for vectorized operations across multiple ships.
    Complex number representations efficiently handle 2D position and velocity vectors.

    State Variables:
        - Position: Current position in world coordinates (complex tensor)
        - Velocity: Current velocity vector (complex tensor)
        - Attitude: Current facing direction (complex tensor, unit vector)
        - Turn Offset: Current turn angle offset from velocity direction (real tensor)
        - Boost: Current boost energy remaining (real tensor)
        - Health: Current health points (real tensor)

    Physics Parameters:
        Thrust System:
            - Base Thrust: Base thrust force applied continuously
            - Forward Boost: Multiplier when forward is pressed
            - Backward Boost: Multiplier when backward is pressed
            - Base Energy Cost: Energy consumed per timestep (neutral)
            - Forward Energy Cost: Energy consumed when moving forward
            - Backward Energy Cost: Energy consumed/regen when moving backward

        Aerodynamic Parameters:
            - No Turn Drag: Drag coefficient when flying straight
            - Normal/Sharp Turn parameters for angle, lift, and drag

        Physical Properties:
            - Collision Radius: For collision detection
            - Max Boost: Maximum energy capacity
            - Max Health: Maximum health points
    """

    # Basic info
    n_ships: int

    # Ship State Variables (persistent)
    position: torch.Tensor  # Complex tensor for 2D position
    velocity: torch.Tensor  # Complex tensor for 2D velocity
    attitude: torch.Tensor  # Complex tensor for facing direction
    turn_offset: torch.Tensor  # Real tensor for turn angle offset
    boost: torch.Tensor  # Current boost energy
    health: torch.Tensor  # Current health points

    # Thrust System Parameters
    thrust: torch.Tensor  # Base thrust force
    forward_boost: torch.Tensor  # Forward thrust multiplier
    backward_boost: torch.Tensor  # Backward thrust multiplier
    base_energy_cost: torch.Tensor  # Neutral energy consumption
    forward_energy_cost: torch.Tensor  # Forward movement energy cost
    backward_energy_cost: torch.Tensor  # Backward movement energy cost

    # Aerodynamic Parameters
    no_turn_drag: torch.Tensor
    normal_turn_angle: torch.Tensor
    normal_turn_lift_coef: torch.Tensor
    normal_turn_drag_coef: torch.Tensor
    sharp_turn_angle: torch.Tensor
    sharp_turn_lift_coef: torch.Tensor
    sharp_turn_drag_coef: torch.Tensor

    # Physical Properties
    collision_radius: torch.Tensor
    max_boost: torch.Tensor
    max_health: torch.Tensor

    def __post_init__(self):
        # Initialize state variables that depend on other fields
        self.boost = self.max_boost.clone()
        self.health = self.max_health.clone()
        self.turn_offset = torch.zeros(self.n_ships)

    @classmethod
    def from_scalars(
        cls,
        n_ships: int,
        world_size: tuple = (800, 600),
        random_positions: bool = True,
        initial_position: complex = 0 + 0j,
        initial_velocity: complex = 0 + 0j,
        initial_attitude: complex = 1 + 0j,  # Facing right
        # Thrust system defaults
        thrust: float = 300.0,
        forward_boost: float = 3.0,
        backward_boost: float = 2.0,
        base_energy_cost: float = 0.0,
        forward_energy_cost: float = 5.0,
        backward_energy_cost: float = -1.0,
        # Aerodynamic defaults
        no_turn_drag: float = 0.008,
        normal_turn_angle: float = np.deg2rad(5.0),
        normal_turn_lift_coef: float = 1.0,
        normal_turn_drag_coef: float = 0.01,
        sharp_turn_angle: float = np.deg2rad(15.0),
        sharp_turn_lift_coef: float = 1.5,
        sharp_turn_drag_coef: float = 0.03,
        # Physical property defaults
        collision_radius: float = 10.0,
        max_boost: float = 100.0,
        max_health: float = 100.0,
    ) -> "Ships":
        """Create Ships instance from scalar values.

        Args:
            n_ships: Number of ships to create
            world_size: (width, height) of the world for random positioning
            random_positions: If True, place ships randomly; else use initial_position
            initial_position: Starting position if not random
            initial_velocity: Starting velocity (default: at rest)
            initial_attitude: Starting facing direction (default: right)
            **kwargs: Physics parameters (see class docstring for details)
        """

        # Helper function to create parameter tensors
        def make_tensor(value: float) -> torch.Tensor:
            return torch.full((n_ships,), value)

        # Create initial state tensors
        if random_positions:
            position_tensor = torch.complex(
                torch.rand(n_ships) * world_size[0], torch.rand(n_ships) * world_size[1]
            )
        else:
            position_tensor = torch.full(
                (n_ships,), initial_position, dtype=torch.complex64
            )

        velocity_tensor = torch.full(
            (n_ships,), initial_velocity, dtype=torch.complex64
        )
        attitude_tensor = torch.full(
            (n_ships,), initial_attitude, dtype=torch.complex64
        )

        return cls(
            n_ships=n_ships,
            # State variables (will be initialized in __post_init__)
            position=position_tensor,
            velocity=velocity_tensor,
            attitude=attitude_tensor,
            turn_offset=torch.zeros(n_ships),
            boost=torch.zeros(n_ships),
            health=torch.zeros(n_ships),
            # Thrust system parameters
            thrust=make_tensor(thrust),
            forward_boost=make_tensor(forward_boost),
            backward_boost=make_tensor(backward_boost),
            base_energy_cost=make_tensor(base_energy_cost),
            forward_energy_cost=make_tensor(forward_energy_cost),
            backward_energy_cost=make_tensor(backward_energy_cost),
            # Aerodynamic parameters
            no_turn_drag=make_tensor(no_turn_drag),
            normal_turn_angle=make_tensor(normal_turn_angle),
            normal_turn_lift_coef=make_tensor(normal_turn_lift_coef),
            normal_turn_drag_coef=make_tensor(normal_turn_drag_coef),
            sharp_turn_angle=make_tensor(sharp_turn_angle),
            sharp_turn_lift_coef=make_tensor(sharp_turn_lift_coef),
            sharp_turn_drag_coef=make_tensor(sharp_turn_drag_coef),
            # Physical properties
            collision_radius=make_tensor(collision_radius),
            max_boost=make_tensor(max_boost),
            max_health=make_tensor(max_health),
        )
