import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class Ships:
    """Physics-based ship simulation with vectorized operations for multiple ships.

    Represents a collection of        # Projectile system defaults
        max_projectiles: int = 20,
        projectile_speed: float = 500.0,
        projectile_damage: float = 10.0,
        projectile_lifetime: float = 2.0,
        firing_cooldown: float = 0.2,
        max_ammo: float = 10.0,
        ammo_regen_rate: float = 0.5,  # Regenerate 0.5 ammo per second
        # Aerodynamic defaultswith both state variables and physics parameters.
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

    Lookup Tables:
        - Turn Angles: [2, 2, 2, n_ships] indexed by [left, right, sharp]
        - Thrust Multipliers: [2, 2, n_ships] indexed by [forward, backward]
        - Energy Costs: [2, 2, n_ships] indexed by [forward, backward]
        - Drag Coefficients: [2, 2, n_ships] indexed by [turning, sharp]
        - Lift Coefficients: [2, n_ships] indexed by [sharp]
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

    # Projectile System State
    projectile_index: torch.Tensor  # Current projectile firing index per ship
    projectile_cooldown: torch.Tensor  # Current firing cooldown timer per ship
    projectiles_position: torch.Tensor  # Complex tensor [n_ships, max_projectiles]
    projectiles_velocity: torch.Tensor  # Complex tensor [n_ships, max_projectiles]
    projectiles_active: torch.Tensor  # Boolean tensor [n_ships, max_projectiles]
    projectiles_lifetime: torch.Tensor  # Remaining lifetime [n_ships, max_projectiles]

    # Thrust System Parameters
    thrust: torch.Tensor  # Base thrust force
    forward_boost: torch.Tensor  # Forward thrust multiplier
    backward_boost: torch.Tensor  # Backward thrust multiplier
    base_energy_cost: torch.Tensor  # Neutral energy consumption
    forward_energy_cost: torch.Tensor  # Forward movement energy cost
    backward_energy_cost: torch.Tensor  # Backward movement energy cost

    # Projectile System Parameters
    max_projectiles: int  # Maximum number of projectiles per ship
    projectile_speed: torch.Tensor  # Base speed of fired projectiles
    projectile_damage: torch.Tensor  # Damage dealt by projectiles
    projectile_lifetime: torch.Tensor  # Maximum lifetime of projectiles
    firing_cooldown: torch.Tensor  # Minimum time between shots
    max_ammo: torch.Tensor  # Maximum ammo capacity
    ammo_count: torch.Tensor  # Current ammo count (can be fractional)
    ammo_regen_rate: torch.Tensor  # Ammo regeneration per second
    projectile_spread: torch.Tensor  # Random angle spread when firing (in radians)

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

    # Lookup Tables (computed in __post_init__)
    turn_angles: torch.Tensor  # [2, 2, 2, n_ships] - [left, right, sharp]
    thrust_multipliers: torch.Tensor  # [2, 2, n_ships] - [forward, backward]
    energy_costs: torch.Tensor  # [2, 2, n_ships] - [forward, backward]
    drag_coefficients: torch.Tensor  # [2, 2, n_ships] - [turning, sharp]
    lift_coefficients: torch.Tensor  # [2, n_ships] - [sharp]

    def __post_init__(self):
        # Initialize state variables that depend on other fields
        self.boost = self.max_boost.clone()
        self.health = self.max_health.clone()
        self.turn_offset = torch.zeros(self.n_ships)

        # Build lookup tables from individual parameters
        self._build_lookup_tables()
        self.ship_arange = torch.arange(self.n_ships)
        self.min_velocity_threshold = 1e-3  # Minimum speed to consider for physics
        self.id = torch.arange(self.n_ships)  # Unique ID for each ship

    def _build_lookup_tables(self):
        """Build efficient lookup tables for physics parameters."""

        # Turn angles lookup table [2, 2, 2, n_ships]
        # Indexed by [left, right, sharp] -> turn angle
        self.turn_angles = torch.zeros(2, 2, 2, self.n_ships)

        # Based on the control system lookup table from requirements:
        # Pattern (L, R, S) -> Turn Offset
        self.turn_angles[0, 0, 0, :] = 0  # 000: None -> 0°
        self.turn_angles[0, 1, 0, :] = self.normal_turn_angle  # 001: R -> +normal_angle
        self.turn_angles[1, 0, 0, :] = (
            -self.normal_turn_angle
        )  # 010: L -> -normal_angle
        self.turn_angles[1, 1, 0, :] = 0  # 011: LR -> maintain (use 0° for lookup)
        self.turn_angles[0, 0, 1, :] = 0  # 100: S -> 0°
        self.turn_angles[0, 1, 1, :] = self.sharp_turn_angle  # 101: SR -> +sharp_angle
        self.turn_angles[1, 0, 1, :] = -self.sharp_turn_angle  # 110: SL -> -sharp_angle
        self.turn_angles[1, 1, 1, :] = 0  # 111: SLR -> maintain (use 0° for lookup)

        # Thrust multipliers lookup table [2, 2, n_ships]
        # Indexed by [forward, backward] -> thrust multiplier
        self.thrust_multipliers = torch.zeros(2, 2, self.n_ships)
        self.thrust_multipliers[0, 0, :] = 1.0  # 00: Neither -> base thrust (1.0x)
        self.thrust_multipliers[0, 1, :] = self.backward_boost  # 01: Backward only
        self.thrust_multipliers[1, 0, :] = self.forward_boost  # 10: Forward only
        self.thrust_multipliers[1, 1, :] = 1.0  # 11: Both -> cancel out to base

        # Energy costs lookup table [2, 2, n_ships]
        # Indexed by [forward, backward] -> energy cost per timestep
        self.energy_costs = torch.zeros(2, 2, self.n_ships)
        self.energy_costs[0, 0, :] = self.base_energy_cost  # 00: Neither
        self.energy_costs[0, 1, :] = self.backward_energy_cost  # 01: Backward
        self.energy_costs[1, 0, :] = self.forward_energy_cost  # 10: Forward
        self.energy_costs[1, 1, :] = self.base_energy_cost  # 11: Both -> cancel to base

        # Drag coefficients lookup table [2, 2, n_ships]
        # Indexed by [turning, sharp] -> drag coefficient
        self.drag_coefficients = torch.zeros(2, 2, self.n_ships)
        self.drag_coefficients[0, 0, :] = self.no_turn_drag  # 00: No turn, normal
        self.drag_coefficients[0, 1, :] = (
            self.no_turn_drag
        )  # 01: No turn, sharp (sharp ignored)
        self.drag_coefficients[1, 0, :] = (
            self.normal_turn_drag_coef
        )  # 10: Turning, normal
        self.drag_coefficients[1, 1, :] = (
            self.sharp_turn_drag_coef
        )  # 11: Turning, sharp

        # Lift coefficients lookup table [2, n_ships]
        # Indexed by [sharp] -> lift coefficient
        self.lift_coefficients = torch.zeros(2, self.n_ships)
        self.lift_coefficients[0, :] = self.normal_turn_lift_coef  # 0: Normal turn
        self.lift_coefficients[1, :] = self.sharp_turn_lift_coef  # 1: Sharp turn

    def get_turn_angle(
        self, left: torch.Tensor, right: torch.Tensor, sharp: torch.Tensor
    ) -> torch.Tensor:
        return self.turn_angles[
            left.long(), right.long(), sharp.long(), self.ship_arange
        ]

    def get_thrust_multiplier(
        self, forward: torch.Tensor, backward: torch.Tensor
    ) -> torch.Tensor:
        return self.thrust_multipliers[
            forward.long(), backward.long(), self.ship_arange
        ]

    def get_energy_cost(
        self, forward: torch.Tensor, backward: torch.Tensor
    ) -> torch.Tensor:
        return self.energy_costs[forward.long(), backward.long(), self.ship_arange]

    def get_drag_coefficient(
        self, turning: torch.Tensor, sharp: torch.Tensor
    ) -> torch.Tensor:
        return self.drag_coefficients[turning.long(), sharp.long(), self.ship_arange]

    def get_lift_coefficient(self, sharp: torch.Tensor) -> torch.Tensor:
        return self.lift_coefficients[sharp.long(), self.ship_arange]

    @classmethod
    def from_scalars(
        cls,
        n_ships: int,
        world_size: tuple = (1200, 800),
        random_positions: bool = True,
        initial_position: complex = 0 + 0j,
        initial_velocity: complex = 100 + 0j,  # Moving right
        initial_attitude: complex = 1 + 0j,  # Facing right
        # Thrust system defaults
        thrust: float = 10.0,
        forward_boost: float = 5.0,
        backward_boost: float = 0.0,
        base_energy_cost: float = -10.0,
        forward_energy_cost: float = 50.0,
        backward_energy_cost: float = -10.0,
        # Projectile system defaults
        max_projectiles: int = 16,
        projectile_speed: float = 500.0,
        projectile_damage: float = 20.0,
        projectile_lifetime: float = 1.0,
        firing_cooldown: float = 0.04,
        max_ammo: float = 32.0,
        ammo_regen_rate: float = 4.0,
        projectile_spread: float = np.deg2rad(3.0),
        # Aerodynamic defaults
        no_turn_drag: float = 8e-4,
        normal_turn_angle: float = np.deg2rad(5.0),
        normal_turn_lift_coef: float = 15e-3,
        normal_turn_drag_coef: float = 1e-3,
        sharp_turn_angle: float = np.deg2rad(15.0),
        sharp_turn_lift_coef: float = 30e-3,
        sharp_turn_drag_coef: float = 3e-3,
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
            # Projectile System State
            projectile_index=torch.zeros(n_ships, dtype=torch.long),
            projectile_cooldown=torch.zeros(n_ships),
            projectiles_position=torch.zeros(
                (n_ships, max_projectiles), dtype=torch.complex64
            ),
            projectiles_velocity=torch.zeros(
                (n_ships, max_projectiles), dtype=torch.complex64
            ),
            projectiles_active=torch.zeros(
                (n_ships, max_projectiles), dtype=torch.bool
            ),
            projectiles_lifetime=torch.zeros((n_ships, max_projectiles)),
            # Projectile System Parameters
            max_projectiles=max_projectiles,
            projectile_speed=make_tensor(projectile_speed),
            projectile_damage=make_tensor(projectile_damage),
            projectile_lifetime=make_tensor(projectile_lifetime),
            firing_cooldown=make_tensor(firing_cooldown),
            max_ammo=make_tensor(max_ammo),
            ammo_count=make_tensor(max_ammo),  # Start with full ammo
            ammo_regen_rate=make_tensor(ammo_regen_rate),
            projectile_spread=make_tensor(projectile_spread),
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
            # Lookup tables (will be computed in __post_init__)
            turn_angles=torch.zeros(1),  # Placeholder, will be overwritten
            thrust_multipliers=torch.zeros(1),
            energy_costs=torch.zeros(1),
            drag_coefficients=torch.zeros(1),
            lift_coefficients=torch.zeros(1),
        )
