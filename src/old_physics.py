# Old physics implementation using torchdiffeq for adaptive ODE integration.

from dataclasses import dataclass
import torch
import torch.nn as nn
from torchdiffeq import odeint

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
    team_id: torch.Tensor  # Team ID for each ship (integer tensor)
    active: torch.Tensor  # Boolean tensor indicating which ships are alive/active

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
        # Initialize active mask - all ships start alive
        if not hasattr(self, "active") or self.active is None:
            self.active = torch.ones(self.n_ships, dtype=torch.bool)

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

    def update_active_status(self):
        """Update active mask based on ship health. Ships with health <= 0 become inactive."""
        self.active = self.health > 0

    def get_active_mask(self) -> torch.Tensor:
        """Get boolean mask of active (alive) ships."""
        return self.active

    @classmethod
    def from_scalars(
        cls,
        n_ships: int,
        world_size: tuple = (1200, 800),
        random_positions: bool = True,
        initial_position: complex = 0 + 0j,
        initial_velocity: complex = 100 + 0j,  # Moving right
        initial_attitude: complex = 1 + 0j,  # Facing right
        team_ids: list = None,  # List of team IDs for each ship
        # Thrust system defaults
        thrust: float = 10.0,
        forward_boost: float = 8.0,
        backward_boost: float = 0.0,
        base_energy_cost: float = -10.0,
        forward_energy_cost: float = 40.0,
        backward_energy_cost: float = -20.0,
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

        # Team IDs
        if team_ids is None:
            team_id_tensor = torch.zeros(n_ships, dtype=torch.long)
        else:
            assert len(team_ids) == n_ships, "team_ids length must match n_ships"
            team_id_tensor = torch.tensor(team_ids, dtype=torch.long)

        return cls(
            n_ships=n_ships,
            # State variables (will be initialized in __post_init__)
            position=position_tensor,
            velocity=velocity_tensor,
            attitude=attitude_tensor,
            turn_offset=torch.zeros(n_ships),
            boost=torch.zeros(n_ships),
            health=torch.zeros(n_ships),
            team_id=team_id_tensor,
            active=torch.ones(n_ships, dtype=torch.bool),
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


@dataclass
class ActionStates:
    """Dataclass to hold extracted action states."""

    left: torch.Tensor
    right: torch.Tensor
    forward: torch.Tensor
    backward: torch.Tensor
    sharp_turn: torch.Tensor


@dataclass
class TurnStates:
    """Dataclass to hold turning-related states."""

    is_turning: torch.Tensor
    turn_direction: torch.Tensor
    is_sharp_turn: torch.Tensor
    turn_angle: torch.Tensor


@dataclass
class ForceComponents:
    """Dataclass to hold force components."""

    aero_force: torch.Tensor
    thrust_force: torch.Tensor
    total_acceleration: torch.Tensor


class ShipPhysics(nn.Module):
    """Physics-based ship simulation with adaptive ODE integration.

    Implements the complete physics system from the requirements including:
    - Turn offset lookup table with bit patterns
    - Energy consumption and regeneration
    - Velocity-based attitude calculation
    - Adaptive ODE integration via torchdiffeq
    - Proper aerodynamic forces
    """

    def __init__(
        self,
        target_timestep: float = 0.02,  # 50 FPS
        solver: str = "dopri5",
        rtol: float = 1e-7,
        atol: float = 1e-9,
        max_steps: int = 1000,
        min_velocity_threshold: float = 1e-6,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.target_timestep = target_timestep
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps
        self.min_velocity_threshold = min_velocity_threshold

    def extract_action_states(self, actions: torch.Tensor) -> ActionStates:
        return ActionStates(
            left=actions[:, Actions.left].bool(),
            right=actions[:, Actions.right].bool(),
            forward=actions[:, Actions.forward].bool(),
            backward=actions[:, Actions.backward].bool(),
            sharp_turn=actions[:, Actions.sharp_turn].bool(),
        )

    def update_turn_offset(self, ships: Ships, action_states: ActionStates) -> None:
        """Update turn offset using lookup table based on button combinations."""
        # Get turn angle from ships' lookup table
        # This handles all bit patterns including maintain-current cases
        turn_angles = ships.get_turn_angle(
            action_states.left, action_states.right, action_states.sharp_turn
        )

        # Handle special cases where both L and R are pressed (maintain current offset)
        both_lr_pressed = action_states.left & action_states.right

        # Update turn offset, maintaining current when both L and R are pressed
        ships.turn_offset = torch.where(
            both_lr_pressed,
            ships.turn_offset,  # Keep current
            turn_angles,  # Update to new angle from lookup
        )

    def update_attitude(self, ships: Ships) -> None:
        """Update ship attitude as velocity direction + turn offset.

        Attitude = velocity_direction * exp(i * turn_offset)
        Only update if velocity is above minimum threshold.
        """
        speed = torch.abs(ships.velocity)

        # Calculate velocity direction (normalized velocity)
        velocity_direction = ships.velocity / speed

        # Apply turn offset to velocity direction
        ships.attitude = velocity_direction * torch.exp(1j * ships.turn_offset)

    def update_energy(self, ships: Ships, action_states: ActionStates) -> None:
        """Update boost energy based on action states and energy costs."""
        # Get energy cost from ships' lookup table
        energy_cost = ships.get_energy_cost(
            action_states.forward, action_states.backward
        )

        # Update boost energy (subtract cost, negative cost = regeneration)
        ships.boost = ships.boost - energy_cost * self.target_timestep

        # Clamp energy between 0 and max capacity
        ships.boost = torch.clamp(ships.boost, torch.tensor(0.0), ships.max_boost)

    def calculate_aerodynamic_coefficients(
        self, ships: Ships, action_states: ActionStates
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate drag and lift coefficients based on turning state."""

        # Get drag coefficient from ships' lookup table
        drag_coef = ships.get_drag_coefficient(
            action_states.left | action_states.right, action_states.sharp_turn
        )

        # Get lift coefficient from ships' lookup table
        base_lift_coef = ships.get_lift_coefficient(action_states.sharp_turn)

        # Apply turn direction to lift coefficient (right = +1, left = -1)
        turn_direction = action_states.right.float() - action_states.left.float()

        lift_coef = torch.where(
            action_states.left ^ action_states.right,
            turn_direction * base_lift_coef,
            0.0,
        )

        return drag_coef, lift_coef

    def calculate_forces(
        self, ships: Ships, action_states: ActionStates
    ) -> ForceComponents:
        """Calculate all force components."""
        # Thrust forces
        thrust_multiplier = ships.get_thrust_multiplier(
            action_states.forward, action_states.backward
        )
        thrust_multiplier = torch.where(ships.boost > 0, thrust_multiplier, 1.0)
        thrust_force = ships.thrust * thrust_multiplier * ships.attitude

        # Aerodynamic forces
        drag_coef, lift_coef = self.calculate_aerodynamic_coefficients(
            ships, action_states
        )
        speed = torch.abs(ships.velocity)

        # Drag opposes velocity, lift is perpendicular (90° rotation = multiply by i)
        aero_force = (
            -drag_coef * ships.velocity * speed
            + 1j * lift_coef * ships.velocity * speed
        )

        # Total acceleration (assuming unit mass)
        total_acceleration = thrust_force + aero_force

        return ForceComponents(
            aero_force=aero_force,
            thrust_force=thrust_force,
            total_acceleration=total_acceleration,
        )

    def dynamics_function(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """ODE dynamics function for torchdiffeq integration.

        State vector: [position_real, position_imag, velocity_real, velocity_imag]
        Returns: [velocity_real, velocity_imag, acceleration_real, acceleration_imag]
        """
        # Extract state components
        pos_real, pos_imag, vel_real, vel_imag = state.chunk(4, dim=0)

        # Reconstruct complex tensors
        position = torch.complex(pos_real, pos_imag)
        velocity = torch.complex(vel_real, vel_imag)

        # Update ships state (temporary for force calculation)
        self.temp_ships.position = position
        self.temp_ships.velocity = velocity

        # Calculate forces using current actions
        forces = self.calculate_forces(self.temp_ships, self.current_actions)

        # Return derivatives: [velocity, acceleration]
        return torch.cat(
            [
                vel_real,
                vel_imag,  # d/dt position = velocity
                forces.total_acceleration.real,  # d/dt velocity = acceleration
                forces.total_acceleration.imag,
            ]
        )

    def integrate_step(self, ships: Ships, actions: torch.Tensor) -> None:
        """Perform one integration step using torchdiffeq."""
        # Store current actions and ships for dynamics function
        self.current_actions = self.extract_action_states(actions)
        self.temp_ships = ships  # Temporary reference for dynamics function

        # Prepare state vector for ODE integration
        initial_state = torch.cat(
            [
                ships.position.real,
                ships.position.imag,
                ships.velocity.real,
                ships.velocity.imag,
            ]
        )

        # Time span for integration
        t_span = torch.tensor([0.0, self.target_timestep])

        # Perform adaptive ODE integration
        solution = odeint(
            self.dynamics_function,
            initial_state,
            t_span,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
            options={"max_num_steps": self.max_steps},
        )

        # Extract final state
        final_state = solution[-1]  # Last time step

        pos_real, pos_imag, vel_real, vel_imag = final_state.chunk(4, dim=0)

        # Update ships with integrated results
        ships.position = torch.complex(pos_real, pos_imag)
        ships.velocity = torch.complex(vel_real, vel_imag)

    def forward(self, ships: Ships, actions: torch.Tensor) -> Ships:
        """Main forward pass - orchestrates all physics calculations.

        Args:
            ships: Ships dataclass containing current state
            actions: MultiBinary(5) tensor with shape (n_ships, 5)

        Returns:
            Updated ships with new state after physics simulation
        """
        # Update active status based on health
        ships.update_active_status()

        # Only process active ships
        active_mask = ships.get_active_mask()

        # If no ships are active, return early
        if not torch.any(active_mask):
            return ships

        action_states = self.extract_action_states(actions)

        # Update turn offset based on action combinations (only for active ships)
        self.update_turn_offset_masked(ships, action_states, active_mask)

        # Update attitude based on velocity direction + turn offset (only for active ships)
        self.update_attitude_masked(ships, active_mask)

        # Update energy based on actions (only for active ships)
        self.update_energy_masked(ships, action_states, active_mask)

        # Perform physics integration (only for active ships)
        self.integrate_step_masked(ships, actions, active_mask)

        # Update attitude again after integration (velocity may have changed)
        self.update_attitude_masked(ships, active_mask)

        return ships

    def update_turn_offset_masked(
        self, ships: Ships, action_states: ActionStates, active_mask: torch.Tensor
    ) -> None:
        """Update turn offset for active ships only."""
        if not torch.any(active_mask):
            return

        # Get turn angle from ships' lookup table
        turn_angles = ships.get_turn_angle(
            action_states.left, action_states.right, action_states.sharp_turn
        )

        # Handle special cases where both L and R are pressed (maintain current offset)
        both_lr_pressed = action_states.left & action_states.right

        # Update turn offset only for active ships
        new_turn_offset = torch.where(
            both_lr_pressed,
            ships.turn_offset,  # Keep current
            turn_angles,  # Update to new angle from lookup
        )

        ships.turn_offset = torch.where(
            active_mask,
            new_turn_offset,
            ships.turn_offset,  # Keep unchanged for inactive ships
        )

    def update_attitude_masked(self, ships: Ships, active_mask: torch.Tensor) -> None:
        """Update attitude for active ships only."""
        if not torch.any(active_mask):
            return

        speed = torch.abs(ships.velocity)

        # Only update attitude for active ships with sufficient velocity
        update_mask = active_mask & (speed > self.min_velocity_threshold)

        if torch.any(update_mask):
            # Calculate velocity direction (normalized velocity)
            velocity_direction = ships.velocity / speed
            # Apply turn offset to velocity direction
            new_attitude = velocity_direction * torch.exp(1j * ships.turn_offset)

            ships.attitude = torch.where(
                update_mask,
                new_attitude,
                ships.attitude,  # Keep unchanged for inactive ships
            )

    def update_energy_masked(
        self, ships: Ships, action_states: ActionStates, active_mask: torch.Tensor
    ) -> None:
        """Update energy for active ships only."""
        if not torch.any(active_mask):
            return

        # Get energy cost from ships' lookup table
        energy_cost = ships.get_energy_cost(
            action_states.forward, action_states.backward
        )

        # Calculate new boost energy
        new_boost = ships.boost - energy_cost * self.target_timestep
        new_boost = torch.clamp(new_boost, torch.tensor(0.0), ships.max_boost)

        # Update energy only for active ships
        ships.boost = torch.where(
            active_mask, new_boost, ships.boost  # Keep unchanged for inactive ships
        )

    def integrate_step_masked(
        self, ships: Ships, actions: torch.Tensor, active_mask: torch.Tensor
    ) -> None:
        """Perform integration only for active ships."""
        if not torch.any(active_mask):
            return

        # For simplicity, we'll still integrate all ships but only apply results to active ones
        # This avoids complex indexing in the ODE solver

        # Store original state for inactive ships
        original_position = ships.position.clone()
        original_velocity = ships.velocity.clone()

        # Perform full integration (including inactive ships)
        self.integrate_step(ships, actions)

        # Restore original state for inactive ships
        ships.position = torch.where(
            active_mask,
            ships.position,  # Use new integrated position
            original_position,  # Restore original position
        )

        ships.velocity = torch.where(
            active_mask,
            ships.velocity,  # Use new integrated velocity
            original_velocity,  # Restore original velocity
        )
