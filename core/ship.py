import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional

from utils.config import Actions


@dataclass
class ActionStates:
    """Dataclass to hold extracted action states."""

    left: torch.Tensor
    right: torch.Tensor
    forward_action: torch.Tensor
    backward_action: torch.Tensor
    is_sharp_turn: torch.Tensor


@dataclass
class TurnStates:
    """Dataclass to hold turning-related states."""

    is_turn_drag: torch.Tensor
    is_turning: torch.Tensor
    turn_direction: torch.Tensor


@dataclass
class ForceComponents:
    """Dataclass to hold force components."""

    aero_force: torch.Tensor
    thrust_force: torch.Tensor
    total_acceleration: torch.Tensor


class Ships:
    def __init__(
        self,
        n_ships: int,
        # Ship parameters as tensors (required - each ship can have different values)
        thrust: torch.Tensor,
        forward_boost: torch.Tensor,
        backward_boost: torch.Tensor,
        no_turn_drag: torch.Tensor,
        normal_turn_angle: torch.Tensor,
        normal_turn_lift_coef: torch.Tensor,
        normal_turn_drag_coef: torch.Tensor,
        sharp_turn_angle: torch.Tensor,
        sharp_turn_lift_coef: torch.Tensor,
        sharp_turn_drag_coef: torch.Tensor,
        collision_radius: torch.Tensor,
        max_boost: torch.Tensor,
        max_health: torch.Tensor,
        # Initial state parameters (required)
        initial_positions: torch.Tensor,
        initial_velocity_directions: torch.Tensor,
        initial_speeds: torch.Tensor,
        initial_orientations: torch.Tensor,
    ):
        """
        Initialize ships with all parameters as tensors.

        Args:
            n_ships: Number of ships to create
            thrust: Base thrust force for each ship (tensor)
            forward_boost: Additional thrust when moving forward for each ship (tensor)
            backward_boost: Additional thrust when moving backward for each ship (tensor)
            no_turn_drag: Drag coefficient when not turning for each ship (tensor)
            normal_turn_angle: Turn angle for normal turns (radians) for each ship (tensor)
            normal_turn_lift_coef: Lift coefficient for normal turns for each ship (tensor)
            normal_turn_drag_coef: Drag coefficient for normal turns for each ship (tensor)
            sharp_turn_angle: Turn angle for sharp turns (radians) for each ship (tensor)
            sharp_turn_lift_coef: Lift coefficient for sharp turns for each ship (tensor)
            sharp_turn_drag_coef: Drag coefficient for sharp turns for each ship (tensor)
            collision_radius: Ship collision radius for each ship (tensor)
            max_boost: Maximum boost energy for each ship (tensor)
            max_health: Maximum ship health for each ship (tensor)
            initial_positions: Initial positions (complex tensor)
            initial_velocity_directions: Initial velocity directions (complex tensor)
            initial_speeds: Initial speeds (float tensor)
            initial_orientations: Initial orientations (complex tensor)
        """
        # Static properties
        self.id = torch.arange(0, n_ships, 1)

        # Ship parameters - all required as tensors
        self.thrust = thrust
        self.forward_boost = forward_boost
        self.backward_boost = backward_boost

        self.no_turn_drag = no_turn_drag

        self.normal_turn_angle = normal_turn_angle
        self.normal_turn_lift_coef = normal_turn_lift_coef
        self.normal_turn_drag_coef = normal_turn_drag_coef

        self.sharp_turn_angle = sharp_turn_angle
        self.sharp_turn_lift_coef = sharp_turn_lift_coef
        self.sharp_turn_drag_coef = sharp_turn_drag_coef

        self.collision_radius = collision_radius

        # Dynamic state - all required as tensors
        self.position = initial_positions.clone()
        self.velocity_direction = initial_velocity_directions.clone()
        self.speed = initial_speeds.clone()
        self.orientation = initial_orientations.clone()

        # Resource parameters
        self.boost = max_boost.clone()  # Start with full boost
        self.health = max_health.clone()  # Start with full health

    @classmethod
    def from_scalars(
        cls,
        n_ships: int,
        world_size: tuple,
        # Ship parameters as scalars with defaults (all ships identical)
        thrust: float = 300.0,
        forward_boost: float = 3.0,
        backward_boost: float = 3.0,
        no_turn_drag: float = 0.008,
        normal_turn_angle: float = np.deg2rad(5.0),
        normal_turn_lift_coef: float = 1.0,
        normal_turn_drag_coef: float = 0.01,
        sharp_turn_angle: float = np.deg2rad(15.0),
        sharp_turn_lift_coef: float = 1.5,
        sharp_turn_drag_coef: float = 0.03,
        collision_radius: float = 10.0,
        max_boost: float = 100.0,
        max_health: float = 100.0,
        # Initial state parameters with defaults
        random_positions: bool = True,
        initial_position: complex = 0 + 0j,
        initial_velocity_direction: complex = 1 + 0j,
        initial_speed: float = 0.0,
        initial_orientation: complex = 1 + 0j,
    ) -> "Ships":
        """
        Convenience constructor that creates ships from scalar parameters.
        All ships will have identical parameters.

        Args:
            n_ships: Number of ships to create
            world_size: (width, height) of the world
            thrust: Base thrust force (same for all ships)
            forward_boost: Additional thrust when moving forward (same for all ships)
            backward_boost: Additional thrust when moving backward (same for all ships)
            no_turn_drag: Drag coefficient when not turning (same for all ships)
            normal_turn_angle: Turn angle for normal turns in radians (same for all ships)
            normal_turn_lift_coef: Lift coefficient for normal turns (same for all ships)
            normal_turn_drag_coef: Drag coefficient for normal turns (same for all ships)
            sharp_turn_angle: Turn angle for sharp turns in radians (same for all ships)
            sharp_turn_lift_coef: Lift coefficient for sharp turns (same for all ships)
            sharp_turn_drag_coef: Drag coefficient for sharp turns (same for all ships)
            collision_radius: Ship collision radius (same for all ships)
            max_boost: Maximum boost energy (same for all ships)
            max_health: Maximum ship health (same for all ships)
            random_positions: If True, use random positions; if False, use initial_position
            initial_position: Starting position if not random (same for all ships)
            initial_velocity_direction: Starting velocity direction (same for all ships)
            initial_speed: Starting speed (same for all ships)
            initial_orientation: Starting orientation (same for all ships)

        Returns:
            Ships instance with identical parameters for all ships
        """
        # Create tensors from scalar values
        thrust_tensor = torch.full((n_ships,), thrust)
        forward_boost_tensor = torch.full((n_ships,), forward_boost)
        backward_boost_tensor = torch.full((n_ships,), backward_boost)
        no_turn_drag_tensor = torch.full((n_ships,), no_turn_drag)

        normal_turn_angle_tensor = torch.full((n_ships,), normal_turn_angle)
        normal_turn_lift_coef_tensor = torch.full((n_ships,), normal_turn_lift_coef)
        normal_turn_drag_coef_tensor = torch.full((n_ships,), normal_turn_drag_coef)

        sharp_turn_angle_tensor = torch.full((n_ships,), sharp_turn_angle)
        sharp_turn_lift_coef_tensor = torch.full((n_ships,), sharp_turn_lift_coef)
        sharp_turn_drag_coef_tensor = torch.full((n_ships,), sharp_turn_drag_coef)

        collision_radius_tensor = torch.full((n_ships,), collision_radius)
        max_boost_tensor = torch.full((n_ships,), max_boost)
        max_health_tensor = torch.full((n_ships,), max_health)

        # Create initial state tensors
        if random_positions:
            initial_positions_tensor = torch.complex(
                torch.rand(n_ships) * world_size[0], torch.rand(n_ships) * world_size[1]
            )
        else:
            initial_positions_tensor = torch.full(
                (n_ships,), initial_position, dtype=torch.complex64
            )

        initial_velocity_directions_tensor = torch.full(
            (n_ships,), initial_velocity_direction, dtype=torch.complex64
        )
        initial_speeds_tensor = torch.full(
            (n_ships,), initial_speed, dtype=torch.float32
        )
        initial_orientations_tensor = torch.full(
            (n_ships,), initial_orientation, dtype=torch.complex64
        )

        # Call the main constructor
        return cls(
            n_ships=n_ships,
            world_size=world_size,
            thrust=thrust_tensor,
            forward_boost=forward_boost_tensor,
            backward_boost=backward_boost_tensor,
            no_turn_drag=no_turn_drag_tensor,
            normal_turn_angle=normal_turn_angle_tensor,
            normal_turn_lift_coef=normal_turn_lift_coef_tensor,
            normal_turn_drag_coef=normal_turn_drag_coef_tensor,
            sharp_turn_angle=sharp_turn_angle_tensor,
            sharp_turn_lift_coef=sharp_turn_lift_coef_tensor,
            sharp_turn_drag_coef=sharp_turn_drag_coef_tensor,
            collision_radius=collision_radius_tensor,
            max_boost=max_boost_tensor,
            max_health=max_health_tensor,
            initial_positions=initial_positions_tensor,
            initial_velocity_directions=initial_velocity_directions_tensor,
            initial_speeds=initial_speeds_tensor,
            initial_orientations=initial_orientations_tensor,
        )

    @property
    def velocity(self) -> torch.Tensor:
        """Computed property: velocity = direction * speed"""
        return self.velocity_direction * self.speed.unsqueeze(-1)

    def set_velocity(self, new_velocity: torch.Tensor) -> None:
        """Set velocity by decomposing into direction and speed."""
        self.speed = torch.abs(new_velocity).float()
        self.velocity_direction = torch.where(
            self.speed > 1e-6,
            new_velocity / self.speed.unsqueeze(-1),
            self.velocity_direction,  # Keep current direction if speed is zero
        )


class ShipPhysics(nn.Module):
    def __init__(self, dt=0.016, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dt = dt

    def extract_action_states(self, actions: torch.Tensor) -> ActionStates:
        """Extract boolean action states from action tensor."""
        return ActionStates(
            left=actions[:, Actions.left],
            right=actions[:, Actions.right],
            forward_action=actions[:, Actions.forward],
            backward_action=actions[:, Actions.backward],
            is_sharp_turn=actions[:, Actions.sharp_turn],
        )

    def calculate_turn_states(self, action_states: ActionStates) -> TurnStates:
        """Calculate turning-related boolean states."""
        return TurnStates(
            is_turn_drag=action_states.left | action_states.right,
            is_turning=action_states.left ^ action_states.right,
            turn_direction=action_states.left.int()
            - action_states.right.int(),  # -1 for right, +1 for left
        )

    def calculate_turn_offset(
        self, ships: Ships, action_states: ActionStates, turn_states: TurnStates
    ) -> torch.Tensor:
        """Calculate the current turn angle offset from velocity direction."""
        # Determine which turn angle to use (normal or sharp)
        turn_angle_magnitude = torch.where(
            action_states.is_sharp_turn, ships.sharp_turn_angle, ships.normal_turn_angle
        )

        # Calculate current turn offset
        # - If turning left/right: apply offset in that direction
        # - If both pressed or neither pressed: maintain current offset (no change)
        current_offset = torch.angle(ships.orientation / ships.velocity_direction)

        new_offset = torch.where(
            turn_states.is_turning,
            turn_states.turn_direction.float() * turn_angle_magnitude,
            current_offset,  # Maintain current offset when not actively turning
        )

        return new_offset

    def update_orientation(
        self, ships: Ships, action_states: ActionStates, turn_states: TurnStates
    ) -> None:
        """Update ship orientation as velocity direction + turn offset."""
        turn_offset = self.calculate_turn_offset(ships, action_states, turn_states)
        ships.orientation = ships.velocity_direction * torch.exp(1j * turn_offset)

    def calculate_drag_coefficient(
        self, ships: Ships, action_states: ActionStates, turn_states: TurnStates
    ) -> torch.Tensor:
        """Calculate drag coefficient based on turning state."""
        return torch.where(
            turn_states.is_turn_drag & action_states.is_sharp_turn,
            ships.sharp_turn_drag_coef,
            torch.where(
                turn_states.is_turn_drag,
                ships.normal_turn_drag_coef,
                ships.no_turn_drag,
            ),
        )

    def calculate_lift_coefficient(
        self, ships: Ships, action_states: ActionStates, turn_states: TurnStates
    ) -> torch.Tensor:
        """Calculate lift coefficient for turning."""
        lift_coef = torch.where(
            action_states.is_sharp_turn,
            ships.sharp_turn_lift_coef,
            ships.normal_turn_lift_coef,
        )

        return torch.where(
            turn_states.is_turning,
            turn_states.turn_direction.float() * lift_coef,
            torch.zeros_like(lift_coef),
        )

    def calculate_aero_forces(
        self, ships: Ships, drag_coef: torch.Tensor, lift_coef: torch.Tensor
    ) -> torch.Tensor:
        """Calculate aerodynamic forces (drag and lift)."""
        velocity = ships.velocity
        speed = ships.speed
        return (-drag_coef + 1j * lift_coef) * velocity * speed.unsqueeze(-1)

    def calculate_thrust_forces(
        self, ships: Ships, action_states: ActionStates
    ) -> torch.Tensor:
        """Calculate thrust forces based on forward/backward actions."""
        boost_magnitude = torch.where(
            action_states.forward_action,
            ships.forward_boost,
            torch.where(
                action_states.backward_action,
                ships.backward_boost,
                torch.zeros_like(ships.forward_boost),
            ),
        )

        thrust_magnitude = ships.thrust + boost_magnitude
        return thrust_magnitude.unsqueeze(-1) * ships.orientation

    def integrate_motion(self, ships: Ships, acceleration: torch.Tensor) -> None:
        """Perform Euler integration to update velocity and position."""
        new_velocity = ships.velocity + acceleration * self.dt
        ships.set_velocity(new_velocity)
        ships.position += ships.velocity * self.dt

    def calculate_forces(
        self, ships: Ships, action_states: ActionStates, turn_states: TurnStates
    ) -> ForceComponents:
        """Calculate all force components and total acceleration."""
        drag_coef = self.calculate_drag_coefficient(ships, action_states, turn_states)
        lift_coef = self.calculate_lift_coefficient(ships, action_states, turn_states)

        aero_force = self.calculate_aero_forces(ships, drag_coef, lift_coef)
        thrust_force = self.calculate_thrust_forces(ships, action_states)

        # Total acceleration (assuming mass = 1)
        total_acceleration = aero_force + thrust_force

        return ForceComponents(
            aero_force=aero_force,
            thrust_force=thrust_force,
            total_acceleration=total_acceleration,
        )

    def forward(self, ships: Ships, actions: torch.Tensor) -> Ships:
        """Main forward pass - orchestrates all physics calculations."""
        action_states = self.extract_action_states(actions)
        turn_states = self.calculate_turn_states(action_states)

        self.update_orientation(ships, action_states, turn_states)

        forces = self.calculate_forces(ships, action_states, turn_states)

        self.integrate_motion(ships, forces.total_acceleration)

        return ships
