from dataclasses import dataclass
import torch.nn as nn
import torch

from core.ship import Ships
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

    def update_orientation(
        self, ships: Ships, action_states: ActionStates, turn_states: TurnStates
    ) -> None:
        """Update ship orientation as velocity direction + turn offset."""
        # Determine which turn angle to use (normal or sharp)
        turn_angle_magnitude = torch.where(
            action_states.is_sharp_turn, ships.sharp_turn_angle, ships.normal_turn_angle
        )
        ships.attitude = torch.where(
            turn_states.is_turning,
            ships.velocity * torch.exp(1j * turn_angle_magnitude),
            ships.attitude,
        )

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
        return thrust_magnitude.unsqueeze(-1) * ships.attitude

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
