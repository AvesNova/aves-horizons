from dataclasses import dataclass
import torch
import torch.nn as nn
from torchdiffeq import odeint

from core.ships import Ships
from utils.config import Actions


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
        action_states = self.extract_action_states(actions)

        # Update turn offset based on action combinations
        self.update_turn_offset(ships, action_states)

        # Update attitude based on velocity direction + turn offset
        self.update_attitude(ships)

        # Update energy based on actions
        self.update_energy(ships, action_states)

        # Perform physics integration
        self.integrate_step(ships, actions)

        # Update attitude again after integration (velocity may have changed)
        self.update_attitude(ships)

        return ships
