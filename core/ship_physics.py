from dataclasses import dataclass
import torch
import torch.nn as nn
from torchdiffeq import odeint

from core.ship import Ships


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
        target_timestep: float = 0.016,
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

        # Turn offset lookup table based on bit patterns (SLR)
        # Bit pattern: [Sharp, Left, Right] -> turn offset multiplier
        self.turn_lookup = torch.tensor(
            [
                0.0,  # 000: None -> 0°
                -1.0,  # 001: R -> -normal_angle
                1.0,  # 010: L -> +normal_angle
                0.0,  # 011: LR -> Previous (no change, handled separately)
                0.0,  # 100: S -> 0°
                -1.0,  # 101: SR -> -sharp_angle
                1.0,  # 110: SL -> +sharp_angle
                0.0,  # 111: SLR -> Previous (no change, handled separately)
            ]
        )

    def extract_action_states(self, actions: torch.Tensor) -> ActionStates:
        """Extract boolean action states from MultiBinary(5) action tensor.

        Action indices:
        0 - Left (L)
        1 - Right (R)
        2 - Forward (F)
        3 - Backward (B)
        4 - Sharp Turn (S)
        """
        return ActionStates(
            left=actions[:, 0].bool(),
            right=actions[:, 1].bool(),
            forward=actions[:, 2].bool(),
            backward=actions[:, 3].bool(),
            sharp_turn=actions[:, 4].bool(),
        )

    def update_turn_offset(self, ships: Ships, action_states: ActionStates) -> None:
        """Update turn offset using lookup table based on button combinations."""
        # Create bit pattern: [Sharp, Left, Right]
        bit_pattern = (
            action_states.sharp_turn.int() * 4
            + action_states.left.int() * 2
            + action_states.right.int()
        )

        # Get turn multipliers from lookup table
        turn_multipliers = self.turn_lookup[bit_pattern]

        # Handle special cases where both L and R are pressed (maintain current offset)
        both_lr_pressed = action_states.left & action_states.right

        # Select appropriate turn angle based on sharp turn modifier
        base_angle = torch.where(
            action_states.sharp_turn, ships.sharp_turn_angle, ships.normal_turn_angle
        )

        # Calculate new turn offset
        new_turn_offset = turn_multipliers * base_angle

        # Maintain current offset when both L and R are pressed
        ships.turn_offset = torch.where(
            both_lr_pressed,
            ships.turn_offset,  # Keep current
            new_turn_offset,  # Update to new
        )

    def update_attitude(self, ships: Ships) -> None:
        """Update ship attitude as velocity direction + turn offset.

        Attitude = velocity_direction * exp(i * turn_offset)
        Only update if velocity is above minimum threshold.
        """
        speed = torch.abs(ships.velocity)
        above_threshold = speed > self.min_velocity_threshold

        # Calculate velocity direction (normalized velocity)
        velocity_direction = torch.where(
            above_threshold.unsqueeze(-1),
            ships.velocity / speed.unsqueeze(-1),
            ships.attitude,  # Keep current attitude if too slow
        )

        # Apply turn offset to velocity direction
        new_attitude = velocity_direction * torch.exp(1j * ships.turn_offset)

        # Only update attitude if above speed threshold
        ships.attitude = torch.where(
            above_threshold.unsqueeze(-1), new_attitude, ships.attitude
        )

    def update_energy(self, ships: Ships, action_states: ActionStates) -> None:
        """Update boost energy based on action states and energy costs."""
        # Determine energy cost based on actions
        # If both forward and backward pressed, treat as neither
        both_fb_pressed = action_states.forward & action_states.backward

        energy_cost = torch.where(
            both_fb_pressed,
            ships.base_energy_cost,  # Both pressed = neutral
            torch.where(
                action_states.forward,
                ships.forward_energy_cost,  # Forward cost
                torch.where(
                    action_states.backward,
                    ships.backward_energy_cost,  # Backward cost (can be negative for regen)
                    ships.base_energy_cost,  # Neither pressed = base cost
                ),
            ),
        )

        # Update boost energy (subtract cost, negative cost = regeneration)
        ships.boost = ships.boost - energy_cost * self.target_timestep

        # Clamp energy between 0 and max capacity
        ships.boost = torch.clamp(ships.boost, 0.0, ships.max_boost)

    def calculate_thrust_multiplier(
        self, ships: Ships, action_states: ActionStates
    ) -> torch.Tensor:
        """Calculate thrust multiplier based on forward/backward actions."""
        # If both forward and backward pressed, treat as neither (multiplier = 1.0)
        both_fb_pressed = action_states.forward & action_states.backward

        multiplier = torch.where(
            both_fb_pressed,
            torch.ones_like(ships.forward_boost),  # Both = no boost
            torch.where(
                action_states.forward,
                ships.forward_boost,  # Forward boost
                torch.where(
                    action_states.backward,
                    ships.backward_boost,  # Backward boost
                    torch.ones_like(ships.forward_boost),  # Neither = no boost
                ),
            ),
        )

        return multiplier

    def calculate_aerodynamic_coefficients(
        self, ships: Ships, action_states: ActionStates
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate drag and lift coefficients based on turning state."""
        # Determine if turning (L XOR R, not both)
        is_turning = action_states.left ^ action_states.right

        # Select coefficients based on sharp turn and turning state
        drag_coef = torch.where(
            is_turning & action_states.sharp_turn,
            ships.sharp_turn_drag_coef,
            torch.where(is_turning, ships.normal_turn_drag_coef, ships.no_turn_drag),
        )

        # Lift coefficient (with direction: left = +1, right = -1)
        turn_direction = action_states.left.float() - action_states.right.float()
        base_lift_coef = torch.where(
            action_states.sharp_turn,
            ships.sharp_turn_lift_coef,
            ships.normal_turn_lift_coef,
        )

        lift_coef = torch.where(
            is_turning,
            turn_direction * base_lift_coef,
            torch.zeros_like(base_lift_coef),
        )

        return drag_coef, lift_coef

    def calculate_forces(
        self, ships: Ships, action_states: ActionStates
    ) -> ForceComponents:
        """Calculate all force components."""
        # Thrust forces
        thrust_multiplier = self.calculate_thrust_multiplier(ships, action_states)
        thrust_force = ships.thrust * thrust_multiplier.unsqueeze(-1) * ships.attitude

        # Aerodynamic forces
        drag_coef, lift_coef = self.calculate_aerodynamic_coefficients(
            ships, action_states
        )
        speed = torch.abs(ships.velocity)

        # Drag opposes velocity, lift is perpendicular (90° rotation = multiply by i)
        aero_force = -drag_coef.unsqueeze(-1) * ships.velocity * speed.unsqueeze(
            -1
        ) + 1j * lift_coef.unsqueeze(-1) * ships.velocity * speed.unsqueeze(-1)

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
        n_ships = state.shape[0] // 4

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
        n_ships = final_state.shape[0] // 4

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
