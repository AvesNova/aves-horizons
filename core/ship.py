import torch
import torch.nn as nn
import numpy as np

from utils.config import Actions


class Ships:
    def __init__(self, n_ships: int, world_size: tuple):
        # Static
        self.id = torch.arange(0, n_ships, 1)

        self.thrust = torch.full((n_ships,), 1.0)
        self.forward_boost = torch.full((n_ships,), 1.0)
        self.backward_boost = torch.full((n_ships,), 1.0)

        self.no_turn_drag = torch.full((n_ships,), 0.008)

        self.normal_turn_angle = torch.full((n_ships,), np.deg2rad(5.0))
        self.normal_turn_lift_coef = torch.full((n_ships,), 1.0)
        self.normal_turn_drag_coef = torch.full((n_ships,), 0.01)

        self.sharp_turn_angle = torch.full((n_ships,), np.deg2rad(15.0))
        self.sharp_turn_lift_coef = torch.full((n_ships,), 1.5)
        self.sharp_turn_drag_coef = torch.full((n_ships,), 0.03)

        self.collision_radius = torch.full((n_ships,), 10.0)

        # Dynamic
        self.position = torch.complex(
            torch.rand(n_ships) * world_size[0], torch.rand(n_ships) * world_size[1]
        )
        self.velocity = torch.zeros(n_ships, dtype=torch.complex64)
        self.orientation = torch.zeros(n_ships, dtype=torch.complex64)

        self.boost = torch.full((n_ships,), 100.0)
        self.health = torch.full((n_ships,), 100.0)


class ShipPhysics(nn.Module):
    def __init__(self, dt=0.016, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dt = dt

    def forward(self, ships: Ships, actions: torch.Tensor):
        left = actions[:, Actions.left]
        right = actions[:, Actions.right]

        is_turn_drag = left | right
        is_turning = left ^ right
        turn_direction = left.int() - right.int()

        is_sharp_turn = actions[:, Actions.sharp_turn]
        speed = torch.abs(ships.velocity)

        # Aero forces
        forward = (
            ~is_turn_drag * ships.no_turn_drag + is_turn_drag
            & ~is_sharp_turn * ships.normal_turn_drag_coef + is_turn_drag
            & is_sharp_turn * ships.sharp_turn_drag_coef
        )

        perpendicular = (is_turning.int() * turn_direction) * (
            ~is_sharp_turn * ships.normal_turn_lift_coef + is_turn_drag
            & is_sharp_turn * ships.sharp_turn_lift_coef
        )

        aero_force = (forward + 1j * perpendicular) * ships.velocity * speed

        # Thrust forces
        thrust_magnitude = ships.thrust + (
            ships.forward_boost * actions[:, Actions.forward]
            + ships.backward_boost * actions[:, Actions.backward]
        )

        turn_angle = (
            ships.normal_turn_angle * turn_direction * ~is_sharp_turn
            + ships.sharp_turn_angle * turn_direction * is_sharp_turn
        )

        velocity_direction = torch.where(
            speed > 1e-6,
            ships.velocity / speed,
            torch.ones_like(ships.velocity),  # Default direction when stationary
        )

        ships.orientation[is_turning] = velocity_direction[is_turning] * torch.exp(
            1j * turn_angle[is_turning]
        )

        thrust_force = thrust_magnitude * ships.orientation

        acceleration = aero_force + thrust_force  # m = 1 for now

        # Euler integration
        ships.velocity += acceleration * self.dt
        ships.position += ships.velocity * self.dt

        return ships
