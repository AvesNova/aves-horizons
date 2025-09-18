from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

from globals import Actions


@dataclass
class ShipConfig:
    # Physical Parameters
    collision_radius: float
    max_health: float
    max_power: float

    # Thrust System Parameters
    base_thrust: float
    boost_thrust: float
    reverse_thrust: float
    base_energy_cost: float
    boost_energy_cost: float
    reverse_energy_cost: float

    # Aerodynamic Parameters
    no_turn_drag_coeff: float
    normal_turn_angle: float
    normal_turn_drag_coeff: float
    normal_turn_lift_coeff: float
    sharp_turn_angle: float
    sharp_turn_drag_coeff: float
    sharp_turn_lift_coeff: float


@dataclass
class ActionStates:
    forward: bool
    backward: bool
    left: bool
    right: bool
    sharp_turn: bool


class Ship(nn.Module):
    def __init__(
        self,
        ship_id: int,
        team_id: int,
        ship_config: ShipConfig,
        initial_x: float,
        initial_y: float,
        initial_vx: float,
        initial_vy: float,
    ):
        self.ship_id = ship_id
        self.team_id = team_id
        self.config = ship_config

        self.alive = True
        self.health = ship_config.max_health
        self.power = ship_config.max_power
        self.turn_offset = 0.0

        self.position = initial_x + 1j * initial_y
        self.velocity = initial_vx + 1j * initial_vy
        self.speed = np.linalg.norm(self.velocity)
        assert self.speed > 0, "Initial velocity cannot be zero vector"
        self.attitude = self.velocity / self.speed

        self._build_lookup_tables(ship_config)

    def _build_lookup_tables(self, ship_config: ShipConfig) -> None:
        # Indexed by [left, right, sharp] -> turn offset
        self.turn_offset_table = np.zeros(2, 2, 2, dtype=np.float32)

        self.turn_offset_table[0, 0, 0] = 0  # None
        self.turn_offset_table[0, 1, 0] = ship_config.normal_turn_angle  # R
        self.turn_offset_table[1, 0, 0] = -ship_config.normal_turn_angle  # L
        self.turn_offset_table[1, 1, 0] = 0  # LR
        self.turn_offset_table[0, 0, 1] = 0  # S
        self.turn_offset_table[0, 1, 1] = ship_config.sharp_turn_angle  # SR
        self.turn_offset_table[1, 0, 1] = -ship_config.sharp_turn_angle  # SL
        self.turn_offset_table[1, 1, 1] = 0  # SLR

        # Indexed by [forward, backward] -> thrust
        self.thrust_table = np.zeros(2, 2, dtype=np.float32)
        self.thrust_table[0, 0] = ship_config.base_thrust  # Neither
        self.thrust_table[0, 1] = ship_config.boost_thrust  # Boost only
        self.thrust_table[1, 0] = ship_config.reverse_thrust  # Forward only
        self.thrust_table[1, 1] = 1.0  # Both -> cancel out to base

        # Indexed by [forward, backward] -> energy cost
        self.energy_cost_table = np.zeros(2, 2, dtype=np.float32)
        self.energy_cost_table[0, 0] = ship_config.base_energy_cost  # Base
        self.energy_cost_table[0, 1] = ship_config.reverse_energy_cost  # Backward
        self.energy_cost_table[1, 0] = ship_config.boost_energy_cost  # Forward
        self.energy_cost_table[1, 1] = ship_config.base_energy_cost  # Both -> base

        # Indexed by [turning, sharp] -> drag coefficient
        self.drag_coeff_table = np.zeros(2, 2, dtype=np.float32)
        self.drag_coeff_table[0, 0] = ship_config.no_turn_drag_coeff  # No turn
        self.drag_coeff_table[0, 1] = ship_config.no_turn_drag_coeff  # No turn
        self.drag_coeff_table[1, 0] = ship_config.normal_turn_drag_coeff  # Normal turn
        self.drag_coeff_table[1, 1] = ship_config.sharp_turn_drag_coeff  # Sharp turn

        # Indexed by [sharp] -> lift coefficient
        self.lift_coeff_table = np.zeros(2, dtype=np.float32)
        self.lift_coeff_table[0] = ship_config.normal_turn_lift_coeff  # 0: Normal turn
        self.lift_coeff_table[1] = ship_config.sharp_turn_lift_coeff  # 1: Sharp turn

    def extract_action_states(self, actions: torch.Tensor) -> ActionStates:
        return ActionStates(
            left=actions[:, Actions.left].bool(),
            right=actions[:, Actions.right].bool(),
            forward=actions[:, Actions.forward].bool(),
            backward=actions[:, Actions.backward].bool(),
            sharp_turn=actions[:, Actions.sharp_turn].bool(),
        )

    def update_power(self, actions: ActionStates, delta_t: float) -> None:
        energy_cost = self.energy_cost_table[actions.forward, actions.backward]
        self.power -= energy_cost * delta_t
        self.power = max(0.0, min(self.power, self.config.max_power))

    def update_attitude(self, actions: ActionStates) -> None:
        if not (actions.left and actions.right):
            self.turn_offset = self.turn_offset_table[
                actions.left, actions.right, actions.sharp_turn
            ]
        self.attitude = self.velocity / self.speed * np.exp(1j * self.turn_offset)

    def calculate_forces(self, actions: ActionStates) -> complex:
        thrust = self.thrust_table[actions.forward, actions.backward]
        thrust_force = thrust * self.attitude

        turning = actions.left or actions.right
        drag_coeff = self.drag_coeff_table[turning, actions.sharp_turn]
        drag_force = -drag_coeff * self.speed * self.velocity

        lift_coeff = self.lift_coeff_table[actions.sharp_turn]
        lift_vector = self.velocity * 1j  # 90 degrees counter-clockwise
        lift_force = lift_coeff * self.speed * lift_vector

        total_force = thrust_force + drag_force + lift_force
        return total_force

    def update_kinematics(self, actions: ActionStates, delta_t: float) -> None:
        total_force = self.calculate_forces(actions)
        acceleration = total_force  # Assuming mass = 1

        self.velocity += acceleration * delta_t
        self.position += self.velocity * delta_t
        self.speed = np.linalg.norm(self.velocity)

    def forward(self, action_vector: torch.Tensor, delta_t: float) -> None:
        if self.health <= 0:
            self.alive = False
        if not self.alive:
            return

        actions = self.extract_action_states(action_vector)

        self.update_power(actions, delta_t)
        self.update_attitude(actions)
        self.update_kinematics(actions, delta_t)
