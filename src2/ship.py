from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

from globals import Actions
from src2.bullets import Bullets


@dataclass
class ShipConfig:
    # Physical Parameters
    collision_radius: float = 10.0
    max_health: float = 100.0
    max_power: float = 100.0

    # Thrust System Parameters
    base_thrust: float = 10.0
    boost_thrust: float = 80.0
    reverse_thrust: float = -10.0
    base_energy_cost: float = -10.0
    boost_energy_cost: float = 40.0
    reverse_energy_cost: float = -20.0

    # Aerodynamic Parameters
    no_turn_drag_coeff: float = 8e-4
    normal_turn_angle: float = np.deg2rad(5.0)
    normal_turn_drag_coeff: float = 1e-3
    normal_turn_lift_coeff: float = 15e-3
    sharp_turn_angle: float = np.deg2rad(15.0)
    sharp_turn_drag_coeff: float = 3e-3
    sharp_turn_lift_coeff: float = 30e-3

    # Bullet System Parameters
    bullet_speed: float = 500.0
    bullet_energy_cost: float = 3.0
    bullet_damage: float = 10.0
    bullet_lifetime: float = 1.0
    bullet_spread: float = 12.0
    firing_cooldown: float = 0.1


default_ship_config = ShipConfig()


@dataclass
class ActionStates:
    forward: bool
    backward: bool
    left: bool
    right: bool
    sharp_turn: bool
    shoot: bool


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
        rng: np.random.Generator = np.random.default_rng(),
    ):
        self.ship_id = ship_id
        self.team_id = team_id
        self.config = ship_config
        self.rng = rng

        self.alive = True
        self.health = ship_config.max_health
        self.power = ship_config.max_power
        self.turn_offset = 0.0
        self.last_fired_time = 0.0

        self.position = initial_x + 1j * initial_y
        self.velocity = initial_vx + 1j * initial_vy
        self.speed = np.linalg.norm(self.velocity)
        assert self.speed > 1e-6, "Initial velocity cannot be too small"
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

    def _extract_action_states(self, actions: torch.Tensor) -> ActionStates:
        return ActionStates(
            left=actions[:, Actions.left].bool(),
            right=actions[:, Actions.right].bool(),
            forward=actions[:, Actions.forward].bool(),
            backward=actions[:, Actions.backward].bool(),
            sharp_turn=actions[:, Actions.sharp_turn].bool(),
        )

    def _update_power(self, actions: ActionStates, delta_t: float) -> None:
        energy_cost = self.energy_cost_table[actions.forward, actions.backward]
        self.power -= energy_cost * delta_t
        self.power = max(0.0, min(self.power, self.config.max_power))

    def _update_attitude(self, actions: ActionStates) -> None:
        if not (actions.left and actions.right):
            self.turn_offset = self.turn_offset_table[
                actions.left, actions.right, actions.sharp_turn
            ]
        self.attitude = self.velocity / self.speed * np.exp(1j * self.turn_offset)

    def _calculate_forces(self, actions: ActionStates) -> complex:
        if self.power > 0:
            thrust = self.thrust_table[actions.forward, actions.backward]
            thrust_force = thrust * self.attitude
        else:
            thrust_force = 0 + 0j

        turning = actions.left or actions.right
        drag_coeff = self.drag_coeff_table[turning, actions.sharp_turn]
        drag_force = -drag_coeff * self.speed * self.velocity

        lift_coeff = self.lift_coeff_table[actions.sharp_turn]
        lift_vector = self.velocity * 1j  # 90 degrees counter-clockwise
        lift_force = lift_coeff * self.speed * lift_vector

        total_force = thrust_force + drag_force + lift_force
        return total_force

    def _update_kinematics(self, actions: ActionStates, delta_t: float) -> None:
        total_force = self._calculate_forces(actions)
        acceleration = total_force  # Assuming mass = 1

        self.velocity += acceleration * delta_t
        self.position += self.velocity * delta_t
        self.speed = np.linalg.norm(self.velocity)
        if self.speed < 1e-6:
            self.speed = 1e-6
            self.velocity = self.speed * self.attitude

    def _shoot_bullet(
        self, actions: ActionStates, bullets: Bullets, current_time: float
    ) -> None:
        if (
            actions.shoot
            and current_time - self.last_fired_time >= self.config.firing_cooldown
            and self.power >= self.config.bullet_energy_cost
        ):
            self.last_fired_time = current_time
            self.power -= self.config.bullet_energy_cost
            self.is_shooting = True

            bullet_x = self.position.real
            bullet_y = self.position.imag

            bullet_vx = (
                self.velocity.real
                + self.config.bullet_speed * self.attitude.real
                + self.rng.normal(0, self.config.bullet_spread)
            )
            bullet_vy = (
                self.velocity.imag
                + self.config.bullet_speed * self.attitude.imag
                + self.rng.normal(0, self.config.bullet_spread)
            )

            bullets.add_bullet(
                ship_id=self.ship_id,
                x=bullet_x,
                y=bullet_y,
                vx=bullet_vx,
                vy=bullet_vy,
                lifetime=self.config.bullet_lifetime,
            )

        self.is_shooting = False

    def forward(
        self,
        action_vector: torch.Tensor,
        bullets: Bullets,
        current_time: float,
        delta_t: float,
    ) -> None:
        if self.health <= 0:
            self.alive = False
        if not self.alive:
            return

        actions = self._extract_action_states(action_vector)

        self._shoot_bullet(actions, bullets, current_time)
        self._update_attitude(actions)
        self._update_kinematics(actions, delta_t)
        self._update_power(actions, delta_t)

    def get_state(self) -> dict:
        return {
            "ship_id": self.ship_id,
            "team_id": self.team_id,
            "alive": self.alive,
            "health": self.health,
            "power": self.power,
            "position": (self.position.real, self.position.imag),
            "velocity": (self.velocity.real, self.velocity.imag),
            "speed": self.speed,
            "attitude": (self.attitude.real, self.attitude.imag),
            "is_shooting": self.is_shooting,
        }
