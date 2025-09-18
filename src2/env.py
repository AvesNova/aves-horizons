from collections import deque
from copy import deepcopy
import gymnasium as gym
import numpy as np
import torch

from src2.bullets import Bullets
from src2.ship import Ship, default_ship_config


class Snapshot:
    def __init__(self, ships: dict[str, Ship]) -> None:
        self.ships = ships

        max_bullets = np.sum(ship.max_bullets for ship in ships.values())
        self.bullets = Bullets(max_bullets=max_bullets)


class Environment(gym.Env):
    def __init__(
        self,
        render_mode: str = "human",
        world_size: tuple[int, int] | None = (1200, 800),
        memory_size: int | None = 1,
        n_ships: int | None = 2,
        agent_dt: float = 0.1,
        physics_dt: float = 0.02,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.world_size = world_size
        self.memory_size = memory_size
        self.n_ships = n_ships
        self.agent_dt = agent_dt
        self.physics_dt = physics_dt
        assert (
            agent_dt / physics_dt
        ) % 1 == 0, "agent_dt must be multiple of physics_dt"
        self.physics_substeps = int(agent_dt / physics_dt)

    def one_vs_one_reset(self) -> Snapshot:
        ship_0 = Ship(
            ship_id=0,
            team_id=0,
            ship_config=default_ship_config,
            initial_x=self.world_size[0] / 4,
            initial_y=self.world_size[1] / 2,
            initial_vx=100.0,
            initial_vy=0.0,
        )

        ship_1 = Ship(
            ship_id=1,
            team_id=1,
            ship_config=default_ship_config,
            initial_x=3 * self.world_size[0] / 4,
            initial_y=self.world_size[1] / 2,
            initial_vx=-100.0,
            initial_vy=0.0,
        )

        ships = {0: ship_0, 1: ship_1}

        return Snapshot(ships=ships)

    def get_observation(self) -> dict:
        raise NotImplementedError

    def reset(self, game_mode: str = "1v1") -> dict:
        self.current_time = 0.0
        self.state: deque[Snapshot] = deque(maxlen=self.memory_size)

        if game_mode == "1v1":
            self.state.append(self.one_vs_one_reset())
        else:
            raise ValueError(f"Unknown game mode: {game_mode}")

        return self.get_observation()

    def _wrap_ship_position(self, position: complex) -> complex:
        """Wrap ship position to toroidal world boundaries"""
        wrapped_real = position.real % self.world_size[0]
        wrapped_imag = position.imag % self.world_size[1]
        return wrapped_real + 1j * wrapped_imag

    def _wrap_bullet_positions(self, bullets: Bullets) -> None:
        """Wrap bullet positions to toroidal world boundaries"""
        if bullets.num_active == 0:
            return

        active_slice = slice(0, bullets.num_active)
        bullets.x[active_slice] %= self.world_size[0]
        bullets.y[active_slice] %= self.world_size[1]

    def _ship_actions(self, actions: dict, snapshot: Snapshot) -> None:
        for ship_id, ship in snapshot.ships.items():
            if ship.alive:
                ship.forward(
                    actions[ship_id],
                    snapshot.bullets,
                    self.current_time,
                    self.physics_dt,
                )
                ship.position = self._wrap_ship_position(ship.position)

    def _bullet_actions(self, bullets: Bullets) -> None:
        bullets.update_all(self.physics_dt)
        self._wrap_bullet_positions(bullets)

    def _ship_bullet_collisions(self, ships: dict[str, Ship], bullets: Bullets):
        if bullets.num_active == 0:
            return

        bx, by, bullet_ship_ids = bullets.get_active_positions()

        # Check each ship against all bullets
        for ship in ships.values():
            if not ship.alive:
                continue

            # Vectorized distance calculation
            dx = bx - ship.position.real
            dy = by - ship.position.imag
            distances_sq = dx * dx + dy * dy

            # Find hits
            hit_mask = (distances_sq < ship.collision_radius_squared) & (
                bullet_ship_ids != ship.ship_id
            )

            if np.any(hit_mask):
                # Process hits
                ship.damage_ship(np.sum(hit_mask) * ship.config.bullet_damage)

                # Remove hit bullets (batch removal)
                hit_indices = np.where(hit_mask)[0]
                for idx in reversed(sorted(hit_indices)):  # Remove from back to front
                    bullets.remove_bullet(idx)

    def step(self, actions: dict) -> dict:
        current_snapshot = deepcopy(self.state[-1])

        for _ in range(self.physics_substeps):
            self._ship_actions(actions, current_snapshot)
            self._bullet_actions(current_snapshot.bullets)
            self._ship_bullet_collisions(
                current_snapshot.ships, current_snapshot.bullets
            )

            self.current_time += self.physics_dt

        self.state.append(current_snapshot)
        return self.get_observation()
