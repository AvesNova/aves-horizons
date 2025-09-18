from collections import deque
from copy import deepcopy
import gymnasium as gym
import numpy as np
import torch

from src2.bullets import Bullets
from src2.ship import Ship, default_ship_config


class Snapshot:
    def __init__(self, ships, bullets: Bullets | None = Bullets()) -> None:
        self.ships: dict[str, Ship] = ships
        self.bullets: Bullets = bullets


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

    def ship_bullet_collision(self, ships: Ship, bullets: Bullets): ...

    def step(self, actions: dict) -> dict:
        current_snapshot = deepcopy(self.state[-1])

        for _ in range(self.physics_substeps):
            for ship_id, ship in current_snapshot.ships.items():
                if ship.alive:
                    ship.forward(actions[ship_id], self.physics_dt)
