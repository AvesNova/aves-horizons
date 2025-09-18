from collections import deque
from copy import deepcopy
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from bullets import Bullets
from ship import Ship, default_ship_config
from renderer import create_renderer
from enums import Actions
from snapshot import Snapshot


class Environment(gym.Env):
    def __init__(
        self,
        render_mode: str | None = None,
        world_size: tuple[int, int] = (1200, 800),
        memory_size: int = 1,
        n_ships: int = 2,
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
        self.target_fps = 1 / physics_dt

        assert (
            agent_dt / physics_dt
        ) % 1 == 0, "agent_dt must be multiple of physics_dt"
        self.physics_substeps = int(agent_dt / physics_dt)

        # Lazy-loaded renderer
        self._renderer = None

        # Initialize state
        self.current_time = 0.0
        self.state: deque[Snapshot] = deque(maxlen=memory_size)

    @property
    def renderer(self):
        """Lazy-load the renderer only when needed"""
        if self._renderer is None and self.render_mode == "human":
            self._renderer = create_renderer(self.world_size, self.target_fps)
            if self._renderer is None:
                raise ImportError("pygame is required for human rendering mode")
        return self._renderer

    def add_human_player(self, ship_id: int) -> None:
        """Register a ship to be controlled by human input"""
        if self.render_mode == "human":
            self.renderer.add_human_player(ship_id)

    def remove_human_player(self, ship_id: int) -> None:
        """Remove human control from a ship"""
        if self.render_mode == "human":
            self.renderer.remove_human_player(ship_id)

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

    def reset(self, game_mode: str = "1v1") -> tuple[dict, dict]:
        self.current_time = 0.0
        self.state.clear()

        if game_mode == "1v1":
            self.state.append(self.one_vs_one_reset())
        else:
            raise ValueError(f"Unknown game mode: {game_mode}")

        return self.get_observation(), {}

    def render(self, snapshot: Snapshot) -> None:
        """Render current game state"""
        if self.render_mode == "human" and len(self.state) > 0:
            self.renderer.render(snapshot)

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

    def _ship_actions(
        self, actions: dict[int, torch.Tensor], snapshot: Snapshot
    ) -> None:
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

    def _ship_bullet_collisions(self, ships: dict[int, Ship], bullets: Bullets):
        if bullets.num_active == 0:
            return

        bx, by, bullet_ship_ids = bullets.get_active_positions()

        for ship in ships.values():
            if not ship.alive:
                continue

            dx = bx - ship.position.real
            dy = by - ship.position.imag
            distances_sq = dx * dx + dy * dy

            hit_mask = (distances_sq < ship.collision_radius_squared) & (
                bullet_ship_ids != ship.ship_id
            )

            if np.any(hit_mask):
                ship.damage_ship(np.sum(hit_mask) * ship.config.bullet_damage)

                hit_indices = np.where(hit_mask)[0]
                for idx in reversed(sorted(hit_indices)):
                    bullets.remove_bullet(idx)

    def _calculate_rewards(self, snapshot: Snapshot) -> dict[int, float]:
        """Calculate basic rewards for each ship"""
        rewards = {}

        for ship_id, ship in snapshot.ships.items():
            reward = 0.0

            if ship.alive:
                reward += 1.0
            else:
                reward -= 100.0

            reward += (ship.health / ship.config.max_health) * 0.5
            rewards[ship_id] = reward

        return rewards

    def _check_termination(self, snapshot: Snapshot) -> tuple[bool, dict[int, bool]]:
        """Check if episode should terminate and which agents are done"""
        alive_ships = [ship for ship in snapshot.ships.values() if ship.alive]
        alive_teams = set(ship.team_id for ship in alive_ships)

        terminated = len(alive_teams) <= 1
        done = {ship_id: not ship.alive for ship_id, ship in snapshot.ships.items()}

        return terminated, done

    def step(
        self, actions: dict[int, torch.Tensor]
    ) -> tuple[dict, dict[int, float], bool, bool, dict]:

        # Handle events if in human mode
        if self.render_mode == "human":
            if not self.renderer.handle_events():
                # User closed window - could handle this gracefully
                pass

        current_snapshot = deepcopy(self.state[-1])

        # Run physics substeps with rendering
        for substep in range(self.physics_substeps):
            # Update human input at physics rate
            if self.render_mode == "human":
                self.renderer.update_human_actions()
                human_actions = self.renderer.get_human_actions()
                # Merge AI and human actions
                merged_actions = {**actions, **human_actions}
            else:
                merged_actions = actions

            # Physics step
            self._ship_actions(merged_actions, current_snapshot)
            self._bullet_actions(current_snapshot.bullets)
            self._ship_bullet_collisions(
                current_snapshot.ships, current_snapshot.bullets
            )
            self.current_time += self.physics_dt

            if self.render_mode == "human":
                self.render(current_snapshot)

        # Save final state
        self.state.append(current_snapshot)

        # Calculate rewards and termination
        rewards = self._calculate_rewards(current_snapshot)
        terminated, done = self._check_termination(current_snapshot)

        info = {
            "current_time": self.current_time,
            "active_bullets": current_snapshot.bullets.num_active,
            "ship_states": {
                ship_id: ship.get_state()
                for ship_id, ship in current_snapshot.ships.items()
            },
            "individual_done": done,
        }

        # Add human control info if in human mode
        if self.render_mode == "human":
            info["human_controlled"] = list(self.renderer.human_ship_ids)

        return self.get_observation(), rewards, terminated, False, info

    def close(self):
        """Clean up resources"""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def get_observation(self) -> dict:
        """Extract observations for each ship from current state"""
        if not self.state:
            # Return empty observations if no state
            return {
                ship_id: self._get_empty_observation()
                for ship_id in range(self.n_ships)
            }

        current_snapshot = self.state[-1]
        observations = {}

        for ship_id, ship in current_snapshot.ships.items():
            observations[ship_id] = self._get_ship_observation(
                ship_id, current_snapshot
            )

        return observations

    def _get_empty_observation(self) -> dict:
        """Empty observation for reset state"""
        return {
            "self_state": np.zeros(
                8, dtype=np.float32
            ),  # [x, y, vx, vy, health, power, attitude_x, attitude_y]
            "enemy_state": np.zeros(8, dtype=np.float32),
            "bullets": np.zeros(
                (20, 6), dtype=np.float32
            ),  # Max 20 visible bullets: [x, y, vx, vy, time_left, is_enemy]
            "world_bounds": np.array(self.world_size, dtype=np.float32),
            "time": 0.0,
        }

    def _get_ship_observation(self, ship_id: int, snapshot: Snapshot) -> dict:
        """Get observation for a specific ship"""
        ship = snapshot.ships[ship_id]

        # Self state: position, velocity, health, power, attitude
        self_state = np.array(
            [
                ship.position.real / self.world_size[0],  # Normalize to [0,1]
                ship.position.imag / self.world_size[1],
                ship.velocity.real / 1000.0,  # Normalize velocity
                ship.velocity.imag / 1000.0,
                ship.health / ship.config.max_health,  # Health ratio
                ship.power / ship.config.max_power,  # Power ratio
                ship.attitude.real,  # Attitude unit vector
                ship.attitude.imag,
            ],
            dtype=np.float32,
        )

        # Enemy state (find the first enemy ship)
        enemy_state = np.zeros(8, dtype=np.float32)
        for enemy_id, enemy_ship in snapshot.ships.items():
            if enemy_id != ship_id and enemy_ship.team_id != ship.team_id:
                if enemy_ship.alive:
                    enemy_state = np.array(
                        [
                            enemy_ship.position.real / self.world_size[0],
                            enemy_ship.position.imag / self.world_size[1],
                            enemy_ship.velocity.real / 1000.0,
                            enemy_ship.velocity.imag / 1000.0,
                            enemy_ship.health / enemy_ship.config.max_health,
                            enemy_ship.power / enemy_ship.config.max_power,
                            enemy_ship.attitude.real,
                            enemy_ship.attitude.imag,
                        ],
                        dtype=np.float32,
                    )
                break

        # Bullet observations (relative to this ship)
        bullets = np.zeros((20, 6), dtype=np.float32)  # Max 20 bullets
        if snapshot.bullets.num_active > 0:
            bx, by, bullet_ship_ids = snapshot.bullets.get_active_positions()

            # Calculate distances to sort by closest
            dx = bx - ship.position.real
            dy = by - ship.position.imag
            distances_sq = dx * dx + dy * dy

            # Sort by distance, take closest 20
            if len(distances_sq) > 0:
                closest_indices = np.argsort(distances_sq)[:20]

                for i, bullet_idx in enumerate(closest_indices):
                    if i >= 20:
                        break

                    bullets[i] = [
                        dx[bullet_idx] / self.world_size[0],  # Relative x
                        dy[bullet_idx] / self.world_size[1],  # Relative y
                        snapshot.bullets.vx[bullet_idx] / 1000.0,  # Velocity x
                        snapshot.bullets.vy[bullet_idx] / 1000.0,  # Velocity y
                        snapshot.bullets.time_remaining[bullet_idx],  # Time left
                        (
                            1.0 if bullet_ship_ids[bullet_idx] != ship_id else 0.0
                        ),  # Is enemy bullet
                    ]

        return {
            "self_state": self_state,
            "enemy_state": enemy_state,
            "bullets": bullets,
            "world_bounds": np.array(self.world_size, dtype=np.float32),
            "time": self.current_time,
        }

    @property
    def action_space(self) -> spaces.Space:
        """Define the action space for each ship"""
        # Multi-discrete for binary actions: [forward, backward, left, right, sharp_turn, shoot]
        return spaces.MultiBinary(len(Actions))

    @property
    def observation_space(self) -> spaces.Space:
        """Define the observation space for each ship"""
        return spaces.Dict(
            {
                "self_state": spaces.Box(
                    low=-1.0, high=1.0, shape=(8,), dtype=np.float32
                ),
                "enemy_state": spaces.Box(
                    low=-1.0, high=1.0, shape=(8,), dtype=np.float32
                ),
                "bullets": spaces.Box(
                    low=-1.0, high=1.0, shape=(20, 6), dtype=np.float32
                ),
                "world_bounds": spaces.Box(
                    low=0, high=10000, shape=(2,), dtype=np.float32
                ),
                "time": spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
            }
        )
