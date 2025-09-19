from collections import deque
from copy import deepcopy
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from bullets import Bullets
from ship import Ship, default_ship_config
from renderer import create_renderer
from constants import Actions
from state import State


class Environment(gym.Env):
    def __init__(
        self,
        render_mode: str | None = None,
        world_size: tuple[int, int] = (1200, 800),
        memory_size: int = 1,
        max_ships: int = 2,
        agent_dt: float = 0.02,
        physics_dt: float = 0.02,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.world_size = world_size
        self.memory_size = memory_size
        self.max_ships = max_ships
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
        self.state: deque[State] = deque(maxlen=memory_size)

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

    def one_vs_one_reset(self) -> State:
        ship_0 = Ship(
            ship_id=0,
            team_id=0,
            ship_config=default_ship_config,
            initial_x=0.25 * self.world_size[0],
            initial_y=0.40 * self.world_size[1],
            initial_vx=100.0,
            initial_vy=0.0,
            world_size=self.world_size,
        )

        ship_1 = Ship(
            ship_id=1,
            team_id=1,
            ship_config=default_ship_config,
            initial_x=0.75 * self.world_size[0],
            initial_y=0.60 * self.world_size[1],
            initial_vx=-100.0,
            initial_vy=0.0,
            world_size=self.world_size,
        )

        ships = {0: ship_0, 1: ship_1}
        return State(ships=ships)

    def reset(self, game_mode: str = "1v1") -> tuple[dict, dict]:
        self.current_time = 0.0
        self.state.clear()

        if game_mode == "1v1":
            self.state.append(self.one_vs_one_reset())
        else:
            raise ValueError(f"Unknown game mode: {game_mode}")

        return self.get_observation(), {}

    def render(self, state: State) -> None:
        """Render current game state"""
        if self.render_mode == "human" and len(self.state) > 0:
            self.renderer.render(state)

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

    def _ship_actions(self, actions: dict[int, torch.Tensor], state: State) -> None:
        for ship_id, ship in state.ships.items():
            if ship.alive:
                ship.forward(
                    actions[ship_id],
                    state.bullets,
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

    def _calculate_rewards(self, state: State) -> dict[int, float]:
        """Calculate basic rewards for each ship"""
        rewards = {}

        for ship_id, ship in state.ships.items():
            reward = 0.0

            if ship.alive:
                reward += 1.0
            else:
                reward -= 100.0

            reward += (ship.health / ship.config.max_health) * 0.5
            rewards[ship_id] = reward

        return rewards

    def _check_termination(self, state: State) -> tuple[bool, dict[int, bool]]:
        """Check if episode should terminate and which agents are done"""
        alive_ships = [ship for ship in state.ships.values() if ship.alive]
        alive_teams = set(ship.team_id for ship in alive_ships)

        terminated = len(alive_teams) <= 1
        done = {ship_id: not ship.alive for ship_id, ship in state.ships.items()}

        return terminated, done

    def step(
        self, actions: dict[int, torch.Tensor]
    ) -> tuple[dict, dict[int, float], bool, bool, dict]:

        # Handle events if in human mode
        if self.render_mode == "human":
            if not self.renderer.handle_events():
                # User closed window - could handle this gracefully
                pass

        current_state = deepcopy(self.state[-1])
        current_state.time += self.agent_dt

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
            self._ship_actions(merged_actions, current_state)
            self._bullet_actions(current_state.bullets)
            self._ship_bullet_collisions(current_state.ships, current_state.bullets)
            self.current_time += self.physics_dt

            if self.render_mode == "human":
                self.render(current_state)

        # Save final state
        self.state.append(current_state)

        # Calculate rewards and termination
        rewards = self._calculate_rewards(current_state)
        terminated, done = self._check_termination(current_state)

        info = {
            "current_time": self.current_time,
            "active_bullets": current_state.bullets.num_active,
            "ship_states": {
                ship_id: ship.get_state()
                for ship_id, ship in current_state.ships.items()
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
        observations = self._get_empty_observation()

        if not self.state:
            return observations

        current_state = self.state[-1]

        for ship_id, ship in current_state.ships.items():
            local_obs = ship.get_state()
            for key, value in local_obs.items():
                if key == "token":
                    # Token is already a tensor, store directly
                    observations[key][ship_id, :] = value
                else:
                    observations[key][ship_id, :] = torch.tensor(value)

        # Create tokens matrix for transformer model
        observations["tokens"] = observations["token"]
        
        return observations

    def _get_empty_observation(self) -> dict:
        """Empty observation for reset state"""
        return {
            "ship_id": torch.zeros((self.max_ships, 1), dtype=torch.int64),
            "team_id": torch.zeros((self.max_ships, 1), dtype=torch.int64),
            "alive": torch.zeros((self.max_ships, 1), dtype=torch.int64),
            "health": torch.zeros((self.max_ships, 1), dtype=torch.int64),
            "power": torch.zeros((self.max_ships, 1), dtype=torch.float32),
            "position": torch.zeros((self.max_ships, 1), dtype=torch.complex64),
            "velocity": torch.zeros((self.max_ships, 1), dtype=torch.complex64),
            "speed": torch.zeros((self.max_ships, 1), dtype=torch.float32),
            "attitude": torch.zeros((self.max_ships, 1), dtype=torch.complex64),
            "is_shooting": torch.zeros((self.max_ships, 1), dtype=torch.int64),
            "token": torch.zeros((self.max_ships, 10), dtype=torch.float32),
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
                "tokens": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.max_ships, 10),
                    dtype=np.float32,
                )
            }
        )
