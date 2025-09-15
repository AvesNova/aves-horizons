import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from typing import Any, Dict, Tuple, Optional

from core.environment import Environment
from utils.config import Actions


class ShipEnv(gym.Env):
    """
    Gymnasium wrapper for the ship combat environment.

    This implements a proper gym interface for the environment with:
    - Transformer-friendly observation space (sequence of ship states)
    - Proper action space using MultiBinary
    - Combat-focused reward shaping
    - Support for multiple agents
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode=None, n_ships=8, n_obstacles=0):
        super().__init__()

        # Create underlying environment
        self.env = Environment(n_ships=n_ships, n_obstacles=n_obstacles)
        self.n_ships = n_ships
        self.render_mode = render_mode

        # Each ship has 6 binary actions: forward, backward, left, right, sharp_turn, shoot
        self.action_space = spaces.MultiBinary(len(Actions))

        # Observation space per ship:
        # - position (2)
        # - velocity (2)
        # - attitude (2)
        # - turn offset (1)
        # - boost normalized (1)
        # - health normalized (1)
        # - ammo normalized (1)
        # = 10 values per ship
        self.obs_per_ship = 10
        obs_low = np.array([-np.inf] * (self.obs_per_ship * n_ships), dtype=np.float32)
        obs_high = np.array([np.inf] * (self.obs_per_ship * n_ships), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

    def _get_obs(self) -> np.ndarray:
        """Convert current state to transformer-friendly observation."""
        ships = self.env.ships

        # Convert complex tensors to real pairs
        positions = torch.stack([ships.position.real, ships.position.imag], dim=1)
        velocities = torch.stack([ships.velocity.real, ships.velocity.imag], dim=1)
        attitudes = torch.stack([ships.attitude.real, ships.attitude.imag], dim=1)

        # Stack all ship states [n_ships, obs_per_ship]
        ship_states = torch.cat(
            [
                positions,  # [n_ships, 2]
                velocities,  # [n_ships, 2]
                attitudes,  # [n_ships, 2]
                ships.turn_offset.unsqueeze(1),  # [n_ships, 1]
                (ships.boost / ships.max_boost).unsqueeze(1),  # [n_ships, 1]
                (ships.health / ships.max_health).unsqueeze(1),  # [n_ships, 1]
                (ships.ammo_count / ships.max_ammo).unsqueeze(1),  # [n_ships, 1]
            ],
            dim=1,
        )

        # Flatten to [n_ships * obs_per_ship]
        return ship_states.flatten().cpu().numpy()

    def _get_info(self) -> Dict:
        """Return current environment info."""
        return {
            "alive_ships": (self.env.ships.health > 0).sum().item(),
            "total_ships": self.n_ships,
        }

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Reset underlying environment
        self.env.reset()

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take one environment step.

        Args:
            action: Binary action array [forward, backward, left, right, sharp_turn, shoot]

        Returns:
            observation: Flattened state array
            reward: Shaped reward value
            terminated: Whether episode is terminated
            truncated: Whether episode is truncated
            info: Additional information
        """
        # Convert numpy action to torch tensor
        action_tensor = torch.as_tensor(action, dtype=torch.bool)

        # Create full action tensor for all ships (this ship's action for index 0)
        actions = torch.zeros((self.n_ships, len(Actions)), dtype=torch.bool)
        actions[0] = action_tensor

        # Other ships take random actions
        for i in range(1, self.n_ships):
            actions[i] = torch.randint(0, 2, (len(Actions),), dtype=torch.bool)

        # Take environment step
        _, _, done = self.env.step(actions)

        # Get new observation
        obs = self._get_obs()

        # Calculate reward (example reward shaping)
        reward = self._calculate_reward(action_tensor)

        # Get additional info
        info = self._get_info()

        return obs, reward, done, False, info

    def _calculate_reward(self, action: torch.Tensor) -> float:
        """Calculate shaped reward for the learning agent (ship 0)."""
        reward = 0.0
        ships = self.env.ships

        # Survival reward
        if ships.health[0] > 0:
            reward += 0.1

        # Damage reward (when hitting other ships)
        for i in range(1, self.n_ships):
            if ships.health[i] <= 0:
                reward += 10.0  # Big reward for eliminating opponents

        # Energy efficiency penalty
        if action[Actions.forward]:
            reward -= 0.01  # Small penalty for using boost

        # Movement reward
        speed = torch.abs(ships.velocity[0])
        if speed > 1.0:
            reward += 0.01  # Small reward for maintaining speed

        return reward

    def render(self):
        """Render current environment state."""
        if not hasattr(self, "renderer"):
            from rendering.pygame_renderer import PygameRenderer

            self.renderer = PygameRenderer(world_size=self.env.world_size.tolist())

        self.renderer.render(self.env.ships, self.env.projectiles, self.env.obstacles)

    def close(self):
        """Clean up environment resources."""
        if hasattr(self, "renderer"):
            self.renderer.close()
