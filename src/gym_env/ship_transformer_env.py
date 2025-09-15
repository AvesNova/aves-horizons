"""
Enhanced Ship Environment with Transformer Support.

This environment extends the base ShipEnv to support the transformer model's
multi-agent architecture with temporal sequences and state history tracking.
"""

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from typing import Any, Dict, List, Tuple, Optional, Union

from core.environment import Environment
from utils.config import Actions
from models.state_history import StateHistory
from models.token_encoder import ShipTokenEncoder


class ShipTransformerEnv(gym.Env):
    """
    Transformer-enhanced ship combat environment.
    
    Features:
    - Temporal state sequences for transformer input
    - Multi-agent coordination support
    - Controlled team vs opponent team dynamics
    - State history tracking
    - Token-based observations
    
    The environment supports the transformer's multi-agent output strategy:
    - Model predicts actions for all ships
    - Only executes actions for controlled team
    - Opponent actions can be provided or use built-in AI
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        n_ships: int = 8,
        n_obstacles: int = 0,
        controlled_team_size: int = 4,
        sequence_length: int = 6,
        world_size: Tuple[float, float] = (1200.0, 800.0),
        normalize_coordinates: bool = True,
        opponent_policy: str = "random",  # "random", "heuristic", "model"
        team_assignment: str = "alternating"  # "alternating", "first_half", "custom"
    ):
        super().__init__()
        
        self.n_ships = n_ships
        self.controlled_team_size = controlled_team_size
        self.sequence_length = sequence_length
        self.world_size = world_size
        self.opponent_policy = opponent_policy
        self.team_assignment = team_assignment
        self.render_mode = render_mode
        
        # Create underlying environment
        self.env = Environment(n_ships=n_ships, n_obstacles=n_obstacles)
        
        # Initialize state history tracking
        self.state_history = StateHistory(
            sequence_length=sequence_length,
            max_ships=n_ships,
            world_size=world_size,
            normalize_coordinates=normalize_coordinates
        )
        
        # Initialize token encoder
        self.token_encoder = ShipTokenEncoder(
            world_size=world_size,
            normalize_coordinates=normalize_coordinates,
            max_ships=n_ships
        )
        
        # Determine team assignments
        self.controlled_ships, self.opponent_ships = self._assign_teams()
        
        # Define action space: actions for all controlled ships
        # Each ship has 6 binary actions
        self.action_space = spaces.MultiBinary(controlled_team_size * len(Actions))
        
        # Define observation space: temporal token sequences
        # Format: [sequence_length * n_ships, token_dim]
        token_dim = self.token_encoder.get_token_dim()  # 12
        sequence_tokens = sequence_length * n_ships
        obs_low = np.full(sequence_tokens * token_dim, -np.inf, dtype=np.float32)
        obs_high = np.full(sequence_tokens * token_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # Track episode state
        self.episode_steps = 0
        self.max_episode_steps = 2000
        
    def _assign_teams(self) -> Tuple[List[int], List[int]]:
        """Assign ships to controlled and opponent teams."""
        if self.team_assignment == "alternating":
            # Alternate assignment: 0,2,4,6 vs 1,3,5,7
            controlled = list(range(0, self.n_ships, 2))[:self.controlled_team_size]
            opponent = list(range(1, self.n_ships, 2))[:self.n_ships - self.controlled_team_size]
        elif self.team_assignment == "first_half":
            # First half vs second half
            controlled = list(range(self.controlled_team_size))
            opponent = list(range(self.controlled_team_size, self.n_ships))
        else:
            # Default to first ships as controlled
            controlled = list(range(self.controlled_team_size))
            opponent = list(range(self.controlled_team_size, self.n_ships))
        
        return controlled, opponent
    
    def _get_obs(self) -> np.ndarray:
        """Get transformer-compatible observation from state history."""
        if not self.state_history.is_ready():
            # If not enough history, return zeros
            token_dim = self.token_encoder.get_token_dim()
            sequence_tokens = self.sequence_length * self.n_ships
            return np.zeros(sequence_tokens * token_dim, dtype=np.float32)
        
        # Get token sequence from state history
        tokens, ship_ids = self.state_history.get_token_sequence()
        
        # Flatten for gym observation
        return tokens.flatten().cpu().numpy().astype(np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Return current environment info."""
        controlled_alive = sum(1 for ship_id in self.controlled_ships 
                             if self.env.ships.health[ship_id] > 0)
        opponent_alive = sum(1 for ship_id in self.opponent_ships 
                           if self.env.ships.health[ship_id] > 0)
        
        return {
            "controlled_alive": controlled_alive,
            "opponent_alive": opponent_alive,
            "controlled_ships": self.controlled_ships,
            "opponent_ships": self.opponent_ships,
            "episode_steps": self.episode_steps,
            "state_history_ready": self.state_history.is_ready(),
        }
    
    def _generate_opponent_actions(self) -> torch.Tensor:
        """Generate actions for opponent ships."""
        opponent_actions = torch.zeros(len(self.opponent_ships), len(Actions), dtype=torch.bool)
        
        if self.opponent_policy == "random":
            # Random actions for opponents
            opponent_actions = torch.randint(0, 2, opponent_actions.shape, dtype=torch.bool)
            
        elif self.opponent_policy == "heuristic":
            # Simple heuristic AI for opponents
            for i, ship_id in enumerate(self.opponent_ships):
                if self.env.ships.health[ship_id] <= 0:
                    continue  # Dead ship, no actions
                
                # Simple behavior: move forward and occasionally turn/shoot
                if np.random.random() < 0.8:
                    opponent_actions[i, Actions.forward] = True
                if np.random.random() < 0.1:
                    opponent_actions[i, Actions.left] = True
                if np.random.random() < 0.1:
                    opponent_actions[i, Actions.right] = True
                if np.random.random() < 0.3:
                    opponent_actions[i, Actions.shoot] = True
        
        return opponent_actions
    
    def _create_full_action_tensor(
        self, 
        controlled_actions: np.ndarray
    ) -> torch.Tensor:
        """
        Create full action tensor for all ships.
        
        Args:
            controlled_actions: Flattened actions for controlled ships
            
        Returns:
            actions: [n_ships, 6] Action tensor for all ships
        """
        # Initialize actions for all ships
        full_actions = torch.zeros(self.n_ships, len(Actions), dtype=torch.bool)
        
        # Reshape controlled actions
        controlled_actions_reshaped = controlled_actions.reshape(
            self.controlled_team_size, len(Actions)
        )
        
        # Apply controlled ship actions
        for i, ship_id in enumerate(self.controlled_ships):
            if i < controlled_actions_reshaped.shape[0]:
                full_actions[ship_id] = torch.from_numpy(
                    controlled_actions_reshaped[i]
                ).bool()
        
        # Generate opponent actions
        opponent_actions = self._generate_opponent_actions()
        for i, ship_id in enumerate(self.opponent_ships):
            if i < opponent_actions.shape[0]:
                full_actions[ship_id] = opponent_actions[i]
        
        return full_actions
    
    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Reset underlying environment
        self.env.reset()
        
        # Reset state history
        self.state_history.reset()
        
        # Reset episode tracking
        self.episode_steps = 0
        
        # Add initial state to history
        self.state_history.add_state(self.env.ships)
        
        return self._get_obs(), self._get_info()
    
    def step(
        self, 
        action: Union[np.ndarray, List]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take one environment step.
        
        Args:
            action: Actions for controlled ships [controlled_team_size * 6]
            
        Returns:
            observation: Token sequence observation
            reward: Shaped reward value
            terminated: Whether episode is terminated
            truncated: Whether episode is truncated
            info: Additional information
        """
        # Convert action to numpy if needed
        if isinstance(action, list):
            action = np.array(action)
        
        # Create full action tensor for all ships
        full_actions = self._create_full_action_tensor(action)
        
        # Take environment step
        _, _, done = self.env.step(full_actions)
        
        # Add new state to history
        self.state_history.add_state(self.env.ships, full_actions)
        
        # Get new observation
        obs = self._get_obs()
        
        # Calculate reward
        reward = self._calculate_reward(action, full_actions)
        
        # Update episode tracking
        self.episode_steps += 1
        
        # Check termination conditions
        terminated = done or self._check_team_elimination()
        truncated = self.episode_steps >= self.max_episode_steps
        
        # Get additional info
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _check_team_elimination(self) -> bool:
        """Check if either team has been eliminated."""
        controlled_alive = sum(1 for ship_id in self.controlled_ships 
                             if self.env.ships.health[ship_id] > 0)
        opponent_alive = sum(1 for ship_id in self.opponent_ships 
                           if self.env.ships.health[ship_id] > 0)
        
        return controlled_alive == 0 or opponent_alive == 0
    
    def _calculate_reward(
        self, 
        controlled_actions: np.ndarray,
        full_actions: torch.Tensor
    ) -> float:
        """
        Calculate shaped reward for the controlled team.
        
        Args:
            controlled_actions: Actions taken by controlled ships
            full_actions: Actions for all ships
            
        Returns:
            reward: Calculated reward value
        """
        reward = 0.0
        ships = self.env.ships
        
        # Team survival bonus
        controlled_alive = sum(1 for ship_id in self.controlled_ships 
                             if ships.health[ship_id] > 0)
        opponent_alive = sum(1 for ship_id in self.opponent_ships 
                           if ships.health[ship_id] > 0)
        
        # Base survival reward
        reward += controlled_alive * 0.1
        
        # Elimination bonuses
        if opponent_alive == 0 and controlled_alive > 0:
            reward += 100.0  # Major victory bonus
        elif controlled_alive == 0:
            reward -= 50.0   # Major defeat penalty
        
        # Health-based rewards
        for ship_id in self.controlled_ships:
            ship_health_norm = ships.health[ship_id] / ships.max_health[ship_id]
            reward += ship_health_norm * 0.05  # Health maintenance bonus
        
        # Combat engagement rewards
        for i, ship_id in enumerate(self.controlled_ships):
            if ships.health[ship_id] > 0:
                # Shooting bonus (encourages engagement)
                if i < len(controlled_actions) // len(Actions):
                    action_idx = i * len(Actions)
                    if controlled_actions[action_idx + Actions.shoot]:
                        reward += 0.02
                
                # Movement bonus (encourages active play)
                speed = torch.abs(ships.velocity[ship_id])
                if speed > 10.0:
                    reward += 0.01
        
        # Energy efficiency penalty
        energy_penalty = 0.0
        for ship_id in self.controlled_ships:
            if ships.boost[ship_id] < ships.max_boost[ship_id] * 0.1:
                energy_penalty += 0.01  # Penalty for very low energy
        reward -= energy_penalty
        
        return reward
    
    def render(self):
        """Render current environment state."""
        if not hasattr(self, "renderer"):
            from rendering.pygame_renderer import PygameRenderer
            self.renderer = PygameRenderer(world_size=list(self.world_size))
        
        self.renderer.render(self.env.ships, self.env.projectiles, self.env.obstacles)
    
    def close(self):
        """Clean up environment resources."""
        if hasattr(self, "renderer"):
            self.renderer.close()
    
    def get_token_sequence(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get current token sequence for direct transformer input.
        
        Returns:
            tokens: [seq_len, 12] Token sequence
            ship_ids: [seq_len] Ship IDs
        """
        return self.state_history.get_token_sequence()
    
    def set_opponent_policy(self, policy: str):
        """Set the opponent AI policy."""
        self.opponent_policy = policy
    
    def get_controlled_ships(self) -> List[int]:
        """Get list of controlled ship IDs."""
        return self.controlled_ships.copy()
    
    def get_opponent_ships(self) -> List[int]:
        """Get list of opponent ship IDs."""
        return self.opponent_ships.copy()


class MultiGameShipTransformerEnv:
    """
    Wrapper for managing multiple ShipTransformerEnv instances for batch training.
    
    This allows running multiple games in parallel for more efficient data collection
    during training.
    """
    
    def __init__(
        self,
        num_envs: int,
        env_kwargs: Dict[str, Any] = None
    ):
        self.num_envs = num_envs
        self.env_kwargs = env_kwargs or {}
        
        # Create individual environments
        self.envs = [
            ShipTransformerEnv(**self.env_kwargs) 
            for _ in range(num_envs)
        ]
        
        # Use first environment as template for spaces
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
    
    def reset(self) -> Tuple[List[np.ndarray], List[Dict]]:
        """Reset all environments."""
        observations = []
        infos = []
        
        for env in self.envs:
            obs, info = env.reset()
            observations.append(obs)
            infos.append(info)
        
        return observations, infos
    
    def step(
        self, 
        actions: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[float], List[bool], List[bool], List[Dict]]:
        """Step all environments."""
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []
        
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
        
        return observations, rewards, terminateds, truncateds, infos
    
    def render(self, env_idx: int = 0):
        """Render specific environment."""
        if 0 <= env_idx < self.num_envs:
            self.envs[env_idx].render()
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()
    
    def get_token_sequences(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get token sequences from all environments."""
        return [env.get_token_sequence() for env in self.envs]