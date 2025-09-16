"""
StableBaselines3-compatible environment wrapper for Aves Horizons.

Provides Gymnasium-compatible environment for PPO training with deathmatch self-play.
"""

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional, List
import random

from game_modes.deathmatch import create_deathmatch_game
from utils.config import ModelConfig, Actions
from .config import TrainingConfig
from .selfplay import OpponentPool


class DeathmatchSelfPlayEnv(gym.Env):
    """
    Gymnasium environment for deathmatch self-play training.
    
    Compatible with StableBaselines3 PPO algorithm.
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }
    
    def __init__(self, config: TrainingConfig, render_mode: Optional[str] = None):
        super().__init__()
        
        self.config = config
        self.render_mode = render_mode
        
        # Create deathmatch game environment
        self.env = create_deathmatch_game(
            n_teams=config.n_teams,
            ships_per_team=config.ships_per_team,
            world_size=config.world_size,
            use_continuous_collision=True
        )
        
        # Store episode state for temporal sequences
        self.episode_states = []
        self.max_history_length = config.sequence_length
        
        # Team assignments (we control team 0)
        self.controlled_ships = list(range(config.ships_per_team))
        self.opponent_ships = list(range(config.ships_per_team, config.total_ships))
        
        # Opponent management for self-play
        self.opponent_pool = OpponentPool(config.max_opponent_pool_size)
        self.current_opponent_policy = "random"
        self.current_opponent_model = None
        
        # Episode tracking
        self.episode_count = 0
        self.step_count = 0
        
        # Define action space: actions for controlled ships only
        # Each ship has 6 binary actions (forward, backward, left, right, sharp_turn, shoot)
        self.action_space = spaces.MultiBinary(config.controlled_ships * len(Actions))
        
        # Define observation space: flattened temporal token sequence
        obs_dim = config.sequence_length * config.total_ships * ModelConfig.TOKEN_DIM
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # Initialize renderer if needed
        self.renderer = None
        if render_mode == "human":
            from rendering.pygame_renderer import PygameRenderer
            self.renderer = PygameRenderer(world_size=list(config.world_size))
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Update opponent policy periodically
        if self.episode_count % 20 == 0:
            self._update_opponent_policy()
        
        # Reset game environment
        self.env.reset()
        self.episode_states.clear()
        self.step_count = 0
        
        # Fill episode states with initial state repeated
        initial_state = self._encode_ships_to_tokens()
        for _ in range(self.config.sequence_length):
            self.episode_states.append(initial_state)
        
        self.episode_count += 1
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step the environment with the given action."""
        # Convert action to full action tensor for all ships
        actions_tensor = self._process_action(action)
        
        # Step the underlying environment
        _, rewards, done = self.env.step(actions_tensor)
        
        # Update episode states
        current_state = self._encode_ships_to_tokens()
        self.episode_states.append(current_state)
        if len(self.episode_states) > self.max_history_length:
            self.episode_states.pop(0)
        
        # Calculate reward for controlled team
        reward = self._calculate_reward(rewards)
        
        self.step_count += 1
        
        # Check termination conditions
        terminated = done or self._check_team_elimination()
        truncated = self.step_count >= self.config.max_episode_steps
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human" and self.renderer is not None:
            self.renderer.render(self.env.ships, self.env.projectiles, self.env.obstacles)
    
    def close(self):
        """Close the environment."""
        if self.renderer is not None:
            self.renderer.close()
    
    def _process_action(self, action: np.ndarray) -> torch.Tensor:
        """Convert SB3 action to full action tensor for all ships."""
        # Reshape action to [controlled_ships, n_actions]
        controlled_actions = action.reshape(self.config.controlled_ships, len(Actions))
        
        # Create full actions tensor for all ships
        full_actions = torch.zeros(self.config.total_ships, len(Actions), dtype=torch.bool)
        
        # Set controlled ship actions
        for i, ship_id in enumerate(self.controlled_ships):
            if i < controlled_actions.shape[0]:
                full_actions[ship_id] = torch.from_numpy(controlled_actions[i]).bool()
        
        # Generate opponent actions
        for ship_id in self.opponent_ships:
            if self.env.ships.health[ship_id] > 0:  # Only for alive ships
                full_actions[ship_id] = self._get_opponent_action(ship_id)
        
        return full_actions
    
    def _get_opponent_action(self, ship_id: int) -> torch.Tensor:
        """Get action for opponent ship based on current policy."""
        if self.current_opponent_policy == "random":
            return torch.randint(0, 2, (len(Actions),), dtype=torch.bool)
        elif self.current_opponent_policy == "heuristic":
            return self._get_heuristic_action(ship_id)
        elif self.current_opponent_policy == "selfplay" and self.current_opponent_model:
            # TODO: Load and use opponent model
            return torch.randint(0, 2, (len(Actions),), dtype=torch.bool)
        else:
            return torch.zeros(len(Actions), dtype=torch.bool)
    
    def _get_heuristic_action(self, ship_id: int) -> torch.Tensor:
        """Simple heuristic AI for opponent ships."""
        action = torch.zeros(len(Actions), dtype=torch.bool)
        
        # Simple behavior: move forward most of the time, occasionally turn and shoot
        if np.random.random() < 0.8:
            action[Actions.forward] = True
        if np.random.random() < 0.1:
            action[Actions.left] = True
        elif np.random.random() < 0.1:
            action[Actions.right] = True
        if np.random.random() < 0.3:
            action[Actions.shoot] = True
        
        return action
    
    def _update_opponent_policy(self):
        """Update opponent policy based on configuration."""
        choice = np.random.choice(
            list(self.config.opponent_selection_probs.keys()),
            p=list(self.config.opponent_selection_probs.values())
        )
        
        if choice == "selfplay" and len(self.opponent_pool) > 0:
            self.current_opponent_model = self.opponent_pool.sample_opponent()
            self.current_opponent_policy = "selfplay"
        elif choice == "heuristic":
            self.current_opponent_policy = "heuristic"
            self.current_opponent_model = None
        else:  # random
            self.current_opponent_policy = "random"
            self.current_opponent_model = None
    
    def _encode_ships_to_tokens(self) -> np.ndarray:
        """Encode current ship states to token array."""
        ships = self.env.ships
        n_ships = self.config.total_ships
        tokens = np.zeros((n_ships, ModelConfig.TOKEN_DIM), dtype=np.float32)
        
        # Normalize coordinates
        world_width, world_height = self.config.world_size
        
        for i in range(n_ships):
            if i < len(ships.position):
                pos = ships.position[i]
                vel = ships.velocity[i]
                att = ships.attitude[i]
                
                # Basic token features
                tokens[i, 0] = pos.real / world_width        # pos_x normalized
                tokens[i, 1] = pos.imag / world_height       # pos_y normalized
                tokens[i, 2] = vel.real / 300.0              # vel_x normalized
                tokens[i, 3] = vel.imag / 300.0              # vel_y normalized
                tokens[i, 4] = att.real                      # attitude_x
                tokens[i, 5] = att.imag                      # attitude_y
                tokens[i, 6] = ships.turn_offset[i]          # turn_offset
                tokens[i, 7] = ships.boost[i] / 100.0        # boost_norm
                tokens[i, 8] = ships.health[i] / 100.0       # health_norm
                tokens[i, 9] = ships.ammo_count[i] / 32.0    # ammo_norm
                tokens[i, 10] = 0.0                          # is_shooting (simplified)
                tokens[i, 11] = ships.team_id[i]             # team_id
                tokens[i, 12] = 0.0                          # timestep_offset (current)
        
        return tokens
        
    def _get_observation(self) -> np.ndarray:
        """Get flattened observation for the model."""
        if len(self.episode_states) < self.config.sequence_length:
            # Return zeros if history not ready
            obs_dim = self.config.sequence_length * self.config.total_ships * ModelConfig.TOKEN_DIM
            return np.zeros(obs_dim, dtype=np.float32)
        
        # Stack recent states in time-major order
        stacked_tokens = np.stack(self.episode_states[-self.config.sequence_length:], axis=0)
        # Shape: [sequence_length, n_ships, token_dim]
        
        # Reorder to time-major: [ship0_t-n, ship1_t-n, ..., ship0_t-0, ship1_t-0]
        time_major_tokens = []
        for t in range(self.config.sequence_length):
            for ship in range(self.config.total_ships):
                time_major_tokens.append(stacked_tokens[t, ship, :])
        
        return np.concatenate(time_major_tokens).astype(np.float32)
    
    def _calculate_reward(self, raw_rewards: torch.Tensor) -> float:
        """Calculate shaped reward for controlled team."""
        # Base reward from controlled ships
        controlled_reward = raw_rewards[self.controlled_ships].sum().item()
        
        # Team survival bonus
        controlled_alive = sum(1 for ship_id in self.controlled_ships 
                              if self.env.ships.health[ship_id] > 0)
        opponent_alive = sum(1 for ship_id in self.opponent_ships 
                            if self.env.ships.health[ship_id] > 0)
        
        # Strong win/loss rewards
        if controlled_alive > 0 and opponent_alive == 0:
            controlled_reward += 100.0  # Victory!
        elif controlled_alive == 0 and opponent_alive > 0:
            controlled_reward -= 50.0   # Defeat
        
        # Survival bonus
        controlled_reward += controlled_alive * 2.0
        
        return controlled_reward
    
    def _check_team_elimination(self) -> bool:
        """Check if either team has been eliminated."""
        controlled_alive = sum(1 for ship_id in self.controlled_ships 
                              if self.env.ships.health[ship_id] > 0)
        opponent_alive = sum(1 for ship_id in self.opponent_ships 
                            if self.env.ships.health[ship_id] > 0)
        
        return controlled_alive == 0 or opponent_alive == 0
    
    def _get_info(self) -> Dict:
        """Get environment info."""
        controlled_alive = sum(1 for ship_id in self.controlled_ships 
                              if self.env.ships.health[ship_id] > 0)
        opponent_alive = sum(1 for ship_id in self.opponent_ships 
                            if self.env.ships.health[ship_id] > 0)
        
        return {
            "controlled_alive": controlled_alive,
            "opponent_alive": opponent_alive,
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "opponent_policy": self.current_opponent_policy,
            "total_ships": self.config.total_ships
        }