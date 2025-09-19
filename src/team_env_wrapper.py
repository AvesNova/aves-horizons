"""
Team Environment Wrapper for StableBaselines3 integration.
Converts multi-agent team environment to single-agent interface.
"""

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from typing import Dict, List, Optional, Union, Any
import random
from collections import deque
from copy import deepcopy

from env import Environment
from scripted_agent import ScriptedAgent
from team_transformer_model import TeamTransformerModel, TeamController


class TeamEnvironmentWrapper(gym.Env):
    """
    Wrapper that converts the multi-agent team environment to a single-agent
    interface for StableBaselines3 training.

    Features:
    - Handles team-based control where one policy controls multiple ships
    - Supports self-play with opponent memory
    - Supports training against scripted agents
    - Compatible with SB3's single-agent interface
    """

    def __init__(
        self,
        env_config: dict = None,
        team_id: int = 0,
        team_assignments: dict = None,
        opponent_type: str = "self_play",  # "self_play", "scripted", "mixed"
        scripted_agent_config: dict = None,
        self_play_memory_size: int = 100,
        opponent_update_freq: int = 1000,  # Steps between opponent updates
        scripted_mix_ratio: float = 0.3,  # Fraction of episodes vs scripted
    ):
        super().__init__()

        # Environment setup
        env_config = env_config or {}
        self.base_env = Environment(**env_config)

        # Team configuration
        self.team_id = team_id
        self.team_assignments = team_assignments or {0: [0], 1: [1]}  # Default 1v1
        self.team_controller = TeamController(self.team_assignments)
        self.controlled_ships = self.team_assignments[team_id]

        # Opponent configuration
        self.opponent_type = opponent_type
        self.scripted_mix_ratio = scripted_mix_ratio
        self.current_episode_type = "self_play"

        # Self-play memory for storing past policies
        self.self_play_memory = deque(maxlen=self_play_memory_size)
        self.opponent_update_freq = opponent_update_freq
        self.steps_since_opponent_update = 0
        self.current_opponent_model = None

        # Scripted agent setup
        scripted_config = scripted_agent_config or {
            "max_shooting_range": 500.0,
            "angle_threshold": 5.0,
            "bullet_speed": 500.0,
            "target_radius": 10.0,
            "radius_multiplier": 1.5,
        }

        self.scripted_agents = {}
        for opponent_team_id, ship_ids in self.team_assignments.items():
            if opponent_team_id != self.team_id:  # Opponent teams
                for ship_id in ship_ids:
                    self.scripted_agents[ship_id] = ScriptedAgent(
                        controlled_ship_id=ship_id,
                        world_size=env_config.get("world_size", (1200, 800)),
                        **scripted_config,
                    )

        # Observation and action spaces
        self._setup_spaces()

        # Episode tracking
        self.episode_count = 0
        self.wins = 0
        self.losses = 0

    def _setup_spaces(self):
        """Setup observation and action spaces for SB3"""
        max_ships = self.base_env.max_ships
        token_dim = 10  # From ship.get_token()

        # Observation space: token matrix for all ships
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(max_ships, token_dim), dtype=np.float32
        )

        # Action space: actions for our controlled ships only
        # Flatten to single vector for SB3 compatibility
        num_controlled_ships = len(self.controlled_ships)
        num_actions_per_ship = 6  # From Actions enum

        self.action_space = spaces.MultiBinary(
            num_controlled_ships * num_actions_per_ship
        )

    def _choose_episode_type(self) -> str:
        """Choose opponent type for this episode"""
        if self.opponent_type == "scripted":
            return "scripted"
        elif self.opponent_type == "self_play":
            return "self_play"
        elif self.opponent_type == "mixed":
            return (
                "scripted" if random.random() < self.scripted_mix_ratio else "self_play"
            )
        else:
            return "self_play"

    def _get_opponent_actions(self, observation: np.ndarray) -> dict:
        """Get actions for opponent ships"""
        opponent_actions = {}

        if self.current_episode_type == "scripted":
            # Use scripted agents
            for ship_id, agent in self.scripted_agents.items():
                if self._is_ship_alive(observation, ship_id):
                    # Convert observation to dict format for scripted agent
                    obs_dict = self._convert_observation_for_scripted(observation)
                    action = agent(obs_dict)
                    opponent_actions[ship_id] = action
                else:
                    opponent_actions[ship_id] = torch.zeros(6, dtype=torch.float32)

        elif self.current_episode_type == "self_play":
            # Use stored opponent model
            if self.current_opponent_model is not None:
                try:
                    # Get actions for all ships from opponent model
                    with torch.no_grad():
                        ship_mask = self._create_ship_mask(observation)
                        obs_batch = {
                            "tokens": torch.from_numpy(observation).unsqueeze(0).float()
                        }
                        all_actions = self.current_opponent_model.get_actions(
                            obs_batch, ship_mask, deterministic=False
                        )["actions"][
                            0
                        ]  # Remove batch dim

                        # Extract opponent team actions
                        for opponent_team_id, ship_ids in self.team_assignments.items():
                            if opponent_team_id != self.team_id:
                                team_actions = (
                                    self.team_controller.extract_team_actions(
                                        all_actions.unsqueeze(0), opponent_team_id
                                    )
                                )
                                opponent_actions.update(team_actions)

                except Exception as e:
                    print(f"Error with opponent model: {e}, falling back to random")
                    # Fallback to random actions
                    for team_id, ship_ids in self.team_assignments.items():
                        if team_id != self.team_id:
                            for ship_id in ship_ids:
                                opponent_actions[ship_id] = torch.randint(
                                    0, 2, (6,), dtype=torch.float32
                                )
            else:
                # No opponent model yet, use random actions
                for team_id, ship_ids in self.team_assignments.items():
                    if team_id != self.team_id:
                        for ship_id in ship_ids:
                            opponent_actions[ship_id] = torch.randint(
                                0, 2, (6,), dtype=torch.float32
                            )

        return opponent_actions

    def _convert_observation_for_scripted(self, observation: np.ndarray) -> dict:
        """Convert token observation back to dict format for scripted agents"""
        # Create the observation dict that scripted agents expect
        obs_dict = {
            "ship_id": torch.from_numpy(observation[:, 0:1]).long(),
            "team_id": torch.from_numpy(
                observation[:, 0:1]
            ).long(),  # Will be overridden
            "alive": (torch.from_numpy(observation[:, 1:2]) > 0).long(),
            "health": torch.from_numpy(observation[:, 1:2]).float()
            * 100,  # Denormalize
            "power": torch.from_numpy(observation[:, 2:3]).float() * 100,  # Denormalize
            "position": torch.complex(
                torch.from_numpy(observation[:, 3]).float()
                * self.base_env.world_size[0],
                torch.from_numpy(observation[:, 4]).float()
                * self.base_env.world_size[1],
            ).unsqueeze(-1),
            "velocity": torch.complex(
                torch.from_numpy(observation[:, 5]).float() * 180.0,
                torch.from_numpy(observation[:, 6]).float() * 180.0,
            ).unsqueeze(-1),
            "attitude": torch.complex(
                torch.from_numpy(observation[:, 7]).float(),
                torch.from_numpy(observation[:, 8]).float(),
            ).unsqueeze(-1),
            "is_shooting": torch.from_numpy(observation[:, 9:10]).long(),
        }

        return obs_dict

    def _is_ship_alive(self, observation: np.ndarray, ship_id: int) -> bool:
        """Check if a ship is alive"""
        if ship_id >= observation.shape[0]:
            return False
        # Health is normalized, so > 0 means alive
        return observation[ship_id, 1] > 0

    def _create_ship_mask(self, observation: np.ndarray) -> torch.Tensor:
        """Create mask for inactive ships"""
        batch_size = 1
        max_ships = observation.shape[0]
        mask = torch.ones(batch_size, max_ships, dtype=torch.bool)

        for ship_id in range(max_ships):
            if self._is_ship_alive(observation, ship_id):
                mask[0, ship_id] = False

        return mask

    def _flatten_team_actions(self, team_actions: dict) -> np.ndarray:
        """Convert team action dict to flattened array for SB3"""
        flattened = []
        for ship_id in sorted(self.controlled_ships):
            if ship_id in team_actions:
                action = team_actions[ship_id]
                if isinstance(action, torch.Tensor):
                    action = action.numpy()
                flattened.extend(action)
            else:
                flattened.extend([0] * 6)  # Default no-action
        return np.array(flattened, dtype=np.float32)

    def _unflatten_actions(self, flattened_actions: np.ndarray) -> dict:
        """Convert flattened SB3 actions back to ship action dict"""
        actions = {}
        for i, ship_id in enumerate(sorted(self.controlled_ships)):
            start_idx = i * 6
            end_idx = start_idx + 6
            ship_action = flattened_actions[start_idx:end_idx]
            actions[ship_id] = torch.from_numpy(ship_action).float()
        return actions

    def reset(self, **kwargs):
        """Reset environment and choose opponent type"""
        self.episode_count += 1
        self.current_episode_type = self._choose_episode_type()

        # Reset base environment
        obs_dict, info = self.base_env.reset()
        observation = obs_dict["tokens"].numpy()

        # Add episode type to info
        info.update(
            {
                "episode_type": self.current_episode_type,
                "episode_count": self.episode_count,
                "win_rate": self.wins / max(1, self.episode_count - 1),
            }
        )

        return observation, info

    def step(self, action):
        """Step environment with team-based actions"""
        # Convert SB3 action to team actions
        team_actions = self._unflatten_actions(action)

        # Get current observation for opponent actions
        obs_dict = self.base_env.get_observation()
        observation = obs_dict["tokens"].numpy()
        opponent_actions = self._get_opponent_actions(observation)

        # Combine all actions
        all_actions = {**team_actions, **opponent_actions}

        # Step environment
        obs_dict, rewards, terminated, truncated, info = self.base_env.step(all_actions)

        # Calculate team reward (sum of controlled ships)
        team_reward = sum(rewards.get(ship_id, 0) for ship_id in self.controlled_ships)

        # Track wins/losses
        if terminated:
            our_team_alive = any(
                info["ship_states"][ship_id]["alive"]
                for ship_id in self.controlled_ships
                if ship_id in info["ship_states"]
            )
            opponent_teams_alive = any(
                info["ship_states"][ship_id]["alive"]
                for team_id, ship_ids in self.team_assignments.items()
                if team_id != self.team_id
                for ship_id in ship_ids
                if ship_id in info["ship_states"]
            )

            if our_team_alive and not opponent_teams_alive:
                self.wins += 1
            elif not our_team_alive and opponent_teams_alive:
                self.losses += 1
            # else: draw (both dead or both alive - shouldn't happen)

        # Track opponent updates
        self.steps_since_opponent_update += 1

        # Convert observation
        observation = obs_dict["tokens"].numpy()

        # Add extra info
        info.update(
            {
                "episode_type": self.current_episode_type,
                "team_reward": team_reward,
                "controlled_ships_alive": sum(
                    1
                    for ship_id in self.controlled_ships
                    if ship_id in info.get("ship_states", {})
                    and info["ship_states"][ship_id]["alive"]
                ),
            }
        )

        return observation, team_reward, terminated, truncated, info

    def add_model_to_memory(self, model: TeamTransformerModel):
        """Add a model snapshot to self-play memory"""
        model_copy = deepcopy(model.state_dict())
        self.self_play_memory.append(model_copy)

        # Update current opponent if it's time
        if (
            self.steps_since_opponent_update >= self.opponent_update_freq
            or self.current_opponent_model is None
        ):
            self._update_opponent_model()
            self.steps_since_opponent_update = 0

    def _update_opponent_model(self):
        """Update the current opponent model from memory"""
        if len(self.self_play_memory) > 0:
            # Choose random model from memory
            opponent_state_dict = random.choice(self.self_play_memory)

            # Create new model instance and load weights
            if self.current_opponent_model is None:
                # Create opponent model with same architecture
                from team_transformer_model import create_team_model

                self.current_opponent_model = create_team_model(
                    {
                        "max_ships": self.base_env.max_ships,
                    }
                )

            self.current_opponent_model.load_state_dict(opponent_state_dict)
            self.current_opponent_model.eval()
            print(f"Updated opponent model (memory size: {len(self.self_play_memory)})")

    def get_win_rate(self) -> float:
        """Get current win rate"""
        if self.episode_count <= 1:
            return 0.5
        return self.wins / (self.episode_count - 1)

    def close(self):
        """Close environment"""
        self.base_env.close()
