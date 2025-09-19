"""
Clean Team Environment Wrapper for StableBaselines3 integration.
Converts multi-agent team environment to single-agent interface.
"""

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from typing import Dict, Optional
import random
from collections import deque
from copy import deepcopy

from env import Environment
from scripted_agent import ScriptedAgent
from team_transformer_model import TeamTransformerModel, TeamController


class OpponentController:
    """Handles different types of opponents (scripted, self-play, random)"""

    def __init__(
        self,
        team_assignments: dict,
        opponent_team_ids: list,
        world_size: tuple,
        scripted_config: dict,
    ):
        self.team_assignments = team_assignments
        self.opponent_team_ids = opponent_team_ids
        self.scripted_agents = self._create_scripted_agents(world_size, scripted_config)

        # Self-play components
        self.self_play_model = None
        self.team_controller = TeamController(team_assignments)

    def _create_scripted_agents(
        self, world_size: tuple, config: dict
    ) -> Dict[int, ScriptedAgent]:
        """Create scripted agents for all opponent ships"""
        agents = {}
        for team_id in self.opponent_team_ids:
            for ship_id in self.team_assignments[team_id]:
                agents[ship_id] = ScriptedAgent(
                    controlled_ship_id=ship_id, world_size=world_size, **config
                )
        return agents

    def get_scripted_actions(self, obs_dict: dict) -> Dict[int, torch.Tensor]:
        """Get actions from scripted agents"""
        actions = {}
        for ship_id, agent in self.scripted_agents.items():
            if self._is_ship_alive(obs_dict, ship_id):
                actions[ship_id] = agent(obs_dict)
            else:
                actions[ship_id] = torch.zeros(6, dtype=torch.float32)
        return actions

    def get_selfplay_actions(self, obs_dict: dict) -> Dict[int, torch.Tensor]:
        """Get actions from self-play model"""
        if self.self_play_model is None:
            return self._get_random_actions()

        try:
            tokens = obs_dict["tokens"].numpy()
            ship_mask = self._create_ship_mask(tokens)

            with torch.no_grad():
                obs_batch = {"tokens": torch.from_numpy(tokens).unsqueeze(0).float()}
                all_actions = self.self_play_model.get_actions(
                    obs_batch, ship_mask, deterministic=False
                )["actions"][0]

                # Extract opponent actions
                opponent_actions = {}
                for team_id in self.opponent_team_ids:
                    team_actions = self.team_controller.extract_team_actions(
                        all_actions.unsqueeze(0), team_id
                    )
                    opponent_actions.update(team_actions)

                return opponent_actions

        except Exception as e:
            print(f"Self-play model error: {e}, using random actions")
            return self._get_random_actions()

    def get_random_actions(self) -> Dict[int, torch.Tensor]:
        """Get random actions for all opponent ships"""
        return self._get_random_actions()

    def _get_random_actions(self) -> Dict[int, torch.Tensor]:
        """Helper for random actions"""
        actions = {}
        for team_id in self.opponent_team_ids:
            for ship_id in self.team_assignments[team_id]:
                actions[ship_id] = torch.randint(0, 2, (6,), dtype=torch.float32)
        return actions

    def _is_ship_alive(self, obs_dict: dict, ship_id: int) -> bool:
        """Check if ship is alive"""
        if ship_id >= obs_dict["alive"].shape[0]:
            return False
        return obs_dict["alive"][ship_id, 0].item() > 0

    def _create_ship_mask(self, tokens: np.ndarray) -> torch.Tensor:
        """Create mask for inactive ships"""
        batch_size = 1
        max_ships = tokens.shape[0]
        mask = torch.ones(batch_size, max_ships, dtype=torch.bool)

        # Mark alive ships as False (not masked)
        for ship_id in range(max_ships):
            if tokens[ship_id, 1] > 0:  # Health > 0
                mask[0, ship_id] = False

        return mask

    def update_selfplay_model(self, model: TeamTransformerModel):
        """Update the self-play opponent model"""
        self.self_play_model = deepcopy(model)
        self.self_play_model.eval()


class EpisodeManager:
    """Manages episode types and opponent selection"""

    def __init__(self, opponent_type: str, scripted_mix_ratio: float):
        self.opponent_type = opponent_type
        self.scripted_mix_ratio = scripted_mix_ratio
        self.current_episode_type = "scripted"

        # Stats tracking
        self.episode_count = 0
        self.wins = 0
        self.losses = 0

    def start_new_episode(self) -> str:
        """Start new episode and choose opponent type"""
        self.episode_count += 1
        self.current_episode_type = self._choose_opponent_type()
        return self.current_episode_type

    def _choose_opponent_type(self) -> str:
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
            return "scripted"

    def record_outcome(self, outcome_reward: float):
        """Record win/loss from outcome reward"""
        if outcome_reward > 0.5:
            self.wins += 1
        elif outcome_reward < -0.5:
            self.losses += 1

    def get_win_rate(self) -> float:
        """Get current win rate"""
        if self.episode_count <= 1:
            return 0.5
        return self.wins / (self.episode_count - 1)


class TeamEnvironmentWrapper(gym.Env):
    """
    Clean wrapper that converts multi-agent team environment to single-agent interface.
    """

    def __init__(
        self,
        env_config: dict = None,
        team_id: int = 0,
        team_assignments: dict = None,
        opponent_type: str = "scripted",  # "scripted", "self_play", "mixed"
        scripted_agent_config: dict = None,
        self_play_memory_size: int = 100,
        opponent_update_freq: int = 1000,
        scripted_mix_ratio: float = 0.3,
    ):
        super().__init__()

        # Environment setup
        env_config = env_config or {}
        self.base_env = Environment(**env_config)

        # Team configuration
        self.team_id = team_id
        self.team_assignments = team_assignments or {0: [0], 1: [1]}
        self.controlled_ships = self.team_assignments[team_id]

        # Find opponent team IDs
        opponent_team_ids = [
            tid for tid in self.team_assignments.keys() if tid != team_id
        ]

        # Initialize components
        scripted_config = scripted_agent_config or {
            "max_shooting_range": 500.0,
            "angle_threshold": 5.0,
            "bullet_speed": 500.0,
            "target_radius": 10.0,
            "radius_multiplier": 1.5,
        }

        self.opponent_controller = OpponentController(
            self.team_assignments,
            opponent_team_ids,
            env_config.get("world_size", (1200, 800)),
            scripted_config,
        )

        self.episode_manager = EpisodeManager(opponent_type, scripted_mix_ratio)

        # Self-play memory
        self.self_play_memory = deque(maxlen=self_play_memory_size)
        self.opponent_update_freq = opponent_update_freq
        self.steps_since_opponent_update = 0

        # Observation and action spaces
        self._setup_spaces()

    def _setup_spaces(self):
        """Setup observation and action spaces for SB3"""
        max_ships = self.base_env.max_ships
        token_dim = 10

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(max_ships, token_dim), dtype=np.float32
        )

        num_controlled_ships = len(self.controlled_ships)
        num_actions_per_ship = 6

        self.action_space = spaces.MultiBinary(
            num_controlled_ships * num_actions_per_ship
        )

    def _get_opponent_actions(self, obs_dict: dict) -> Dict[int, torch.Tensor]:
        """Get opponent actions - now clean and simple!"""
        episode_type = self.episode_manager.current_episode_type

        if episode_type == "scripted":
            return self.opponent_controller.get_scripted_actions(obs_dict)
        elif episode_type == "self_play":
            return self.opponent_controller.get_selfplay_actions(obs_dict)
        else:
            return self.opponent_controller.get_random_actions()

    def _unflatten_actions(
        self, flattened_actions: np.ndarray
    ) -> Dict[int, torch.Tensor]:
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
        episode_type = self.episode_manager.start_new_episode()

        # Reset base environment
        obs_dict, info = self.base_env.reset(
            game_mode="nvn"
        )  # Always use nvn for variety
        observation = obs_dict["tokens"].numpy()

        # Add episode info
        info.update(
            {
                "episode_type": episode_type,
                "episode_count": self.episode_manager.episode_count,
                "win_rate": self.episode_manager.get_win_rate(),
            }
        )

        return observation, info

    def step(self, action):
        """Step environment - much cleaner now!"""
        # Get our team's actions
        team_actions = self._unflatten_actions(action)

        # Get full observation and opponent actions
        obs_dict = self.base_env.get_observation()
        opponent_actions = self._get_opponent_actions(obs_dict)

        # Combine and step
        all_actions = {**team_actions, **opponent_actions}
        obs_dict, _, terminated, truncated, info = self.base_env.step(all_actions)

        # Calculate rewards
        current_state = self.base_env.state[-1]
        team_reward = self.base_env._calculate_team_reward(
            current_state, self.team_id, episode_ended=terminated
        )

        # Record episode outcome
        if terminated:
            outcome_reward = self.base_env._calculate_outcome_rewards(
                current_state, self.team_id
            )
            self.episode_manager.record_outcome(outcome_reward)

        # Track self-play updates
        self.steps_since_opponent_update += 1

        # Prepare return values
        observation = obs_dict["tokens"].numpy()
        info.update(
            {
                "episode_type": self.episode_manager.current_episode_type,
                "team_reward": team_reward,
                "controlled_ships_alive": sum(
                    1
                    for ship_id in self.controlled_ships
                    if ship_id < obs_dict["alive"].shape[0]
                    and obs_dict["alive"][ship_id, 0].item() > 0
                ),
            }
        )

        return observation, team_reward, terminated, truncated, info

    def add_model_to_memory(self, model: TeamTransformerModel):
        """Add a model snapshot to self-play memory"""
        model_state = deepcopy(model.state_dict())
        self.self_play_memory.append(model_state)

        # Update opponent model if it's time
        if (
            self.steps_since_opponent_update >= self.opponent_update_freq
            or self.opponent_controller.self_play_model is None
        ):
            self._update_opponent_model()
            self.steps_since_opponent_update = 0

    def _update_opponent_model(self):
        """Update the opponent model from memory"""
        if len(self.self_play_memory) > 0:
            # Choose random model from memory
            opponent_state = random.choice(self.self_play_memory)

            # Create and load model
            if self.opponent_controller.self_play_model is None:
                from team_transformer_model import create_team_model

                model = create_team_model({"max_ships": self.base_env.max_ships})
            else:
                model = self.opponent_controller.self_play_model

            model.load_state_dict(opponent_state)
            self.opponent_controller.update_selfplay_model(model)

            print(f"Updated opponent model (memory size: {len(self.self_play_memory)})")

    def get_win_rate(self) -> float:
        """Get current win rate"""
        return self.episode_manager.get_win_rate()

    def close(self):
        """Close environment"""
        self.base_env.close()
