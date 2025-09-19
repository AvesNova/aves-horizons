"""
Custom SB3 policy that integrates the transformer model.
"""

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import BernoulliDistribution
from stable_baselines3 import PPO
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Type, Union

from team_transformer_model import TeamTransformerModel


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor that uses the transformer model.
    Extracts ship embeddings for the policy and value networks.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        transformer_config: dict = None,
        team_id: int = 0,
        team_assignments: dict = None,
    ):
        # Calculate features dim based on controlled ships
        self.team_assignments = team_assignments or {0: [0], 1: [1]}
        self.controlled_ships = self.team_assignments[team_id]
        self.num_controlled_ships = len(self.controlled_ships)
        self.team_id = team_id

        transformer_config = transformer_config or {}
        embed_dim = transformer_config.get("embed_dim", 64)

        # Features = embeddings of controlled ships flattened
        features_dim = self.num_controlled_ships * embed_dim

        super().__init__(observation_space, features_dim)

        # Create transformer model
        self.transformer = TeamTransformerModel(**transformer_config)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features using transformer.

        Args:
            observations: (batch_size, max_ships, token_dim)

        Returns:
            features: (batch_size, features_dim) - flattened embeddings of controlled ships
        """
        batch_size = observations.shape[0]

        # Prepare observation dict for transformer
        obs_dict = {"tokens": observations}

        # Create ship mask (True for inactive ships)
        max_ships = observations.shape[1]
        ship_mask = torch.ones(
            batch_size, max_ships, dtype=torch.bool, device=observations.device
        )

        # Mark active ships as False (not masked)
        for ship_id in range(max_ships):
            # Check if ship is alive (health > 0, assuming health is token[1])
            alive_mask = observations[:, ship_id, 1] > 0  # (batch_size,)
            ship_mask[:, ship_id] = ~alive_mask

        # Forward through transformer
        output = self.transformer(obs_dict, ship_mask)
        ship_embeddings = output[
            "ship_embeddings"
        ]  # (batch_size, max_ships, embed_dim)

        # Extract embeddings for controlled ships only
        controlled_embeddings = []
        for ship_id in sorted(self.controlled_ships):
            if ship_id < max_ships:
                controlled_embeddings.append(ship_embeddings[:, ship_id, :])
            else:
                # Pad with zeros if ship_id out of bounds
                embed_dim = ship_embeddings.shape[-1]
                controlled_embeddings.append(
                    torch.zeros(batch_size, embed_dim, device=observations.device)
                )

        # Flatten controlled ship embeddings
        features = torch.cat(
            controlled_embeddings, dim=1
        )  # (batch_size, num_controlled * embed_dim)

        return features


class TransformerActorCriticPolicy(BasePolicy):
    """
    Custom Actor-Critic policy using transformer features.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        transformer_config: dict = None,
        team_id: int = 0,
        team_assignments: dict = None,
        **kwargs
    ):
        self.transformer_config = transformer_config or {}
        self.team_id = team_id
        self.team_assignments = team_assignments or {0: [0], 1: [1]}
        self.controlled_ships = self.team_assignments[team_id]

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class=TransformerFeaturesExtractor,
            features_extractor_kwargs={
                "transformer_config": transformer_config,
                "team_id": team_id,
                "team_assignments": team_assignments,
            },
            **kwargs
        )

        # Action space should be MultiBinary
        assert isinstance(action_space, spaces.MultiBinary)
        self.action_dim = action_space.n

        # Get features dimension
        features_dim = self.features_extractor.features_dim

        # Actor network (policy)
        self.action_net = nn.Sequential(
            nn.Linear(features_dim, features_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim // 2, self.action_dim),
        )

        # Critic network (value function)
        self.value_net = nn.Sequential(
            nn.Linear(features_dim, features_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim // 2, 1),
        )

        # Action distribution
        self.action_dist = BernoulliDistribution(self.action_dim)

    def _get_constructor_parameters(self) -> Dict[str, any]:
        data = super()._get_constructor_parameters()
        data.update(
            {
                "transformer_config": self.transformer_config,
                "team_id": self.team_id,
                "team_assignments": self.team_assignments,
            }
        )
        return data

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for both actor and critic.

        Returns:
            actions, values, log_probs
        """
        features = self.extract_features(obs)

        # Get action logits and value
        action_logits = self.action_net(features)
        values = self.value_net(features)

        # Create distribution and sample actions
        self.action_dist = self.action_dist.proba_distribution(action_logits)
        actions = self.action_dist.get_actions(deterministic=deterministic)
        log_probs = self.action_dist.log_prob(actions)

        return actions, values, log_probs

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO updates.

        Returns:
            values, log_probs, entropy
        """
        features = self.extract_features(obs)

        action_logits = self.action_net(features)
        values = self.value_net(features)

        self.action_dist = self.action_dist.proba_distribution(action_logits)
        log_probs = self.action_dist.log_prob(actions)
        entropy = self.action_dist.entropy()

        return values, log_probs, entropy

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Predict values for critic.
        """
        features = self.extract_features(obs)
        return self.value_net(features)

    def get_transformer_model(self) -> TeamTransformerModel:
        """Get the underlying transformer model for self-play memory."""
        return self.features_extractor.transformer


def create_team_ppo_model(
    env,
    transformer_config: dict = None,
    team_id: int = 0,
    team_assignments: dict = None,
    ppo_config: dict = None,
):
    """
    Create PPO model with transformer policy.

    Args:
        env: The wrapped environment
        transformer_config: Config for transformer model
        team_id: Which team this policy controls
        team_assignments: Team assignments dict
        ppo_config: Config for PPO algorithm

    Returns:
        PPO model ready for training
    """

    # Default configs
    transformer_config = transformer_config or {
        "token_dim": 10,
        "embed_dim": 64,
        "num_heads": 4,
        "num_layers": 3,
        "max_ships": 4,
        "num_actions": 6,
        "dropout": 0.1,
    }

    ppo_config = ppo_config or {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }

    # Create policy kwargs
    policy_kwargs = {
        "transformer_config": transformer_config,
        "team_id": team_id,
        "team_assignments": team_assignments,
    }

    # Create PPO model
    model = PPO(
        policy=TransformerActorCriticPolicy,
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        **ppo_config
    )

    return model
