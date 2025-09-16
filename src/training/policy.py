"""
Custom PPO policy using transformer architecture for Aves Horizons.

Integrates the ShipTransformer model with StableBaselines3 PPO.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional, List
import gymnasium as gym
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule

from models.ship_nn import ShipNN
from utils.config import ModelConfig, Actions
from .config import TrainingConfig


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor using the ShipTransformer model.
    
    Converts flattened temporal observations into transformer features.
    """
    
    def __init__(
        self, 
        observation_space: gym.Space,
        config: TrainingConfig
    ):
        # Calculate the features dimension (transformer output size)
        features_dim = config.hidden_dim * config.total_ships
        super().__init__(observation_space, features_dim)
        
        self.config = config
        
        # Store dimensions for reshaping
        self.seq_len = config.sequence_length * config.total_ships
        self.token_dim = ModelConfig.TOKEN_DIM
        
        # Create the ShipNN model
        self.transformer = ShipNN(
            input_dim=self.token_dim,
            hidden_dim=config.hidden_dim,
            output_dim=6,  # Actions per ship
            max_ships=config.total_ships,
            sequence_length=config.sequence_length,
            encoder_layers=config.encoder_layers,
            transformer_layers=config.transformer_layers,
            decoder_layers=config.decoder_layers,
            n_heads=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=0.1,
            negative_slope=0.01
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from observations using the transformer.
        
        Args:
            observations: Flattened observations [batch_size, obs_dim]
            
        Returns:
            features: Transformer features [batch_size, features_dim]
        """
        batch_size = observations.shape[0]
        
        # Reshape observations to token format [batch_size, seq_len, token_dim]
        tokens = observations.view(batch_size, self.seq_len, self.token_dim)
        
        # Create ship IDs for transformer (time-major order)
        ship_ids = torch.zeros(batch_size, self.seq_len, dtype=torch.long, device=observations.device)
        for t in range(self.config.sequence_length):
            for ship_id in range(self.config.total_ships):
                idx = t * self.config.total_ships + ship_id
                if idx < self.seq_len:
                    ship_ids[:, idx] = ship_id
        
        # Pass through transformer with features extraction mode
        transformer_output = self.transformer(tokens, ship_ids, return_features=True)  # [batch, n_ships, d_model]
        
        # Flatten transformer output for use as features
        features = transformer_output.view(batch_size, -1)  # [batch, n_ships * d_model]
        
        return features


class TransformerActorCriticPolicy(ActorCriticPolicy):
    """
    Custom ActorCritic policy using transformer features extractor.
    
    This policy integrates the ShipTransformer model with PPO training.
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Schedule,
        config: TrainingConfig,
        **kwargs
    ):
        self.config = config
        
        # Use our custom features extractor
        features_extractor_class = TransformerFeaturesExtractor
        features_extractor_kwargs = {"config": config}
        
        # Network architecture for policy and value heads
        # Since we have rich transformer features, we can use simpler heads
        net_arch = {
            "pi": [256, 128],  # Policy head
            "vf": [256, 128],  # Value head
        }
        
        super().__init__(
            observation_space,
            action_space, 
            lr_schedule,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            net_arch=net_arch,
            activation_fn=nn.ReLU,
            **kwargs
        )
    
    def extract_features(self, obs: torch.Tensor, features_extractor: nn.Module = None) -> torch.Tensor:
        """
        Extract features from observations.
        
        Args:
            obs: Observations
            features_extractor: The features extractor to use (if None, uses self.features_extractor)
            
        Returns:
            features: Extracted features
        """
        if features_extractor is None:
            features_extractor = self.features_extractor
        return features_extractor(obs)
    
    def get_action_logits(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get action logits for controlled ships only.
        
        Args:
            obs: Observations
            
        Returns:
            action_logits: Logits for controlled ships [batch, controlled_ships * n_actions]
        """
        # Extract features using transformer
        features = self.extract_features(obs)
        
        # Get policy logits
        latent_pi = self.mlp_extractor.forward_actor(features)
        action_logits = self.action_net(latent_pi)
        
        return action_logits
    
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get value predictions.
        
        Args:
            obs: Observations
            
        Returns:
            values: Value predictions [batch, 1]
        """
        # Extract features using transformer
        features = self.extract_features(obs)
        
        # Get value predictions
        latent_vf = self.mlp_extractor.forward_critic(features)
        values = self.value_net(latent_vf)
        
        return values
    
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for both action prediction and value estimation.
        
        Args:
            obs: Observations
            deterministic: Whether to use deterministic actions
            
        Returns:
            actions: Sampled actions
            values: Value predictions  
            log_probs: Log probabilities of actions
        """
        # Get action distribution
        distribution = self.get_distribution(obs)
        actions = distribution.get_actions(deterministic=deterministic)
        log_probs = distribution.log_prob(actions)
        
        # Get values
        values = self.predict_values(obs)
        
        return actions, values, log_probs
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO updates.
        
        Args:
            obs: Observations
            actions: Actions to evaluate
            
        Returns:
            values: Value predictions
            log_probs: Log probabilities of actions
            entropy: Action distribution entropy
        """
        # Get action distribution
        distribution = self.get_distribution(obs)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        # Get values
        values = self.predict_values(obs)
        
        return values, log_probs, entropy


def create_policy_class(config: TrainingConfig):
    """
    Create a policy class with the given configuration.
    
    This allows us to pass configuration to the policy constructor.
    """
    class ConfiguredTransformerPolicy(TransformerActorCriticPolicy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, config=config, **kwargs)
    
    return ConfiguredTransformerPolicy