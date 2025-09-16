"""
Training Infrastructure for ShipTransformer Model.

This module provides comprehensive training support for the transformer-based
ship combat AI, including temporal sequence handling, multi-agent coordination,
and specialized policy implementations.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import os
import json
from pathlib import Path
from dataclasses import asdict

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.distributions import DiagGaussianDistribution, MultiCategoricalDistribution
from stable_baselines3.common.type_aliases import Schedule
from gymnasium import spaces
import torch.nn.functional as F

from gym_env.ship_transformer_env import ShipTransformerEnv, MultiGameShipTransformerEnv
from models.ship_transformer import ShipTransformerMVP
from models.state_history import StateHistory
from models.token_encoder import ShipTokenEncoder
from utils.config import Actions, ModelConfig


class ShipTransformerPolicy(ActorCriticPolicy):
    """
    Custom policy for ShipTransformer that handles temporal sequences.
    
    This policy integrates the transformer model into the stable-baselines3
    framework, handling the conversion between gym observations and
    transformer token sequences.
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        # Transformer parameters
        d_model: int = ModelConfig.DEFAULT_D_MODEL,
        nhead: int = ModelConfig.DEFAULT_N_HEAD,
        num_layers: int = ModelConfig.DEFAULT_NUM_LAYERS,
        sequence_length: int = ModelConfig.DEFAULT_SEQUENCE_LENGTH,
        n_ships: int = ModelConfig.DEFAULT_N_SHIPS,
        controlled_team_size: int = ModelConfig.DEFAULT_CONTROLLED_TEAM_SIZE,
        world_size: Tuple[float, float] = ModelConfig.DEFAULT_WORLD_SIZE,
        **kwargs
    ):
        # Store transformer parameters
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.n_ships = n_ships
        self.controlled_team_size = controlled_team_size
        self.world_size = world_size
        
        # Calculate feature dimensions
        token_dim = ModelConfig.TOKEN_DIM  # Base token dimension
        self.seq_len = sequence_length * n_ships
        self.obs_dim = self.seq_len * token_dim
        
        # Initialize token encoder for observation processing
        self.token_encoder = ShipTokenEncoder(
            world_size=world_size,
            normalize_coordinates=True,
            max_ships=n_ships
        )
        
        # Create network architecture
        net_arch = {
            "pi": [256, 128],  # Policy network
            "vf": [256, 128],  # Value network
        }
        
        # Filter kwargs for parent class
        parent_kwargs = {k: v for k, v in kwargs.items() 
                        if k not in ['d_model', 'nhead', 'num_layers', 'sequence_length', 
                                   'n_ships', 'controlled_team_size', 'world_size']}
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=nn.ReLU,
            **parent_kwargs
        )
        
        # Create transformer model after parent initialization
        self.transformer = ShipTransformerMVP(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        )
        
        # Override the features dimension for the transformer output
        self.features_dim = d_model * n_ships
    
    def _build_mlp_extractor(self) -> None:
        """Build MLP networks with correct input size for transformer features."""
        from stable_baselines3.common.torch_layers import MlpExtractor
        
        self.mlp_extractor = MlpExtractor(
            feature_dim=self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )
    
    def _process_observation(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process gym observation into transformer-compatible format.
        
        Args:
            obs: Flattened observation tensor [batch_size, obs_dim]
            
        Returns:
            tokens: [batch_size, seq_len, {}] Token tensor
            ship_ids: [batch_size, seq_len] Ship ID tensor".format(ModelConfig.TOKEN_DIM)
        """
        batch_size = obs.shape[0] if len(obs.shape) > 1 else 1
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        
        # Reshape observation to token format
        tokens = obs.view(batch_size, self.seq_len, ModelConfig.TOKEN_DIM)
        
        # Create ship IDs (time-major order)
        ship_ids = torch.zeros(batch_size, self.seq_len, dtype=torch.long, device=obs.device)
        for t in range(self.sequence_length):
            for ship_id in range(self.n_ships):
                idx = t * self.n_ships + ship_id
                if idx < self.seq_len:
                    ship_ids[:, idx] = ship_id
        
        return tokens, ship_ids
    
    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract features using transformer."""
        tokens, ship_ids = self._process_observation(obs)
        
        # Pass through transformer
        # The transformer outputs actions for all ships, but we extract features
        transformer_output = self.transformer(tokens, ship_ids)  # [batch, n_ships, 6]
        
        # Flatten transformer output to use as features
        features = transformer_output.view(obs.shape[0], -1)  # [batch, n_ships * 6]
        
        return features
    
    def _predict_action_logits(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Predict action logits directly from transformer.
        
        Args:
            obs: Observation tensor
            
        Returns:
            action_logits: [batch_size, controlled_team_size * 6] Action logits
        """
        tokens, ship_ids = self._process_observation(obs)
        
        # Get transformer output for all ships
        all_ship_actions = self.transformer(tokens, ship_ids)  # [batch, n_ships, 6]
        
        # Extract actions for controlled ships only
        # Assuming controlled ships are the first controlled_team_size ships
        controlled_actions = all_ship_actions[:, :self.controlled_team_size, :]  # [batch, controlled_team_size, 6]
        
        # Flatten to match action space
        action_logits = controlled_actions.view(obs.shape[0], -1)  # [batch, controlled_team_size * 6]
        
        return action_logits
    
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for policy.
        
        Args:
            obs: Observation tensor
            deterministic: Whether to use deterministic actions
            
        Returns:
            actions: Sampled actions
            values: State values
            log_probs: Log probabilities of actions
        """
        # Get action logits directly from transformer
        action_logits = self._predict_action_logits(obs)
        
        # Convert to probabilities (sigmoid for multi-binary)
        action_probs = torch.sigmoid(action_logits)
        
        # Sample actions
        if deterministic:
            actions = (action_probs > 0.5).float()
        else:
            actions = torch.bernoulli(action_probs)
        
        # Calculate log probabilities
        log_probs = torch.sum(
            actions * torch.log(action_probs + 1e-8) + 
            (1 - actions) * torch.log(1 - action_probs + 1e-8),
            dim=1
        )
        
        # Get value estimate (use transformer features)
        features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        values = self.value_net(latent_vf)
        
        return actions, values, log_probs
    
    def get_distribution(self, obs: torch.Tensor):
        """Get action distribution (required by stable-baselines3)."""
        action_logits = self._predict_action_logits(obs)
        
        # Create a custom distribution for multi-binary actions
        return CustomMultiBinaryDistribution(action_logits)
    
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict state values."""
        features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for training."""
        # Get action logits
        action_logits = self._predict_action_logits(obs)
        action_probs = torch.sigmoid(action_logits)
        
        # Calculate log probabilities
        log_probs = torch.sum(
            actions * torch.log(action_probs + 1e-8) + 
            (1 - actions) * torch.log(1 - action_probs + 1e-8),
            dim=1
        )
        
        # Get values
        values = self.predict_values(obs)
        
        # Calculate entropy
        entropy = -torch.sum(
            action_probs * torch.log(action_probs + 1e-8) + 
            (1 - action_probs) * torch.log(1 - action_probs + 1e-8),
            dim=1
        )
        
        return values, log_probs, entropy


class CustomMultiBinaryDistribution:
    """Custom distribution for multi-binary action spaces."""
    
    def __init__(self, action_logits: torch.Tensor):
        self.action_logits = action_logits
        self.action_probs = torch.sigmoid(action_logits)
    
    def sample(self) -> torch.Tensor:
        """Sample actions from distribution."""
        return torch.bernoulli(self.action_probs)
    
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Calculate log probability of actions."""
        return torch.sum(
            actions * torch.log(self.action_probs + 1e-8) + 
            (1 - actions) * torch.log(1 - self.action_probs + 1e-8),
            dim=1
        )
    
    def entropy(self) -> torch.Tensor:
        """Calculate entropy of distribution."""
        return -torch.sum(
            self.action_probs * torch.log(self.action_probs + 1e-8) + 
            (1 - self.action_probs) * torch.log(1 - self.action_probs + 1e-8),
            dim=1
        )
    
    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        """Get actions (deterministic or stochastic)."""
        if deterministic:
            return (self.action_probs > 0.5).float()
        else:
            return self.sample()


class TransformerTrainingCallback(BaseCallback):
    """Custom callback for transformer training with additional logging."""
    
    def __init__(self, eval_env, eval_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.eval_episode = 0
    
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_model()
        return True
    
    def _evaluate_model(self):
        """Evaluate model performance."""
        obs, _ = self.eval_env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        self.eval_episode += 1
        
        # Log results
        self.logger.record("eval/episode_reward", episode_reward)
        self.logger.record("eval/episode", self.eval_episode)
        self.logger.record("eval/controlled_alive", info.get("controlled_alive", 0))
        self.logger.record("eval/opponent_alive", info.get("opponent_alive", 0))
        
        if self.verbose > 0:
            print(f"Eval Episode {self.eval_episode}: Reward = {episode_reward:.2f}")


def make_transformer_env(
    n_ships: int = 8,
    controlled_team_size: int = 4,
    sequence_length: int = 6,
    render_mode: Optional[str] = None,
    **kwargs
) -> ShipTransformerEnv:
    """Create a ShipTransformerEnv with specified parameters."""
    return ShipTransformerEnv(
        n_ships=n_ships,
        controlled_team_size=controlled_team_size,
        sequence_length=sequence_length,
        render_mode=render_mode,
        **kwargs
    )


def train_ship_transformer(
    total_timesteps: int = 1000000,
    n_ships: int = 8,
    controlled_team_size: int = 4,
    sequence_length: int = 6,
    # Model parameters
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 3,
    # Training parameters
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    n_steps: int = 2048,
    n_epochs: int = 10,
    # Environment parameters
    opponent_policy: str = "heuristic",
    world_size: Tuple[float, float] = (1200.0, 800.0),
    # Logging parameters
    log_dir: str = "./logs/",
    model_save_path: str = "./models/",
    eval_freq: int = 10000,
    save_freq: int = 50000,
    verbose: int = 1
) -> PPO:
    """
    Train a ShipTransformer model using PPO.
    
    Args:
        total_timesteps: Total training timesteps
        n_ships: Number of ships in environment
        controlled_team_size: Number of ships controlled by agent
        sequence_length: Length of temporal sequences
        d_model: Transformer model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        n_steps: Steps per rollout
        n_epochs: Training epochs per rollout
        opponent_policy: Policy for opponent ships
        world_size: Environment world size
        log_dir: Directory for logs
        model_save_path: Path to save models
        eval_freq: Frequency of evaluation
        save_freq: Frequency of model saving
        verbose: Verbosity level
        
    Returns:
        Trained PPO model
    """
    # Create directories
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    
    # Create training environment
    train_env = make_transformer_env(
        n_ships=n_ships,
        controlled_team_size=controlled_team_size,
        sequence_length=sequence_length,
        opponent_policy=opponent_policy,
        world_size=world_size
    )
    
    # Create evaluation environment
    eval_env = make_transformer_env(
        n_ships=n_ships,
        controlled_team_size=controlled_team_size,
        sequence_length=sequence_length,
        opponent_policy=opponent_policy,
        world_size=world_size
    )
    
    # Initialize PPO with custom policy
    model = PPO(
        policy=ShipTransformerPolicy,
        env=train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "sequence_length": sequence_length,
            "n_ships": n_ships,
            "controlled_team_size": controlled_team_size,
            "world_size": world_size,
        },
        verbose=verbose,
        tensorboard_log=log_dir,
    )
    
    # Set up callbacks
    transformer_callback = TransformerTrainingCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        verbose=verbose
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=os.path.join(model_save_path, "checkpoints"),
        name_prefix="ship_transformer"
    )
    
    # Train the model
    print(f"Starting training with {total_timesteps} timesteps...")
    print(f"Model architecture: d_model={d_model}, nhead={nhead}, layers={num_layers}")
    print(f"Environment: {n_ships} ships, {controlled_team_size} controlled, {sequence_length} sequence length")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[transformer_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(model_save_path, "ship_transformer_final")
    model.save(final_model_path)
    
    # Save training configuration
    config = {
        "total_timesteps": total_timesteps,
        "n_ships": n_ships,
        "controlled_team_size": controlled_team_size,
        "sequence_length": sequence_length,
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "n_steps": n_steps,
        "n_epochs": n_epochs,
        "opponent_policy": opponent_policy,
        "world_size": world_size,
    }
    
    config_path = os.path.join(model_save_path, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Training completed! Model saved to {final_model_path}")
    print(f"Configuration saved to {config_path}")
    
    return model


def evaluate_transformer_model(
    model_path: str,
    n_episodes: int = 10,
    render: bool = True,
    deterministic: bool = True
) -> Dict[str, float]:
    """
    Evaluate a trained transformer model.
    
    Args:
        model_path: Path to saved model
        n_episodes: Number of episodes to evaluate
        render: Whether to render episodes
        deterministic: Whether to use deterministic actions
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Load model
    model = PPO.load(model_path)
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(model_path), "training_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create evaluation environment
    env = make_transformer_env(
        n_ships=config["n_ships"],
        controlled_team_size=config["controlled_team_size"],
        sequence_length=config["sequence_length"],
        opponent_policy=config.get("opponent_policy", "heuristic"),
        world_size=tuple(config["world_size"]),
        render_mode="human" if render else None
    )
    
    # Run evaluation episodes
    episode_rewards = []
    episode_lengths = []
    wins = 0
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Check if controlled team won
        if info.get("opponent_alive", 1) == 0 and info.get("controlled_alive", 0) > 0:
            wins += 1
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
        print(f"  Controlled alive: {info.get('controlled_alive', 0)}, Opponent alive: {info.get('opponent_alive', 0)}")
    
    env.close()
    
    # Calculate metrics
    metrics = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "win_rate": wins / n_episodes,
        "total_episodes": n_episodes
    }
    
    print(f"\nEvaluation Results ({n_episodes} episodes):")
    print(f"  Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"  Mean Length: {metrics['mean_length']:.1f}")
    print(f"  Win Rate: {metrics['win_rate']:.2%}")
    
    return metrics


if __name__ == "__main__":
    # Train model with MVP configuration
    print("Training ShipTransformer MVP...")
    
    model = train_ship_transformer(
        total_timesteps=500000,  # Start with shorter training for MVP
        n_ships=8,
        controlled_team_size=4,
        sequence_length=6,
        d_model=64,
        nhead=4,
        num_layers=3,
        opponent_policy="heuristic",
        verbose=1
    )
    
    # Evaluate trained model
    print("\nEvaluating trained model...")
    evaluate_transformer_model(
        model_path="./models/ship_transformer_final",
        n_episodes=5,
        render=True
    )