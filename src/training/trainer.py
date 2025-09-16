"""
Main training module for Aves Horizons using StableBaselines3.

Provides high-level training interface with PPO and self-play.
"""

import torch
from typing import Optional, Dict, Any
from pathlib import Path
import json
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy

from .config import TrainingConfig
from .environment import DeathmatchSelfPlayEnv  
from .policy import create_policy_class
from .selfplay import SelfPlayCallback
from .callbacks import TrainingMetricsCallback, SelfPlayManagementCallback


class AvesHorizonsTrainer:
    """
    Main trainer class for Aves Horizons using StableBaselines3 PPO.
    
    Handles environment creation, model initialization, training loop, and self-play management.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Create training environment
        self.env = DeathmatchSelfPlayEnv(config)
        
        # Create evaluation environment
        self.eval_env = DeathmatchSelfPlayEnv(config)
        
        # Create custom policy class with our configuration
        policy_class = create_policy_class(config)
        
        # Create PPO model with custom policy
        self.model = PPO(
            policy=policy_class,
            env=self.env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            clip_range_vf=config.clip_range_vf,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            target_kl=config.target_kl,
            tensorboard_log=str(config.tensorboard_log),
            device=config.device,
            verbose=1
        )
        
        # Setup logging
        self._setup_logging()
        
        # Training state
        self.total_timesteps = 0
        self.best_mean_reward = float('-inf')
        
    def _setup_logging(self):
        """Setup logging configuration."""
        # Configure SB3 logger
        new_logger = configure(str(self.config.log_dir), ["stdout", "csv", "tensorboard"])
        self.model.set_logger(new_logger)
        
        print(f"Logging to: {self.config.log_dir}")
        print(f"Tensorboard logs: {self.config.tensorboard_log}")
    
    def _create_callbacks(self):
        """Create training callbacks."""
        callbacks = []
        
        # Checkpoint callback - saves model periodically
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.save_freq,
            save_path=str(self.config.model_dir / "checkpoints"),
            name_prefix="aves_horizons_model",
            save_replay_buffer=False,  # We don't use replay buffer in PPO
            save_vecnormalize=False
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback - evaluates and saves best model
        eval_callback = EvalCallback(
            eval_env=self.eval_env,
            best_model_save_path=str(self.config.model_dir / "best_model"),
            log_path=str(self.config.log_dir / "evaluations"),
            eval_freq=self.config.eval_freq,
            n_eval_episodes=self.config.eval_episodes,
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
        
        # Self-play management callback
        selfplay_callback = SelfPlayManagementCallback(
            env=self.env,
            config=self.config
        )
        callbacks.append(selfplay_callback)
        
        # Training metrics callback
        metrics_callback = TrainingMetricsCallback(
            config=self.config
        )
        callbacks.append(metrics_callback)
        
        return CallbackList(callbacks)
    
    def train(self, total_timesteps: int = None):
        """
        Run training for specified number of timesteps.
        
        Args:
            total_timesteps: Number of timesteps to train for. If None, uses config default.
        """
        if total_timesteps is None:
            total_timesteps = self.config.total_timesteps
            
        print(f"Starting training for {total_timesteps:,} timesteps")
        print(f"Environment: Deathmatch Self-Play ({self.config.n_teams} teams, {self.config.ships_per_team} ships/team)")
        print(f"Model: Transformer (d_model={self.config.d_model}, layers={self.config.num_layers}, heads={self.config.n_head})")
        print(f"Training: PPO (lr={self.config.learning_rate}, batch_size={self.config.batch_size}, n_steps={self.config.n_steps})")
        
        # Create callbacks
        callbacks = self._create_callbacks()
        
        # Start training
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                log_interval=self.config.log_interval,
                reset_num_timesteps=False  # Continue from current timestep count
            )
            
            self.total_timesteps += total_timesteps
            
            print(f"Training completed! Total timesteps: {self.total_timesteps:,}")
            
            # Save final model
            final_model_path = self.config.model_dir / "final_model"
            self.model.save(str(final_model_path))
            print(f"Final model saved: {final_model_path}")
            
        except KeyboardInterrupt:
            print("Training interrupted by user")
            # Save current model
            interrupt_model_path = self.config.model_dir / f"interrupted_model_step_{self.total_timesteps}"
            self.model.save(str(interrupt_model_path))
            print(f"Model saved before interruption: {interrupt_model_path}")
            raise
    
    def evaluate(self, n_eval_episodes: int = None, deterministic: bool = True) -> Dict[str, float]:
        """
        Evaluate the current model.
        
        Args:
            n_eval_episodes: Number of episodes to evaluate. If None, uses config default.
            deterministic: Whether to use deterministic actions during evaluation.
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        if n_eval_episodes is None:
            n_eval_episodes = self.config.eval_episodes
            
        print(f"Evaluating model for {n_eval_episodes} episodes...")
        
        mean_reward, std_reward = evaluate_policy(
            model=self.model,
            env=self.eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            render=False,
            return_episode_rewards=False
        )
        
        results = {
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "n_eval_episodes": n_eval_episodes
        }
        
        print(f"Evaluation results: Mean reward = {mean_reward:.2f} ± {std_reward:.2f}")
        
        return results
    
    def save_model(self, path: Path) -> Path:
        """
        Save the current model.
        
        Args:
            path: Path where to save the model
            
        Returns:
            Full path to saved model
        """
        path = Path(path)
        if path.suffix != '.zip':
            path = path.with_suffix('.zip')
            
        self.model.save(str(path))
        print(f"Model saved: {path}")
        return path
    
    def load_model(self, path: Path):
        """
        Load a model from disk.
        
        Args:
            path: Path to the saved model
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
            
        self.model = PPO.load(str(path), env=self.env)
        print(f"Model loaded: {path}")
    
    def get_training_info(self) -> Dict[str, Any]:
        """
        Get information about the current training state.
        
        Returns:
            Dictionary with training information.
        """
        return {
            "total_timesteps": self.total_timesteps,
            "config": self.config.__dict__,
            "opponent_pool_size": len(self.env.opponent_pool),
            "current_opponent_policy": self.env.current_opponent_policy,
            "episode_count": self.env.episode_count
        }
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'env'):
            self.env.close()
        if hasattr(self, 'eval_env'):
            self.eval_env.close()


def create_trainer_from_config_file(config_file: Path) -> AvesHorizonsTrainer:
    """
    Create trainer from a JSON configuration file.
    
    Args:
        config_file: Path to JSON configuration file
        
    Returns:
        Configured trainer instance
    """
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    
    config = TrainingConfig(**config_dict)
    return AvesHorizonsTrainer(config)