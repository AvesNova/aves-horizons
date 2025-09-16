"""
Custom callbacks for Aves Horizons training.

Provides specialized callbacks for self-play management and metrics tracking.
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback

from .config import TrainingConfig
from .selfplay import SelfPlayCallback


class SelfPlayManagementCallback(BaseCallback):
    """
    Callback to manage self-play opponent pool during training.
    
    Automatically saves models to the opponent pool at specified intervals.
    """
    
    def __init__(self, env, config: TrainingConfig, verbose: int = 1):
        super().__init__(verbose)
        self.env = env  # The training environment
        self.config = config
        
        # Initialize self-play callback
        self.selfplay_callback = SelfPlayCallback(
            opponent_pool=env.opponent_pool,
            save_freq=config.selfplay_update_freq,
            model_save_path=config.model_dir / "selfplay",
            verbose=verbose > 0
        )
    
    def _on_step(self) -> bool:
        """
        Called after each environment step.
        
        Returns:
            True to continue training, False to stop.
        """
        # Check if we should save current model to opponent pool
        if self.selfplay_callback.should_save_model(self.num_timesteps):
            # Get current training metrics for metadata
            additional_metadata = {}
            if hasattr(self.locals, 'infos') and self.locals['infos']:
                # Extract useful metrics from the last episode info
                last_info = self.locals['infos'][-1]
                if 'episode' in last_info:
                    additional_metadata = {
                        'episode_reward': last_info['episode']['r'],
                        'episode_length': last_info['episode']['l']
                    }
            
            # Save model to opponent pool
            self.selfplay_callback.save_model_to_pool(
                model=self.model,
                timestep=self.num_timesteps,
                additional_metadata=additional_metadata
            )
        
        return True


class TrainingMetricsCallback(BaseCallback):
    """
    Callback to track and log custom training metrics.
    
    Logs additional metrics specific to Aves Horizons training.
    """
    
    def __init__(self, config: TrainingConfig, verbose: int = 1):
        super().__init__(verbose)
        self.config = config
        self.metrics_history = []
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Metrics file
        self.metrics_file = config.log_dir / "custom_metrics.json"
    
    def _on_step(self) -> bool:
        """
        Called after each environment step.
        
        Returns:
            True to continue training, False to stop.
        """
        # Collect episode metrics when episodes end
        if hasattr(self.locals, 'infos') and self.locals['infos']:
            for info in self.locals['infos']:
                if 'episode' in info:
                    # Episode completed
                    episode_reward = info['episode']['r']
                    episode_length = info['episode']['l']
                    
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    
                    # Log to tensorboard if available
                    if self.logger:
                        self.logger.record("custom/episode_reward", episode_reward)
                        self.logger.record("custom/episode_length", episode_length)
                        
                        # Log environment-specific metrics
                        if 'controlled_alive' in info:
                            self.logger.record("custom/controlled_alive", info['controlled_alive'])
                        if 'opponent_alive' in info:
                            self.logger.record("custom/opponent_alive", info['opponent_alive'])
                        if 'opponent_policy' in info:
                            # Log opponent policy as a numeric value for tensorboard
                            policy_map = {"random": 0, "heuristic": 1, "selfplay": 2}
                            policy_value = policy_map.get(info['opponent_policy'], -1)
                            self.logger.record("custom/opponent_policy", policy_value)
        
        return True
    
    def _on_training_start(self) -> None:
        """Called when training starts."""
        if self.verbose > 0:
            print(f"Starting metrics tracking. Metrics will be saved to: {self.metrics_file}")
    
    def _on_training_end(self) -> None:
        """Called when training ends."""
        # Save final metrics to file
        final_metrics = {
            "total_episodes": len(self.episode_rewards),
            "mean_episode_reward": float(sum(self.episode_rewards) / len(self.episode_rewards)) if self.episode_rewards else 0,
            "mean_episode_length": float(sum(self.episode_lengths) / len(self.episode_lengths)) if self.episode_lengths else 0,
            "episode_rewards": self.episode_rewards[-100:],  # Last 100 episodes
            "episode_lengths": self.episode_lengths[-100:],
            "final_timesteps": self.num_timesteps
        }
        
        with open(self.metrics_file, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        if self.verbose > 0:
            print(f"Training metrics saved to: {self.metrics_file}")


class PerformanceMonitoringCallback(BaseCallback):
    """
    Callback to monitor performance and detect potential training issues.
    
    Provides early warnings for common training problems.
    """
    
    def __init__(self, config: TrainingConfig, verbose: int = 1):
        super().__init__(verbose)
        self.config = config
        self.recent_rewards = []
        self.reward_window_size = 100
        self.stagnation_threshold = 50  # Episodes without improvement
        self.stagnation_counter = 0
        self.best_recent_mean = float('-inf')
    
    def _on_step(self) -> bool:
        """
        Called after each environment step.
        
        Returns:
            True to continue training, False to stop.
        """
        # Collect reward data
        if hasattr(self.locals, 'infos') and self.locals['infos']:
            for info in self.locals['infos']:
                if 'episode' in info:
                    reward = info['episode']['r']
                    self.recent_rewards.append(reward)
                    
                    # Maintain window size
                    if len(self.recent_rewards) > self.reward_window_size:
                        self.recent_rewards.pop(0)
                    
                    # Check for performance stagnation
                    if len(self.recent_rewards) >= self.reward_window_size:
                        current_mean = sum(self.recent_rewards) / len(self.recent_rewards)
                        
                        if current_mean > self.best_recent_mean:
                            self.best_recent_mean = current_mean
                            self.stagnation_counter = 0
                        else:
                            self.stagnation_counter += 1
                        
                        # Warn about stagnation
                        if self.stagnation_counter >= self.stagnation_threshold and self.verbose > 0:
                            print(f"Warning: Performance may be stagnating. "
                                  f"No improvement for {self.stagnation_counter} episodes. "
                                  f"Current mean reward: {current_mean:.2f}")
                            self.stagnation_counter = 0  # Reset to avoid spam
        
        return True