"""
Training Infrastructure for Deathmatch Self-Play.

This module provides self-play training support for the deathmatch game mode,
allowing agents to learn by playing against evolving versions of themselves.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import os
import json
from pathlib import Path
import random
import copy

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.type_aliases import Schedule

from gym_env.ship_transformer_env import ShipTransformerEnv, MultiGameShipTransformerEnv
from game_modes.deathmatch import create_deathmatch_game
from models.ship_transformer import ShipTransformerMVP
from train_ship_transformer import ShipTransformerPolicy, TransformerTrainingCallback


class DeathmatchSelfPlayEnv(ShipTransformerEnv):
    """
    Deathmatch environment wrapper for self-play training.
    
    This environment dynamically switches between different opponent policies
    to ensure the agent learns to handle diverse strategies.
    """
    
    def __init__(
        self,
        n_teams: int = 2,
        ships_per_team: int = 2,
        sequence_length: int = 6,
        world_size: Tuple[float, float] = (1200.0, 800.0),
        render_mode: Optional[str] = None,
        opponent_model_pool: Optional[List[str]] = None,
        **kwargs
    ):
        # Create deathmatch game
        deathmatch_env = create_deathmatch_game(
            n_teams=n_teams,
            ships_per_team=ships_per_team,
            world_size=world_size
        )
        
        # Calculate controlled ships (assume we control team 0)
        controlled_ships = list(range(ships_per_team))
        opponent_ships = list(range(ships_per_team, n_teams * ships_per_team))
        
        # Initialize parent with deathmatch environment
        super().__init__(
            env=deathmatch_env,
            controlled_ships=controlled_ships,
            opponent_ships=opponent_ships,
            sequence_length=sequence_length,
            render_mode=render_mode,
            **kwargs
        )
        
        # Self-play specific attributes
        self.n_teams = n_teams
        self.ships_per_team = ships_per_team
        self.opponent_model_pool = opponent_model_pool or []
        self.current_opponent_model = None
        self.episode_count = 0
        
        # Track win rates against different opponents
        self.win_stats = {}
        
    def reset(self, **kwargs):
        """Reset environment and potentially switch opponent policy."""
        # Every few episodes, switch opponent policy
        if self.episode_count % 10 == 0:
            self._update_opponent_policy()
        
        self.episode_count += 1
        return super().reset(**kwargs)
    
    def _update_opponent_policy(self):
        """Update opponent policy for variety in training."""
        if len(self.opponent_model_pool) > 0:
            # 70% chance to use a random model from pool, 30% chance to use heuristic
            if random.random() < 0.7:
                model_path = random.choice(self.opponent_model_pool)
                self.current_opponent_model = model_path
                self.opponent_policy = "learned"
                print(f"Switching to learned opponent: {os.path.basename(model_path)}")
            else:
                self.current_opponent_model = None
                self.opponent_policy = "heuristic"
                print("Switching to heuristic opponent")
        else:
            self.opponent_policy = "heuristic"
    
    def step(self, action):
        """Step with win tracking."""
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Track wins when episode ends
        if terminated or truncated:
            opponent_key = self.current_opponent_model or "heuristic"
            if opponent_key not in self.win_stats:
                self.win_stats[opponent_key] = {"wins": 0, "total": 0}
            
            self.win_stats[opponent_key]["total"] += 1
            
            # Check if controlled team won (team 0 alive and team 1 dead)
            controlled_alive = info.get("controlled_alive", 0)
            opponent_alive = info.get("opponent_alive", 0)
            
            if controlled_alive > 0 and opponent_alive == 0:
                self.win_stats[opponent_key]["wins"] += 1
        
        return obs, reward, terminated, truncated, info
    
    def add_opponent_model(self, model_path: str):
        """Add a model to the opponent pool."""
        if model_path not in self.opponent_model_pool:
            self.opponent_model_pool.append(model_path)
            print(f"Added opponent model: {os.path.basename(model_path)}")
    
    def get_win_rates(self) -> Dict[str, float]:
        """Get current win rates against different opponents."""
        win_rates = {}
        for opponent, stats in self.win_stats.items():
            if stats["total"] > 0:
                win_rates[opponent] = stats["wins"] / stats["total"]
            else:
                win_rates[opponent] = 0.0
        return win_rates


class SelfPlayCallback(BaseCallback):
    """Callback for self-play training with model pool management."""
    
    def __init__(
        self,
        env: DeathmatchSelfPlayEnv,
        save_freq: int = 50000,
        model_save_path: str = "./models/selfplay/",
        max_pool_size: int = 10,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.env = env
        self.save_freq = save_freq
        self.model_save_path = model_save_path
        self.max_pool_size = max_pool_size
        self.saved_models = []
        
        # Create save directory
        os.makedirs(model_save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        # Save model periodically and add to opponent pool
        if self.n_calls % self.save_freq == 0 and self.n_calls > 0:
            self._save_and_add_to_pool()
        
        # Log win rates periodically
        if self.n_calls % 10000 == 0:
            self._log_win_rates()
        
        return True
    
    def _save_and_add_to_pool(self):
        """Save current model and add to opponent pool."""
        model_name = f"selfplay_model_step_{self.n_calls}"
        model_path = os.path.join(self.model_save_path, model_name)
        
        # Save the model
        self.model.save(model_path)
        
        # Add to saved models list
        self.saved_models.append(model_path)
        
        # Add to environment's opponent pool
        self.env.add_opponent_model(model_path)
        
        # Remove oldest model if pool is too large
        if len(self.saved_models) > self.max_pool_size:
            old_model = self.saved_models.pop(0)
            if old_model in self.env.opponent_model_pool:
                self.env.opponent_model_pool.remove(old_model)
            
            # Optionally delete the old model file to save space
            try:
                os.remove(old_model + ".zip")
            except FileNotFoundError:
                pass
        
        if self.verbose > 0:
            print(f"Saved model to opponent pool: {model_name}")
            print(f"Pool size: {len(self.env.opponent_model_pool)}")
    
    def _log_win_rates(self):
        """Log current win rates against different opponents."""
        win_rates = self.env.get_win_rates()
        
        for opponent, win_rate in win_rates.items():
            opponent_name = os.path.basename(opponent) if opponent != "heuristic" else "heuristic"
            self.logger.record(f"selfplay/win_rate_{opponent_name}", win_rate)
        
        if self.verbose > 0 and win_rates:
            print("\nCurrent Win Rates:")
            for opponent, win_rate in win_rates.items():
                opponent_name = os.path.basename(opponent) if opponent != "heuristic" else "heuristic"
                print(f"  vs {opponent_name}: {win_rate:.2%}")


def train_deathmatch_selfplay(
    total_timesteps: int = 2000000,
    n_teams: int = 2,
    ships_per_team: int = 2,
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
    # Self-play parameters
    model_save_freq: int = 50000,
    max_pool_size: int = 10,
    # Environment parameters
    world_size: Tuple[float, float] = (1200.0, 800.0),
    # Logging parameters
    log_dir: str = "./logs/selfplay/",
    model_save_path: str = "./models/selfplay/",
    verbose: int = 1
) -> PPO:
    """
    Train a deathmatch agent using self-play.
    
    Args:
        total_timesteps: Total training timesteps
        n_teams: Number of teams in deathmatch
        ships_per_team: Ships per team
        sequence_length: Length of temporal sequences
        d_model: Transformer model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        n_steps: Steps per rollout
        n_epochs: Training epochs per rollout
        model_save_freq: How often to save models for opponent pool
        max_pool_size: Maximum size of opponent model pool
        world_size: Environment world size
        log_dir: Directory for logs
        model_save_path: Directory for model checkpoints
        verbose: Verbosity level
        
    Returns:
        Trained PPO model
    """
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)
    
    # Create training environment
    env = DeathmatchSelfPlayEnv(
        n_teams=n_teams,
        ships_per_team=ships_per_team,
        sequence_length=sequence_length,
        world_size=world_size
    )
    
    # Create evaluation environment
    eval_env = DeathmatchSelfPlayEnv(
        n_teams=n_teams,
        ships_per_team=ships_per_team,
        sequence_length=sequence_length,
        world_size=world_size
    )
    
    print(f"Training environment: {n_teams} teams, {ships_per_team} ships per team")
    print(f"Controlled ships: {env.controlled_ships}")
    print(f"Opponent ships: {env.opponent_ships}")
    print(f"Total ships: {n_teams * ships_per_team}")
    
    # Calculate action space size
    n_controlled = len(env.controlled_ships)
    action_space_size = n_controlled * 6  # 6 actions per ship
    
    # Create PPO model with custom transformer policy
    model = PPO(
        ShipTransformerPolicy,
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        policy_kwargs={
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "sequence_length": sequence_length,
            "n_ships": n_teams * ships_per_team,
            "controlled_team_size": n_controlled,
            "world_size": world_size,
        },
        verbose=verbose,
        tensorboard_log=log_dir,
    )
    
    # Set up callbacks
    selfplay_callback = SelfPlayCallback(
        env=env,
        save_freq=model_save_freq,
        model_save_path=model_save_path,
        max_pool_size=max_pool_size,
        verbose=verbose
    )
    
    transformer_callback = TransformerTrainingCallback(
        eval_env=eval_env,
        eval_freq=10000,
        verbose=verbose
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=model_save_freq,
        save_path=os.path.join(model_save_path, "checkpoints"),
        name_prefix="deathmatch_selfplay"
    )
    
    # Train the model
    print(f"Starting self-play training with {total_timesteps} timesteps...")
    print(f"Model architecture: d_model={d_model}, nhead={nhead}, layers={num_layers}")
    print(f"Self-play: save_freq={model_save_freq}, max_pool_size={max_pool_size}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[selfplay_callback, transformer_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(model_save_path, "deathmatch_selfplay_final")
    model.save(final_model_path)
    
    # Save training configuration
    config = {
        "total_timesteps": total_timesteps,
        "n_teams": n_teams,
        "ships_per_team": ships_per_team,
        "sequence_length": sequence_length,
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "n_steps": n_steps,
        "n_epochs": n_epochs,
        "model_save_freq": model_save_freq,
        "max_pool_size": max_pool_size,
        "world_size": world_size,
        "training_type": "selfplay_deathmatch"
    }
    
    config_path = os.path.join(model_save_path, "selfplay_training_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Print final win rates
    print("\nFinal Win Rates:")
    win_rates = env.get_win_rates()
    for opponent, win_rate in win_rates.items():
        opponent_name = os.path.basename(opponent) if opponent != "heuristic" else "heuristic"
        print(f"  vs {opponent_name}: {win_rate:.2%}")
    
    print(f"Training completed! Model saved to {final_model_path}")
    print(f"Configuration saved to {config_path}")
    
    return model


def evaluate_selfplay_model(
    model_path: str,
    n_episodes: int = 20,
    render: bool = False,
    deterministic: bool = True
) -> Dict[str, float]:
    """
    Evaluate a trained self-play model against various opponents.
    
    Args:
        model_path: Path to saved model
        n_episodes: Number of episodes to evaluate per opponent type
        render: Whether to render episodes
        deterministic: Whether to use deterministic actions
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Load model
    model = PPO.load(model_path)
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(model_path), "selfplay_training_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    results = {}
    
    # Evaluate against heuristic opponent
    print("Evaluating against heuristic opponent...")
    env = DeathmatchSelfPlayEnv(
        n_teams=config["n_teams"],
        ships_per_team=config["ships_per_team"],
        sequence_length=config["sequence_length"],
        world_size=tuple(config["world_size"])
    )
    env.opponent_policy = "heuristic"
    
    heuristic_results = _run_evaluation_episodes(model, env, n_episodes, render, deterministic)
    results["heuristic"] = heuristic_results
    
    env.close()
    
    print(f"\nEvaluation Results ({n_episodes} episodes per opponent):")
    for opponent, metrics in results.items():
        print(f"  vs {opponent}:")
        print(f"    Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"    Win Rate: {metrics['win_rate']:.2%}")
        print(f"    Mean Length: {metrics['mean_length']:.1f}")
    
    return results


def _run_evaluation_episodes(model, env, n_episodes: int, render: bool, deterministic: bool) -> Dict[str, float]:
    """Run evaluation episodes and return metrics."""
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
    
    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "win_rate": wins / n_episodes,
        "total_episodes": n_episodes
    }


if __name__ == "__main__":
    # Train deathmatch self-play model
    print("Training Deathmatch Self-Play Agent...")
    
    model = train_deathmatch_selfplay(
        total_timesteps=1000000,  # Start with 1M timesteps
        n_teams=2,
        ships_per_team=2,  # 2v2 deathmatch
        sequence_length=6,
        d_model=64,
        nhead=4,
        num_layers=3,
        model_save_freq=25000,  # Save every 25k steps for more diverse pool
        max_pool_size=8,  # Keep 8 previous versions
        verbose=1
    )
    
    # Evaluate trained model
    print("\nEvaluating trained model...")
    evaluate_selfplay_model(
        model_path="./models/selfplay/deathmatch_selfplay_final",
        n_episodes=10,
        render=False
    )