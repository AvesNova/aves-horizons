#!/usr/bin/env python3
"""
Main training script for team-based self-play RL.

Usage:
    python main.py --config configs/default.yaml
    python main.py --config configs/1v1.yaml
    python main.py --config configs/2v2.yaml
"""

import argparse
import os
import yaml
import torch
import wandb
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from team_env_wrapper import TeamEnvironmentWrapper
from transformer_policy import create_team_ppo_model
from callbacks import SelfPlayCallback, EvalAgainstScriptedCallback
from evaluation import TournamentEvaluator


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_training_env(config: dict) -> TeamEnvironmentWrapper:
    """Create the training environment."""
    env_config = config["environment"]
    training_config = config["training"]

    env = TeamEnvironmentWrapper(
        env_config=env_config,
        team_id=training_config["learning_team_id"],
        team_assignments=training_config["team_assignments"],
        opponent_type=training_config["opponent_type"],
        scripted_mix_ratio=training_config.get("scripted_mix_ratio", 0.2),
        self_play_memory_size=training_config.get("self_play_memory_size", 50),
        opponent_update_freq=training_config.get("opponent_update_freq", 10000),
    )

    return env


def create_eval_env(config: dict) -> TeamEnvironmentWrapper:
    """Create evaluation environment (always vs scripted agents)."""
    env_config = config["environment"].copy()
    training_config = config["training"]

    # Evaluation always vs scripted agents for consistent metrics
    eval_env = TeamEnvironmentWrapper(
        env_config=env_config,
        team_id=training_config["learning_team_id"],
        team_assignments=training_config["team_assignments"],
        opponent_type="scripted",
        scripted_mix_ratio=1.0,  # Always scripted for evaluation
    )

    return eval_env


def setup_wandb(config: dict, run_name: str):
    """Initialize Weights & Biases logging."""
    if config.get("wandb", {}).get("enabled", False):
        wandb.init(
            project=config["wandb"]["project"],
            name=run_name,
            config=config,
            tags=config["wandb"].get("tags", []),
        )


def main():
    parser = argparse.ArgumentParser(description="Train team-based RL agent")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name for logging and checkpoints",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="Only run evaluation, no training"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create run name
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        game_mode = config["training"]["game_mode"]
        args.run_name = f"{game_mode}_{timestamp}"

    # Setup directories
    checkpoint_dir = Path(f"checkpoints/{args.run_name}")
    log_dir = Path(f"logs/{args.run_name}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(checkpoint_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Starting run: {args.run_name}")
    print(f"Config: {args.config}")
    print(f"Game mode: {config['training']['game_mode']}")
    print(f"Team assignments: {config['training']['team_assignments']}")

    # Setup wandb
    setup_wandb(config, args.run_name)

    # Create environments
    print("Creating environments...")
    train_env = create_training_env(config)
    eval_env = create_eval_env(config)

    # Wrap for monitoring
    train_env = Monitor(train_env, str(log_dir / "train"))
    eval_env = Monitor(eval_env, str(log_dir / "eval"))

    # Vectorize environments (SB3 requirement)
    train_env = DummyVecEnv([lambda: train_env])
    eval_env = DummyVecEnv([lambda: eval_env])

    # Create model
    print("Creating model...")
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        model = PPO.load(args.resume, env=train_env)
    else:
        model = create_team_ppo_model(
            env=train_env,
            transformer_config=config["model"]["transformer"],
            team_id=config["training"]["learning_team_id"],
            team_assignments=config["training"]["team_assignments"],
            ppo_config=config["model"]["ppo"],
        )

    if args.eval_only:
        print("Running evaluation only...")
        # Create tournament evaluator
        evaluator = TournamentEvaluator(config)
        results = evaluator.evaluate_model(model)
        print("Evaluation results:")
        for metric, value in results.items():
            print(f"  {metric}: {value}")
        return

    # Setup callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config["training"].get("checkpoint_freq", 50000),
        save_path=str(checkpoint_dir),
        name_prefix="model",
    )
    callbacks.append(checkpoint_callback)

    # Self-play callback
    self_play_callback = SelfPlayCallback(
        env_wrapper=train_env.envs[0],  # Get original env from DummyVecEnv
        save_freq=config["training"].get("self_play_update_freq", 20000),
        min_save_steps=config["training"].get("min_steps_before_selfplay", 50000),
    )
    callbacks.append(self_play_callback)

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir / "best_model"),
        log_path=str(log_dir),
        eval_freq=config["training"].get("eval_freq", 25000),
        n_eval_episodes=config["training"].get("eval_episodes", 10),
        deterministic=True,
    )
    callbacks.append(eval_callback)

    # Scripted evaluation callback
    scripted_eval_callback = EvalAgainstScriptedCallback(
        eval_freq=config["training"].get("scripted_eval_freq", 50000),
        n_eval_episodes=config["training"].get("scripted_eval_episodes", 20),
    )
    callbacks.append(scripted_eval_callback)

    # Start training
    print("Starting training...")
    print(f"Total timesteps: {config['training']['total_timesteps']:,}")

    try:
        model.learn(
            total_timesteps=config["training"]["total_timesteps"],
            callback=callbacks,
            progress_bar=True,
            tb_log_name=args.run_name,
        )

        # Save final model
        final_path = checkpoint_dir / "final_model"
        model.save(str(final_path))
        print(f"Final model saved to: {final_path}")

        # Run final evaluation
        print("Running final evaluation...")
        evaluator = TournamentEvaluator(config)
        final_results = evaluator.evaluate_model(model)

        print("Final evaluation results:")
        for metric, value in final_results.items():
            print(f"  {metric}: {value}")

        if wandb.run:
            wandb.log({"final_" + k: v for k, v in final_results.items()})

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

        # Save interrupted model
        interrupted_path = checkpoint_dir / "interrupted_model"
        model.save(str(interrupted_path))
        print(f"Interrupted model saved to: {interrupted_path}")

    finally:
        # Cleanup
        train_env.close()
        eval_env.close()

        if wandb.run:
            wandb.finish()

        print("Training complete!")


if __name__ == "__main__":
    main()
