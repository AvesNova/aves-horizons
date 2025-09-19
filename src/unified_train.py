#!/usr/bin/env python3
"""
Unified Training Script
Supports BC pretraining and RL training with the unified system
"""

import argparse
import yaml
import torch
from pathlib import Path
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_wrapper import create_unified_rl_env
from transformer_policy import create_team_ppo_model
from bc_training import train_bc_model, create_bc_model
from callbacks import SelfPlayCallback


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_directories(run_name: str) -> tuple[Path, Path]:
    """Setup output directories"""
    checkpoint_dir = Path(f"checkpoints/{run_name}")
    log_dir = Path(f"logs/{run_name}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    return checkpoint_dir, log_dir


def train_bc_phase(config: dict, run_name: str) -> str:
    """
    Phase 1: Behavior Cloning pretraining

    Returns:
        Path to trained BC model
    """
    print("=" * 60)
    print("PHASE 1: BEHAVIOR CLONING PRETRAINING")
    print("=" * 60)

    # Check if BC data exists
    data_dir = Path(config["data_collection"]["bc_data"]["output_dir"])
    data_files = list(data_dir.glob("*.pkl*"))

    if not data_files:
        print("No BC training data found. Please run data collection first:")
        print(
            "python scripts/collect_data.py collect_bc --config configs/unified_training.yaml"
        )
        return None

    print(f"Found BC training data: {len(data_files)} files")

    # Create BC model
    model_config = config["model"]["transformer"]
    bc_model = create_bc_model(model_config)

    # Train BC model
    bc_config = config["model"]["bc"]
    checkpoint_dir, _ = setup_directories(f"{run_name}_bc")

    trained_model_path = train_bc_model(
        model=bc_model,
        data_files=data_files,
        config=bc_config,
        output_dir=checkpoint_dir,
        run_name=f"{run_name}_bc",
    )

    print(f"BC pretraining complete! Model saved to: {trained_model_path}")
    return trained_model_path


def train_rl_phase(
    config: dict, run_name: str, bc_model_path: str | None = None
) -> str:
    """
    Phase 2: RL training (optionally starting from BC model)

    Args:
        bc_model_path: Path to BC model for initialization (optional)

    Returns:
        Path to trained RL model
    """
    print("=" * 60)
    print("PHASE 2: REINFORCEMENT LEARNING TRAINING")
    print("=" * 60)

    if bc_model_path:
        print(f"Initializing from BC model: {bc_model_path}")
    else:
        print("Training from scratch")

    # Setup directories
    checkpoint_dir, log_dir = setup_directories(f"{run_name}_rl")

    # Create training environment
    env_config = config["environment"]
    training_config = config["training"]["rl"]

    train_env = create_unified_rl_env(
        env_config=env_config,
        learning_team_id=training_config["learning_team_id"],
        opponent_config=training_config["opponent"],
    )

    # Create evaluation environment (always vs scripted for consistent metrics)
    eval_opponent_config = training_config["opponent"].copy()
    eval_opponent_config["type"] = "scripted"
    eval_opponent_config["scripted_mix_ratio"] = 1.0

    eval_env = create_unified_rl_env(
        env_config=env_config,
        learning_team_id=training_config["learning_team_id"],
        opponent_config=eval_opponent_config,
    )

    # Wrap environments for SB3
    train_env = Monitor(train_env, str(log_dir / "train"))
    eval_env = Monitor(eval_env, str(log_dir / "eval"))
    train_env = DummyVecEnv([lambda: train_env])
    eval_env = DummyVecEnv([lambda: eval_env])

    # Create PPO model
    model_config = config["model"]["transformer"]
    ppo_config = config["model"]["ppo"]

    if bc_model_path:
        # Initialize PPO with BC weights
        model = create_team_ppo_model_from_bc(
            env=train_env,
            bc_model_path=bc_model_path,
            transformer_config=model_config,
            team_id=training_config["learning_team_id"],
            ppo_config=ppo_config,
        )
    else:
        # Create fresh PPO model
        model = create_team_ppo_model(
            env=train_env,
            transformer_config=model_config,
            team_id=training_config["learning_team_id"],
            team_assignments={0: [0, 1], 1: [2, 3]},  # Will be updated by nvn
            ppo_config=ppo_config,
        )

    # Setup callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=training_config["checkpoint_freq"],
        save_path=str(checkpoint_dir),
        name_prefix="rl_model",
    )
    callbacks.append(checkpoint_callback)

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir / "best_model"),
        log_path=str(log_dir),
        eval_freq=training_config["eval_freq"],
        n_eval_episodes=training_config["eval_episodes"],
        deterministic=True,
    )
    callbacks.append(eval_callback)

    # Self-play callback (if using self-play)
    if training_config["opponent"]["type"] in ["self_play", "mixed"]:
        selfplay_callback = SelfPlayCallback(
            env_wrapper=train_env.envs[0],
            save_freq=training_config["selfplay_update_freq"],
            min_save_steps=training_config["min_steps_before_selfplay"],
        )
        callbacks.append(selfplay_callback)

    # Start training
    print(
        f"Starting RL training for {training_config['total_timesteps']:,} timesteps..."
    )

    try:
        model.learn(
            total_timesteps=training_config["total_timesteps"],
            callback=callbacks,
            progress_bar=True,
        )

        # Save final model
        final_model_path = checkpoint_dir / "final_rl_model"
        model.save(str(final_model_path))
        print(f"RL training complete! Final model saved to: {final_model_path}")

        return str(final_model_path)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        interrupted_path = checkpoint_dir / "interrupted_rl_model"
        model.save(str(interrupted_path))
        print(f"Interrupted model saved to: {interrupted_path}")
        return str(interrupted_path)

    finally:
        train_env.close()
        eval_env.close()


def create_team_ppo_model_from_bc(
    env, bc_model_path: str, transformer_config: dict, team_id: int, ppo_config: dict
):
    """Create PPO model initialized with BC weights"""
    from team_transformer_model import create_team_model

    # Load BC model
    bc_model = create_team_model(transformer_config)
    bc_model.load_state_dict(torch.load(bc_model_path, map_location="cpu"))

    # Create PPO model
    model = create_team_ppo_model(
        env=env,
        transformer_config=transformer_config,
        team_id=team_id,
        team_assignments={0: [0, 1], 1: [2, 3]},  # Will be updated
        ppo_config=ppo_config,
    )

    # Transfer weights from BC model to PPO policy
    try:
        # Get the transformer from PPO policy
        ppo_transformer = model.policy.get_transformer_model()

        # Copy weights from BC model
        ppo_transformer.load_state_dict(bc_model.state_dict())

        print("Successfully initialized PPO with BC weights")

    except Exception as e:
        print(f"Warning: Could not transfer BC weights: {e}")
        print("Continuing with random initialization")

    return model


def evaluate_final_model(model_path: str, config: dict, run_name: str):
    """Evaluate the final trained model"""
    print("=" * 60)
    print("FINAL MODEL EVALUATION")
    print("=" * 60)

    from collect_data import evaluate_model

    eval_config = config["evaluation"].copy()
    eval_config["model_config"] = config["model"]["transformer"]

    stats = evaluate_model(model_path, eval_config)

    # Save evaluation results
    results_dir = Path(f"results/{run_name}")
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "final_evaluation.yaml"
    with open(results_file, "w") as f:
        yaml.dump(stats, f)

    print(f"Evaluation results saved to: {results_file}")


def run_full_pipeline(config: dict, run_name: str, skip_bc: bool = False):
    """Run the complete training pipeline"""
    print("=" * 60)
    print(f"UNIFIED TRAINING PIPELINE: {run_name}")
    print("=" * 60)

    bc_model_path = None

    # Phase 1: BC Pretraining (optional)
    if not skip_bc:
        bc_model_path = train_bc_phase(config, run_name)
        if bc_model_path is None:
            print("BC training failed or skipped")
    else:
        print("Skipping BC pretraining phase")

    # Phase 2: RL Training
    rl_model_path = train_rl_phase(config, run_name, bc_model_path)

    # Phase 3: Final Evaluation
    evaluate_final_model(rl_model_path, config, run_name)

    print("=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"BC Model: {bc_model_path or 'None'}")
    print(f"RL Model: {rl_model_path}")


def main():
    parser = argparse.ArgumentParser(description="Unified training pipeline")
    parser.add_argument(
        "mode",
        choices=["bc", "rl", "full"],
        help="Training mode: bc (behavior cloning), rl (reinforcement learning), full (both)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/unified_training.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--run-name", type=str, default=None, help="Run name for checkpoints and logs"
    )
    parser.add_argument(
        "--bc-model",
        type=str,
        default=None,
        help="Path to BC model for RL initialization",
    )
    parser.add_argument(
        "--skip-bc", action="store_true", help="Skip BC pretraining in full mode"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Generate run name
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"unified_{args.mode}_{timestamp}"

    print(f"Starting {args.mode} training with run name: {args.run_name}")
    print(f"Config: {args.config}")

    # Save config copy
    checkpoint_dir, _ = setup_directories(args.run_name)
    config_copy_path = checkpoint_dir / "config.yaml"
    with open(config_copy_path, "w") as f:
        yaml.dump(config, f)

    # Run selected mode
    if args.mode == "bc":
        train_bc_phase(config, args.run_name)
    elif args.mode == "rl":
        train_rl_phase(config, args.run_name, args.bc_model)
    elif args.mode == "full":
        run_full_pipeline(config, args.run_name, args.skip_bc)


if __name__ == "__main__":
    main()
