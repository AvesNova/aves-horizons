#!/usr/bin/env python3
"""
Unified Data Collection Script
Supports BC pretraining data, human play, RL evaluation, and self-play data
"""

import argparse
import pickle
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
from datetime import datetime

from game_runner import create_standard_runner, create_human_runner
from agents import (
    create_scripted_agent,
    create_human_agent,
    create_rl_agent,
    create_selfplay_agent,
)


def compute_mc_returns(rewards: list[float], gamma: float = 0.99) -> list[float]:
    """Compute Monte Carlo returns from reward sequence"""
    returns = []
    G = 0.0

    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)

    return list(reversed(returns))


def add_mc_returns(episode_data: dict, gamma: float = 0.99) -> dict:
    """Add Monte Carlo returns to episode data"""
    episode_data = episode_data.copy()
    episode_data["mc_returns"] = {}

    for team_id, rewards in episode_data["rewards"].items():
        returns = compute_mc_returns(rewards, gamma)
        episode_data["mc_returns"][team_id] = returns

    return episode_data


def save_episodes(episodes: list[dict], filepath: Path, compress: bool = True):
    """Save episodes to disk with optional compression"""
    print(f"Saving {len(episodes)} episodes to {filepath}")

    if compress and filepath.suffix == ".pkl":
        import gzip

        with gzip.open(str(filepath) + ".gz", "wb") as f:
            pickle.dump(episodes, f)
        print(f"Compressed and saved to {filepath}.gz")
    else:
        with open(filepath, "wb") as f:
            pickle.dump(episodes, f)


def load_episodes(filepath: Path) -> list[dict]:
    """Load episodes from disk"""
    if filepath.suffix == ".gz":
        import gzip

        with gzip.open(filepath, "rb") as f:
            return pickle.load(f)
    else:
        with open(filepath, "rb") as f:
            return pickle.load(f)


def collect_bc_data(config: dict):
    """Collect behavior cloning data (scripted vs scripted)"""
    print("Starting BC data collection...")
    print(f"Target: {sum(config['episodes_per_mode'].values())} total episodes")

    # Setup runner
    runner = create_standard_runner(
        world_size=tuple(config.get("world_size", [1200, 800])),
        max_ships=config.get("max_ships", 8),
    )
    runner.setup_environment()

    # Create scripted agents for both teams
    scripted_agent = create_scripted_agent(
        world_size=tuple(config.get("world_size", [1200, 800])),
        config=config.get("scripted_config", {}),
    )

    runner.assign_agent(0, scripted_agent)
    runner.assign_agent(1, scripted_agent)

    # Collection setup
    episodes_per_mode = config["episodes_per_mode"]
    game_modes = config["game_modes"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    all_episodes = []
    stats = {
        mode: {"episodes": 0, "total_length": 0, "outcomes": []} for mode in game_modes
    }

    total_episodes = sum(episodes_per_mode[mode] for mode in game_modes)

    def progress_callback(episode_num, total, episode_data):
        mode = episode_data["game_mode"]
        stats[mode]["episodes"] += 1
        stats[mode]["total_length"] += episode_data["episode_length"]
        if episode_data.get("outcome"):
            stats[mode]["outcomes"].append(episode_data["outcome"])

    try:
        pbar = tqdm(total=total_episodes, desc="Collecting episodes")

        for game_mode in game_modes:
            print(
                f"\nCollecting {episodes_per_mode[game_mode]} {game_mode} episodes..."
            )

            episodes = runner.run_multiple_episodes(
                n_episodes=episodes_per_mode[game_mode],
                game_mode=game_mode,
                collect_data=True,
                progress_callback=lambda i, total, ep_data: (
                    pbar.update(1),
                    pbar.set_postfix(
                        {
                            "mode": ep_data["game_mode"],
                            "length": ep_data["episode_length"],
                            "total": len(all_episodes) + i,
                        }
                    ),
                    progress_callback(i, total, ep_data),
                )[
                    0
                ],  # Return None from lambda
            )

            # Add MC returns to episodes
            episodes_with_returns = []
            for episode in episodes:
                episode_with_returns = add_mc_returns(
                    episode, config.get("gamma", 0.99)
                )
                episodes_with_returns.append(episode_with_returns)
                all_episodes.append(episode_with_returns)

            # Save checkpoint for this game mode
            checkpoint_path = output_dir / f"{game_mode}_episodes.pkl"
            save_episodes(episodes_with_returns, checkpoint_path, compress=True)

        pbar.close()

        # Save all episodes together
        final_path = output_dir / "bc_training_data.pkl"
        save_episodes(all_episodes, final_path, compress=True)

        # Save collection stats
        stats_path = output_dir / "collection_stats.yaml"
        with open(stats_path, "w") as f:
            yaml.dump(stats, f)

        # Print final statistics
        print(f"\n{'='*60}")
        print("DATA COLLECTION COMPLETE")
        print(f"{'='*60}")
        print(f"Total episodes: {len(all_episodes)}")
        print(f"Total samples: {sum(ep['episode_length'] for ep in all_episodes)}")
        print(f"Saved to: {final_path}")

        for mode, mode_stats in stats.items():
            if mode_stats["episodes"] > 0:
                avg_length = mode_stats["total_length"] / mode_stats["episodes"]
                print(
                    f"{mode}: {mode_stats['episodes']} episodes, avg length {avg_length:.1f}"
                )

        return all_episodes

    finally:
        runner.close()


def run_human_play(config: dict):
    """Run human play session"""
    print("Starting human play session...")
    print("Controls: WASD/Arrow Keys (move), Space (shoot), Shift (sharp turn)")
    print("Close window or Ctrl+C to quit")

    # Setup runner
    runner = create_human_runner(
        world_size=tuple(config.get("world_size", [1200, 800]))
    )
    env = runner.setup_environment(render_mode="human")

    # Create agents
    human_agent = create_human_agent(env.renderer)
    scripted_agent = create_scripted_agent(
        world_size=tuple(config.get("world_size", [1200, 800])),
        config=config.get("scripted_config", {}),
    )

    runner.assign_agent(0, human_agent)  # Human team
    runner.assign_agent(1, scripted_agent)  # Enemy team

    try:
        game_count = 0
        human_wins = 0

        while True:
            game_count += 1
            print(f"\n--- Game {game_count} ---")

            episode_data = runner.run_episode(game_mode="nvn", collect_data=False)

            if episode_data["terminated"] or episode_data["truncated"]:
                outcome = episode_data.get("outcome", {}).get(0, 0)
                length = episode_data["episode_length"]

                if episode_data["truncated"]:
                    print(f"⏱️  Game ended (timeout after {length} steps)")
                elif outcome > 0.5:
                    print(f"🎉 You won! (Game lasted {length} steps)")
                    human_wins += 1
                elif outcome < -0.5:
                    print(f"💀 You lost! (Game lasted {length} steps)")
                else:
                    print(f"🤝 Draw! (Game lasted {length} steps)")

                # Show statistics
                if game_count > 1:
                    win_rate = human_wins / game_count * 100
                    print(f"📊 Stats: {human_wins}/{game_count} wins ({win_rate:.1f}%)")

                # Ask to continue
                continue_game = input("Play again? (y/n): ").lower().startswith("y")
                if not continue_game:
                    break
            else:
                print("Game ended unexpectedly")
                break

    finally:
        runner.close()


def evaluate_model(model_path: str, config: dict):
    """Evaluate a model against scripted agents"""
    print(f"Evaluating model: {model_path}")

    # Load model
    model_type = config.get("model_type", "transformer")

    if model_type == "ppo":
        from stable_baselines3 import PPO

        model = PPO.load(model_path)
    else:
        # Transformer-based model (BC or direct transformer)
        from team_transformer_model import create_team_model, TeamController

        model_config = config.get("model_config", {"max_ships": 8})
        model = create_team_model(model_config)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

    # Setup runner
    runner = create_standard_runner(
        world_size=tuple(config.get("world_size", [1200, 800])),
        max_ships=config.get("max_ships", 8),
    )
    runner.setup_environment()

    # Create agents
    team_assignments = {0: [0, 1], 1: [2, 3]}  # Default 2v2

    if model_type == "ppo":
        model_agent = create_rl_agent(model, None, "ppo")
    else:
        from team_transformer_model import TeamController

        team_controller = TeamController(team_assignments)
        model_agent = create_rl_agent(model, team_controller, "transformer")

    scripted_agent = create_scripted_agent(
        world_size=tuple(config.get("world_size", [1200, 800])),
        config=config.get("scripted_config", {}),
    )

    runner.assign_agent(0, model_agent)
    runner.assign_agent(1, scripted_agent)

    # Run evaluation episodes
    n_episodes = config.get("eval_episodes", 100)
    game_mode = config.get("eval_game_mode", "2v2")

    print(f"Running {n_episodes} episodes in {game_mode} mode...")

    episodes = runner.run_multiple_episodes(
        n_episodes=n_episodes,
        game_mode=game_mode,
        collect_data=False,
        progress_callback=lambda i, total, ep_data: (
            print(
                f"Episode {i}/{total}: Length={ep_data['episode_length']}, "
                f"Outcome={ep_data.get('outcome', {}).get(0, 'N/A'):.2f}"
            )
            if i % 10 == 0 or i == total
            else None
        ),
    )

    # Calculate results
    stats = runner.get_win_stats(episodes, team_id=0)

    runner.close()

    # Print results
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Episodes: {len(episodes)}")
    print(f"Game mode: {game_mode}")
    print(f"Wins: {stats['wins']}")
    print(f"Losses: {stats['losses']}")
    print(f"Draws: {stats['draws']}")
    print(f"Win Rate: {stats['win_rate']*100:.1f}%")
    print(f"Average Episode Length: {stats['avg_length']:.1f}")

    return stats


def collect_selfplay_data(config: dict):
    """Collect self-play data for RL training"""
    print("Starting self-play data collection...")

    # Load models for self-play
    model_paths = config.get("model_paths", [])
    if not model_paths:
        print("No model paths provided, using scripted vs scripted")
        return collect_bc_data(config)

    # Setup runner
    runner = create_standard_runner(
        world_size=tuple(config.get("world_size", [1200, 800])),
        max_ships=config.get("max_ships", 8),
    )
    runner.setup_environment()

    # Load models
    from team_transformer_model import create_team_model, TeamController

    models = []
    for model_path in model_paths:
        model = create_team_model(config.get("model_config", {"max_ships": 8}))
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        models.append(model)

    team_assignments = {0: [0, 1], 1: [2, 3]}
    team_controller = TeamController(team_assignments)

    # Create self-play agent
    selfplay_agent = create_selfplay_agent(team_controller, memory_size=len(models))

    # Add all models to memory
    for model in models:
        selfplay_agent.add_model_to_memory(model)

    # Update opponent for initial setup
    selfplay_agent.update_opponent(
        type(models[0]), config.get("model_config", {"max_ships": 8})
    )

    runner.assign_agent(0, selfplay_agent)
    runner.assign_agent(1, selfplay_agent)

    # Collection parameters
    n_episodes = config.get("total_episodes", 1000)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting {n_episodes} self-play episodes...")

    episodes = runner.run_multiple_episodes(
        n_episodes=n_episodes,
        game_mode=config.get("game_mode", "nvn"),
        collect_data=True,
        progress_callback=lambda i, total, ep_data: (
            # Update opponent every 50 episodes
            (
                selfplay_agent.update_opponent(
                    type(models[0]), config.get("model_config", {"max_ships": 8})
                )
                if i % 50 == 0
                else None
            ),
            (
                print(f"Episode {i}/{total}: Length={ep_data['episode_length']}")
                if i % 100 == 0
                else None
            ),
        )[
            1
        ],  # Return the print result
    )

    # Add MC returns
    episodes_with_returns = []
    for episode in episodes:
        episode_with_returns = add_mc_returns(episode, config.get("gamma", 0.99))
        episodes_with_returns.append(episode_with_returns)

    # Save data
    final_path = output_dir / "selfplay_data.pkl"
    save_episodes(episodes_with_returns, final_path, compress=True)

    runner.close()

    print(f"\nSelf-play data collection complete!")
    print(f"Saved {len(episodes_with_returns)} episodes to {final_path}")

    return episodes_with_returns


def main():
    parser = argparse.ArgumentParser(
        description="Unified data collection and evaluation"
    )
    parser.add_argument(
        "mode",
        choices=["collect_bc", "human_play", "evaluate_model", "collect_selfplay"],
        help="What to run",
    )
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--model", type=str, help="Model path for evaluation")
    parser.add_argument("--output", type=str, help="Output directory override")

    args = parser.parse_args()

    # Load config
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        # Default config
        config = {
            "world_size": [1200, 800],
            "max_ships": 8,
            "episodes_per_mode": {"1v1": 2500, "2v2": 2500, "3v3": 2500, "4v4": 2500},
            "game_modes": ["1v1", "2v2", "3v3", "4v4"],
            "output_dir": "data/bc_pretraining",
            "gamma": 0.99,
            "eval_episodes": 100,
            "eval_game_mode": "2v2",
            "model_type": "transformer",
            "scripted_config": {
                "max_shooting_range": 500.0,
                "angle_threshold": 5.0,
                "bullet_speed": 500.0,
                "target_radius": 10.0,
                "radius_multiplier": 1.5,
            },
        }

    # Override output directory if specified
    if args.output:
        config["output_dir"] = args.output

    # Add timestamp to output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if "output_dir" in config:
        config["output_dir"] = f"{config['output_dir']}_{timestamp}"

    # Run selected mode
    if args.mode == "collect_bc":
        collect_bc_data(config)
    elif args.mode == "human_play":
        run_human_play(config)
    elif args.mode == "evaluate_model":
        if not args.model:
            parser.error("--model required for evaluate_model mode")
        evaluate_model(args.model, config)
    elif args.mode == "collect_selfplay":
        collect_selfplay_data(config)


if __name__ == "__main__":
    main()
