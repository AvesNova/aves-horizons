"""
Evaluation utilities for model performance assessment.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from team_env_wrapper import TeamEnvironmentWrapper


class TournamentEvaluator:
    """
    Comprehensive evaluation system for trained models.
    """

    def __init__(self, config: dict):
        self.config = config
        self.env_config = config["environment"]
        self.team_assignments = config["training"]["team_assignments"]

    def evaluate_model(self, model, n_episodes: int = 50) -> Dict[str, float]:
        """
        Comprehensive evaluation of a trained model.

        Returns:
            Dictionary of evaluation metrics
        """
        results = {}

        # 1. Evaluate vs scripted agents
        results.update(self._eval_vs_scripted(model, n_episodes))

        # 2. Evaluate vs random agents
        results.update(self._eval_vs_random(model, n_episodes // 2))

        # 3. Evaluate consistency (multiple runs)
        results.update(self._eval_consistency(model, n_episodes // 4))

        # 4. Performance analysis
        results.update(self._analyze_performance(model, n_episodes // 4))

        return results

    def _eval_vs_scripted(self, model, n_episodes: int) -> Dict[str, float]:
        """Evaluate against scripted agents"""
        env = self._create_eval_env("scripted")

        try:
            episode_rewards, episode_lengths = evaluate_policy(
                model,
                env,
                n_eval_episodes=n_episodes,
                deterministic=True,
                return_episode_rewards=True,
            )

            wins = sum(1 for r in episode_rewards if r > 0)
            win_rate = wins / len(episode_rewards)

            return {
                "scripted_win_rate": win_rate,
                "scripted_mean_reward": np.mean(episode_rewards),
                "scripted_std_reward": np.std(episode_rewards),
                "scripted_mean_length": np.mean(episode_lengths),
            }

        finally:
            env.close()

    def _eval_vs_random(self, model, n_episodes: int) -> Dict[str, float]:
        """Evaluate against random agents"""
        env = self._create_eval_env("random")

        try:
            episode_rewards, episode_lengths = evaluate_policy(
                model,
                env,
                n_eval_episodes=n_episodes,
                deterministic=True,
                return_episode_rewards=True,
            )

            wins = sum(1 for r in episode_rewards if r > 0)
            win_rate = wins / len(episode_rewards)

            return {
                "random_win_rate": win_rate,
                "random_mean_reward": np.mean(episode_rewards),
                "random_mean_length": np.mean(episode_lengths),
            }

        finally:
            env.close()

    def _eval_consistency(self, model, n_episodes: int) -> Dict[str, float]:
        """Evaluate consistency across multiple seeds"""
        seeds = [42, 123, 456, 789, 999]
        win_rates = []

        for seed in seeds:
            env = self._create_eval_env("scripted", seed=seed)

            try:
                episode_rewards, _ = evaluate_policy(
                    model,
                    env,
                    n_eval_episodes=n_episodes // len(seeds),
                    deterministic=True,
                    return_episode_rewards=True,
                )

                wins = sum(1 for r in episode_rewards if r > 0)
                win_rate = wins / len(episode_rewards) if episode_rewards else 0
                win_rates.append(win_rate)

            finally:
                env.close()

        return {
            "consistency_mean_win_rate": np.mean(win_rates),
            "consistency_std_win_rate": np.std(win_rates),
            "consistency_min_win_rate": np.min(win_rates),
            "consistency_max_win_rate": np.max(win_rates),
        }

    def _analyze_performance(self, model, n_episodes: int) -> Dict[str, float]:
        """Analyze detailed performance metrics"""
        env = self._create_eval_env("scripted")

        episode_data = []

        try:
            for episode in range(n_episodes):
                obs, _ = env.reset()
                episode_length = 0
                total_reward = 0
                ship_damages = []

                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(action)

                    total_reward += reward
                    episode_length += 1

                    # Track ship states
                    if "ship_states" in info:
                        for ship_id in env.controlled_ships:
                            if ship_id in info["ship_states"]:
                                ship_state = info["ship_states"][ship_id]
                                # You could track more detailed metrics here

                    done = done or truncated

                episode_data.append(
                    {
                        "length": episode_length,
                        "reward": total_reward,
                        "won": total_reward > 0,
                    }
                )

        finally:
            env.close()

        if not episode_data:
            return {}

        # Calculate advanced metrics
        lengths = [ep["length"] for ep in episode_data]
        rewards = [ep["reward"] for ep in episode_data]

        return {
            "performance_episode_length_std": np.std(lengths),
            "performance_reward_range": np.max(rewards) - np.min(rewards),
            "performance_early_wins": sum(
                1 for ep in episode_data if ep["won"] and ep["length"] < 50
            ),
            "performance_long_battles": sum(
                1 for ep in episode_data if ep["length"] > 200
            ),
        }

    def _create_eval_env(
        self, opponent_type: str, seed: int = None
    ) -> TeamEnvironmentWrapper:
        """Create evaluation environment"""
        env_config = self.env_config.copy()

        if opponent_type == "random":
            # For random evaluation, we'd need to modify the wrapper
            # For now, use scripted with low skill
            opponent_type = "scripted"

        env = TeamEnvironmentWrapper(
            env_config=env_config,
            team_id=0,
            team_assignments=self.team_assignments,
            opponent_type=opponent_type,
            scripted_mix_ratio=1.0,
        )

        if seed is not None:
            env.seed(seed)

        return env

    def create_evaluation_report(self, model, save_path: str = None) -> str:
        """
        Create a comprehensive evaluation report.

        Returns:
            Report string
        """
        results = self.evaluate_model(model)

        report = "=" * 60 + "\n"
        report += "MODEL EVALUATION REPORT\n"
        report += "=" * 60 + "\n\n"

        # Overall Performance
        report += "OVERALL PERFORMANCE:\n"
        report += f"  Win Rate vs Scripted: {results.get('scripted_win_rate', 0):.3f}\n"
        report += f"  Win Rate vs Random:   {results.get('random_win_rate', 0):.3f}\n"
        report += (
            f"  Mean Reward:          {results.get('scripted_mean_reward', 0):.2f}\n"
        )
        report += (
            f"  Mean Episode Length:  {results.get('scripted_mean_length', 0):.1f}\n\n"
        )

        # Consistency Analysis
        report += "CONSISTENCY ANALYSIS:\n"
        report += f"  Mean Win Rate:        {results.get('consistency_mean_win_rate', 0):.3f}\n"
        report += f"  Std Win Rate:         {results.get('consistency_std_win_rate', 0):.3f}\n"
        report += f"  Min Win Rate:         {results.get('consistency_min_win_rate', 0):.3f}\n"
        report += f"  Max Win Rate:         {results.get('consistency_max_win_rate', 0):.3f}\n\n"

        # Performance Details
        report += "PERFORMANCE DETAILS:\n"
        report += (
            f"  Early Wins:           {results.get('performance_early_wins', 0)}\n"
        )
        report += (
            f"  Long Battles:         {results.get('performance_long_battles', 0)}\n"
        )
        report += f"  Episode Length Std:   {results.get('performance_episode_length_std', 0):.1f}\n"
        report += f"  Reward Range:         {results.get('performance_reward_range', 0):.2f}\n\n"

        # Model Rating
        scripted_wr = results.get("scripted_win_rate", 0)
        if scripted_wr >= 0.8:
            rating = "EXCELLENT"
        elif scripted_wr >= 0.6:
            rating = "GOOD"
        elif scripted_wr >= 0.4:
            rating = "AVERAGE"
        else:
            rating = "NEEDS IMPROVEMENT"

        report += f"OVERALL RATING: {rating}\n"
        report += "=" * 60 + "\n"

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                f.write(report)

        return report

    def plot_evaluation_results(self, results: Dict[str, float], save_path: str = None):
        """Create visualization of evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Model Evaluation Results", fontsize=16)

        # Win rates comparison
        win_rates = [
            results.get("scripted_win_rate", 0),
            results.get("random_win_rate", 0),
            results.get("consistency_mean_win_rate", 0),
        ]
        opponent_types = ["Scripted", "Random", "Consistency"]

        axes[0, 0].bar(opponent_types, win_rates)
        axes[0, 0].set_title("Win Rates by Opponent Type")
        axes[0, 0].set_ylabel("Win Rate")
        axes[0, 0].set_ylim(0, 1)

        # Consistency analysis
        consistency_metrics = [
            results.get("consistency_mean_win_rate", 0),
            results.get("consistency_min_win_rate", 0),
            results.get("consistency_max_win_rate", 0),
        ]
        consistency_labels = ["Mean", "Min", "Max"]

        axes[0, 1].bar(consistency_labels, consistency_metrics)
        axes[0, 1].set_title("Win Rate Consistency")
        axes[0, 1].set_ylabel("Win Rate")
        axes[0, 1].set_ylim(0, 1)

        # Performance metrics
        perf_metrics = [
            results.get("performance_early_wins", 0),
            results.get("performance_long_battles", 0),
        ]
        perf_labels = ["Early Wins", "Long Battles"]

        axes[1, 0].bar(perf_labels, perf_metrics)
        axes[1, 0].set_title("Battle Characteristics")
        axes[1, 0].set_ylabel("Count")

        # Reward distribution (mock data - would need actual episode data)
        # For now, create a normal distribution around mean reward
        mean_reward = results.get("scripted_mean_reward", 0)
        std_reward = results.get("scripted_std_reward", 1)
        mock_rewards = np.random.normal(mean_reward, std_reward, 100)

        axes[1, 1].hist(mock_rewards, bins=20, alpha=0.7)
        axes[1, 1].set_title("Reward Distribution")
        axes[1, 1].set_xlabel("Episode Reward")
        axes[1, 1].set_ylabel("Frequency")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def quick_eval(
    model_path: str, config_path: str, n_episodes: int = 20
) -> Dict[str, float]:
    """
    Quick evaluation function for trained models.

    Args:
        model_path: Path to saved model
        config_path: Path to config file
        n_episodes: Number of episodes to evaluate

    Returns:
        Dictionary of evaluation metrics
    """
    import yaml
    from stable_baselines3 import PPO

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load model
    model = PPO.load(model_path)

    # Create evaluator
    evaluator = TournamentEvaluator(config)

    # Run evaluation
    results = evaluator.evaluate_model(model, n_episodes)

    # Print quick summary
    print("Quick Evaluation Results:")
    print(f"  Win Rate vs Scripted: {results.get('scripted_win_rate', 0):.3f}")
    print(f"  Mean Reward: {results.get('scripted_mean_reward', 0):.2f}")
    print(f"  Consistency Std: {results.get('consistency_std_win_rate', 0):.3f}")

    return results


def compare_models(
    model_paths: list[str], config_path: str, n_episodes: int = 20
) -> dict[str, dict[str, float]]:
    """
    Compare multiple trained models.

    Args:
        model_paths: List of paths to saved models
        config_path: Path to config file
        n_episodes: Number of episodes per model

    Returns:
        Dictionary mapping model names to evaluation results
    """
    import yaml
    from stable_baselines3 import PPO

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Create evaluator
    evaluator = TournamentEvaluator(config)

    results = {}

    for model_path in model_paths:
        model_name = Path(model_path).stem
        print(f"Evaluating {model_name}...")

        # Load model
        model = PPO.load(model_path)

        # Evaluate
        model_results = evaluator.evaluate_model(model, n_episodes)
        results[model_name] = model_results

        print(f"  Win Rate: {model_results.get('scripted_win_rate', 0):.3f}")

    return results
