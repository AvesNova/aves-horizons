"""
Unified Game Runner - supports all game modes and agent types
"""

from typing import Any, Optional
import numpy as np

from env import Environment
from agents import Agent, HumanAgentProvider


class UnifiedGameRunner:
    """
    Unified game runner that supports:
    - Human play (human vs scripted/RL)
    - Data collection (scripted vs scripted)
    - RL training (RL vs scripted/self-play)
    - BC evaluation (BC vs scripted)
    """

    def __init__(self, env_config: dict, team_assignments: dict[int, list[int]]):
        self.env_config = env_config
        self.team_assignments = team_assignments
        self.env: Optional[Environment] = None
        self.agents: dict[int, Agent] = {}  # team_id -> Agent

    def setup_environment(self, render_mode: str | None = None) -> Environment:
        """Setup environment with optional rendering"""
        config = self.env_config.copy()
        config["render_mode"] = render_mode
        self.env = Environment(**config)
        return self.env

    def assign_agent(self, team_id: int, agent: Agent):
        """Assign an agent to control a team"""
        self.agents[team_id] = agent

        # If human agent, register with renderer
        if (
            isinstance(agent, HumanAgentProvider)
            and self.env
            and self.env.render_mode == "human"
        ):
            for ship_id in self.team_assignments[team_id]:
                self.env.add_human_player(ship_id)

    def run_episode(
        self, game_mode: str = "nvn", collect_data: bool = False, max_steps: int = 10000
    ) -> dict[str, Any]:
        """
        Run a single episode and optionally collect data

        Args:
            game_mode: Game mode to run ("1v1", "2v2", "3v3", "4v4", "nvn")
            collect_data: Whether to collect full episode data
            max_steps: Maximum steps before truncation

        Returns:
            episode_data: Dict with observations, actions, rewards, etc.
        """
        if self.env is None:
            raise ValueError("Environment not setup. Call setup_environment() first.")

        # Reset environment - this will set team_assignments for nvn mode
        obs_dict, info = self.env.reset(game_mode=game_mode)

        # Get actual team assignments from environment state
        actual_teams = self._get_actual_team_assignments()

        # Data collection storage
        episode_data = {
            "game_mode": game_mode,
            "team_assignments": actual_teams,
            "agent_types": {
                tid: self.agents[tid].get_agent_type()
                for tid in self.agents.keys()
                if tid in actual_teams
            },
            "observations": [obs_dict] if collect_data else [],
            "actions": {tid: [] for tid in actual_teams.keys()} if collect_data else {},
            "rewards": {tid: [] for tid in actual_teams.keys()} if collect_data else {},
            "episode_length": 0,
            "terminated": False,
            "truncated": False,
            "outcome": {},
        }

        terminated = False
        truncated = False
        step_count = 0

        while not (terminated or truncated) and step_count < max_steps:
            # Get actions from all agents
            all_actions = {}
            step_team_actions = {}

            for team_id, agent in self.agents.items():
                if team_id in actual_teams:
                    ship_ids = actual_teams[team_id]
                    team_actions = agent.get_actions(obs_dict, ship_ids)
                    all_actions.update(team_actions)

                    if collect_data:
                        step_team_actions[team_id] = team_actions

            # Step environment
            obs_dict, env_rewards, terminated, truncated, info = self.env.step(
                all_actions
            )
            step_count += 1

            # Collect data if requested
            if collect_data:
                episode_data["observations"].append(obs_dict)

                # Store actions by team
                for team_id, team_actions in step_team_actions.items():
                    episode_data["actions"][team_id].append(team_actions)

                # Calculate and store team rewards
                for team_id in actual_teams.keys():
                    team_reward = self.env._calculate_team_reward(
                        self.env.state[-1], team_id, episode_ended=terminated
                    )
                    episode_data["rewards"][team_id].append(team_reward)

            # Handle rendering
            if self.env.render_mode == "human":
                if not self.env.renderer.handle_events():
                    truncated = True  # User closed window

        # Finalize episode data
        episode_data["episode_length"] = step_count
        episode_data["terminated"] = terminated
        episode_data["truncated"] = truncated

        if collect_data and terminated:
            # Calculate final outcomes
            for team_id in actual_teams.keys():
                outcome_reward = self.env._calculate_outcome_rewards(
                    self.env.state[-1], team_id
                )
                episode_data["outcome"][team_id] = outcome_reward

        return episode_data

    def _get_actual_team_assignments(self) -> dict[int, list[int]]:
        """Get actual team assignments from current environment state"""
        if not self.env or not self.env.state:
            return self.team_assignments

        current_state = self.env.state[-1]
        actual_teams = {}

        for ship_id, ship in current_state.ships.items():
            team_id = ship.team_id
            if team_id not in actual_teams:
                actual_teams[team_id] = []
            actual_teams[team_id].append(ship_id)

        return actual_teams

    def run_multiple_episodes(
        self,
        n_episodes: int,
        game_mode: str = "nvn",
        collect_data: bool = False,
        progress_callback=None,
        max_steps: int = 10000,
    ) -> list[dict]:
        """Run multiple episodes with progress tracking"""
        episodes = []

        for i in range(n_episodes):
            episode_data = self.run_episode(game_mode, collect_data, max_steps)
            episodes.append(episode_data)

            if progress_callback:
                progress_callback(i + 1, n_episodes, episode_data)

        return episodes

    def get_win_stats(self, episodes: list[dict], team_id: int = 0) -> dict[str, float]:
        """Calculate win statistics for a team across episodes"""
        wins = 0
        losses = 0
        draws = 0
        total_length = 0

        for episode in episodes:
            if episode["terminated"] and team_id in episode["outcome"]:
                outcome = episode["outcome"][team_id]
                total_length += episode["episode_length"]

                if outcome > 0.5:
                    wins += 1
                elif outcome < -0.5:
                    losses += 1
                else:
                    draws += 1

        total = wins + losses + draws
        if total == 0:
            return {
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "win_rate": 0.0,
                "avg_length": 0.0,
            }

        return {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / (wins + losses) if (wins + losses) > 0 else 0.0,
            "avg_length": total_length / total,
        }

    def close(self):
        """Clean up resources"""
        if self.env:
            self.env.close()
            self.env = None


def create_standard_runner(
    world_size: tuple[int, int] = (1200, 800), max_ships: int = 8
) -> UnifiedGameRunner:
    """Create runner with standard configuration"""
    env_config = {
        "world_size": world_size,
        "max_ships": max_ships,
        "agent_dt": 0.04,
        "physics_dt": 0.02,
    }

    # Default team assignments (will be overridden by nvn mode)
    team_assignments = {0: [0, 1], 1: [2, 3]}

    return UnifiedGameRunner(env_config, team_assignments)


def create_human_runner(world_size: tuple[int, int] = (1200, 800)) -> UnifiedGameRunner:
    """Create runner optimized for human play"""
    env_config = {
        "render_mode": "human",
        "world_size": world_size,
        "max_ships": 8,
        "agent_dt": 0.04,
        "physics_dt": 0.02,
    }

    team_assignments = {0: [0], 1: [1, 2, 3]}  # Human vs multiple opponents
    return UnifiedGameRunner(env_config, team_assignments)
