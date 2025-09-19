#!/usr/bin/env python3
"""
Human vs Scripted Agent Game Entry Point

Controls for Human Player (Ship 0):
- Arrow Keys or WASD: Movement
- Shift: Sharp turn mode
- Space: Shoot

The scripted agent (Ship 1) will intelligently target and shoot at the human player.
"""

import torch
import numpy as np
from env import Environment
from constants import Actions
from scripted_agent import ScriptedAgent


def print_game_info():
    """Print game instructions"""
    print("=" * 60)
    print("SPACE COMBAT - Team Battle")
    print("=" * 60)
    print("Controls for Human Player (Team 0, Ship 0):")
    print("  Arrow Keys or WASD: Move")
    print("  Shift: Sharp turn mode (more agile but uses more energy)")
    print("  Space: Shoot")
    print("\nObjective:")
    print("  Work with your AI teammates to destroy the enemy team!")
    print("  Watch your health and power levels.")
    print("\nPress any key to start, ESC or close window to quit.")
    print("=" * 60)


def print_game_stats(info: dict, episode_time: float):
    """Print current game statistics"""
    ship_states = info.get("ship_states", {})

    print(f"\rTime: {episode_time:.1f}s | ", end="")

    for ship_id, state in ship_states.items():
        if state["alive"]:
            player_type = (
                "Human" if ship_id in info.get("human_controlled", []) else "AI"
            )
            print(
                f"Ship {ship_id}({player_type}): HP={state['health']:.1f} PWR={state['power']:.1f} | ",
                end="",
            )
        else:
            player_type = (
                "Human" if ship_id in info.get("human_controlled", []) else "AI"
            )
            print(f"Ship {ship_id}({player_type}): DESTROYED | ", end="")

    print(f"Bullets: {info.get('active_bullets', 0)}", end="", flush=True)


def main():
    """Main game loop"""
    print_game_info()

    # Create environment with human rendering
    env = Environment(
        render_mode="human",
        world_size=(1200, 800),
        memory_size=1,
        max_ships=8,  # Support up to 4v4 mode with 8 ships
        agent_dt=0.04,
        physics_dt=0.02,
    )

    # Create scripted agents dynamically based on max ships
    def create_scripted_agents(max_ships: int) -> dict:
        """Create scripted agents for all ships except ship 0 (human player)"""
        agents = {}
        for ship_id in range(1, max_ships):
            agents[ship_id] = ScriptedAgent(
                controlled_ship_id=ship_id,
                max_shooting_range=500.0,
                angle_threshold=5.0,
                bullet_speed=500.0,
                target_radius=10.0,
                radius_multiplier=1.5,
                world_size=(1200, 800),
            )
        return agents

    scripted_agents = create_scripted_agents(env.max_ships)

    try:
        # Reset environment
        observation, info = env.reset(game_mode="nvn")

        # Add human control for ship 0
        env.add_human_player(ship_id=0)

        print("Game started! Fight!")

        # Initialize dynamic reward tracking
        episode_reward = {}
        episode_time = 0.0

        while True:
            # Get actions
            actions = {}

            # Check which ships are alive
            alive_ships = set()
            for ship_idx in range(observation["alive"].shape[0]):
                ship_id = observation["ship_id"][ship_idx, 0].item()
                if observation["alive"][ship_idx, 0].item():
                    alive_ships.add(ship_id)

            # AI actions for all scripted agents
            for ship_id, agent in scripted_agents.items():
                if ship_id in alive_ships:
                    actions[ship_id] = agent(observation)
                else:
                    actions[ship_id] = torch.zeros(len(Actions), dtype=torch.float32)

            # Human controls ship 0 (handled by renderer)
            actions[0] = torch.zeros(
                len(Actions)
            )  # Placeholder - renderer will override

            # Step environment
            observation, rewards, terminated, truncated, info = env.step(actions)

            # Update episode stats
            episode_time = info.get("current_time", 0.0)
            for ship_id in rewards:
                if ship_id not in episode_reward:
                    episode_reward[ship_id] = 0.0
                episode_reward[ship_id] += rewards[ship_id]

            # Print stats every few frames
            if int(episode_time * 10) % 5 == 0:  # Every 0.5 seconds
                print_game_stats(info, episode_time)

            # Check for game end
            if terminated:
                print("\n" + "=" * 60)

                # Determine winner based on teams (dynamic)
                ship_states = info.get("ship_states", {})

                # Group ships by team
                teams_alive = {}
                for ship_id, ship_state in ship_states.items():
                    team_id = ship_state.get("team_id", 0)
                    if team_id not in teams_alive:
                        teams_alive[team_id] = False
                    if ship_state.get("alive", False):
                        teams_alive[team_id] = True

                # Determine winner
                alive_teams = [
                    team_id for team_id, alive in teams_alive.items() if alive
                ]

                if len(alive_teams) == 1:
                    winning_team = alive_teams[0]
                    if winning_team == 0:
                        print("🎉 VICTORY! Team 0 (Your team) wins!")
                    else:
                        print(f"💀 DEFEAT! Team {winning_team} wins!")
                elif len(alive_teams) == 0:
                    print("🤝 DRAW! All teams were eliminated!")
                else:
                    print("🤝 DRAW! Multiple teams survived!")

                print(f"\nFinal Scores:")
                # Group ships by team for display
                teams_scores = {}
                for ship_id, ship_state in ship_states.items():
                    team_id = ship_state.get("team_id", 0)
                    if team_id not in teams_scores:
                        teams_scores[team_id] = []

                    player_type = "Human" if ship_id == 0 else "AI"
                    score = episode_reward.get(ship_id, 0.0)
                    teams_scores[team_id].append((ship_id, player_type, score))

                # Display scores by team
                for team_id in sorted(teams_scores.keys()):
                    for ship_id, player_type, score in sorted(teams_scores[team_id]):
                        print(
                            f"  Team {team_id} - Ship {ship_id} ({player_type}): {score:.1f}"
                        )
                print(f"  Battle Duration: {episode_time:.1f} seconds")

                # Ask to play again
                print("\nPress any key to play again, or close window to quit...")

                # Reset for next game
                observation, info = env.reset(game_mode="nvn")
                env.add_human_player(ship_id=0)
                episode_reward = {}
                episode_time = 0.0
                print("New game started!")

    except KeyboardInterrupt:
        print("\n\nGame interrupted by user.")
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback

        traceback.print_exc()
    finally:
        env.close()
        print("Game ended. Thanks for playing!")


if __name__ == "__main__":
    main()
