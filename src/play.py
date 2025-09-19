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
    print("SPACE COMBAT - 2v2 Team Battle")
    print("=" * 60)
    print("Controls for Human Player (Team 0, Ship 0):")
    print("  Arrow Keys or WASD: Move")
    print("  Shift: Sharp turn mode (more agile but uses more energy)")
    print("  Space: Shoot")
    print("\nObjective:")
    print("  Work with your AI teammate (Ship 1) to destroy Team 1 (Ships 2 & 3)!")
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
        max_ships=4,  # Support 2v2 mode with 4 ships
        agent_dt=0.04,
        physics_dt=0.02,
    )

    # Create scripted agents for AI ships in 2v2 mode
    scripted_agents = {
        1: ScriptedAgent(  # Team 0, Ship 1 (AI teammate)
            controlled_ship_id=1,
            max_shooting_range=500.0,
            angle_threshold=5.0,
            bullet_speed=500.0,
            target_radius=10.0,
            radius_multiplier=1.5,
            world_size=(1200, 800),
        ),
        2: ScriptedAgent(  # Team 1, Ship 2 (AI enemy)
            controlled_ship_id=2,
            max_shooting_range=500.0,
            angle_threshold=5.0,
            bullet_speed=500.0,
            target_radius=10.0,
            radius_multiplier=1.5,
            world_size=(1200, 800),
        ),
        3: ScriptedAgent(  # Team 1, Ship 3 (AI enemy)
            controlled_ship_id=3,
            max_shooting_range=500.0,
            angle_threshold=5.0,
            bullet_speed=500.0,
            target_radius=10.0,
            radius_multiplier=1.5,
            world_size=(1200, 800),
        ),
    }

    try:
        # Reset environment
        observation, info = env.reset(game_mode="2v2")

        # Add human control for ship 0
        env.add_human_player(ship_id=0)

        print("Game started! Fight!")

        episode_reward = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
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
                episode_reward[ship_id] += rewards[ship_id]

            # Print stats every few frames
            if int(episode_time * 10) % 5 == 0:  # Every 0.5 seconds
                print_game_stats(info, episode_time)

            # Check for game end
            if terminated:
                print("\n" + "=" * 60)

                # Determine winner based on teams
                ship_states = info.get("ship_states", {})
                team_0_alive = (
                    ship_states.get(0, {}).get("alive", False) or
                    ship_states.get(1, {}).get("alive", False)
                )
                team_1_alive = (
                    ship_states.get(2, {}).get("alive", False) or
                    ship_states.get(3, {}).get("alive", False)
                )

                if team_0_alive and not team_1_alive:
                    print("🎉 VICTORY! Team 0 (Your team) wins!")
                elif team_1_alive and not team_0_alive:
                    print("💀 DEFEAT! Team 1 (Enemy team) wins!")
                else:
                    print("🤝 DRAW! Both teams were eliminated!")

                print(f"\nFinal Scores:")
                print(f"  Team 0 - Ship 0 (Human): {episode_reward[0]:.1f}")
                print(f"  Team 0 - Ship 1 (AI): {episode_reward[1]:.1f}")
                print(f"  Team 1 - Ship 2 (AI): {episode_reward[2]:.1f}")
                print(f"  Team 1 - Ship 3 (AI): {episode_reward[3]:.1f}")
                print(f"  Battle Duration: {episode_time:.1f} seconds")

                # Ask to play again
                print("\nPress any key to play again, or close window to quit...")

                # Reset for next game
                observation, info = env.reset(game_mode="2v2")
                env.add_human_player(ship_id=0)
                episode_reward = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
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
