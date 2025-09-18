#!/usr/bin/env python3
"""
Human vs Random Agent Game Entry Point

Controls for Human Player (Ship 0):
- Arrow Keys or WASD: Movement
- Shift: Sharp turn mode
- Space: Shoot

The random agent ( Ship 1) will move and shoot randomly.
"""

import torch
import numpy as np
from env import Environment
from enums import Actions


class DummyAgent:
    """Simple random agent for testing"""

    def __init__(self, ship_id: int, action_prob: float = 0.3, seed: int = None):
        self.ship_id = ship_id
        self.action_prob = action_prob
        self.rng = np.random.default_rng(seed)

    def get_action(self, observation=None) -> torch.Tensor:
        return torch.zeros(len(Actions))


def print_game_info():
    """Print game instructions"""
    print("=" * 60)
    print("SPACE COMBAT - Human vs Random Agent")
    print("=" * 60)
    print("Controls for Human Player (Blue Ship):")
    print("  Arrow Keys or WASD: Move")
    print("  Shift: Sharp turn mode (more agile but uses more energy)")
    print("  Space: Shoot")
    print("\nObjective:")
    print("  Destroy the enemy ship (Red) while staying alive!")
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
        n_ships=2,
        agent_dt=0.02,
        physics_dt=0.02,
    )

    # Create random agent for ship 1
    random_agent = DummyAgent(ship_id=1, action_prob=0.4, seed=42)

    try:
        # Reset environment
        observation, info = env.reset(game_mode="1v1")

        # Add human control for ship 0
        env.add_human_player(ship_id=0)

        print("Game started! Fight!")

        episode_reward = {0: 0.0, 1: 0.0}
        episode_time = 0.0

        while True:
            # Get actions
            actions = {}

            # Random agent action for ship 1
            actions[1] = random_agent.get_action(observation)

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

                # Determine winner
                ship_states = info.get("ship_states", {})
                human_alive = ship_states.get(0, {}).get("alive", False)
                ai_alive = ship_states.get(1, {}).get("alive", False)

                if human_alive and not ai_alive:
                    print("🎉 VICTORY! You destroyed the enemy ship!")
                elif ai_alive and not human_alive:
                    print("💀 DEFEAT! The enemy destroyed your ship!")
                else:
                    print("🤝 DRAW! Both ships were destroyed!")

                print(f"\nFinal Scores:")
                print(f"  Human (Ship 0): {episode_reward[0]:.1f}")
                print(f"  AI (Ship 1): {episode_reward[1]:.1f}")
                print(f"  Battle Duration: {episode_time:.1f} seconds")

                # Ask to play again
                print("\nPress any key to play again, or close window to quit...")

                # Reset for next game
                observation, info = env.reset(game_mode="1v1")
                env.add_human_player(ship_id=0)
                episode_reward = {0: 0.0, 1: 0.0}
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
