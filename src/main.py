import pygame
import torch
import argparse
import sys
from core.environment import Environment
from rendering.pygame_renderer import PygameRenderer
from game_modes.deathmatch import create_deathmatch_game
from utils.config import Actions


def create_environment(game_mode, n_ships, ships_per_team, n_obstacles, use_continuous_collision=True):
    """Create environment based on game mode."""
    if game_mode == 'deathmatch':
        n_teams = 2  # Default to 2 teams for deathmatch
        if ships_per_team is None:
            ships_per_team = n_ships // n_teams
        env = create_deathmatch_game(
            n_teams=n_teams,
            ships_per_team=ships_per_team,
            world_size=(1200.0, 800.0),
            use_continuous_collision=use_continuous_collision
        )
    else:  # 'standard' or default
        env = Environment(
            n_ships=n_ships, 
            n_obstacles=n_obstacles,
            use_continuous_collision=use_continuous_collision
        )
    
    return env


def main(game_mode='standard', n_ships=8, ships_per_team=None, n_obstacles=0, controlled_ship=0, use_continuous_collision=True):
    """Main game loop with configurable parameters."""
    # Initialize environment and renderer
    env = create_environment(game_mode, n_ships, ships_per_team, n_obstacles, use_continuous_collision)
    renderer = PygameRenderer(world_size=env.world_size.tolist())

    # Reset environment
    env.reset()

    # Display game mode and controls
    print(f"Starting {game_mode} mode with {env.n_ships if hasattr(env, 'n_ships') else env.total_ships} ships")
    if game_mode == 'deathmatch':
        print(f"Teams: {env.config.n_teams}, Ships per team: {env.config.ships_per_team}")
    print(f"Controlled ship: {controlled_ship}")
    print("Controls: WASD=move, Shift=sharp turn, Space=shoot, ESC=quit")
    
    running = True

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get keyboard state for controlled ship
        keys = pygame.key.get_pressed()
        total_ships = env.n_ships if hasattr(env, 'n_ships') else env.total_ships
        actions = torch.zeros((total_ships, len(Actions)), dtype=torch.bool)

        # Control first ship with keyboard
        actions[controlled_ship, Actions.forward] = keys[pygame.K_w]
        actions[controlled_ship, Actions.backward] = keys[pygame.K_s]
        actions[controlled_ship, Actions.left] = keys[pygame.K_a]
        actions[controlled_ship, Actions.right] = keys[pygame.K_d]
        actions[controlled_ship, Actions.sharp_turn] = keys[pygame.K_LSHIFT]
        actions[controlled_ship, Actions.shoot] = keys[pygame.K_SPACE]

        # Other ships take random actions
        for i in range(total_ships):
            if i != controlled_ship:
                actions[i] = torch.randint(0, 2, (len(Actions),), dtype=torch.bool)

        # Step environment
        observation, rewards, done = env.step(actions)

        # Render
        renderer.render(env.ships, env.projectiles, env.obstacles)

        # Reset if done
        if done:
            env.reset()

    renderer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Aves Horizons Space Battle Simulator',
        epilog='Examples:\n'
               '  %(prog)s --game-mode standard --n-ships 10\n'
               '  %(prog)s --game-mode deathmatch --ships-per-team 4\n'
               '  %(prog)s --game-mode deathmatch --ships-per-team 3 --controlled-ship 2\n',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--game-mode', choices=['standard', 'deathmatch'], 
                        default='standard', help='Game mode to play')
    parser.add_argument('--n-ships', type=int, default=8, 
                        help='Total number of ships (for standard mode)')
    parser.add_argument('--ships-per-team', type=int, default=None, 
                        help='Ships per team (for deathmatch mode)')
    parser.add_argument('--n-obstacles', type=int, default=0, 
                        help='Number of obstacles in the environment (recommend 0 for training since ships cannot see obstacles yet)')
    parser.add_argument('--controlled-ship', type=int, default=0, 
                        help='Which ship to control with keyboard (0-indexed)')
    parser.add_argument('--disable-continuous-collision', action='store_true',
                        help='Disable continuous collision detection (may cause projectiles to phase through ships at low frame rates)')
    
    args = parser.parse_args()
    
    # Validate arguments based on game mode
    if args.game_mode == 'deathmatch':
        if args.ships_per_team is None:
            args.ships_per_team = args.n_ships // 2  # Default to 2 teams
        total_ships = 2 * args.ships_per_team  # Assuming 2 teams
    else:
        total_ships = args.n_ships
    
    if args.controlled_ship >= total_ships:
        print(f"Error: controlled-ship ({args.controlled_ship}) must be less than total ships ({total_ships})")
        sys.exit(1)
    
    main(game_mode=args.game_mode, 
         n_ships=args.n_ships, 
         ships_per_team=args.ships_per_team,
         n_obstacles=args.n_obstacles, 
         controlled_ship=args.controlled_ship,
         use_continuous_collision=not args.disable_continuous_collision)
