import pygame
import torch
from rendering.pygame_renderer import PygameRenderer
from utils.config import Actions, ModelConfig
from utils.entry_points import (
    entry_point_manager, setup_environment_from_config, 
    print_config, handle_common_errors
)


@handle_common_errors
def main_game_loop(config):
    """Main game loop with unified configuration."""
    # Initialize environment and renderer
    env = setup_environment_from_config(config)
    renderer = PygameRenderer(world_size=list(config['world_size']))

    # Reset environment
    env.reset()

    # Display game mode and controls
    print_config(config, "Game Configuration")
    print(f"Starting {config['game_mode']} mode with {config['total_ships']} ships")
    if config['game_mode'] == 'deathmatch':
        print(f"Teams: {ModelConfig.DEFAULT_N_TEAMS}, Ships per team: {config['ships_per_team']}")
    
    controlled_ship = config.get('controlled_ship', 0)
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
        total_ships = config['total_ships']
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
    # Create parser with rendering-specific options
    parser = entry_point_manager.create_base_parser(
        'Aves Horizons Space Battle Simulator\n\n'
        'Examples:\n'
        '  %(prog)s --game-mode standard --n-ships 10\n'
        '  %(prog)s --game-mode deathmatch --ships-per-team 4\n'
        '  %(prog)s --game-mode deathmatch --ships-per-team 3 --controlled-ship 2\n'
    )
    entry_point_manager.add_rendering_args(parser)
    
    args = parser.parse_args()
    config = entry_point_manager.validate_args(args)
    
    # Run the game
    main_game_loop(config)
