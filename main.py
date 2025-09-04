import pygame
import torch
from core.environent import Environment
from rendering.pygame_renderer import PygameRenderer
from utils.config import Actions


def main():
    # Initialize environment and renderer
    env = Environment(n_ships=32, n_obstacles=0)
    renderer = PygameRenderer(world_size=env.world_size.tolist())

    # Reset environment
    env.reset()

    # For debugging: control one ship with keyboard
    controlled_ship = 0
    running = True

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get keyboard state for controlled ship
        keys = pygame.key.get_pressed()
        actions = torch.zeros((env.n_ships, len(Actions)), dtype=torch.bool)

        # Control first ship with keyboard
        actions[controlled_ship, Actions.forward] = keys[pygame.K_w]
        actions[controlled_ship, Actions.backward] = keys[pygame.K_s]
        actions[controlled_ship, Actions.left] = keys[pygame.K_a]
        actions[controlled_ship, Actions.right] = keys[pygame.K_d]
        actions[controlled_ship, Actions.sharp_turn] = keys[pygame.K_LSHIFT]
        actions[controlled_ship, Actions.shoot] = keys[pygame.K_SPACE]

        # Other ships take random actions
        for i in range(env.n_ships):
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
    main()
