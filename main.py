import pygame
import torch
from core.environent import Environment
from rendering.pygame_renderer import PygameRenderer

def main():
    # Initialize environment and renderer
    env = Environment(n_ships=2, n_obstacles=5)
    renderer = PygameRenderer()
    
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
        actions = torch.zeros((env.n_ships, 4))  # [thrust, rotate_left, rotate_right, shoot]
        
        # Control first ship with keyboard
        actions[controlled_ship, 0] = 1.0 if keys[pygame.K_w] else 0.0  # Thrust
        actions[controlled_ship, 1] = 1.0 if keys[pygame.K_a] else 0.0  # Rotate left
        actions[controlled_ship, 2] = 1.0 if keys[pygame.K_d] else 0.0  # Rotate right
        actions[controlled_ship, 3] = 1.0 if keys[pygame.K_SPACE] else 0.0  # Shoot
        
        # Other ships take random actions
        for i in range(env.n_ships):
            if i != controlled_ship:
                actions[i] = torch.randint(0, 2, (4,)).float()
        
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