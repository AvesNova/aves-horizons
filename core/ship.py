import torch

def create_ships(n_ships, world_size):
    ships = {
        'id': torch.arange(0, n_ships, 1),
        'position': torch.rand(n_ships, 2) * torch.tensor([world_size[0], world_size[1]]),
        'velocity': torch.zeros(n_ships, 2),
        'rotation': torch.rand(n_ships) * 2 * torch.pi,
        'rotation_speed': torch.full((n_ships,), 0.5),  # radians per second
        'acceleration': torch.full((n_ships,), 10.0),    # units per second squared
        'radius': torch.full((n_ships,), 10.0),         # collision radius
        'health': torch.full((n_ships,), 100.0),
        'cooldown': torch.zeros(n_ships),
        'fire_cooldown': torch.full((n_ships,), 0.5),   # seconds between shots
        'projectile_speed': torch.full((n_ships,), 300.0),  # units per second
        'projectile_lifetime': torch.full((n_ships,), 2.0), # seconds
    }
    return ships