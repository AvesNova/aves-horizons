import torch

def create_projectiles(n_ships, max_projectiles_per_ship):
    projectiles = {
        'fire_index': torch.arange(0, n_ships * max_projectiles_per_ship, max_projectiles_per_ship),
        'active': torch.zeros(n_ships * max_projectiles_per_ship, dtype=torch.bool),
        'lifetime': torch.zeros(n_ships * max_projectiles_per_ship),
        'position': torch.zeros(n_ships * max_projectiles_per_ship, 2),
        'velocity': torch.zeros(n_ships * max_projectiles_per_ship, 2),
    }
    return projectiles