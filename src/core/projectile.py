import torch
from typing import Tuple


def update_projectiles(ships, dt: float) -> None:
    """Update projectile positions, lifetimes, and ammo regeneration.

    Args:
        ships: Ships dataclass containing projectile state
        dt: Time step for integration
    """
    # Update positions using current velocities
    ships.projectiles_position += ships.projectiles_velocity * dt

    # Update lifetimes and deactivate expired projectiles
    ships.projectiles_lifetime -= dt
    ships.projectiles_active &= ships.projectiles_lifetime > 0

    # Regenerate ammo
    ships.ammo_count = torch.clamp(
        ships.ammo_count + ships.ammo_regen_rate * dt,
        min=torch.zeros_like(ships.ammo_count),
        max=ships.max_ammo,
    )


def fire_projectile(ships, ship_idx: int) -> bool:
    """Attempt to fire a projectile from the specified ship.

    Args:
        ships: Ships dataclass containing ship and projectile state
        ship_idx: Index of the ship firing

    Returns:
        bool: True if projectile was fired, False if on cooldown, insufficient ammo, or ship is dead
    """
    # Check if ship is active (alive)
    if hasattr(ships, 'active') and not ships.active[ship_idx]:
        return False
        
    if ships.projectile_cooldown[ship_idx] > 0:
        return False

    # Check if we have enough ammo
    if ships.ammo_count[ship_idx] < 1.0:
        return False

    # Get next projectile index for this ship
    idx = ships.projectile_index[ship_idx]

    # Reset projectile state
    ships.projectiles_active[ship_idx, idx] = True
    ships.projectiles_lifetime[ship_idx, idx] = ships.projectile_lifetime[ship_idx]
    ships.projectiles_position[ship_idx, idx] = ships.position[ship_idx]

    # Calculate random spread angle
    spread = (torch.rand(1) * 2 - 1) * ships.projectile_spread[ship_idx]
    spread_direction = torch.exp(1j * spread)

    # Set velocity as ship velocity plus bullet speed in ship's attitude direction with spread
    ships.projectiles_velocity[ship_idx, idx] = (
        ships.velocity[ship_idx]
        + ships.projectile_speed[ship_idx] * ships.attitude[ship_idx] * spread_direction
    )

    # Update firing index, cooldown, and ammo count
    ships.projectile_index[ship_idx] = (idx + 1) % ships.max_projectiles
    ships.projectile_cooldown[ship_idx] = ships.firing_cooldown[ship_idx]
    ships.ammo_count[ship_idx] -= 1.0

    return True
