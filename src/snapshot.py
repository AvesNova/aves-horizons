import numpy as np
from bullets import Bullets
from ship import Ship


class Snapshot:
    """Represents a snapshot of the game state at a specific time"""
    def __init__(self, ships: dict[int, Ship]) -> None:
        self.ships = ships

        max_bullets = np.sum(ship.max_bullets for ship in ships.values())
        self.bullets = Bullets(max_bullets=max_bullets)