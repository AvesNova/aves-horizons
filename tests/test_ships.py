"""
Test basic Ships functionality.
"""
import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.ships import Ships


class TestShips:
    """Test the Ships class."""
    
    def test_ships_creation(self):
        """Test basic ships creation."""
        ships = Ships.from_scalars(n_ships=2)
        assert ships.n_ships == 2
        assert ships.position.shape == (2,)
        assert ships.velocity.shape == (2,)
        assert ships.attitude.shape == (2,)
        
    def test_ships_with_random_positions(self):
        """Test ships creation with random positions."""
        ships = Ships.from_scalars(
            n_ships=4,
            world_size=(1200, 800),
            random_positions=True
        )
        assert ships.n_ships == 4
        assert ships.position.shape == (4,)
        
        # Check positions are within bounds
        pos_real = ships.position.real
        pos_imag = ships.position.imag
        assert torch.all(pos_real >= 0) and torch.all(pos_real <= 1200)
        assert torch.all(pos_imag >= 0) and torch.all(pos_imag <= 800)
        
    def test_ships_initial_state(self):
        """Test that ships have correct initial state."""
        ships = Ships.from_scalars(
            n_ships=3,
            max_health=100.0,
            max_boost=50.0,
            max_ammo=20.0
        )
        
        # Check that health, boost, and ammo are initialized to max values
        assert torch.allclose(ships.health, torch.full((3,), 100.0))
        assert torch.allclose(ships.boost, torch.full((3,), 50.0))
        assert torch.allclose(ships.ammo_count, torch.full((3,), 20.0))
        
    def test_turn_offset_lookup(self):
        """Test turn offset lookup functionality."""
        ships = Ships.from_scalars(n_ships=2)
        
        # Test basic turn operations
        left = torch.tensor([True, False])
        right = torch.tensor([False, True])
        sharp = torch.tensor([False, False])
        
        turn_angles = ships.get_turn_angle(left, right, sharp)
        assert turn_angles.shape == (2,)
        
        # Left turn should be negative, right turn should be positive
        assert turn_angles[0] < 0  # Left turn
        assert turn_angles[1] > 0  # Right turn
        
    def test_thrust_multiplier_lookup(self):
        """Test thrust multiplier lookup."""
        ships = Ships.from_scalars(n_ships=2)
        
        forward = torch.tensor([True, False])
        backward = torch.tensor([False, True])
        
        thrust_mults = ships.get_thrust_multiplier(forward, backward)
        assert thrust_mults.shape == (2,)
        
        # Forward should use forward_boost, backward should use backward_boost
        assert thrust_mults[0] == ships.forward_boost[0]
        assert thrust_mults[1] == ships.backward_boost[1]
        
    def test_energy_cost_lookup(self):
        """Test energy cost lookup."""
        ships = Ships.from_scalars(n_ships=2)
        
        forward = torch.tensor([True, False])
        backward = torch.tensor([False, False])
        
        energy_costs = ships.get_energy_cost(forward, backward)
        assert energy_costs.shape == (2,)
        
        # Forward should have higher energy cost
        assert energy_costs[0] > energy_costs[1]