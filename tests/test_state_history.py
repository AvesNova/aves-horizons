"""
Test StateHistory functionality.
"""
import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.state_history import StateHistory
from core.ships import Ships
from utils.config import ModelConfig


class TestStateHistory:
    """Test the StateHistory class."""
    
    def test_state_history_creation(self):
        """Test basic state history creation."""
        history = StateHistory(
            sequence_length=6,
            max_ships=8,
            world_size=(1200.0, 800.0),
            normalize_coordinates=True
        )
        assert history.sequence_length == 6
        assert history.max_ships == 8
        assert history.world_size == (1200.0, 800.0)
        assert history.normalize_coordinates == True
        
        # Should start with initialized buffer
        assert len(history.state_buffer) == 6
        assert not history.is_ready()  # Need actual states, not just empty ones
        
    def test_add_single_state(self):
        """Test adding a single state."""
        history = StateHistory(sequence_length=3, max_ships=4)
        
        # Create test ships
        ships = Ships.from_scalars(
            n_ships=4,
            random_positions=False,
            initial_position=complex(100, 200)
        )
        
        # Add state
        history.add_state(ships)
        
        # Check that timestep was incremented
        assert history.current_timestep == 1
        
        # Check that we can retrieve the state
        current_state = history.get_current_state()
        assert 'position' in current_state
        assert 'velocity' in current_state
        
    def test_rolling_window_behavior(self):
        """Test that state buffer maintains rolling window."""
        history = StateHistory(sequence_length=3, max_ships=2)
        
        # Add more states than sequence_length
        for i in range(5):
            ships = Ships.from_scalars(
                n_ships=2,
                random_positions=False,
                initial_position=complex(i * 10, i * 10)
            )
            history.add_state(ships)
        
        # Should only keep 3 most recent states
        assert len(history.state_buffer) == 3
        assert history.current_timestep == 5
        
        # Check that the buffer contains the most recent states
        # by checking positions (should be from iterations 2, 3, 4)
        states_list = list(history.state_buffer)
        
        # First state in buffer should be from iteration 2 (position 20+20j)
        first_pos = states_list[0]['position'][0]
        assert torch.allclose(first_pos.real, torch.tensor(20.0/1200.0))  # Normalized
        assert torch.allclose(first_pos.imag, torch.tensor(20.0/800.0))
        
    def test_token_sequence_generation(self):
        """Test generating token sequences."""
        history = StateHistory(
            sequence_length=3, 
            max_ships=2,
            world_size=(100.0, 100.0)  # Simple world for easy testing
        )
        
        # Add 3 states
        for i in range(3):
            ships = Ships.from_scalars(
                n_ships=2,
                random_positions=False,
                initial_position=complex(i * 10, i * 5)  # Different positions
            )
            history.add_state(ships)
        
        # Generate token sequence
        tokens, ship_ids = history.get_token_sequence()
        
        # Should have sequence_length * max_ships tokens
        expected_seq_len = 3 * 2
        assert tokens.shape == (expected_seq_len, ModelConfig.TOKEN_DIM)
        assert ship_ids.shape == (expected_seq_len,)
        
        # Check time-major organization
        expected_ship_ids = [0, 1, 0, 1, 0, 1]  # ship0, ship1 for each timestep
        assert torch.equal(ship_ids, torch.tensor(expected_ship_ids))
        
        # Check timestep offsets
        expected_offsets = [-2, -2, -1, -1, 0, 0]
        timestep_offsets = tokens[:, ModelConfig.TOKEN_FEATURES['timestep_offset']]  # timestep_offset column
        assert torch.allclose(timestep_offsets, torch.tensor(expected_offsets, dtype=torch.float32))
        
    def test_active_ships_tracking(self):
        """Test tracking of active (alive) ships."""
        history = StateHistory(sequence_length=2, max_ships=3)
        
        # Create ships with some dead
        ships = Ships.from_scalars(n_ships=3)
        ships.health[1] = 0.0  # Kill ship 1
        
        history.add_state(ships)
        
        active_ships = history.get_active_ships()
        expected_active = {0, 2}  # Ships 0 and 2 should be alive
        assert active_ships == expected_active
        
    def test_is_ready_functionality(self):
        """Test the is_ready check."""
        history = StateHistory(sequence_length=3, max_ships=2)
        
        # Initially not ready (only has empty initialized states)
        assert not history.is_ready()
        
        # Add one real state
        ships = Ships.from_scalars(n_ships=2)
        history.add_state(ships)
        
        # Should be ready after having real states fill the buffer
        # For simplicity, let's just check after adding enough states
        for _ in range(2):
            history.add_state(ships)
        
        # Now should be ready
        assert history.is_ready()
        
    def test_state_with_actions(self):
        """Test state tracking with action information."""
        history = StateHistory(sequence_length=2, max_ships=2)
        
        ships = Ships.from_scalars(n_ships=2)
        actions = torch.tensor([[False, False, False, False, False, True],   # Ship 0 shooting
                               [True, False, False, False, False, False]])   # Ship 1 forward
        
        history.add_state(ships, actions)
        
        # Check that shooting state is recorded
        current_state = history.get_current_state()
        assert current_state['is_shooting'][0] == True   # Ship 0 was shooting
        assert current_state['is_shooting'][1] == False  # Ship 1 was not shooting
        
    def test_reset_functionality(self):
        """Test resetting the state history."""
        history = StateHistory(sequence_length=2, max_ships=2)
        
        # Add some states
        for i in range(3):
            ships = Ships.from_scalars(n_ships=2)
            history.add_state(ships)
        
        assert history.current_timestep == 3
        
        # Reset
        history.reset()
        
        # Should be back to initial state
        assert history.current_timestep == 0
        assert len(history.state_buffer) == 2  # Back to sequence_length
        assert not history.is_ready()