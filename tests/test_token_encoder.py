"""
Test ShipTokenEncoder functionality.
"""
import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.token_encoder import ShipTokenEncoder
from core.ships import Ships
from utils.config import ModelConfig


class TestShipTokenEncoder:
    """Test the ShipTokenEncoder class."""
    
    def test_encoder_creation(self):
        """Test basic encoder creation."""
        encoder = ShipTokenEncoder(
            world_size=(1200.0, 800.0),
            max_speed=300.0,
            normalize_coordinates=True,
            max_ships=8
        )
        assert encoder.world_size == (1200.0, 800.0)
        assert encoder.max_speed == 300.0
        assert encoder.normalize_coordinates == True
        assert encoder.max_ships == 8
        
    def test_encode_ships_basic(self):
        """Test basic ship encoding."""
        encoder = ShipTokenEncoder(
            world_size=(1200.0, 800.0),
            normalize_coordinates=True,
            max_ships=8
        )
        
        # Create test ships
        ships = Ships.from_scalars(
            n_ships=4,
            world_size=(1200, 800),
            random_positions=False,
            initial_position=complex(600, 400),  # Center of world
            initial_velocity=complex(100, 0),    # Moving right
        )
        
        # Encode ships to tokens
        tokens = encoder.encode_ships_to_tokens(ships, timestep_offset=-1.0)
        
        # Check token shape and basic properties
        assert tokens.shape == (4, ModelConfig.TOKEN_DIM), f"Expected (4, {ModelConfig.TOKEN_DIM}), got {tokens.shape}"
        
        # Check that timestep offset is correctly encoded
        assert torch.allclose(tokens[:, ModelConfig.TOKEN_FEATURES['timestep_offset']], torch.full((4,), -1.0))
        
        # Check normalized positions (should be 0.5, 0.5 for center)
        assert torch.allclose(tokens[:, ModelConfig.TOKEN_FEATURES['pos_x']], torch.full((4,), 0.5), atol=1e-6)  # pos_x
        assert torch.allclose(tokens[:, ModelConfig.TOKEN_FEATURES['pos_y']], torch.full((4,), 0.5), atol=1e-6)  # pos_y
        
    def test_encode_with_actions(self):
        """Test encoding with action information."""
        encoder = ShipTokenEncoder()
        ships = Ships.from_scalars(n_ships=2)
        
        # Create actions with shooting
        actions = torch.tensor([[False, False, False, False, False, True],   # Ship 0 shooting
                               [True, False, False, False, False, False]])   # Ship 1 moving forward
        
        tokens = encoder.encode_ships_to_tokens(ships, actions=actions)
        
        # Check shooting state is encoded correctly (is_shooting is at index 10)
        assert tokens[0, ModelConfig.TOKEN_FEATURES['is_shooting']] == 1.0  # Ship 0 is shooting
        assert tokens[1, ModelConfig.TOKEN_FEATURES['is_shooting']] == 0.0  # Ship 1 is not shooting
        
    def test_coordinate_normalization(self):
        """Test coordinate normalization."""
        # Test with normalization
        encoder_norm = ShipTokenEncoder(
            world_size=(1000.0, 500.0),
            normalize_coordinates=True
        )
        
        ships = Ships.from_scalars(
            n_ships=1,
            random_positions=False,
            initial_position=complex(500, 250),  # Center
        )
        
        tokens_norm = encoder_norm.encode_ships_to_tokens(ships)
        
        # Should be normalized to 0.5, 0.5
        assert torch.allclose(tokens_norm[0, ModelConfig.TOKEN_FEATURES['pos_x']], torch.tensor(0.5), atol=1e-6)
        assert torch.allclose(tokens_norm[0, ModelConfig.TOKEN_FEATURES['pos_y']], torch.tensor(0.5), atol=1e-6)
        
        # Test without normalization
        encoder_no_norm = ShipTokenEncoder(
            world_size=(1000.0, 500.0),
            normalize_coordinates=False
        )
        
        tokens_no_norm = encoder_no_norm.encode_ships_to_tokens(ships)
        
        # Should be raw coordinates
        assert torch.allclose(tokens_no_norm[0, ModelConfig.TOKEN_FEATURES['pos_x']], torch.tensor(500.0), atol=1e-6)
        assert torch.allclose(tokens_no_norm[0, ModelConfig.TOKEN_FEATURES['pos_y']], torch.tensor(250.0), atol=1e-6)
        
    def test_token_dimensions(self):
        """Test that tokens have correct dimensions."""
        encoder = ShipTokenEncoder()
        assert encoder.get_token_dim() == ModelConfig.TOKEN_DIM
        
    def test_temporal_sequence_creation(self):
        """Test creating temporal sequences from ship history."""
        encoder = ShipTokenEncoder()
        
        # Create a sequence of ship states
        ships_history = []
        for i in range(3):  # 3 timesteps
            ships = Ships.from_scalars(
                n_ships=2,
                random_positions=False,
                initial_position=complex(i * 10, i * 10)  # Moving diagonally
            )
            ships_history.append(ships)
        
        tokens, ship_ids = encoder.create_temporal_sequence(
            ships_history, 
            sequence_length=3
        )
        
        # Should have 3 timesteps * 2 ships = 6 tokens
        assert tokens.shape == (6, ModelConfig.TOKEN_DIM)
        assert ship_ids.shape == (6,)
        
        # Check ship ID pattern (time-major order)
        expected_ship_ids = [0, 1, 0, 1, 0, 1]  # ship0, ship1 for each timestep
        assert torch.equal(ship_ids, torch.tensor(expected_ship_ids))
        
        # Check timestep offsets
        expected_offsets = [-2, -2, -1, -1, 0, 0]  # Timestep offsets
        timestep_offsets = tokens[:, ModelConfig.TOKEN_FEATURES['timestep_offset']]
        assert torch.allclose(timestep_offsets, torch.tensor(expected_offsets, dtype=torch.float32))
