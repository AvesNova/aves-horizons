"""
Improved integration tests using fixtures and constants.

This demonstrates how the fixture-based approach makes tests more maintainable
and less brittle to changes in model architecture or constants.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.ships import Ships
from .test_constants import *


class TestImprovedIntegration:
    """Improved integration tests using fixtures and constants."""
    
    def test_ships_to_tokens_to_model(self, default_ships, token_encoder, expected_token_shape):
        """Test complete pipeline: Ships -> TokenEncoder -> Model."""
        # Encode ships to tokens
        tokens = token_encoder.encode_ships_to_tokens(default_ships, timestep_offset=0.0)
        
        # Use dynamic shape checking instead of hardcoded values
        assert tokens.shape == expected_token_shape
        
        # Check that positions are normalized correctly (center position)
        expected_x = 0.5  # Center of world
        expected_y = 0.5  # Center of world
        assert torch.allclose(tokens[:, 0], torch.full((DEFAULT_N_SHIPS,), expected_x))
        assert torch.allclose(tokens[:, 1], torch.full((DEFAULT_N_SHIPS,), expected_y))
    
    def test_temporal_sequence_pipeline(self, test_state_history, test_transformer):
        """Test complete temporal sequence pipeline with smaller components."""
        # Add states over time
        for t in range(3):
            ships = Ships.from_scalars(
                n_ships=4,
                world_size=TEST_WORLD_SIZE,
                random_positions=False,
                initial_position=complex(t * 10, t * 5),  # Moving
                team_ids=[0, 0, 1, 1]
            )
            test_state_history.add_state(ships)
        
        # Generate token sequence
        tokens, ship_ids = test_state_history.get_token_sequence()
        
        # Use constants instead of hardcoded values
        expected_seq_len = 3 * 4  # 3 timesteps * 4 ships
        assert tokens.shape == (expected_seq_len, TOKEN_DIM)
        assert ship_ids.shape == (expected_seq_len,)
        
        # Test model forward pass
        tokens_batch = tokens.unsqueeze(0)
        ship_ids_batch = ship_ids.unsqueeze(0)
        
        with torch.no_grad():
            actions = test_transformer(tokens_batch, ship_ids_batch)
        
        # Model always outputs for MAX_SHIPS, but we only have 4 active ships
        assert actions.shape == (1, MAX_SHIPS, ACTION_DIM)
        assert torch.isfinite(actions).all()
    
    def test_realistic_game_simulation(self, state_history, test_transformer):
        """Test realistic game simulation with configurable parameters."""
        n_ships = 6
        
        # Simulate game timesteps
        for t in range(TEST_SIMULATION_STEPS):
            # Create ships with some variation
            ships = Ships.from_scalars(
                n_ships=n_ships,
                world_size=DEFAULT_WORLD_SIZE,
                random_positions=True,
                team_ids=[0, 0, 0, 1, 1, 1]  # Two teams of 3
            )
            
            # Simulate some damage over time
            ships.health -= torch.rand(n_ships) * 2
            ships.health = torch.clamp(ships.health, 0, DEFAULT_HEALTH)
            
            # Create random actions
            actions = torch.randint(0, 2, (n_ships, ACTION_DIM), dtype=torch.bool)
            
            # Add to history
            state_history.add_state(ships, actions)
            
            # Once we have enough history, test model inference
            if state_history.is_ready():
                tokens, ship_ids = state_history.get_token_sequence()
                
                # Batch for model
                tokens_batch = tokens.unsqueeze(0)
                ship_ids_batch = ship_ids.unsqueeze(0)
                
                # Model forward pass
                with torch.no_grad():
                    predicted_actions = test_transformer(tokens_batch, ship_ids_batch)
                
                # Verify output with dynamic shapes
                assert predicted_actions.shape == (1, MAX_SHIPS, ACTION_DIM)
                assert torch.isfinite(predicted_actions).all()
                
                # Convert to probabilities and sample
                action_probs = torch.sigmoid(predicted_actions)
                sampled_actions = torch.bernoulli(action_probs)
                
                assert sampled_actions.shape == (1, MAX_SHIPS, ACTION_DIM)
                assert torch.all((sampled_actions == 0) | (sampled_actions == 1))
    
    def test_token_encoder_compatibility(self, test_token_encoder, test_state_history):
        """Test that TokenEncoder and StateHistory produce compatible outputs."""
        # Create ships
        ships = Ships.from_scalars(
            n_ships=3,
            world_size=TEST_WORLD_SIZE,
            random_positions=False,
            initial_position=complex(TEST_WORLD_SIZE[0]/2, TEST_WORLD_SIZE[1]/2),
            team_ids=[0, 1, 0]
        )
        
        # Method 1: Direct encoding
        direct_tokens = test_token_encoder.encode_ships_to_tokens(ships, timestep_offset=-1.0)
        
        # Method 2: Through StateHistory
        # Add state twice to get a sequence
        test_state_history.add_state(ships)
        test_state_history.add_state(ships)
        
        history_tokens, ship_ids = test_state_history.get_token_sequence()
        
        # Extract tokens for the last timestep from history
        max_ships_in_history = 4  # From test_state_history fixture
        last_timestep_tokens = history_tokens[max_ships_in_history:max_ships_in_history+3]
        
        # Compare positions and other non-temporal fields (but not timestep_offset)
        # Direct tokens have timestep_offset = -1, history tokens have timestep_offset = 0
        assert torch.allclose(direct_tokens[:3, :-1], last_timestep_tokens[:, :-1])
    
    def test_gradient_flow(self, test_transformer, mock_token_sequence, test_tolerance):
        """Test gradient flow through transformer."""
        tokens, ship_ids = mock_token_sequence
        
        # Create input that requires gradients
        tokens_with_grad = tokens.clone().requires_grad_(True)
        tokens_batch = tokens_with_grad.unsqueeze(0)
        ship_ids_batch = ship_ids.unsqueeze(0)
        
        # Forward pass
        actions = test_transformer(tokens_batch, ship_ids_batch)
        
        # Create a simple loss
        loss = torch.sum(actions ** 2)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist and are reasonable
        assert tokens_with_grad.grad is not None
        assert torch.isfinite(tokens_with_grad.grad).all()
        assert tokens_with_grad.grad.abs().max() < 10.0  # Gradients shouldn't explode
        
        # Check model parameter gradients
        total_grad_norm = 0.0
        for param in test_transformer.parameters():
            if param.requires_grad and param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
        
        total_grad_norm = total_grad_norm ** 0.5
        assert total_grad_norm > test_tolerance  # Should have non-zero gradients
        assert total_grad_norm < 1000  # Gradients shouldn't explode too much