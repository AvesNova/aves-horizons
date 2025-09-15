"""
Integration tests for the full ShipTransformer pipeline.

Tests that all components work together correctly.
"""
import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.ship_transformer import ShipTransformerMVP
from models.state_history import StateHistory
from models.token_encoder import ShipTokenEncoder
from core.ships import Ships


class TestIntegration:
    """Test full pipeline integration."""
    
    def test_ships_to_tokens_to_model(self):
        """Test complete pipeline: Ships -> TokenEncoder -> StateHistory -> Model."""
        # Create test ships
        ships = Ships.from_scalars(
            n_ships=4,
            world_size=(1000, 500),
            random_positions=False,
            initial_position=complex(500, 250)  # Center
        )
        
        # Create token encoder
        encoder = ShipTokenEncoder(
            world_size=(1000.0, 500.0),
            normalize_coordinates=True,
            max_ships=8
        )
        
        # Encode ships to tokens
        tokens = encoder.encode_ships_to_tokens(ships, timestep_offset=0.0)
        
        assert tokens.shape == (4, 12)
        
        # Check that positions are normalized correctly
        assert torch.allclose(tokens[:, 0], torch.tensor(0.5))  # pos_x
        assert torch.allclose(tokens[:, 1], torch.tensor(0.5))  # pos_y
        
    def test_full_temporal_sequence_pipeline(self):
        """Test the complete temporal sequence pipeline."""
        # Create state history
        history = StateHistory(
            sequence_length=3,
            max_ships=4,
            world_size=(100.0, 100.0),
            normalize_coordinates=True
        )
        
        # Create model
        model = ShipTransformerMVP(
            d_model=16,
            nhead=2, 
            num_layers=1,
            max_ships=4,
            sequence_length=3
        )
        
        # Add states over time
        for t in range(3):
            ships = Ships.from_scalars(
                n_ships=4,
                world_size=(100, 100),
                random_positions=False,
                initial_position=complex(t * 10, t * 5)  # Moving
            )
            history.add_state(ships)
        
        # Generate token sequence
        tokens, ship_ids = history.get_token_sequence()
        
        assert tokens.shape == (12, 12)  # 3 timesteps * 4 ships = 12 tokens
        assert ship_ids.shape == (12,)
        
        # Add batch dimension for model
        tokens_batch = tokens.unsqueeze(0)  # [1, 12, 12]
        ship_ids_batch = ship_ids.unsqueeze(0)  # [1, 12]
        
        # Forward through model
        with torch.no_grad():
            actions = model(tokens_batch, ship_ids_batch)
        
        assert actions.shape == (1, 4, 6)  # batch=1, ships=4, actions=6
        assert torch.isfinite(actions).all()
        
    def test_realistic_game_simulation(self):
        """Test a more realistic game simulation scenario."""
        # Setup
        n_ships = 6
        sequence_length = 4
        n_timesteps = 10
        
        history = StateHistory(
            sequence_length=sequence_length,
            max_ships=8,
            world_size=(1200.0, 800.0),
            normalize_coordinates=True
        )
        
        model = ShipTransformerMVP(
            d_model=32,
            nhead=2,
            num_layers=2,
            max_ships=8,
            sequence_length=sequence_length
        )
        
        # Simulate game timesteps
        for t in range(n_timesteps):
            # Create ships with some variation
            ships = Ships.from_scalars(
                n_ships=n_ships,
                world_size=(1200, 800),
                random_positions=True
            )
            
            # Simulate some damage over time
            ships.health -= torch.rand(n_ships) * 2
            ships.health = torch.clamp(ships.health, 0, 100)
            
            # Create random actions
            actions = torch.randint(0, 2, (n_ships, 6), dtype=torch.bool)
            
            # Add to history
            history.add_state(ships, actions)
            
            # Once we have enough history, test model inference
            if history.is_ready():
                tokens, ship_ids = history.get_token_sequence()
                
                # Batch for model
                tokens_batch = tokens.unsqueeze(0)
                ship_ids_batch = ship_ids.unsqueeze(0)
                
                # Model forward pass
                with torch.no_grad():
                    predicted_actions = model(tokens_batch, ship_ids_batch)
                
                # Verify output
                assert predicted_actions.shape == (1, 8, 6)
                assert torch.isfinite(predicted_actions).all()
                
                # Convert to probabilities and sample
                action_probs = torch.sigmoid(predicted_actions)
                sampled_actions = torch.bernoulli(action_probs)
                
                assert sampled_actions.shape == (1, 8, 6)
                assert torch.all((sampled_actions == 0) | (sampled_actions == 1))
                
    def test_multi_batch_processing(self):
        """Test processing multiple game instances simultaneously."""
        batch_size = 3
        n_ships = 4
        sequence_length = 3
        
        # Create multiple histories
        histories = []
        for _ in range(batch_size):
            history = StateHistory(
                sequence_length=sequence_length,
                max_ships=8,
                world_size=(1000.0, 1000.0)
            )
            
            # Fill with states
            for t in range(sequence_length):
                ships = Ships.from_scalars(
                    n_ships=n_ships,
                    world_size=(1000, 1000),
                    random_positions=True
                )
                history.add_state(ships)
            
            histories.append(history)
        
        # Create model
        model = ShipTransformerMVP(
            d_model=16,
            nhead=2,
            num_layers=1,
            sequence_length=sequence_length
        )
        
        # Get token sequences from all histories
        tokens_list = []
        ship_ids_list = []
        
        for history in histories:
            tokens, ship_ids = history.get_token_sequence()
            tokens_list.append(tokens)
            ship_ids_list.append(ship_ids)
        
        # Stack into batches
        tokens_batch = torch.stack(tokens_list, dim=0)
        ship_ids_batch = torch.stack(ship_ids_list, dim=0)
        
        expected_seq_len = sequence_length * 8  # 8 max ships
        assert tokens_batch.shape == (batch_size, expected_seq_len, 12)
        assert ship_ids_batch.shape == (batch_size, expected_seq_len)
        
        # Process through model
        with torch.no_grad():
            actions_batch = model(tokens_batch, ship_ids_batch)
        
        assert actions_batch.shape == (batch_size, 8, 6)
        assert torch.isfinite(actions_batch).all()
        
    def test_token_encoder_state_history_compatibility(self):
        """Test that TokenEncoder and StateHistory produce compatible outputs."""
        # Create ships
        ships = Ships.from_scalars(
            n_ships=3,
            world_size=(200, 200),
            random_positions=False,
            initial_position=complex(100, 100)
        )
        
        # Method 1: Direct encoding
        encoder = ShipTokenEncoder(
            world_size=(200.0, 200.0),
            normalize_coordinates=True,
            max_ships=8
        )
        direct_tokens = encoder.encode_ships_to_tokens(ships, timestep_offset=-1.0)
        
        # Method 2: Through StateHistory
        history = StateHistory(
            sequence_length=2,
            max_ships=8,
            world_size=(200.0, 200.0),
            normalize_coordinates=True
        )
        
        # Add state twice to get a sequence
        history.add_state(ships)
        history.add_state(ships)
        
        history_tokens, ship_ids = history.get_token_sequence()
        
        # Extract tokens for the last timestep from history (should match direct encoding)
        # Last timestep tokens should be at indices 8-15 (second occurrence of each ship)
        last_timestep_tokens = history_tokens[8:11]  # Ships 0-2 from last timestep
        
        # Compare positions and other non-temporal fields (but not timestep_offset)
        # Direct tokens have timestep_offset = -1, history tokens have timestep_offset = 0
        assert torch.allclose(direct_tokens[:3, :-1], last_timestep_tokens[:, :-1])
        
    def test_gradient_flow_through_full_pipeline(self):
        """Test that gradients flow correctly through the full pipeline."""
        # Create model
        model = ShipTransformerMVP(d_model=16, nhead=2, num_layers=1)
        
        # Create input that requires gradients
        tokens = torch.randn(1, 48, 12, requires_grad=True)
        ship_ids = torch.arange(48) % 8
        ship_ids = ship_ids.unsqueeze(0)
        
        # Forward pass
        actions = model(tokens, ship_ids)
        
        # Create a simple loss
        loss = torch.sum(actions ** 2)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist and are reasonable
        assert tokens.grad is not None
        assert torch.isfinite(tokens.grad).all()
        assert tokens.grad.abs().max() < 10.0  # Gradients shouldn't explode
        
        # Check model parameter gradients
        total_grad_norm = 0.0
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
        
        total_grad_norm = total_grad_norm ** 0.5
        assert total_grad_norm > 0  # Should have non-zero gradients
        assert total_grad_norm < 1000  # Gradients shouldn't explode too much
