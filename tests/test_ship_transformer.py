"""
Test ShipTransformer model functionality.
"""
import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.ship_transformer import ShipTransformerMVP


class TestShipTransformer:
    """Test the ShipTransformer model."""
    
    def test_model_creation(self):
        """Test basic model creation."""
        model = ShipTransformerMVP(
            d_model=64,
            nhead=4,
            num_layers=3
        )
        
        assert model.d_model == 64
        assert model.nhead == 4
        assert model.num_layers == 3
        assert model.max_ships == 8  # Default for MVP
        assert model.sequence_length == 6  # Default for MVP
        assert model.base_token_dim == 12
        assert model.action_dim == 6
        
    def test_model_forward_pass(self):
        """Test basic forward pass through the model."""
        model = ShipTransformerMVP(d_model=32, nhead=2, num_layers=2)  # Small model for testing
        
        batch_size = 2
        sequence_length = 6
        n_ships = 8
        token_dim = 12
        
        # Create dummy input in the expected format
        tokens = torch.randn(batch_size, sequence_length * n_ships, token_dim)
        
        # Create ship IDs (time-major order)
        ship_ids = torch.zeros(batch_size, sequence_length * n_ships, dtype=torch.long)
        for b in range(batch_size):
            for t in range(sequence_length):
                for ship in range(n_ships):
                    idx = t * n_ships + ship
                    ship_ids[b, idx] = ship
        
        # Forward pass
        with torch.no_grad():
            actions = model(tokens, ship_ids)
        
        # Check output shape
        expected_shape = (batch_size, n_ships, 6)
        assert actions.shape == expected_shape, f"Expected {expected_shape}, got {actions.shape}"
        
        # Check that outputs are reasonable (not NaN or infinite)
        assert torch.isfinite(actions).all()
        
    def test_ship_embeddings(self):
        """Test that ship embeddings are working."""
        model = ShipTransformerMVP(d_model=16)
        
        # Test that different ships get different embeddings
        ship_ids = torch.tensor([[0, 1, 2, 3]])
        embeddings = model.ship_embeddings(ship_ids)
        
        assert embeddings.shape == (1, 4, 16)
        
        # Embeddings for different ships should be different
        assert not torch.allclose(embeddings[0, 0], embeddings[0, 1])
        assert not torch.allclose(embeddings[0, 0], embeddings[0, 2])
        
    def test_input_projection(self):
        """Test input projection layer."""
        model = ShipTransformerMVP(d_model=32)
        
        # Create dummy tokens
        batch_size = 1
        seq_len = 8
        token_dim = 12
        tokens = torch.randn(batch_size, seq_len, token_dim)
        
        # Test input projection
        projected = model.input_projection(tokens)
        
        assert projected.shape == (batch_size, seq_len, 32)
        assert torch.isfinite(projected).all()
        
    def test_deterministic_output(self):
        """Test that model produces deterministic output."""
        torch.manual_seed(42)
        
        model = ShipTransformerMVP(d_model=16, nhead=2, num_layers=1)
        
        # Create deterministic input
        tokens = torch.ones(1, 48, 12)  # 6 timesteps * 8 ships = 48
        ship_ids = torch.zeros(1, 48, dtype=torch.long)
        for i in range(48):
            ship_ids[0, i] = i % 8
        
        # Two forward passes should be identical
        model.eval()
        with torch.no_grad():
            output1 = model(tokens, ship_ids)
            output2 = model(tokens, ship_ids)
        
        assert torch.allclose(output1, output2)
        
    def test_different_batch_sizes(self):
        """Test model with different batch sizes."""
        model = ShipTransformerMVP(d_model=16, nhead=2, num_layers=1)
        
        seq_len = 48  # 6 * 8
        token_dim = 12
        
        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            tokens = torch.randn(batch_size, seq_len, token_dim)
            ship_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
            for b in range(batch_size):
                for i in range(seq_len):
                    ship_ids[b, i] = i % 8
            
            with torch.no_grad():
                actions = model(tokens, ship_ids)
            
            assert actions.shape == (batch_size, 8, 6)
            assert torch.isfinite(actions).all()
            
    def test_action_head_outputs(self):
        """Test that action head produces reasonable outputs."""
        model = ShipTransformerMVP(d_model=16, nhead=2, num_layers=1)
        
        # Create simple input
        tokens = torch.zeros(1, 48, 12)
        ship_ids = torch.arange(48) % 8
        ship_ids = ship_ids.unsqueeze(0)
        
        with torch.no_grad():
            actions = model(tokens, ship_ids)
        
        # Actions should be logits (can be any real value)
        assert actions.shape == (1, 8, 6)
        assert torch.isfinite(actions).all()
        
        # Convert to probabilities using sigmoid
        probs = torch.sigmoid(actions)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
        
    def test_gradient_flow(self):
        """Test that gradients can flow through the model."""
        model = ShipTransformerMVP(d_model=16, nhead=2, num_layers=1)
        
        tokens = torch.randn(1, 48, 12, requires_grad=True)
        ship_ids = torch.arange(48) % 8
        ship_ids = ship_ids.unsqueeze(0)
        
        # Forward pass
        actions = model(tokens, ship_ids)
        loss = actions.sum()  # Simple loss for testing
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist and are finite
        assert tokens.grad is not None
        assert torch.isfinite(tokens.grad).all()
        
        # Check that model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert torch.isfinite(param.grad).all()
                
    def test_model_parameter_count(self):
        """Test that model has reasonable parameter count."""
        model = ShipTransformerMVP(d_model=64, nhead=4, num_layers=3)
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Should be reasonable for a small transformer
        assert 10000 < total_params < 200000  # Between 10K and 200K parameters