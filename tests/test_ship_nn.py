"""
Test ShipNN model functionality.

Tests the three-stage neural network architecture for ship combat AI.
"""

import pytest
import torch
import numpy as np

from models.ship_nn import ShipNN, create_ship_nn
from utils.config import ModelConfig


class TestShipNN:
    """Test ShipNN model architecture."""
    
    def test_shipnn_creation(self):
        """Test basic ShipNN model creation."""
        model = ShipNN()
        assert model is not None
        assert model.hidden_dim == 128
        assert model.max_ships == 8
        assert model.sequence_length == 6
    
    def test_shipnn_forward_pass(self):
        """Test forward pass through ShipNN."""
        model = ShipNN(
            hidden_dim=64,
            encoder_layers=1,
            transformer_layers=2,
            decoder_layers=1,
            n_heads=2
        )
        
        batch_size = 2
        seq_len = model.sequence_length * model.max_ships
        
        # Create sample input
        tokens = torch.randn(batch_size, seq_len, ModelConfig.TOKEN_DIM)
        ship_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        
        # Fill ship_ids in time-major order
        for t in range(model.sequence_length):
            for ship_id in range(model.max_ships):
                idx = t * model.max_ships + ship_id
                if idx < seq_len:
                    ship_ids[:, idx] = ship_id
        
        # Forward pass
        actions = model(tokens, ship_ids)
        
        assert actions.shape == (batch_size, model.max_ships, 6)
    
    def test_shipnn_features_mode(self):
        """Test ShipNN feature extraction mode."""
        model = ShipNN(hidden_dim=32)
        
        batch_size = 1
        seq_len = model.sequence_length * model.max_ships
        
        tokens = torch.randn(batch_size, seq_len, ModelConfig.TOKEN_DIM)
        ship_ids = torch.arange(model.max_ships).repeat(model.sequence_length).unsqueeze(0)
        
        # Get features instead of actions
        features = model(tokens, ship_ids, return_features=True)
        
        assert features.shape == (batch_size, model.max_ships, model.hidden_dim)
    
    def test_action_probabilities(self):
        """Test action probability computation."""
        model = ShipNN()
        
        # Sample action logits
        action_logits = torch.randn(2, 8, 6)
        
        # Get probabilities
        probs = model.get_action_probabilities(action_logits)
        
        assert probs.shape == action_logits.shape
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
    
    def test_action_sampling(self):
        """Test action sampling."""
        model = ShipNN()
        
        # Sample action logits
        action_logits = torch.randn(2, 8, 6)
        
        # Sample actions
        actions = model.sample_actions(action_logits)
        
        assert actions.shape == action_logits.shape
        assert torch.all(actions.int() == actions)  # Should be binary
        assert torch.all((actions == 0) | (actions == 1))  # 0 or 1 only


class TestCreateShipNN:
    """Test ShipNN factory function."""
    
    def test_default_preset(self):
        """Test default preset configuration."""
        model = create_ship_nn("default")
        assert model.hidden_dim == 128
        assert model.encoder_layers == 2
        assert model.transformer_layers == 3
        assert model.decoder_layers == 2
    
    def test_small_preset(self):
        """Test small preset configuration."""
        model = create_ship_nn("small")
        assert model.hidden_dim == 64
        assert model.encoder_layers == 1
        assert model.transformer_layers == 2
    
    def test_large_preset(self):
        """Test large preset configuration."""
        model = create_ship_nn("large")
        assert model.hidden_dim == 256
        assert model.transformer_layers == 6
    
    def test_preset_override(self):
        """Test preset with parameter overrides."""
        model = create_ship_nn("small", hidden_dim=96)
        assert model.hidden_dim == 96  # Override
        assert model.encoder_layers == 1  # From preset
    
    def test_invalid_preset(self):
        """Test invalid preset raises error."""
        with pytest.raises(ValueError, match="Unknown preset"):
            create_ship_nn("invalid_preset")


class TestModelArchitecture:
    """Test model architecture details."""
    
    def test_encoder_layers(self):
        """Test different encoder layer configurations."""
        # Single layer encoder
        model1 = ShipNN(encoder_layers=1)
        assert model1.encoder.num_layers == 1
        
        # Multi-layer encoder
        model2 = ShipNN(encoder_layers=3)
        assert model2.encoder.num_layers == 3
    
    def test_decoder_layers(self):
        """Test different decoder layer configurations."""
        # Single layer decoder
        model1 = ShipNN(decoder_layers=1)
        assert model1.decoder.num_layers == 1
        
        # Multi-layer decoder
        model2 = ShipNN(decoder_layers=3)
        assert model2.decoder.num_layers == 3
    
    def test_ship_embeddings(self):
        """Test ship identity embeddings."""
        # With ship embeddings (default)
        model1 = ShipNN(use_ship_embeddings=True)
        assert model1.ship_embeddings is not None
        
        # Without ship embeddings
        model2 = ShipNN(use_ship_embeddings=False)
        assert model2.ship_embeddings is None
    
    def test_positional_encoding(self):
        """Test positional encoding."""
        # With positional encoding (default)
        model1 = ShipNN(use_positional_encoding=True)
        assert model1.positional_encoding is not None
        
        # Without positional encoding
        model2 = ShipNN(use_positional_encoding=False)
        assert model2.positional_encoding is None


class TestTrainingCompatibility:
    """Test compatibility with training system."""
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = ShipNN(hidden_dim=32)
        
        batch_size = 2
        seq_len = model.sequence_length * model.max_ships
        
        tokens = torch.randn(batch_size, seq_len, ModelConfig.TOKEN_DIM, requires_grad=True)
        ship_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        
        # Forward pass
        actions = model(tokens, ship_ids)
        
        # Compute dummy loss
        loss = actions.sum()
        loss.backward()
        
        # Check gradients exist
        assert tokens.grad is not None
        
        # Check model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_different_batch_sizes(self):
        """Test model works with different batch sizes."""
        model = ShipNN()
        seq_len = model.sequence_length * model.max_ships
        
        for batch_size in [1, 2, 4, 8]:
            tokens = torch.randn(batch_size, seq_len, ModelConfig.TOKEN_DIM)
            ship_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
            
            actions = model(tokens, ship_ids)
            assert actions.shape == (batch_size, model.max_ships, 6)