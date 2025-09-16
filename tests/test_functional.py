"""
Functional tests that verify end-to-end training capability.
"""
import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.ship_transformer import ShipTransformerMVP
from models.state_history import StateHistory
from core.ships import Ships
from utils.config import ModelConfig


class TestFunctional:
    """Test end-to-end functionality."""
    
    def test_simple_training_loop(self):
        """Test that we can run a simple training loop without errors."""
        # Create small model for fast testing
        model = ShipTransformerMVP(d_model=16, nhead=2, num_layers=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Create state history
        history = StateHistory(sequence_length=3, max_ships=4, world_size=(100, 100))
        
        # Training loop
        losses = []
        for step in range(5):  # Just 5 steps to test
            # Generate some dummy data
            for t in range(3):
                ships = Ships.from_scalars(
                    n_ships=4,
                    world_size=(100, 100),
                    random_positions=True
                )
                # Add some random damage to simulate game progress
                ships.health -= torch.rand(4) * 10
                ships.health = torch.clamp(ships.health, 0, 100)
                
                actions = torch.randint(0, 2, (4, 6), dtype=torch.bool)
                history.add_state(ships, actions)
            
            # Get training data
            if history.is_ready():
                tokens, ship_ids = history.get_token_sequence()
                tokens_batch = tokens.unsqueeze(0)  # Add batch dim
                ship_ids_batch = ship_ids.unsqueeze(0)
                
                # Forward pass
                predicted_actions = model(tokens_batch, ship_ids_batch)
                
                # Simple loss: encourage some action
                target_actions = torch.randint(0, 2, predicted_actions.shape, dtype=torch.float32)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    predicted_actions, target_actions
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                
                # Reset for next iteration
                history.reset()
        
        # Verify that training happened
        assert len(losses) > 0
        assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)
        print(f"Training losses: {losses}")
        
    def test_action_sampling(self):
        """Test that we can sample actions from the model."""
        model = ShipTransformerMVP(d_model=16, nhead=2, num_layers=1)
        
        # Create dummy input
        tokens = torch.randn(1, 12, ModelConfig.TOKEN_DIM)  # 1 batch, 12 tokens, TOKEN_DIM features
        ship_ids = torch.arange(12) % 4  # 4 ships
        ship_ids = ship_ids.unsqueeze(0)
        
        model.eval()
        with torch.no_grad():
            # Get action logits
            action_logits = model(tokens, ship_ids)
            
            # Sample actions
            action_probs = torch.sigmoid(action_logits)
            sampled_actions = torch.bernoulli(action_probs)
            
            # Verify shapes and values (model always outputs for max_ships=8)
            assert action_logits.shape == (1, 8, 6)
            assert action_probs.shape == (1, 8, 6)
            assert sampled_actions.shape == (1, 8, 6)
            
            # Probabilities should be between 0 and 1
            assert torch.all(action_probs >= 0) and torch.all(action_probs <= 1)
            
            # Sampled actions should be 0 or 1
            assert torch.all((sampled_actions == 0) | (sampled_actions == 1))
            
            print(f"Action probabilities sample: {action_probs[0, 0]}")
            print(f"Sampled actions sample: {sampled_actions[0, 0]}")
            
    def test_model_improvement(self):
        """Test that the model can actually improve on a simple task."""
        # Create a simple task: predict constant actions
        model = ShipTransformerMVP(d_model=32, nhead=2, num_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Target: all ships should always move forward (action 0 = True, others False)
        target_pattern = torch.zeros(1, 8, 6)
        target_pattern[0, :, 0] = 1.0  # Forward action for all ships
        
        # Training data: random tokens (shouldn't matter for this simple task)
        tokens = torch.randn(1, 48, ModelConfig.TOKEN_DIM)
        ship_ids = torch.arange(48) % 8
        ship_ids = ship_ids.unsqueeze(0)
        
        # Train for more steps
        initial_loss = None
        final_loss = None
        
        for step in range(50):
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(tokens, ship_ids)
            
            # Loss: encourage forward movement
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                predictions, target_pattern
            )
            
            if step == 0:
                initial_loss = loss.item()
            if step == 49:
                final_loss = loss.item()
                
            # Backward pass
            loss.backward()
            optimizer.step()
        
        # Model should improve (loss should decrease)
        print(f"Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")
        assert final_loss < initial_loss, f"Model didn't improve: {initial_loss} -> {final_loss}"
        
        # Test final predictions
        model.eval()
        with torch.no_grad():
            final_predictions = model(tokens, ship_ids)
            final_probs = torch.sigmoid(final_predictions)
            
            # Should have learned to predict forward movement
            forward_probs = final_probs[0, :, 0]  # Forward action for all ships
            assert torch.mean(forward_probs) > 0.7, f"Didn't learn forward movement: {torch.mean(forward_probs)}"
            
            print(f"Final forward probabilities: {forward_probs}")
            
    def test_different_ship_counts(self):
        """Test that the model handles different numbers of ships correctly."""
        model = ShipTransformerMVP(d_model=16, nhead=2, num_layers=1)
        
        # Test with different ship counts
        for n_ships in [2, 4, 6, 8]:
            history = StateHistory(sequence_length=2, max_ships=8, world_size=(100, 100))
            
            # Add states
            for _ in range(2):
                ships = Ships.from_scalars(n_ships=n_ships, world_size=(100, 100))
                history.add_state(ships)
            
            # Get tokens
            tokens, ship_ids = history.get_token_sequence()
            tokens_batch = tokens.unsqueeze(0)
            ship_ids_batch = ship_ids.unsqueeze(0)
            
            # Forward pass
            with torch.no_grad():
                actions = model(tokens_batch, ship_ids_batch)
            
            # Should always output actions for 8 ships (max_ships)
            assert actions.shape == (1, 8, 6)
            
            # Actions for existing ships should be reasonable
            existing_actions = actions[0, :n_ships, :]
            assert torch.isfinite(existing_actions).all()
            
            # Actions for non-existing ships (should be based on empty tokens)
            if n_ships < 8:
                non_existing_actions = actions[0, n_ships:, :]
                assert torch.isfinite(non_existing_actions).all()
            
            print(f"Tested with {n_ships} ships - OK")