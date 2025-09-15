"""
Test script for ShipTransformer implementation.

This script verifies that all components work together correctly:
- Model architecture
- State history tracking
- Token encoding
- Environment integration
- Training infrastructure
"""

import torch
import numpy as np
from pathlib import Path

from models.ship_transformer import ShipTransformerMVP
from models.state_history import StateHistory
from models.token_encoder import ShipTokenEncoder
from gym_env.ship_transformer_env import ShipTransformerEnv
from core.ships import Ships
from utils.config import Actions


def test_ship_transformer_model():
    """Test the core transformer model."""
    print("Testing ShipTransformer model...")
    
    # Create model
    model = ShipTransformerMVP(
        d_model=64,
        nhead=4,
        num_layers=3
    )
    
    # Test forward pass
    batch_size = 2
    sequence_length = 6
    n_ships = 8
    token_dim = 12
    
    # Create dummy input
    tokens = torch.randn(batch_size, sequence_length * n_ships, token_dim)
    ship_ids = torch.zeros(batch_size, sequence_length * n_ships, dtype=torch.long)
    
    # Fill ship IDs (time-major order)
    for t in range(sequence_length):
        for ship in range(n_ships):
            idx = t * n_ships + ship
            ship_ids[:, idx] = ship
    
    # Forward pass
    with torch.no_grad():
        actions = model(tokens, ship_ids)
    
    print(f"  Input shape: {tokens.shape}")
    print(f"  Output shape: {actions.shape}")
    print(f"  Expected output shape: [{batch_size}, {n_ships}, 6]")
    
    assert actions.shape == (batch_size, n_ships, 6), f"Wrong output shape: {actions.shape}"
    print("✓ Model forward pass successful")


def test_state_history():
    """Test state history tracking."""
    print("\nTesting StateHistory...")
    
    # Create state history
    history = StateHistory(
        sequence_length=6,
        max_ships=8,
        world_size=(1200.0, 800.0),
        normalize_coordinates=True
    )
    
    # Create dummy ships
    ships = Ships.from_scalars(
        n_ships=4,
        world_size=(1200, 800),
        random_positions=True
    )
    
    # Add states to history
    for i in range(10):  # Add more than sequence_length
        # Simulate movement
        ships.position += torch.complex(torch.randn(4) * 10, torch.randn(4) * 10)
        ships.health -= torch.rand(4) * 2  # Gradual health loss
        
        # Create dummy actions
        actions = torch.randint(0, 2, (4, 6), dtype=torch.bool)
        
        history.add_state(ships, actions)
    
    print(f"  History length: {len(history.state_buffer)}")
    print(f"  Is ready: {history.is_ready()}")
    
    # Test token sequence generation
    tokens, ship_ids = history.get_token_sequence()
    print(f"  Token sequence shape: {tokens.shape}")
    print(f"  Ship IDs shape: {ship_ids.shape}")
    
    expected_seq_len = history.sequence_length * history.max_ships
    assert tokens.shape == (expected_seq_len, 12), f"Wrong token shape: {tokens.shape}"
    assert ship_ids.shape == (expected_seq_len,), f"Wrong ship_ids shape: {ship_ids.shape}"
    
    print("✓ StateHistory test successful")


def test_token_encoder():
    """Test token encoder."""
    print("\nTesting ShipTokenEncoder...")
    
    # Create encoder
    encoder = ShipTokenEncoder(
        world_size=(1200.0, 800.0),
        normalize_coordinates=True,
        max_ships=8
    )
    
    # Create dummy ships
    ships = Ships.from_scalars(n_ships=4, world_size=(1200, 800))
    
    # Encode ships
    tokens = encoder.encode_ships_to_tokens(ships, timestep_offset=-2.0)
    
    print(f"  Encoded tokens shape: {tokens.shape}")
    print(f"  Token sample: {tokens[0]}")
    
    assert tokens.shape == (4, 12), f"Wrong token shape: {tokens.shape}"
    assert tokens[0, -1].item() == -2.0, "Timestep offset not encoded correctly"
    
    print("✓ Token encoder test successful")


def test_environment():
    """Test the transformer environment."""
    print("\nTesting ShipTransformerEnv...")
    
    try:
        # Create environment
        env = ShipTransformerEnv(
            n_ships=8,
            controlled_team_size=4,
            sequence_length=6,
            opponent_policy="random"
        )
        
        print(f"  Action space: {env.action_space}")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Controlled ships: {env.get_controlled_ships()}")
        print(f"  Opponent ships: {env.get_opponent_ships()}")
        
        # Reset environment
        obs, info = env.reset()
        print(f"  Initial observation shape: {obs.shape}")
        print(f"  Initial info: {info}")
        
        # Take a few steps
        for step in range(10):
            # Random action for controlled ships
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step == 0:
                print(f"  First step reward: {reward}")
                print(f"  First step observation shape: {obs.shape}")
            
            if terminated or truncated:
                print(f"  Episode ended at step {step + 1}")
                break
        
        env.close()
        print("✓ Environment test successful")
        
    except Exception as e:
        print(f"  Environment test failed: {e}")
        import traceback
        traceback.print_exc()


def test_integration():
    """Test integration of all components."""
    print("\nTesting component integration...")
    
    try:
        # Create model
        model = ShipTransformerMVP()
        
        # Create environment
        env = ShipTransformerEnv(
            n_ships=8,
            controlled_team_size=4,
            sequence_length=6
        )
        
        # Reset environment
        obs, _ = env.reset()
        
        # Process observation through model (simulate policy processing)
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        
        # Reshape to token format
        seq_len = 6 * 8  # sequence_length * n_ships
        tokens = obs_tensor.view(1, seq_len, 12)
        
        # Create ship IDs
        ship_ids = torch.zeros(1, seq_len, dtype=torch.long)
        for t in range(6):
            for ship in range(8):
                idx = t * 8 + ship
                ship_ids[0, idx] = ship
        
        # Forward pass through model
        with torch.no_grad():
            action_logits = model(tokens, ship_ids)
        
        # Extract controlled ship actions
        controlled_actions = action_logits[0, :4, :].flatten()  # First 4 ships
        
        # Convert to binary actions
        actions = (torch.sigmoid(controlled_actions) > 0.5).float().numpy()
        
        # Take environment step
        obs, reward, terminated, truncated, info = env.step(actions)
        
        print(f"  Model output shape: {action_logits.shape}")
        print(f"  Action shape: {actions.shape}")
        print(f"  Step reward: {reward}")
        print("✓ Integration test successful")
        
        env.close()
        
    except Exception as e:
        print(f"  Integration test failed: {e}")
        import traceback
        traceback.print_exc()


def test_model_persistence():
    """Test model saving and loading."""
    print("\nTesting model persistence...")
    
    try:
        from utils.persistence import ModelCheckpoint
        
        # Create and save model
        model = ShipTransformerMVP(d_model=32, nhead=2, num_layers=2)
        
        # Create dummy training state
        checkpoint = ModelCheckpoint(
            model=model,
            epoch=10,
            step=1000,
            best_reward=42.5,
            training_config={"test": True}
        )
        
        # Save checkpoint
        save_path = "test_checkpoint.pt"
        checkpoint.save(save_path)
        
        # Load checkpoint
        loaded_checkpoint = ModelCheckpoint.load(save_path)
        
        print(f"  Original epoch: {checkpoint.epoch}, Loaded epoch: {loaded_checkpoint.epoch}")
        print(f"  Original reward: {checkpoint.best_reward}, Loaded reward: {loaded_checkpoint.best_reward}")
        
        # Test model weights match
        original_params = list(model.parameters())
        loaded_params = list(loaded_checkpoint.model.parameters())
        
        weights_match = all(torch.allclose(p1, p2) for p1, p2 in zip(original_params, loaded_params))
        print(f"  Model weights match: {weights_match}")
        
        # Clean up
        Path(save_path).unlink()
        
        print("✓ Model persistence test successful")
        
    except Exception as e:
        print(f"  Model persistence test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("Running ShipTransformer tests...")
    print("=" * 50)
    
    test_ship_transformer_model()
    test_state_history()
    test_token_encoder()
    test_environment()
    test_integration()
    test_model_persistence()
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    
    print("\nNext steps:")
    print("1. Run training: python train_ship_transformer.py")
    print("2. Monitor training progress in ./logs/")
    print("3. Evaluate trained model using the evaluation functions")


if __name__ == "__main__":
    main()