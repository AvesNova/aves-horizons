#!/usr/bin/env python3
"""
Test the unified training system to ensure it works correctly.
"""

import torch
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent))

from train import TrainingConfig, DeathmatchMode, TrainingEnvironment, Trainer


def test_training_config():
    """Test training configuration creation."""
    config = TrainingConfig()
    assert config.d_model == 64
    assert config.n_teams == 2
    assert config.ships_per_team == 4
    print("✓ TrainingConfig works")


def test_deathmatch_mode():
    """Test deathmatch mode environment creation."""
    config = TrainingConfig()
    mode = DeathmatchMode()
    
    env = mode.create_environment(config)
    assert env is not None
    assert hasattr(env, 'reset')
    assert hasattr(env, 'step')
    print("✓ DeathmatchMode works")


def test_training_environment():
    """Test training environment wrapper."""
    config = TrainingConfig(
        sequence_length=3,  # Shorter for testing
        ships_per_team=2,   # Smaller for testing
    )
    mode = DeathmatchMode()
    train_env = TrainingEnvironment(mode, config)
    
    # Test reset
    obs = train_env.reset()
    assert obs is not None
    assert isinstance(obs, torch.Tensor)
    print(f"✓ TrainingEnvironment reset works, obs shape: {obs.shape}")
    
    # Test step
    n_controlled = config.ships_per_team
    actions = torch.randint(0, 2, (n_controlled, 6), dtype=torch.bool)
    
    next_obs, reward, done, info = train_env.step(actions)
    assert isinstance(next_obs, torch.Tensor)
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    print(f"✓ TrainingEnvironment step works, reward: {reward}")


def test_trainer_creation():
    """Test trainer creation and basic setup."""
    config = TrainingConfig(
        d_model=16,         # Smaller for testing
        n_head=2,
        num_layers=1,
        sequence_length=3,
        ships_per_team=2,
        batch_size=4,
        steps_per_update=8,
        buffer_size=100
    )
    
    trainer = Trainer(config)
    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert trainer.env is not None
    print("✓ Trainer creation works")


def test_full_training_step():
    """Test a few steps of actual training."""
    config = TrainingConfig(
        d_model=16,         # Small model for testing
        n_head=2,
        num_layers=1,
        sequence_length=3,
        ships_per_team=2,
        batch_size=4,
        steps_per_update=8,
        buffer_size=100,
        log_freq=10,
        save_freq=100,
        eval_freq=100,
        selfplay_update_freq=100
    )
    
    trainer = Trainer(config)
    
    # Run a few training steps
    try:
        trainer.train(total_steps=20)
        print("✓ Training loop completes successfully")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        raise


def test_command_line_interface():
    """Test the command line interface."""
    # Test help
    import subprocess
    result = subprocess.run([
        sys.executable, "train.py", "--help"
    ], capture_output=True, text=True, cwd=Path(__file__).parent)
    
    assert result.returncode == 0
    assert "Aves Horizons Unified Training System" in result.stdout
    print("✓ Command line interface works")


if __name__ == "__main__":
    print("Testing unified training system...")
    
    try:
        test_training_config()
        test_deathmatch_mode() 
        test_training_environment()
        test_trainer_creation()
        test_full_training_step()
        test_command_line_interface()
        
        print("\n🎉 All tests passed! The unified training system is working correctly.")
        
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)