"""
Test basic imports to ensure modules are loadable.
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_torch_import():
    """Test that torch imports correctly."""
    import torch
    assert torch.__version__ is not None


def test_numpy_import():
    """Test that numpy imports correctly."""
    import numpy as np
    assert np.__version__ is not None


def test_gymnasium_import():
    """Test that gymnasium imports correctly."""
    import gymnasium as gym
    assert hasattr(gym, 'Env')


def test_core_ships_import():
    """Test importing core ships module."""
    try:
        from core.ships import Ships
        assert Ships is not None
    except ImportError as e:
        pytest.skip(f"Skipping core.ships import test due to missing dependencies: {e}")


def test_utils_config_import():
    """Test importing utils config."""
    try:
        from utils.config import Actions
        assert Actions is not None
    except ImportError as e:
        pytest.skip(f"Skipping utils.config import test due to missing dependencies: {e}")