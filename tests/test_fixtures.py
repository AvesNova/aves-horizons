"""
Test fixtures for Aves Horizons test suite.

This module provides reusable pytest fixtures for common test objects
like ships, models, environments, etc. This reduces code duplication
and makes tests more maintainable.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.ships import Ships
from core.environment import Environment
from models.ship_nn import ShipNN
from game_modes.deathmatch import DeathmatchEnvironment, DeathmatchConfig, create_deathmatch_game
from .test_constants import *


@pytest.fixture
def default_ships():
    """Create a standard set of ships for testing."""
    return Ships.from_scalars(
        n_ships=DEFAULT_N_SHIPS,
        world_size=DEFAULT_WORLD_SIZE,
        random_positions=False,
        initial_position=complex(DEFAULT_WORLD_SIZE[0]/2, DEFAULT_WORLD_SIZE[1]/2),
        team_ids=[0, 0, 1, 1]  # Two teams of 2 ships each
    )


@pytest.fixture
def test_ships():
    """Create a smaller set of ships for faster testing."""
    return Ships.from_scalars(
        n_ships=2,
        world_size=TEST_WORLD_SIZE,
        random_positions=False,
        initial_position=complex(TEST_WORLD_SIZE[0]/2, TEST_WORLD_SIZE[1]/2),
        team_ids=[0, 1]
    )


@pytest.fixture
def random_ships():
    """Create ships with random positions for testing variability."""
    return Ships.from_scalars(
        n_ships=DEFAULT_N_SHIPS,
        world_size=DEFAULT_WORLD_SIZE,
        random_positions=True,
        team_ids=[0, 0, 1, 1]
    )


@pytest.fixture
def test_environment():
    """Create a basic environment for testing."""
    return Environment(n_ships=DEFAULT_N_SHIPS, n_obstacles=0)


@pytest.fixture
def deathmatch_game():
    """Create a deathmatch game for testing team-based gameplay."""
    return create_deathmatch_game(
        n_teams=DEFAULT_N_TEAMS,
        ships_per_team=DEFAULT_SHIPS_PER_TEAM,
        world_size=DEFAULT_WORLD_SIZE
    )


@pytest.fixture
def test_ship_nn():
    """Create a small ShipNN model for testing."""
    return ShipNN(
        hidden_dim=TEST_D_MODEL,
        encoder_layers=1,
        transformer_layers=TEST_NUM_LAYERS,
        decoder_layers=1,
        n_heads=TEST_NHEAD,
        max_ships=MAX_SHIPS,
        sequence_length=DEFAULT_SEQUENCE_LENGTH
    )


@pytest.fixture
def production_ship_nn():
    """Create a full-size ShipNN model for integration testing."""
    return ShipNN(
        hidden_dim=DEFAULT_D_MODEL,
        encoder_layers=2,
        transformer_layers=DEFAULT_NUM_LAYERS,
        decoder_layers=2,
        n_heads=DEFAULT_NHEAD,
        max_ships=MAX_SHIPS,
        sequence_length=DEFAULT_SEQUENCE_LENGTH
    )




@pytest.fixture
def sample_actions():
    """Create sample actions tensor for testing."""
    return torch.zeros((DEFAULT_N_SHIPS, ACTION_DIM), dtype=torch.bool)


@pytest.fixture
def random_actions():
    """Create random actions for testing variability."""
    return torch.randint(0, 2, (DEFAULT_N_SHIPS, ACTION_DIM), dtype=torch.bool)


@pytest.fixture
def combat_scenario_ships():
    """Create ships positioned for combat testing."""
    ships = Ships.from_scalars(
        n_ships=2,
        world_size=TEST_WORLD_SIZE,
        random_positions=False,
        initial_position=complex(100, 100),
        team_ids=[0, 1]
    )
    # Position ships close together for combat
    ships.position[0] = complex(100, 100)  # Ship 1
    ships.position[1] = complex(150, 100)  # Ship 2, nearby
    return ships


@pytest.fixture
def trailing_scenario_ships():
    """Create ships positioned for trailing/pursuit testing."""
    ships = Ships.from_scalars(
        n_ships=2,
        world_size=TEST_WORLD_SIZE,
        random_positions=False,
        team_ids=[0, 1]
    )
    # Set up trailing scenario
    ships.position[0] = complex(100, 100)    # Leading ship
    ships.position[1] = complex(50, 100)     # Trailing ship
    ships.velocity[0] = complex(50, 0)       # Both moving right
    ships.velocity[1] = complex(50, 0)
    ships.attitude[0] = complex(1, 0)        # Both facing right
    ships.attitude[1] = complex(1, 0)
    return ships


# Helper fixtures for common test data

@pytest.fixture
def expected_token_shape():
    """Expected shape for token tensors."""
    return (DEFAULT_N_SHIPS, TOKEN_DIM)


@pytest.fixture
def expected_action_shape():
    """Expected shape for action tensors."""
    return (DEFAULT_N_SHIPS, ACTION_DIM)


@pytest.fixture
def test_tolerance():
    """Standard tolerance for floating point comparisons."""
    return FLOAT_TOLERANCE


# Context managers and utilities

@pytest.fixture
def no_grad():
    """Fixture that provides torch.no_grad context."""
    return torch.no_grad


# Mock data generators

@pytest.fixture
def mock_token_sequence():
    """Generate a mock token sequence for testing."""
    seq_len = DEFAULT_SEQUENCE_LENGTH * MAX_SHIPS
    return torch.randn(seq_len, TOKEN_DIM), torch.arange(seq_len) % MAX_SHIPS


@pytest.fixture
def mock_batch_tokens():
    """Generate mock batch token data."""
    batch_size = 3
    seq_len = DEFAULT_SEQUENCE_LENGTH * MAX_SHIPS
    tokens = torch.randn(batch_size, seq_len, TOKEN_DIM)
    ship_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1) % MAX_SHIPS
    return tokens, ship_ids