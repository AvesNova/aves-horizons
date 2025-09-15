"""
Test constants and configuration for Aves Horizons test suite.

This module centralizes all test configuration constants to make tests
more maintainable and reduce hardcoded values throughout the test suite.
"""

# Model Architecture Constants
TOKEN_DIM = 13  # Base token dimensions including team_id
ACTION_DIM = 6  # Number of actions per ship
MAX_SHIPS = 8   # Maximum ships supported by model
DEFAULT_SEQUENCE_LENGTH = 6  # Default temporal sequence length

# World/Environment Constants  
DEFAULT_WORLD_SIZE = (1200.0, 800.0)
TEST_WORLD_SIZE = (1000.0, 1000.0)  # Smaller world for testing
MAX_SPEED = 300.0  # Maximum speed for normalization

# Model Configuration Constants
DEFAULT_D_MODEL = 64
DEFAULT_NHEAD = 4
DEFAULT_NUM_LAYERS = 3
TEST_D_MODEL = 16  # Smaller model for faster tests
TEST_NHEAD = 2
TEST_NUM_LAYERS = 1

# Ship Configuration Constants
DEFAULT_N_SHIPS = 4
DEFAULT_HEALTH = 100.0
DEFAULT_MAX_AMMO = 32.0

# Team Configuration Constants
DEFAULT_N_TEAMS = 2
DEFAULT_SHIPS_PER_TEAM = 4

# Test Timing Constants
TEST_SIMULATION_STEPS = 10
TEST_MAX_STEPS = 50

# Tolerance Constants
FLOAT_TOLERANCE = 1e-6
POSITION_TOLERANCE = 1e-4

# Test Data Paths (if we need them later)
TEST_DATA_DIR = "tests/test_data"
FIXTURES_DIR = "tests/fixtures"