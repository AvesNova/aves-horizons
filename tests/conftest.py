"""
Shared test fixtures and configuration for the ship game test suite.
"""

import pytest
import numpy as np
import torch
from typing import Generator

# Add src to path for imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ship import Ship, ShipConfig
from bullets import Bullets
from env import Environment
from state import State
from constants import Actions
from derive_ship_parameters import derive_ship_parameters


# --- Ship Derived Parameters Fixtures ---


@pytest.fixture
def derived_ship_parameters() -> dict:
    """Provides the default ship configuration."""
    return derive_ship_parameters()


# --- Ship Configuration Fixtures ---


@pytest.fixture
def default_ship_config() -> ShipConfig:
    """Provides the default ship configuration."""
    return ShipConfig()


@pytest.fixture
def minimal_ship_config() -> ShipConfig:
    """Provides a minimal ship config for faster testing."""
    return ShipConfig(
        collision_radius=5.0,
        max_health=50.0,
        max_power=50.0,
        base_thrust=5.0,
        boost_thrust=40.0,
        reverse_thrust=-5.0,
        bullet_lifetime=0.5,
        firing_cooldown=0.1,
    )


@pytest.fixture
def zero_drag_ship_config() -> ShipConfig:
    """Ship config with no drag for predictable physics tests."""
    return ShipConfig(
        no_turn_drag_coeff=0.0,
        normal_turn_drag_coeff=0.0,
        sharp_turn_drag_coeff=0.0,
        normal_turn_lift_coeff=0.0,
        sharp_turn_lift_coeff=0.0,
    )


# --- Ship Fixtures ---


@pytest.fixture
def basic_ship(default_ship_config) -> Ship:
    """Creates a basic ship at origin moving right."""
    return Ship(
        ship_id=0,
        team_id=0,
        ship_config=default_ship_config,
        initial_x=400.0,
        initial_y=300.0,
        initial_vx=100.0,
        initial_vy=0.0,
        rng=np.random.default_rng(42),
    )


@pytest.fixture
def stationary_ship_attempt():
    """Returns a function that attempts to create a stationary ship (should fail)."""

    def _create():
        return Ship(
            ship_id=0,
            team_id=0,
            ship_config=ShipConfig(),
            initial_x=400.0,
            initial_y=300.0,
            initial_vx=0.0,
            initial_vy=0.0,
        )

    return _create


@pytest.fixture
def two_ships(default_ship_config) -> tuple[Ship, Ship]:
    """Creates two opposing ships."""
    ship1 = Ship(
        ship_id=0,
        team_id=0,
        ship_config=default_ship_config,
        initial_x=200.0,
        initial_y=300.0,
        initial_vx=100.0,
        initial_vy=0.0,
        rng=np.random.default_rng(42),
    )
    ship2 = Ship(
        ship_id=1,
        team_id=1,
        ship_config=default_ship_config,
        initial_x=600.0,
        initial_y=300.0,
        initial_vx=-100.0,
        initial_vy=0.0,
        rng=np.random.default_rng(43),
    )
    return ship1, ship2


# --- Bullet System Fixtures ---


@pytest.fixture
def empty_bullets() -> Bullets:
    """Creates an empty bullet container."""
    return Bullets(max_bullets=10)


@pytest.fixture
def bullets_with_some() -> Bullets:
    """Creates a bullet container with some bullets."""
    bullets = Bullets(max_bullets=10)
    bullets.add_bullet(0, 100.0, 100.0, 50.0, 0.0, 1.0)
    bullets.add_bullet(1, 200.0, 200.0, -50.0, 50.0, 0.5)
    bullets.add_bullet(0, 300.0, 100.0, 0.0, -50.0, 0.75)
    return bullets


# --- Environment Fixtures ---


@pytest.fixture
def basic_env() -> Environment:
    """Creates a basic environment for testing."""
    return Environment(
        render_mode=None,
        world_size=(800, 600),
        memory_size=1,
        n_ships=2,
        agent_dt=0.02,
        physics_dt=0.02,
    )


@pytest.fixture
def env_with_substeps() -> Environment:
    """Environment with multiple physics substeps per agent step."""
    return Environment(
        render_mode=None,
        world_size=(800, 600),
        memory_size=1,
        n_ships=2,
        agent_dt=0.04,  # 2x physics_dt
        physics_dt=0.02,
    )


# --- Action Fixtures ---


@pytest.fixture
def no_action() -> torch.Tensor:
    """Returns a zero action tensor."""
    return torch.zeros(len(Actions))


@pytest.fixture
def forward_action() -> torch.Tensor:
    """Returns an action tensor for moving forward."""
    action = torch.zeros(len(Actions))
    action[Actions.forward] = 1
    return action


@pytest.fixture
def all_actions() -> dict[str, torch.Tensor]:
    """Returns a dictionary of all individual actions."""
    actions = {}
    for action_name in Actions:
        action = torch.zeros(len(Actions))
        action[action_name] = 1
        actions[action_name.name] = action
    return actions


@pytest.fixture
def action_combinations() -> dict[str, torch.Tensor]:
    """Returns common action combinations."""
    combinations = {}

    # Single actions
    for action_name in Actions:
        action = torch.zeros(len(Actions))
        action[action_name] = 1
        combinations[action_name.name] = action

    # Common combinations
    action = torch.zeros(len(Actions))
    action[Actions.forward] = 1
    action[Actions.left] = 1
    combinations["forward_left"] = action

    action = torch.zeros(len(Actions))
    action[Actions.forward] = 1
    action[Actions.right] = 1
    combinations["forward_right"] = action

    action = torch.zeros(len(Actions))
    action[Actions.left] = 1
    action[Actions.right] = 1
    combinations["left_right"] = action

    action = torch.zeros(len(Actions))
    action[Actions.sharp_turn] = 1
    action[Actions.left] = 1
    combinations["sharp_left"] = action

    action = torch.zeros(len(Actions))
    action[Actions.forward] = 1
    action[Actions.shoot] = 1
    combinations["forward_shoot"] = action

    return combinations


# --- State Fixtures ---


@pytest.fixture
def empty_state() -> State:
    """Creates an empty state."""
    return State(ships={})


@pytest.fixture
def combat_state(two_ships) -> State:
    """Creates a state with two ships ready for combat."""
    ship1, ship2 = two_ships
    return State(ships={0: ship1, 1: ship2})


# --- Random Number Generator Fixtures ---


@pytest.fixture
def fixed_rng() -> np.random.Generator:
    """Provides a fixed-seed RNG for deterministic tests."""
    return np.random.default_rng(12345)


# --- Tolerance Fixtures ---


@pytest.fixture
def physics_tolerance() -> float:
    """Tolerance for physics calculations."""
    return 1e-5


@pytest.fixture
def loose_tolerance() -> float:
    """Looser tolerance for aggregate calculations."""
    return 1e-3


# --- Helper Function Fixtures ---


@pytest.fixture
def step_environment():
    """Returns a function to step environment multiple times."""

    def _step(env: Environment, actions: dict, n_steps: int = 1):
        observations = []
        rewards = []
        infos = []

        for _ in range(n_steps):
            obs, rew, term, trunc, info = env.step(actions)
            observations.append(obs)
            rewards.append(rew)
            infos.append(info)

            if term or trunc:
                break

        return observations, rewards, infos

    return _step


@pytest.fixture
def assert_complex_close():
    """Returns a function to assert complex numbers are close."""

    def _assert(actual: complex, expected: complex, tolerance: float = 1e-5):
        assert (
            abs(actual.real - expected.real) < tolerance
        ), f"Real part mismatch: {actual.real} != {expected.real}"
        assert (
            abs(actual.imag - expected.imag) < tolerance
        ), f"Imaginary part mismatch: {actual.imag} != {expected.imag}"

    return _assert


@pytest.fixture
def assert_vector_close():
    """Returns a function to assert numpy arrays are close."""

    def _assert(actual: np.ndarray, expected: np.ndarray, tolerance: float = 1e-5):
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)

    return _assert
