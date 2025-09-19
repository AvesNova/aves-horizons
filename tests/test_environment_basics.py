"""
Tests for basic environment operations and state management.
"""

import pytest
import numpy as np
import torch
from gymnasium import spaces

from constants import Actions


class TestEnvironmentInitialization:
    """Tests for environment initialization."""

    def test_basic_initialization(self, basic_env):
        """Test that environment initializes with correct parameters."""
        assert basic_env.world_size == (800, 600)
        assert basic_env.memory_size == 1
        assert basic_env.n_ships == 2
        assert basic_env.agent_dt == 0.02
        assert basic_env.physics_dt == 0.02
        assert basic_env.physics_substeps == 1

    def test_substep_calculation(self, env_with_substeps):
        """Test physics substep calculation."""
        assert env_with_substeps.agent_dt == 0.04
        assert env_with_substeps.physics_dt == 0.02
        assert env_with_substeps.physics_substeps == 2

    def test_invalid_timestep_ratio(self):
        """Test that non-integer timestep ratios fail."""
        from env import Environment

        with pytest.raises(
            AssertionError, match="agent_dt must be multiple of physics_dt"
        ):
            Environment(agent_dt=0.03, physics_dt=0.02)  # Not a multiple of 0.02

    def test_state_initialization(self, basic_env):
        """Test that state deque initializes correctly."""
        assert len(basic_env.state) == 0
        assert basic_env.state.maxlen == basic_env.memory_size
        assert basic_env.current_time == 0.0


class TestReset:
    """Tests for environment reset."""

    def test_reset_1v1(self, basic_env):
        """Test 1v1 reset creates correct initial state."""
        obs, info = basic_env.reset(game_mode="1v1")

        assert len(basic_env.state) == 1
        state = basic_env.state[0]

        # Check ships were created
        assert len(state.ships) == 2
        assert 0 in state.ships
        assert 1 in state.ships

        # Check initial positions (from one_vs_one_reset)
        ship0 = state.ships[0]
        ship1 = state.ships[1]

        assert ship0.position.real == 0.25 * basic_env.world_size[0]
        assert ship0.position.imag == 0.40 * basic_env.world_size[1]
        assert ship1.position.real == 0.75 * basic_env.world_size[0]
        assert ship1.position.imag == 0.60 * basic_env.world_size[1]

        # Check velocities (opposing)
        assert ship0.velocity.real > 0
        assert ship1.velocity.real < 0

    def test_reset_clears_state(self, basic_env):
        """Test that reset clears previous state."""
        # First reset
        basic_env.reset(game_mode="1v1")

        # Step to add more state
        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        basic_env.step(actions)

        # Reset again
        basic_env.reset(game_mode="1v1")

        assert len(basic_env.state) == 1
        assert basic_env.current_time == 0.0

    def test_reset_returns_observations(self, basic_env):
        """Test that reset returns proper observations."""
        obs, info = basic_env.reset(game_mode="1v1")

        assert isinstance(obs, dict)
        assert 0 in obs
        assert 1 in obs

        # Check observation structure
        for ship_id in [0, 1]:
            ship_obs = obs[ship_id]
            assert "self_state" in ship_obs
            assert "enemy_state" in ship_obs
            assert "bullets" in ship_obs
            assert "world_bounds" in ship_obs
            assert "time" in ship_obs

    def test_invalid_game_mode(self, basic_env):
        """Test that invalid game mode raises error."""
        with pytest.raises(ValueError, match="Unknown game mode"):
            basic_env.reset(game_mode="invalid")


class TestObservations:
    """Tests for observation generation."""

    def test_observation_structure(self, basic_env):
        """Test observation has correct structure."""
        obs, _ = basic_env.reset(game_mode="1v1")

        for ship_id, ship_obs in obs.items():
            # Check all required keys
            assert isinstance(ship_obs["self_state"], np.ndarray)
            assert isinstance(ship_obs["enemy_state"], np.ndarray)
            assert isinstance(ship_obs["bullets"], np.ndarray)
            assert isinstance(ship_obs["world_bounds"], np.ndarray)
            assert isinstance(ship_obs["time"], float)

            # Check shapes
            assert ship_obs["self_state"].shape == (8,)
            assert ship_obs["enemy_state"].shape == (8,)
            assert ship_obs["bullets"].shape == (20, 6)
            assert ship_obs["world_bounds"].shape == (2,)

    def test_self_state_normalization(self, basic_env):
        """Test that self state is properly normalized."""
        obs, _ = basic_env.reset(game_mode="1v1")

        ship0_obs = obs[0]
        self_state = ship0_obs["self_state"]

        # Position should be normalized to [0, 1]
        assert 0 <= self_state[0] <= 1  # x position
        assert 0 <= self_state[1] <= 1  # y position

        # Velocity normalized by 1000
        assert abs(self_state[2]) < 1.0  # vx
        assert abs(self_state[3]) < 1.0  # vy

        # Health and power ratios
        assert 0 <= self_state[4] <= 1  # health
        assert 0 <= self_state[5] <= 1  # power

        # Attitude is unit vector
        attitude_mag = np.sqrt(self_state[6] ** 2 + self_state[7] ** 2)
        assert abs(attitude_mag - 1.0) < 1e-5

    def test_enemy_state_identification(self, basic_env):
        """Test that enemy state correctly identifies opposing team."""
        obs, _ = basic_env.reset(game_mode="1v1")

        # Ship 0 should see ship 1 as enemy
        ship0_enemy = obs[0]["enemy_state"]
        assert np.sum(ship0_enemy) > 0  # Not all zeros

        # Ship 1 should see ship 0 as enemy
        ship1_enemy = obs[1]["enemy_state"]
        assert np.sum(ship1_enemy) > 0  # Not all zeros

    def test_empty_observation_before_reset(self, basic_env):
        """Test that observations before reset are empty."""
        obs = basic_env.get_observation()

        for ship_id in range(basic_env.n_ships):
            assert ship_id in obs
            ship_obs = obs[ship_id]
            assert np.all(ship_obs["self_state"] == 0)
            assert np.all(ship_obs["enemy_state"] == 0)
            assert np.all(ship_obs["bullets"] == 0)

    def test_bullet_observations(self, basic_env, step_environment):
        """Test bullet observations are generated correctly."""
        basic_env.reset(game_mode="1v1")

        # Make ship 0 shoot
        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        actions[0][Actions.shoot] = 1

        # Step multiple times to create bullets
        for _ in range(3):
            basic_env.step(actions)
            basic_env.state[-1].ships[0].last_fired_time = -1.0  # Reset cooldown

        obs = basic_env.get_observation()

        # Ship 1 should see bullets
        ship1_bullets = obs[1]["bullets"]
        non_zero_bullets = np.any(ship1_bullets != 0, axis=1)
        assert np.sum(non_zero_bullets) > 0  # At least one bullet visible

    def test_bullet_relative_positions(self, basic_env):
        """Test that bullet positions are relative to observing ship."""
        basic_env.reset(game_mode="1v1")
        state = basic_env.state[-1]

        # Manually add a bullet
        state.bullets.add_bullet(
            ship_id=0, x=400.0, y=300.0, vx=100.0, vy=0.0, lifetime=1.0
        )

        obs = basic_env.get_observation()

        # Get ship 1's observation of the bullet
        ship1 = state.ships[1]
        ship1_bullets = obs[1]["bullets"]

        # First bullet should have relative position
        rel_x = ship1_bullets[0, 0] * basic_env.world_size[0]
        rel_y = ship1_bullets[0, 1] * basic_env.world_size[1]

        expected_rel_x = 400.0 - ship1.position.real
        expected_rel_y = 300.0 - ship1.position.imag

        assert abs(rel_x - expected_rel_x) < 1.0
        assert abs(rel_y - expected_rel_y) < 1.0


class TestWorldWrapping:
    """Tests for toroidal world wrapping."""

    def test_ship_position_wrapping(self, basic_env):
        """Test that ship positions wrap at boundaries."""
        basic_env.reset(game_mode="1v1")

        # Move ship 0 past right boundary
        ship0 = basic_env.state[-1].ships[0]
        initial_velocity = ship0.velocity
        ship0.position = basic_env.world_size[0] + 50.0 + 300.0j

        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        basic_env.step(actions)

        # Position should wrap, accounting for movement during physics step
        wrapped_ship = basic_env.state[-1].ships[0]
        # Expected position: (initial + velocity*dt) % world_size
        expected_real = (
            basic_env.world_size[0] + 50.0 + initial_velocity.real * basic_env.agent_dt
        ) % basic_env.world_size[0]
        assert abs(wrapped_ship.position.real - expected_real) < 1.0
        assert abs(wrapped_ship.position.imag - 300.0) < 1.0

    def test_ship_wrapping_all_boundaries(self, basic_env):
        """Test wrapping on all four boundaries."""
        basic_env.reset(game_mode="1v1")
        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        ship = basic_env.state[-1].ships[0]
        velocity = ship.velocity
        dt = basic_env.agent_dt

        test_cases = [
            # (initial_position, description)
            (-50.0 + 300.0j, "Left"),
            (basic_env.world_size[0] + 50.0 + 300.0j, "Right"),
            (400.0 - 50.0j, "Top"),
            (400.0 + basic_env.world_size[1] + 50.0j, "Bottom"),
        ]

        for initial, description in test_cases:
            # Reset environment to get clean state for each test case
            basic_env.reset(game_mode="1v1")
            ship = basic_env.state[-1].ships[0]
            velocity = ship.velocity  # Get fresh velocity after reset

            ship.position = initial
            basic_env.step(actions)

            # Calculate expected position after movement and wrapping
            moved_pos = initial + velocity * dt
            expected_real = moved_pos.real % basic_env.world_size[0]
            expected_imag = moved_pos.imag % basic_env.world_size[1]

            wrapped = basic_env.state[-1].ships[0].position
            assert (
                abs(wrapped.real - expected_real) < 1.0
            ), f"{description} boundary wrapping failed"
            assert (
                abs(wrapped.imag - expected_imag) < 1.0
            ), f"{description} boundary wrapping failed"

    def test_bullet_position_wrapping(self, basic_env):
        """Test that bullet positions wrap at boundaries."""
        basic_env.reset(game_mode="1v1")

        # Add bullets at boundaries
        state = basic_env.state[-1]
        state.bullets.add_bullet(0, -10.0, 300.0, 100.0, 0.0, 1.0)
        state.bullets.add_bullet(0, 810.0, 300.0, 100.0, 0.0, 1.0)
        state.bullets.add_bullet(0, 400.0, -10.0, 0.0, 100.0, 1.0)
        state.bullets.add_bullet(0, 400.0, 610.0, 0.0, 100.0, 1.0)

        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        basic_env.step(actions)

        bullets = basic_env.state[-1].bullets

        # Check all bullets wrapped correctly
        assert 0 <= bullets.x[0] < basic_env.world_size[0]
        assert 0 <= bullets.x[1] < basic_env.world_size[0]
        assert 0 <= bullets.y[2] < basic_env.world_size[1]
        assert 0 <= bullets.y[3] < basic_env.world_size[1]


class TestActionSpaceObservationSpace:
    """Tests for Gym space definitions."""

    def test_action_space(self, basic_env):
        """Test action space is correctly defined."""
        action_space = basic_env.action_space

        assert isinstance(action_space, spaces.MultiBinary)
        assert action_space.shape == (len(Actions),)

        # Test sample action
        sample = action_space.sample()
        assert sample.shape == (len(Actions),)
        assert np.all((sample == 0) | (sample == 1))

    def test_observation_space(self, basic_env):
        """Test observation space is correctly defined."""
        obs_space = basic_env.observation_space

        assert isinstance(obs_space, spaces.Dict)

        # Check all components
        assert "self_state" in obs_space.spaces
        assert "enemy_state" in obs_space.spaces
        assert "bullets" in obs_space.spaces
        assert "world_bounds" in obs_space.spaces
        assert "time" in obs_space.spaces

        # Check shapes
        assert obs_space["self_state"].shape == (8,)
        assert obs_space["enemy_state"].shape == (8,)
        assert obs_space["bullets"].shape == (20, 6)
        assert obs_space["world_bounds"].shape == (2,)
        assert obs_space["time"].shape == ()

    def test_observation_matches_space(self, basic_env):
        """Test that actual observations match the defined space."""
        obs, _ = basic_env.reset(game_mode="1v1")
        obs_space = basic_env.observation_space

        for ship_id, ship_obs in obs.items():
            # Use Gym's contains method to check
            assert obs_space.contains(ship_obs)
