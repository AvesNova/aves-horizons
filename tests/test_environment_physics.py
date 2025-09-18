"""
Tests for environment physics integration and multi-step behavior.
"""

import pytest
import numpy as np
import torch

from enums import Actions


class TestPhysicsIntegration:
    """Tests for physics integration in the environment."""

    def test_single_physics_step(self, basic_env):
        """Test single physics step execution."""
        basic_env.reset(game_mode="1v1")
        initial_positions = {
            sid: ship.position for sid, ship in basic_env.state[-1].ships.items()
        }

        # Step with no actions
        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        obs, rewards, terminated, truncated, info = basic_env.step(actions)

        # Ships should have moved due to initial velocity
        for sid, ship in basic_env.state[-1].ships.items():
            assert ship.position != initial_positions[sid]

        # Time should advance
        assert info["current_time"] == basic_env.physics_dt

    def test_multiple_physics_substeps(self, env_with_substeps):
        """Test that multiple physics substeps execute correctly."""
        env_with_substeps.reset(game_mode="1v1")

        # Track ship 0's position over substeps
        ship0 = env_with_substeps.state[-1].ships[0]
        initial_x = ship0.position.real

        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        obs, rewards, terminated, truncated, info = env_with_substeps.step(actions)

        # Time should advance by agent_dt (which is 2 * physics_dt)
        assert info["current_time"] == env_with_substeps.agent_dt

        # Ship should have moved more than single step would allow
        final_x = env_with_substeps.state[-1].ships[0].position.real
        distance_moved = final_x - initial_x

        # Should move approximately velocity * agent_dt
        expected_distance = ship0.velocity.real * env_with_substeps.agent_dt
        assert abs(distance_moved - expected_distance) < 5.0  # Allow for forces

    def test_consistent_physics_determinism(self, basic_env, fixed_rng):
        """Test that physics is deterministic with fixed seeds."""
        # Run simulation twice with same initial conditions
        results = []

        for _ in range(2):
            basic_env.reset(game_mode="1v1")

            # Use fixed RNG for any randomness
            for ship in basic_env.state[-1].ships.values():
                ship.rng = fixed_rng

            # Apply same actions
            actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
            actions[0][Actions.forward] = 1
            actions[0][Actions.left] = 1

            # Step multiple times
            positions = []
            for _ in range(10):
                basic_env.step(actions)
                pos = basic_env.state[-1].ships[0].position
                positions.append((pos.real, pos.imag))

            results.append(positions)

        # Results should be identical
        for i in range(len(results[0])):
            assert abs(results[0][i][0] - results[1][i][0]) < 1e-10
            assert abs(results[0][i][1] - results[1][i][1]) < 1e-10


class TestCollisions:
    """Tests for collision detection between ships and bullets."""

    def test_bullet_ship_collision(self, basic_env):
        """Test that bullets hitting ships cause damage."""
        basic_env.reset(game_mode="1v1")
        snapshot = basic_env.state[-1]

        ship1 = snapshot.ships[1]
        initial_health = ship1.health

        # Place bullet at ship1's position from ship0
        snapshot.bullets.add_bullet(
            ship_id=0,
            x=ship1.position.real,
            y=ship1.position.imag,
            vx=0.0,
            vy=0.0,
            lifetime=1.0,
        )

        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        basic_env.step(actions)

        # Ship1 should take damage
        final_health = basic_env.state[-1].ships[1].health
        damage = ship1.config.bullet_damage
        assert abs(final_health - (initial_health - damage)) < 1e-5

        # Bullet should be removed
        assert basic_env.state[-1].bullets.num_active == 0

    def test_no_friendly_fire(self, basic_env):
        """Test that ships don't damage themselves with their own bullets."""
        basic_env.reset(game_mode="1v1")
        snapshot = basic_env.state[-1]

        ship0 = snapshot.ships[0]
        initial_health = ship0.health

        # Place ship0's bullet at its own position
        snapshot.bullets.add_bullet(
            ship_id=0,
            x=ship0.position.real,
            y=ship0.position.imag,
            vx=0.0,
            vy=0.0,
            lifetime=1.0,
        )

        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        basic_env.step(actions)

        # Ship0 should not take damage
        final_health = basic_env.state[-1].ships[0].health
        assert final_health == initial_health

        # Bullet should still exist
        assert basic_env.state[-1].bullets.num_active == 1

    def test_multiple_bullet_hits(self, basic_env):
        """Test that multiple bullets can hit same ship."""
        basic_env.reset(game_mode="1v1")
        snapshot = basic_env.state[-1]

        ship1 = snapshot.ships[1]
        initial_health = ship1.health

        # Place 3 bullets at ship1's position
        for _ in range(3):
            snapshot.bullets.add_bullet(
                ship_id=0,
                x=ship1.position.real,
                y=ship1.position.imag,
                vx=0.0,
                vy=0.0,
                lifetime=1.0,
            )

        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        basic_env.step(actions)

        # Ship1 should take 3x damage
        final_health = basic_env.state[-1].ships[1].health
        damage = ship1.config.bullet_damage * 3
        assert abs(final_health - (initial_health - damage)) < 1e-5

        # All bullets should be removed
        assert basic_env.state[-1].bullets.num_active == 0

    def test_collision_radius(self, basic_env):
        """Test that collision radius is respected."""
        basic_env.reset(game_mode="1v1")
        snapshot = basic_env.state[-1]

        ship1 = snapshot.ships[1]
        initial_health = ship1.health

        # Place bullet just outside collision radius
        offset = ship1.config.collision_radius + 1.0
        snapshot.bullets.add_bullet(
            ship_id=0,
            x=ship1.position.real + offset,
            y=ship1.position.imag,
            vx=0.0,
            vy=0.0,
            lifetime=1.0,
        )

        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        basic_env.step(actions)

        # Ship1 should not take damage
        final_health = basic_env.state[-1].ships[1].health
        assert final_health == initial_health

        # Bullet should still exist
        assert basic_env.state[-1].bullets.num_active == 1

    def test_dead_ship_no_collision(self, basic_env):
        """Test that dead ships don't interact with bullets."""
        basic_env.reset(game_mode="1v1")
        snapshot = basic_env.state[-1]

        # Kill ship1
        ship1 = snapshot.ships[1]
        ship1.alive = False
        ship1.health = 0

        # Place bullet at dead ship's position
        snapshot.bullets.add_bullet(
            ship_id=0,
            x=ship1.position.real,
            y=ship1.position.imag,
            vx=0.0,
            vy=0.0,
            lifetime=1.0,
        )

        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        basic_env.step(actions)

        # Bullet should still exist (no collision with dead ship)
        assert basic_env.state[-1].bullets.num_active == 1


class TestRewardsAndTermination:
    """Tests for reward calculation and episode termination."""

    def test_basic_rewards(self, basic_env):
        """Test basic reward structure."""
        basic_env.reset(game_mode="1v1")

        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        obs, rewards, terminated, truncated, info = basic_env.step(actions)

        # Both ships alive should get positive rewards
        assert rewards[0] > 0
        assert rewards[1] > 0

        # Rewards should include survival and health components
        for ship_id in [0, 1]:
            # Base survival reward + health bonus
            assert rewards[ship_id] >= 1.0

    def test_death_penalty(self, basic_env):
        """Test that dying gives large negative reward."""
        basic_env.reset(game_mode="1v1")

        # Kill ship 1
        basic_env.state[-1].ships[1].health = 0

        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        obs, rewards, terminated, truncated, info = basic_env.step(actions)

        # Dead ship should get large negative reward
        assert rewards[1] < -50

        # Alive ship should get positive reward
        assert rewards[0] > 0

    def test_health_based_rewards(self, basic_env):
        """Test that rewards scale with health."""
        basic_env.reset(game_mode="1v1")

        # Damage ship 0
        ship0 = basic_env.state[-1].ships[0]
        ship0.health = ship0.config.max_health / 2

        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        obs, rewards, terminated, truncated, info = basic_env.step(actions)

        # Ship with less health should get lower reward than full health ship
        assert rewards[0] < rewards[1]

    def test_termination_one_team_survives(self, basic_env):
        """Test that episode terminates when only one team survives."""
        basic_env.reset(game_mode="1v1")

        # Kill ship 1
        basic_env.state[-1].ships[1].alive = False

        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        obs, rewards, terminated, truncated, info = basic_env.step(actions)

        assert terminated is True
        assert info["individual_done"][1] is True
        assert info["individual_done"][0] is False

    def test_no_termination_both_alive(self, basic_env):
        """Test that episode continues when both teams alive."""
        basic_env.reset(game_mode="1v1")

        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        obs, rewards, terminated, truncated, info = basic_env.step(actions)

        assert terminated is False
        assert info["individual_done"][0] is False
        assert info["individual_done"][1] is False

    def test_termination_mutual_destruction(self, basic_env):
        """Test termination when all ships destroyed."""
        basic_env.reset(game_mode="1v1")

        # Kill both ships
        basic_env.state[-1].ships[0].alive = False
        basic_env.state[-1].ships[1].alive = False

        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        obs, rewards, terminated, truncated, info = basic_env.step(actions)

        assert terminated is True
        assert info["individual_done"][0] is True
        assert info["individual_done"][1] is True
