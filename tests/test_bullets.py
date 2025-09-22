"""
Tests for the bullet management system.
"""

import pytest


class TestBulletAllocation:
    """Tests for bullet allocation and deallocation."""

    def test_empty_initialization(self, empty_bullets):
        """Test that bullet container initializes empty."""
        assert empty_bullets.num_active == 0
        assert empty_bullets.num_free == empty_bullets.max_bullets
        assert len(empty_bullets.x) == empty_bullets.max_bullets

    def test_add_single_bullet(self, empty_bullets):
        """Test adding a single bullet."""
        slot = empty_bullets.add_bullet(
            ship_id=0, x=100.0, y=200.0, vx=50.0, vy=-50.0, lifetime=1.0
        )

        assert slot == 0
        assert empty_bullets.num_active == 1
        assert empty_bullets.num_free == empty_bullets.max_bullets - 1
        assert empty_bullets.x[0] == 100.0
        assert empty_bullets.y[0] == 200.0
        assert empty_bullets.vx[0] == 50.0
        assert empty_bullets.vy[0] == -50.0
        assert empty_bullets.time_remaining[0] == 1.0
        assert empty_bullets.ship_id[0] == 0

    def test_add_multiple_bullets(self, empty_bullets):
        """Test adding multiple bullets."""
        bullets_data = [
            (0, 100.0, 100.0, 10.0, 0.0, 1.0),
            (1, 200.0, 200.0, 0.0, 10.0, 0.5),
            (0, 300.0, 300.0, -10.0, -10.0, 0.75),
        ]

        for i, (ship_id, x, y, vx, vy, lifetime) in enumerate(bullets_data):
            slot = empty_bullets.add_bullet(ship_id, x, y, vx, vy, lifetime)
            assert slot == i

        assert empty_bullets.num_active == 3
        assert empty_bullets.num_free == empty_bullets.max_bullets - 3

    def test_add_bullets_to_capacity(self, empty_bullets):
        """Test adding bullets up to maximum capacity."""
        for i in range(empty_bullets.max_bullets):
            slot = empty_bullets.add_bullet(i % 2, i * 10.0, i * 10.0, 0.0, 0.0, 1.0)
            assert slot == i

        assert empty_bullets.num_active == empty_bullets.max_bullets
        assert empty_bullets.num_free == 0

    def test_add_bullet_when_full(self, empty_bullets):
        """Test that adding bullet when full returns -1."""
        # Fill up the container
        for i in range(empty_bullets.max_bullets):
            empty_bullets.add_bullet(0, 0.0, 0.0, 0.0, 0.0, 1.0)

        # Try to add one more
        slot = empty_bullets.add_bullet(0, 0.0, 0.0, 0.0, 0.0, 1.0)
        assert slot == -1
        assert empty_bullets.num_active == empty_bullets.max_bullets


class TestBulletRemoval:
    """Tests for bullet removal and free slot management."""

    def test_remove_single_bullet(self, bullets_with_some):
        """Test removing a single bullet."""
        initial_active = bullets_with_some.num_active
        bullets_with_some.remove_bullet(0)

        assert bullets_with_some.num_active == initial_active - 1
        assert bullets_with_some.num_free == bullets_with_some.max_bullets - (
            initial_active - 1
        )

    def test_remove_last_bullet(self, bullets_with_some):
        """Test removing the last active bullet."""
        last_idx = bullets_with_some.num_active - 1
        initial_active = bullets_with_some.num_active

        bullets_with_some.remove_bullet(last_idx)
        assert bullets_with_some.num_active == initial_active - 1

    def test_remove_middle_bullet_swaps(self, empty_bullets):
        """Test that removing middle bullet swaps with last."""
        # Add three bullets with distinctive values
        empty_bullets.add_bullet(0, 100.0, 0.0, 0.0, 0.0, 1.0)
        empty_bullets.add_bullet(1, 200.0, 0.0, 0.0, 0.0, 1.0)
        empty_bullets.add_bullet(2, 300.0, 0.0, 0.0, 0.0, 1.0)

        # Remove middle bullet (index 1)
        empty_bullets.remove_bullet(1)

        # Check that last bullet was swapped to position 1
        assert empty_bullets.num_active == 2
        assert empty_bullets.x[0] == 100.0  # First unchanged
        assert empty_bullets.x[1] == 300.0  # Last moved to middle

    def test_remove_already_inactive(self, bullets_with_some):
        """Test that removing inactive bullet does nothing."""
        initial_active = bullets_with_some.num_active
        bullets_with_some.remove_bullet(bullets_with_some.max_bullets - 1)
        assert bullets_with_some.num_active == initial_active

    def test_remove_all_bullets(self, bullets_with_some):
        """Test removing all bullets."""
        while bullets_with_some.num_active > 0:
            bullets_with_some.remove_bullet(0)

        assert bullets_with_some.num_active == 0
        assert bullets_with_some.num_free == bullets_with_some.max_bullets


class TestBulletUpdate:
    """Tests for bullet position and lifetime updates."""

    def test_update_positions(self, empty_bullets):
        """Test that bullet positions update correctly."""
        empty_bullets.add_bullet(0, 100.0, 100.0, 50.0, -25.0, 1.0)
        empty_bullets.add_bullet(1, 200.0, 200.0, -50.0, 25.0, 1.0)

        dt = 0.1
        empty_bullets.update_all(dt)

        assert abs(empty_bullets.x[0] - 105.0) < 1e-5
        assert abs(empty_bullets.y[0] - 97.5) < 1e-5
        assert abs(empty_bullets.x[1] - 195.0) < 1e-5
        assert abs(empty_bullets.y[1] - 202.5) < 1e-5

    def test_update_lifetime(self, empty_bullets):
        """Test that bullet lifetime decreases."""
        empty_bullets.add_bullet(0, 0.0, 0.0, 0.0, 0.0, 1.0)
        empty_bullets.add_bullet(1, 0.0, 0.0, 0.0, 0.0, 0.5)

        dt = 0.1
        empty_bullets.update_all(dt)

        assert abs(empty_bullets.time_remaining[0] - 0.9) < 1e-5
        assert abs(empty_bullets.time_remaining[1] - 0.4) < 1e-5

    def test_remove_expired_single(self, empty_bullets):
        """Test that expired bullets are removed."""
        empty_bullets.add_bullet(0, 0.0, 0.0, 0.0, 0.0, 0.05)
        empty_bullets.add_bullet(1, 0.0, 0.0, 0.0, 0.0, 1.0)

        empty_bullets.update_all(0.1)

        assert empty_bullets.num_active == 1
        assert empty_bullets.time_remaining[0] == 0.9  # Second bullet moved to front

    def test_remove_expired_multiple(self, empty_bullets):
        """Test removing multiple expired bullets."""
        empty_bullets.add_bullet(0, 0.0, 0.0, 0.0, 0.0, 0.05)
        empty_bullets.add_bullet(1, 0.0, 0.0, 0.0, 0.0, 0.08)
        empty_bullets.add_bullet(2, 0.0, 0.0, 0.0, 0.0, 1.0)

        empty_bullets.update_all(0.1)

        assert empty_bullets.num_active == 1
        assert abs(empty_bullets.time_remaining[0] - 0.9) < 1e-5

    def test_remove_all_expired(self, empty_bullets):
        """Test when all bullets expire simultaneously."""
        for i in range(3):
            empty_bullets.add_bullet(i, 0.0, 0.0, 0.0, 0.0, 0.05)

        empty_bullets.update_all(0.1)

        assert empty_bullets.num_active == 0
        assert empty_bullets.num_free == empty_bullets.max_bullets

    def test_update_empty_container(self, empty_bullets):
        """Test updating empty bullet container doesn't crash."""
        empty_bullets.update_all(0.1)
        assert empty_bullets.num_active == 0


class TestBulletActivePositions:
    """Tests for getting active bullet positions."""

    def test_get_positions_empty(self, empty_bullets):
        """Test getting positions from empty container."""
        x, y, ship_ids = empty_bullets.get_active_positions()
        assert len(x) == 0
        assert len(y) == 0
        assert len(ship_ids) == 0

    def test_get_positions_with_bullets(self, bullets_with_some):
        """Test getting positions returns correct arrays."""
        x, y, ship_ids = bullets_with_some.get_active_positions()

        assert len(x) == bullets_with_some.num_active
        assert len(y) == bullets_with_some.num_active
        assert len(ship_ids) == bullets_with_some.num_active

        # Check first bullet values
        assert x[0] == bullets_with_some.x[0]
        assert y[0] == bullets_with_some.y[0]
        assert ship_ids[0] == bullets_with_some.ship_id[0]

    def test_positions_after_removal(self, empty_bullets):
        """Test that positions are correct after removing bullets."""
        # Add bullets with distinct positions
        empty_bullets.add_bullet(0, 100.0, 150.0, 0.0, 0.0, 1.0)
        empty_bullets.add_bullet(1, 200.0, 250.0, 0.0, 0.0, 1.0)
        empty_bullets.add_bullet(2, 300.0, 350.0, 0.0, 0.0, 1.0)

        # Remove middle bullet
        empty_bullets.remove_bullet(1)

        x, y, ship_ids = empty_bullets.get_active_positions()
        assert len(x) == 2
        assert 200.0 not in x  # Middle bullet position gone
        assert 100.0 in x  # First still there
        assert 300.0 in x  # Last still there


class TestBulletEdgeCases:
    """Tests for edge cases and stress conditions."""

    def test_add_remove_interleaved(self, empty_bullets):
        """Test interleaving adds and removes."""
        # Add some bullets
        empty_bullets.add_bullet(0, 0.0, 0.0, 0.0, 0.0, 1.0)
        empty_bullets.add_bullet(1, 0.0, 0.0, 0.0, 0.0, 1.0)
        empty_bullets.add_bullet(2, 0.0, 0.0, 0.0, 0.0, 1.0)

        # Remove middle
        empty_bullets.remove_bullet(1)
        assert empty_bullets.num_active == 2

        # Add new bullet - should reuse freed slot
        slot = empty_bullets.add_bullet(3, 0.0, 0.0, 0.0, 0.0, 1.0)
        assert slot >= 0
        assert empty_bullets.num_active == 3

    def test_rapid_add_remove_cycle(self, empty_bullets):
        """Test rapid cycling of bullets."""
        for cycle in range(10):
            # Add bullets
            for i in range(5):
                empty_bullets.add_bullet(i, 0.0, 0.0, 0.0, 0.0, 1.0)
            assert empty_bullets.num_active == 5

            # Remove all
            while empty_bullets.num_active > 0:
                empty_bullets.remove_bullet(0)
            assert empty_bullets.num_active == 0

    def test_free_list_integrity(self, empty_bullets):
        """Test that free list maintains integrity through operations."""
        max_bullets = empty_bullets.max_bullets

        # Fill halfway
        for i in range(max_bullets // 2):
            empty_bullets.add_bullet(i, 0.0, 0.0, 0.0, 0.0, 1.0)

        # Remove every other bullet
        for i in range(0, empty_bullets.num_active, 2):
            empty_bullets.remove_bullet(i)

        # Fill remaining capacity
        while empty_bullets.num_active < max_bullets:
            result = empty_bullets.add_bullet(99, 0.0, 0.0, 0.0, 0.0, 1.0)
            if result == -1:
                break

        # Should be able to fill to capacity
        assert empty_bullets.num_active == max_bullets
        assert empty_bullets.num_free == 0

    def test_update_with_zero_dt(self, bullets_with_some):
        """Test that zero dt doesn't change state."""
        initial_x = bullets_with_some.x[0]
        initial_lifetime = bullets_with_some.time_remaining[0]

        bullets_with_some.update_all(0.0)

        assert bullets_with_some.x[0] == initial_x
        assert bullets_with_some.time_remaining[0] == initial_lifetime

    def test_negative_lifetime_removal(self, empty_bullets):
        """Test bullets with negative lifetime are removed."""
        empty_bullets.add_bullet(0, 0.0, 0.0, 0.0, 0.0, 0.01)
        empty_bullets.update_all(0.1)  # Way past expiration

        assert empty_bullets.num_active == 0
