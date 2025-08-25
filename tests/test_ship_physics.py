import torch
import pytest
from core.ship import Ships, ShipPhysics
from utils.config import Actions


# Test utilities
def create_test_ship(
    position: complex = 0 + 0j,
    velocity: complex = 0 + 0j,
    orientation: complex = 1 + 0j,
) -> Ships:
    """Create a single ship for testing."""
    ships = Ships(1, (100, 100))
    ships.position = torch.tensor([position], dtype=torch.complex64)
    ships.velocity = torch.tensor([velocity], dtype=torch.complex64)
    ships.orientation = torch.tensor([orientation], dtype=torch.complex64)
    return ships


def create_action_tensor(
    left: bool = False,
    right: bool = False,
    forward: bool = False,
    backward: bool = False,
    sharp_turn: bool = False,
) -> torch.Tensor:
    """Create an action tensor for testing."""
    actions = torch.zeros((1, 5), dtype=torch.bool)  # Assuming 5 actions
    if left:
        actions[0, Actions.left] = True
    if right:
        actions[0, Actions.right] = True
    if forward:
        actions[0, Actions.forward] = True
    if backward:
        actions[0, Actions.backward] = True
    if sharp_turn:
        actions[0, Actions.sharp_turn] = True
    return actions


# Test fixtures
@pytest.fixture
def physics():
    """Fixture to provide a ShipPhysics instance."""
    return ShipPhysics(dt=0.016)


@pytest.fixture
def stationary_ship():
    """Fixture to provide a stationary ship."""
    return create_test_ship(position=50 + 50j, velocity=0 + 0j, orientation=1 + 0j)


@pytest.fixture
def moving_ship():
    """Fixture to provide a moving ship."""
    return create_test_ship(position=50 + 50j, velocity=10 + 5j, orientation=1 + 0j)


class TestShipPhysicsStationary:
    """Tests for stationary ship behavior."""

    def test_stationary_ship_left_turn_normal(self, physics, stationary_ship):
        """Test that a stationary ship turning left changes only orientation."""
        actions = create_action_tensor(left=True)

        initial_position = stationary_ship.position.clone()
        initial_velocity = stationary_ship.velocity.clone()
        initial_orientation = stationary_ship.orientation.clone()

        physics(stationary_ship, actions)

        # Position and velocity should remain unchanged
        torch.testing.assert_close(
            stationary_ship.position, initial_position, rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            stationary_ship.velocity, initial_velocity, rtol=1e-6, atol=1e-6
        )

        # Orientation should change (rotate left by normal_turn_angle)
        expected_orientation = initial_orientation * torch.exp(
            1j * stationary_ship.normal_turn_angle
        )
        torch.testing.assert_close(
            stationary_ship.orientation, expected_orientation, rtol=1e-6, atol=1e-6
        )

    def test_stationary_ship_right_turn_normal(self, physics, stationary_ship):
        """Test that a stationary ship turning right changes only orientation."""
        actions = create_action_tensor(right=True)

        initial_position = stationary_ship.position.clone()
        initial_velocity = stationary_ship.velocity.clone()
        initial_orientation = stationary_ship.orientation.clone()

        physics(stationary_ship, actions)

        # Position and velocity should remain unchanged
        torch.testing.assert_close(
            stationary_ship.position, initial_position, rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            stationary_ship.velocity, initial_velocity, rtol=1e-6, atol=1e-6
        )

        # Orientation should change (rotate right by -normal_turn_angle)
        expected_orientation = initial_orientation * torch.exp(
            -1j * stationary_ship.normal_turn_angle
        )
        torch.testing.assert_close(
            stationary_ship.orientation, expected_orientation, rtol=1e-6, atol=1e-6
        )

    def test_stationary_ship_left_turn_sharp(self, physics, stationary_ship):
        """Test that a stationary ship sharp turning left changes only orientation."""
        actions = create_action_tensor(left=True, sharp_turn=True)

        initial_position = stationary_ship.position.clone()
        initial_velocity = stationary_ship.velocity.clone()
        initial_orientation = stationary_ship.orientation.clone()

        physics(stationary_ship, actions)

        # Position and velocity should remain unchanged
        torch.testing.assert_close(
            stationary_ship.position, initial_position, rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            stationary_ship.velocity, initial_velocity, rtol=1e-6, atol=1e-6
        )

        # Orientation should change (rotate left by sharp_turn_angle)
        expected_orientation = initial_orientation * torch.exp(
            1j * stationary_ship.sharp_turn_angle
        )
        torch.testing.assert_close(
            stationary_ship.orientation, expected_orientation, rtol=1e-6, atol=1e-6
        )

    def test_stationary_ship_right_turn_sharp(self, physics, stationary_ship):
        """Test that a stationary ship sharp turning right changes only orientation."""
        actions = create_action_tensor(right=True, sharp_turn=True)

        initial_position = stationary_ship.position.clone()
        initial_velocity = stationary_ship.velocity.clone()
        initial_orientation = stationary_ship.orientation.clone()

        physics(stationary_ship, actions)

        # Position and velocity should remain unchanged
        torch.testing.assert_close(
            stationary_ship.position, initial_position, rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            stationary_ship.velocity, initial_velocity, rtol=1e-6, atol=1e-6
        )

        # Orientation should change (rotate right by -sharp_turn_angle)
        expected_orientation = initial_orientation * torch.exp(
            -1j * stationary_ship.sharp_turn_angle
        )
        torch.testing.assert_close(
            stationary_ship.orientation, expected_orientation, rtol=1e-6, atol=1e-6
        )

    def test_stationary_ship_no_turn_maintains_orientation(
        self, physics, stationary_ship
    ):
        """Test that stationary ship maintains orientation when not turning."""
        # First turn to change orientation
        actions_turn = create_action_tensor(left=True)
        physics(stationary_ship, actions_turn)

        # Store the turned orientation
        turned_orientation = stationary_ship.orientation.clone()

        # Now release turn keys (no actions)
        actions_no_turn = create_action_tensor()
        physics(stationary_ship, actions_no_turn)

        # For stationary ship, orientation should remain as it was
        # (since velocity direction is undefined when velocity is zero)
        torch.testing.assert_close(
            stationary_ship.orientation, turned_orientation, rtol=1e-6, atol=1e-6
        )


class TestShipPhysicsMoving:
    """Tests for moving ship behavior."""

    def test_moving_ship_no_turn_aligns_orientation_with_velocity(
        self, physics, moving_ship
    ):
        """Test that orientation aligns with velocity when not turning."""
        # No turn actions
        actions = create_action_tensor()

        initial_velocity = moving_ship.velocity.clone()
        physics(moving_ship, actions)

        # Orientation should align with velocity direction
        expected_orientation = initial_velocity / torch.abs(initial_velocity)
        torch.testing.assert_close(
            moving_ship.orientation, expected_orientation, rtol=1e-5, atol=1e-5
        )


class TestShipPhysicsForces:
    """Tests for force calculations."""

    def test_zero_thrust_zero_boost_stationary(self, physics, stationary_ship):
        """Test that stationary ship with zero thrust produces zero thrust force."""
        # Set thrust and boost to zero
        stationary_ship.thrust.fill_(0.0)
        stationary_ship.forward_boost.fill_(0.0)
        stationary_ship.backward_boost.fill_(0.0)

        actions = create_action_tensor()
        action_states = physics.extract_action_states(actions)

        thrust_force = physics.calculate_thrust_forces(stationary_ship, action_states)

        expected_force = torch.zeros_like(thrust_force)
        torch.testing.assert_close(thrust_force, expected_force, rtol=1e-6, atol=1e-6)

    def test_zero_velocity_produces_zero_aero_force(self, physics, stationary_ship):
        """Test that zero velocity produces zero aerodynamic force."""
        actions = create_action_tensor(left=True)  # Turn to get non-zero lift
        action_states = physics.extract_action_states(actions)
        turn_states = physics.calculate_turn_states(action_states)

        drag_coef = physics.calculate_drag_coefficient(
            stationary_ship, action_states, turn_states
        )
        lift_coef = physics.calculate_lift_coefficient(
            stationary_ship, action_states, turn_states
        )

        aero_force = physics.calculate_aero_forces(
            stationary_ship, drag_coef, lift_coef
        )

        expected_force = torch.zeros_like(aero_force)
        torch.testing.assert_close(aero_force, expected_force, rtol=1e-6, atol=1e-6)


class TestShipPhysicsIntegration:
    """Integration tests for complete physics simulation."""

    def test_multiple_turn_steps_accumulate_correctly(self, physics, stationary_ship):
        """Test that multiple turning steps accumulate angle changes correctly."""
        actions = create_action_tensor(left=True)

        # Perform multiple steps
        num_steps = 5
        for _ in range(num_steps):
            physics(stationary_ship, actions)

        # Expected total rotation
        expected_total_angle = num_steps * stationary_ship.normal_turn_angle[0]
        expected_orientation = torch.exp(1j * expected_total_angle)

        torch.testing.assert_close(
            stationary_ship.orientation, expected_orientation, rtol=1e-5, atol=1e-5
        )

    def test_alternating_turns_cancel_out(self, physics, stationary_ship):
        """Test that alternating left and right turns cancel each other out."""
        initial_orientation = stationary_ship.orientation.clone()

        # Alternate left and right turns
        for _ in range(10):
            physics(stationary_ship, create_action_tensor(left=True))
            physics(stationary_ship, create_action_tensor(right=True))

        # Should end up back at initial orientation
        torch.testing.assert_close(
            stationary_ship.orientation, initial_orientation, rtol=1e-5, atol=1e-5
        )


class TestActionStatesExtraction:
    """Test action state extraction."""

    def test_extract_action_states_all_false(self, physics):
        """Test extracting action states when all actions are false."""
        actions = create_action_tensor()
        action_states = physics.extract_action_states(actions)

        assert not action_states.left.item()
        assert not action_states.right.item()
        assert not action_states.forward.item()
        assert not action_states.backward.item()
        assert not action_states.sharp_turn.item()

    def test_extract_action_states_mixed(self, physics):
        """Test extracting action states with mixed true/false values."""
        actions = create_action_tensor(left=True, forward=True, sharp_turn=True)
        action_states = physics.extract_action_states(actions)

        assert action_states.left.item()
        assert not action_states.right.item()
        assert action_states.forward.item()
        assert not action_states.backward.item()
        assert action_states.sharp_turn.item()


class TestTurnStatesCalculation:
    """Test turn state calculations."""

    def test_turn_states_no_input(self, physics):
        """Test turn states when no turn inputs are active."""
        actions = create_action_tensor()
        action_states = physics.extract_action_states(actions)
        turn_states = physics.calculate_turn_states(action_states)

        assert not turn_states.is_turn_drag.item()
        assert not turn_states.is_turning.item()
        assert turn_states.turn_direction.item() == 0

    def test_turn_states_left_only(self, physics):
        """Test turn states when only left is pressed."""
        actions = create_action_tensor(left=True)
        action_states = physics.extract_action_states(actions)
        turn_states = physics.calculate_turn_states(action_states)

        assert turn_states.is_turn_drag.item()
        assert turn_states.is_turning.item()
        assert turn_states.turn_direction.item() == 1

    def test_turn_states_right_only(self, physics):
        """Test turn states when only right is pressed."""
        actions = create_action_tensor(right=True)
        action_states = physics.extract_action_states(actions)
        turn_states = physics.calculate_turn_states(action_states)

        assert turn_states.is_turn_drag.item()
        assert turn_states.is_turning.item()
        assert turn_states.turn_direction.item() == -1

    def test_turn_states_both_pressed(self, physics):
        """Test turn states when both left and right are pressed."""
        actions = create_action_tensor(left=True, right=True)
        action_states = physics.extract_action_states(actions)
        turn_states = physics.calculate_turn_states(action_states)

        assert turn_states.is_turn_drag.item()  # Drag should be applied
        assert not turn_states.is_turning.item()  # But no turning (XOR is false)
        assert turn_states.turn_direction.item() == 0  # No net turn direction
