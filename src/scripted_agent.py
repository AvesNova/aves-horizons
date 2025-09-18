"""
Scripted agent for controlling ship_1 in 1v1 combat.

The agent implements a simple strategy:
1. Turn towards the nearest enemy
2. Shoot if looking within 3 degrees of the enemy and in range
"""

import torch
import torch.nn as nn
import numpy as np
from enums import Actions


class ScriptedAgent(nn.Module):
    """
    Simple scripted agent that turns towards enemies and shoots when aligned.

    This agent is designed to control ship_1 in a 1v1 scenario.
    """

    def __init__(self, max_shooting_range: float = 400.0, angle_threshold: float = 5.0):
        """
        Initialize the scripted agent.

        Args:
            max_shooting_range: Maximum distance to shoot at enemies (in world units)
            angle_threshold: Angle tolerance in degrees for shooting
        """
        super().__init__()
        self.max_shooting_range = max_shooting_range
        self.angle_threshold = np.deg2rad(angle_threshold)  # Convert to radians

        # Register as buffer so it moves with device but isn't a parameter
        self.register_buffer("_dummy", torch.zeros(1))

    def forward(self, observation: dict) -> torch.Tensor:
        """
        Generate action based on observation.

        Args:
            observation: Dictionary containing observation for ship_1

        Returns:
            action: Tensor of shape (6,) with binary actions [forward, backward, left, right, sharp_turn, shoot]
        """
        # Extract relevant information from observation
        self_state = observation[
            "self_state"
        ]  # [x, y, vx, vy, health, power, attitude_x, attitude_y]
        enemy_state = observation["enemy_state"]  # Same format as self_state
        world_bounds = observation["world_bounds"]  # [world_width, world_height]

        # Initialize action (all zeros)
        action = torch.zeros(
            len(Actions), dtype=torch.float32, device=self._dummy.device
        )

        # Check if enemy is alive/visible (non-zero health)
        enemy_health = enemy_state[4]
        if enemy_health <= 0:
            return action  # No enemy to target

        # Get positions in world coordinates
        self_pos = torch.tensor(
            [
                self_state[0] * world_bounds[0],  # Denormalize x
                self_state[1] * world_bounds[1],  # Denormalize y
            ],
            device=self._dummy.device,
        )

        enemy_pos = torch.tensor(
            [
                enemy_state[0] * world_bounds[0],  # Denormalize x
                enemy_state[1] * world_bounds[1],  # Denormalize y
            ],
            device=self._dummy.device,
        )

        # Get current attitude (ship's facing direction)
        self_attitude = torch.tensor(
            [self_state[6], self_state[7]], device=self._dummy.device
        )

        # Calculate vector to enemy (accounting for toroidal world wrapping)
        to_enemy = self._calculate_wrapped_vector(self_pos, enemy_pos, world_bounds)
        distance = torch.norm(to_enemy)

        if distance < 1e-6:  # Avoid division by zero
            return action

        # Normalize direction to enemy
        to_enemy_normalized = to_enemy / distance

        # Calculate angle between current attitude and direction to enemy
        # Use dot product: cos(angle) = a·b / (|a||b|)
        cos_angle = torch.dot(self_attitude, to_enemy_normalized)
        angle_to_enemy = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))

        # Determine turn direction using cross product (2D cross product gives scalar)
        # If cross product is positive, enemy is to the left; if negative, to the right
        cross_product = (
            self_attitude[0] * to_enemy_normalized[1]
            - self_attitude[1] * to_enemy_normalized[0]
        )

        # Decide on turning action
        if angle_to_enemy > self.angle_threshold:
            if cross_product > 0:
                action[Actions.right] = 1.0
            else:
                action[Actions.left] = 1.0

            # Use sharp turn for large angle differences
            if angle_to_enemy > np.deg2rad(15.0):
                action[Actions.sharp_turn] = 1.0

        # Shoot if aligned with enemy and in range
        if (
            angle_to_enemy <= self.angle_threshold
            and distance <= self.max_shooting_range
            and self_state[4] > 0
        ):  # Only shoot if we're alive
            action[Actions.shoot] = 1.0

        # Only boost (forward) if power > 60%, otherwise maintain base thrust
        current_power = self_state[
            5
        ]  # Power is normalized [0,1], so 0.6 = 60% = 60 power units
        if current_power > 0.6:
            action[Actions.forward] = 1.0
        # Note: When power <= 0.6, we don't set forward=1, so ship uses base thrust only

        return action

    def _calculate_wrapped_vector(
        self, pos1: torch.Tensor, pos2: torch.Tensor, world_bounds: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the shortest vector from pos1 to pos2 in a toroidal world.

        Args:
            pos1: Position 1 [x, y]
            pos2: Position 2 [x, y]
            world_bounds: World size [width, height]

        Returns:
            Vector from pos1 to pos2 considering wrapping
        """
        # Calculate direct difference
        diff = pos2 - pos1

        # Handle wrapping for each dimension
        for i in range(2):
            world_size = world_bounds[i]

            # If distance is more than half the world size, wrap around
            if abs(diff[i]) > world_size / 2:
                if diff[i] > 0:
                    diff[i] -= world_size
                else:
                    diff[i] += world_size

        return diff


def create_scripted_agent(**kwargs) -> ScriptedAgent:
    """
    Factory function to create a scripted agent.

    Args:
        **kwargs: Arguments to pass to ScriptedAgent constructor

    Returns:
        ScriptedAgent instance
    """
    return ScriptedAgent(**kwargs)


# Example usage
if __name__ == "__main__":
    # Create agent
    agent = ScriptedAgent()

    # Example observation (mock data)
    observation = {
        "self_state": torch.tensor(
            [0.5, 0.5, 0.1, 0.0, 1.0, 0.8, 1.0, 0.0]
        ),  # Facing right
        "enemy_state": torch.tensor(
            [0.7, 0.6, -0.1, 0.0, 0.9, 0.7, -1.0, 0.0]
        ),  # Enemy facing left
        "bullets": torch.zeros(20, 6),
        "world_bounds": torch.tensor([1200.0, 800.0]),
        "time": 0.0,
    }

    # Get action
    action = agent(observation)
    print("Action:", action)
    print("Forward:", bool(action[Actions.forward]))
    print("Backward:", bool(action[Actions.backward]))
    print("Left:", bool(action[Actions.left]))
    print("Right:", bool(action[Actions.right]))
    print("Sharp turn:", bool(action[Actions.sharp_turn]))
    print("Shoot:", bool(action[Actions.shoot]))
