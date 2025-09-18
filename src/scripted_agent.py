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
    Advanced scripted agent with predictive targeting.

    Features:
    - Calculates bullet travel time based on distance
    - Predicts enemy position when bullet arrives
    - Uses predicted position for both turning and shooting

    This agent is designed to control ship_1 in a 1v1 scenario.
    """

    def __init__(
        self,
        max_shooting_range: float = 400.0,
        angle_threshold: float = 6.0,
        bullet_speed: float = 500.0,
        target_radius: float = 10.0,
        radius_multiplier: float = 1.5,
    ):
        """
        Initialize the scripted agent.

        Args:
            max_shooting_range: Maximum distance to shoot at enemies (in world units)
            angle_threshold: Fallback angle tolerance in degrees for shooting (used at max range)
            bullet_speed: Speed of bullets (used for travel time calculation)
            target_radius: Collision radius of enemy ships (in world units)
            radius_multiplier: Multiplier for angular size calculation (1.5 = shoot within 1.5 radii)
        """
        super().__init__()
        self.max_shooting_range = max_shooting_range
        self.angle_threshold = np.deg2rad(angle_threshold)  # Convert to radians
        self.bullet_speed = bullet_speed
        self.target_radius = target_radius
        self.radius_multiplier = radius_multiplier

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

        # Get enemy velocity (denormalized)
        enemy_velocity = torch.tensor(
            [
                enemy_state[2] * 1000.0,  # Denormalize vx
                enemy_state[3] * 1000.0,  # Denormalize vy
            ],
            device=self._dummy.device,
        )

        # Calculate predicted enemy position when bullet arrives
        predicted_enemy_pos = self._calculate_predicted_position(
            self_pos, enemy_pos, enemy_velocity, world_bounds
        )

        # Get current attitude (ship's facing direction)
        self_attitude = torch.tensor(
            [self_state[6], self_state[7]], device=self._dummy.device
        )

        # Calculate vector to PREDICTED enemy position (accounting for toroidal world wrapping)
        to_target = self._calculate_wrapped_vector(
            self_pos, predicted_enemy_pos, world_bounds
        )
        distance_to_target = torch.norm(to_target)

        # Also calculate distance to current enemy position for range checking
        to_enemy_current = self._calculate_wrapped_vector(
            self_pos, enemy_pos, world_bounds
        )
        current_distance = torch.norm(to_enemy_current)

        if distance_to_target < 1e-6:  # Avoid division by zero
            return action

        # Normalize direction to predicted target position
        to_target_normalized = to_target / distance_to_target

        # Calculate angle between current attitude and direction to predicted target
        # Use dot product: cos(angle) = a·b / (|a||b|)
        cos_angle = torch.dot(self_attitude, to_target_normalized)
        angle_to_target = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))

        # Determine turn direction using cross product (2D cross product gives scalar)
        # If cross product is positive, target is to the left; if negative, to the right
        cross_product = (
            self_attitude[0] * to_target_normalized[1]
            - self_attitude[1] * to_target_normalized[0]
        )

        # Decide on turning action (aim at predicted position)
        # Use small fixed threshold to prevent jitter
        if angle_to_target > self.angle_threshold:
            if cross_product > 0:
                # Target is to the left (counter-clockwise), turn right to face them
                action[Actions.right] = 1.0
            else:
                # Target is to the right (clockwise), turn left to face them
                action[Actions.left] = 1.0

            # Use sharp turn for large angle differences
            if angle_to_target > np.deg2rad(15.0):
                action[Actions.sharp_turn] = 1.0

        # Calculate dynamic shooting angle threshold based on distance to target
        dynamic_shooting_threshold = self._calculate_shooting_angle_threshold(
            current_distance
        )

        # Shoot if aligned with predicted target and current enemy is in range
        if (
            angle_to_target <= dynamic_shooting_threshold
            and current_distance <= self.max_shooting_range
            and self_state[4] > 0
        ):  # Only shoot if we're alive
            action[Actions.shoot] = 1.0

        # Thrust management based on distance and power
        close_range_threshold = (
            2.0 * self.target_radius
        )  # 2 radii = 20 units by default

        if current_distance <= close_range_threshold:
            # Close range: use reverse thrust to maintain distance, regardless of power level
            action[Actions.backward] = 1.0
        else:
            # Normal range: only boost (forward) if power > 80%, otherwise maintain base thrust
            current_power = self_state[
                5
            ]  # Power is normalized [0,1], so 0.6 = 80% = 80 power units
            if current_power > 0.8:
                action[Actions.forward] = 1.0
            # Note: When power <= 0.8, we don't set forward=1, so ship uses base thrust only

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

    def _calculate_predicted_position(
        self,
        self_pos: torch.Tensor,
        enemy_pos: torch.Tensor,
        enemy_velocity: torch.Tensor,
        world_bounds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate where the enemy will be when a bullet fired now reaches them.

        Args:
            self_pos: Our current position [x, y]
            enemy_pos: Enemy's current position [x, y]
            enemy_velocity: Enemy's velocity [vx, vy] (already denormalized)
            world_bounds: World size [width, height]

        Returns:
            Predicted enemy position [x, y] accounting for toroidal wrapping
        """
        # Calculate current distance to enemy (accounting for wrapping)
        to_enemy = self._calculate_wrapped_vector(self_pos, enemy_pos, world_bounds)
        current_distance = torch.norm(to_enemy)

        if current_distance < 1e-6:
            return enemy_pos  # Already at same position

        # Calculate time for bullet to travel current distance
        bullet_travel_time = current_distance / self.bullet_speed

        # Predict where enemy will be after this time
        predicted_displacement = enemy_velocity * bullet_travel_time
        predicted_pos = enemy_pos + predicted_displacement

        # Apply world wrapping to predicted position
        predicted_pos[0] = predicted_pos[0] % world_bounds[0]  # Wrap x
        predicted_pos[1] = predicted_pos[1] % world_bounds[1]  # Wrap y

        return predicted_pos

    def _calculate_shooting_angle_threshold(self, distance: torch.Tensor) -> float:
        """
        Calculate the dynamic shooting angle threshold based on target distance.

        The angle threshold is based on the apparent angular size of the target:
        angular_size = 2 * arctan(radius / distance)

        Args:
            distance: Distance to target

        Returns:
            Angle threshold in radians
        """
        if distance < 1e-6:
            return self.angle_threshold  # Fallback for very close targets

        # Calculate angular size of target (half-angle from center to edge)
        # For a circular target: half_angle = arctan(radius / distance)
        half_angle = torch.atan(self.target_radius / distance)

        # Full angular threshold = radius_multiplier * 2 * half_angle
        # This gives us the angle within which we'll shoot
        dynamic_threshold = self.radius_multiplier * 2 * half_angle

        # Clamp to reasonable bounds
        max_threshold = np.deg2rad(45.0)  # Never more than 45 degrees
        min_threshold = np.deg2rad(1.0)  # Never less than 1 degree

        return float(torch.clamp(dynamic_threshold, min_threshold, max_threshold))


def create_scripted_agent(**kwargs) -> ScriptedAgent:
    """
    Factory function to create a scripted agent.

    Args:
        **kwargs: Arguments to pass to ScriptedAgent constructor

    Returns:
        ScriptedAgent instance
    """
    return ScriptedAgent(**kwargs)
