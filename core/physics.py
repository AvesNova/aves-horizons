import torch

class PhysicsEngine:
    def __init__(self, dt=0.1, drag_coefficient=0.001, turning_penalty=0.005, lift_coefficient=0.1):
        self.dt = dt
        self.drag = drag_coefficient
        self.turning_penalty = turning_penalty
        self.lift_coefficient = lift_coefficient
    
    def update_ships(self, ships, actions):
        # Actions: [thrust, rotate_left, rotate_right, shoot] for each ship
        
        # Apply rotation
        rotate_left = actions[:, 1]
        rotate_right = actions[:, 2]
        rotation_change = (rotate_right - rotate_left) * ships['rotation_speed'] * self.dt
        ships['rotation'] *= torch.exp(1j * rotation_change)
        
        # Engine thrust (forward acceleration)
        thrust = actions[:, 0]
        turning_penalty_factor = 1.0 - torch.abs(rotate_left + rotate_right) * self.turning_penalty
        forward_acceleration_magnitude = thrust * ships['acceleration'] * turning_penalty_factor
        forward_acceleration = forward_acceleration_magnitude * ships['rotation']
        
        # Drag force (opposes velocity direction, proportional to v²)
        velocity_magnitude_squared = torch.abs(ships['velocity']) ** 2
        velocity_direction = torch.where(
            torch.abs(ships['velocity']) > 1e-6,
            ships['velocity'] / torch.abs(ships['velocity']),
            torch.zeros_like(ships['velocity'])
        )
        drag_force = -self.drag * velocity_magnitude_squared * velocity_direction
        
        # Update velocity with thrust and drag (energy-changing forces)
        ships['velocity'] += (forward_acceleration + drag_force) * self.dt
        
        # Apply lift as velocity rotation (energy-conserving)
        turn_input = rotate_right - rotate_left
        velocity_magnitude = torch.abs(ships['velocity'])
        lift_rotation_angle = turn_input * self.lift_coefficient * velocity_magnitude * self.dt
        
        # Rotate velocity vector (preserves magnitude)
        ships['velocity'] *= torch.exp(1j * lift_rotation_angle)
        
        # Update position
        ships['position'] += ships['velocity'] * self.dt
        
        return ships
    
    def _handle_shooting(self, ships, shoot_action):
        return {}
    
    def update_projectiles(self, projectiles):
        if not projectiles:
            return projectiles

        projectiles['position'] += projectiles['velocity'] * self.dt
        projectiles['lifetime'] -= self.dt
        projectiles['active'] = projectiles['lifetime'] > 0
        
        return projectiles