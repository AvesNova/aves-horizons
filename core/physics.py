import torch

class PhysicsEngine:
    def __init__(self, dt=0.1, drag_coefficient=0.02, turning_penalty=0.7):
        self.dt = dt
        self.drag = drag_coefficient
        self.turning_penalty = turning_penalty
    
    def update_ships(self, ships, actions):
        # Actions: [thrust, rotate_left, rotate_right, shoot] for each ship
        n_ships = ships['position'].shape[0]
        
        # Convert actions to tensor if needed
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32)
        
        # Apply rotation
        rotate_left = actions[:, 1]
        rotate_right = actions[:, 2]
        rotation_change = (rotate_right - rotate_left) * ships['rotation_speed'] * self.dt
        ships['rotation'] = (ships['rotation'] + rotation_change) % (2 * torch.pi)
        
        # Apply thrust with turning penalty
        thrust = actions[:, 0]
        turning_penalty_factor = 1.0 - self.turning_penalty * (rotate_left + rotate_right)
        acceleration_magnitude = thrust * ships['acceleration'] * turning_penalty_factor
        
        # Calculate acceleration vector
        acceleration_x = acceleration_magnitude * torch.cos(ships['rotation'])
        acceleration_y = acceleration_magnitude * torch.sin(ships['rotation'])
        acceleration = torch.stack([acceleration_x, acceleration_y], dim=1)
        
        # Update velocity with acceleration and drag
        ships['velocity'] += acceleration * self.dt
        ships['velocity'] *= (1 - self.drag)  # Simple drag
        
        # Update position
        ships['position'] += ships['velocity'] * self.dt
        
        # Handle shooting
        shoot_action = actions[:, 3]
        # new_projectiles = self._handle_shooting(ships, shoot_action)
        
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