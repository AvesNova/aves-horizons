import torch

from core.ship import create_ships
from core.physics import PhysicsEngine

class Environment:
    def __init__(self, world_size=(800, 600), n_ships=2, n_obstacles=5):
        self.world_size = torch.tensor(world_size)
        self.n_ships = n_ships
        self.n_obstacles = n_obstacles
        self.ships = None
        self.projectiles = {}
        self.obstacles = self._generate_obstacles()
        self.physics_engine = PhysicsEngine()
        
    def _generate_obstacles(self):
        obstacles = []
        for _ in range(self.n_obstacles):
            pos = torch.rand(2) * self.world_size
            size = torch.rand(2) * 50 + 30  # Random size between 30-80
            obstacles.append({
                'position': pos,
                'size': size
            })
        return obstacles
    
    def reset(self):
        self.ships = create_ships(self.n_ships, self.world_size)
        self.projectiles = {}
        return self.get_observation()
    
    def step(self, actions):
        # Update ships and get new projectiles
        self.ships = self.physics_engine.update_ships(self.ships, actions)
        
        # Update existing projectiles
        # self.projectiles = self.physics_engine.update_projectiles(self.projectiles)
        
        # Check collisions
        self._check_collisions()
        
        # Get observation
        observation = self.get_observation()
        
        # For now, use dummy rewards
        rewards = torch.zeros(self.n_ships)
        
        # Check if episode is done
        done = torch.any(self.ships['health'] <= 0)
        
        return observation, rewards, done
    
    def _check_collisions(self):
        # Check ship-projectile collisions
        for proj in self.projectiles:
            for i in range(self.n_ships):
                if i != proj['owner']:  # Don't collide with owner
                    dist = torch.norm(self.ships['position'][i] - proj['position'])
                    if dist < self.ships['radius'][i]:
                        self.ships['health'][i] -= 10  # Damage
                        proj['lifetime'] = 0  # Remove projectile
        
        # Check ship-obstacle collisions (simple AABB)
        for ship_idx in range(self.n_ships):
            ship_pos = self.ships['position'][ship_idx]
            ship_rad = self.ships['radius'][ship_idx]
            
            for obs in self.obstacles:
                obs_min = obs['position'] - obs['size'] / 2
                obs_max = obs['position'] + obs['size'] / 2
                
                # Find closest point on obstacle to ship
                closest_x = torch.clamp(ship_pos[0], obs_min[0], obs_max[0])
                closest_y = torch.clamp(ship_pos[1], obs_min[1], obs_max[1])
                
                dist = torch.norm(ship_pos - torch.tensor([closest_x, closest_y]))
                
                if dist < ship_rad:
                    # Simple collision response - bounce
                    self.ships['velocity'][ship_idx] *= -0.5
                    self.ships['health'][ship_idx] -= 1
    
    def get_observation(self):
        # For now, return all ship and projectile data
        # This will be refined in later stages
        return {
            'ships': self.ships,
            'projectiles': self.projectiles,
            'obstacles': self.obstacles
        }