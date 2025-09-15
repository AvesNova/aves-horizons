from collections import deque
import copy
import torch

from core.ships import Ships
from core.ship_physics import ShipPhysics
from core.projectile import update_projectiles, fire_projectile
from utils.config import Actions


class Environment:
    def __init__(
        self, 
        world_size=(1200, 800), 
        n_ships=2, 
        n_obstacles=5, 
        memory_length=8,
        use_continuous_collision=True  # Enable continuous collision detection by default
    ):
        self.world_size = torch.tensor(world_size)
        self.n_ships = n_ships
        self.n_obstacles = n_obstacles
        self.memory_length = memory_length
        self.use_continuous_collision = use_continuous_collision
        
        # Store history of ships states for memory/transformer usage
        self.ships_history: deque[Ships] = deque(maxlen=self.memory_length)
        # Current ships state that physics and collision systems work with
        self.ships: Ships = None
        self.projectiles = {}
        self.obstacles = self._generate_obstacles()
        self.physics_engine = ShipPhysics()
        self.target_dt = self.physics_engine.target_timestep

        # Pre-allocate tensors for collision detection to avoid repeated allocation
        self._collision_temp_tensors = {}

    def _generate_obstacles(self):
        obstacles = []
        for _ in range(self.n_obstacles):
            pos = torch.rand(2) * self.world_size
            size = torch.rand(2) * 50 + 30  # Random size between 30-80
            obstacles.append({"position": pos, "size": size})
        return obstacles

    def reset(self):
        # Create new ships state
        self.ships = Ships.from_scalars(n_ships=self.n_ships, world_size=self.world_size)
        # Add to history
        self.ships_history.append(copy.deepcopy(self.ships))
        self.projectiles = {}
        return self.get_observation()

    def step(self, actions):

        # Update ships
        self.physics_engine(self.ships, actions)

        # Handle ship firing
        self._handle_ship_firing(actions)

        # Update projectiles
        update_projectiles(self.ships, self.target_dt)

        # Apply wrap-around for positions (optimized)
        self._apply_wrap_around()

        # Check all collisions (optimized)
        self._check_all_collisions()
        
        # Update active status after damage from collisions
        self.ships.update_active_status()

        # Add current state to history after all updates
        self.ships_history.append(copy.deepcopy(self.ships))

        # Get observation
        observation = self.get_observation()

        # For now, use dummy rewards
        rewards = torch.zeros(self.n_ships)

        # Check if episode is done
        done = torch.all(self.ships.health <= 0).item()  # Convert to Python bool

        return observation, rewards, done

    def _handle_ship_firing(self, actions):
        """Handle ship firing logic."""
        # Only active (alive) ships can fire
        active_mask = self.ships.get_active_mask()
        fire_mask = (actions[:, Actions.shoot] > 0) & (
            self.ships.projectile_cooldown <= 0
        ) & active_mask  # Add active ship check

        for i in range(self.n_ships):
            if fire_mask[i]:
                fire_projectile(self.ships, i)

        # Update cooldowns
        self.ships.projectile_cooldown = torch.clamp(
            self.ships.projectile_cooldown - self.target_dt, min=0
        )

    def _apply_wrap_around(self):
        """Optimized wrap-around for both ships and projectiles."""
        # Ship positions
        self.ships.position.real = torch.remainder(
            self.ships.position.real, self.world_size[0]
        )
        self.ships.position.imag = torch.remainder(
            self.ships.position.imag, self.world_size[1]
        )

        # Projectile positions
        self.ships.projectiles_position.real = torch.remainder(
            self.ships.projectiles_position.real, self.world_size[0]
        )
        self.ships.projectiles_position.imag = torch.remainder(
            self.ships.projectiles_position.imag, self.world_size[1]
        )

    def _check_all_collisions(self):
        """Optimized collision detection for all object types."""
        # Check bullet-ship collisions (most common, optimize first)
        self._check_bullet_ship_collisions_vectorized()

        # Check ship-obstacle collisions
        self._check_ship_obstacle_collisions()

    def _check_bullet_ship_collisions_vectorized(self):
        """Vectorized bullet-ship collision detection with optional continuous collision detection."""
        if not hasattr(self.ships, "projectiles_active"):
            return

        # Only check collisions for active (alive) ships
        ship_active_mask = self.ships.get_active_mask()
        if not torch.any(ship_active_mask):
            return

        # Get all active projectiles
        active_mask = self.ships.projectiles_active  # (n_ships, max_projectiles)

        if not torch.any(active_mask):
            return

        # Choose collision detection method based on configuration
        if self.use_continuous_collision:
            # Use continuous collision detection (ray casting) - better for low frame rates
            self._check_continuous_projectile_collisions(active_mask, ship_active_mask)
        else:
            # Use discrete collision detection - faster but can miss fast projectiles
            self._check_discrete_projectile_collisions(active_mask, ship_active_mask)

    def _check_continuous_projectile_collisions(self, active_mask, ship_active_mask):
        """Continuous collision detection using swept collision detection."""
        # Calculate previous positions based on current position and velocity
        prev_projectile_positions = (
            self.ships.projectiles_position - 
            self.ships.projectiles_velocity * self.target_dt
        )
        current_projectile_positions = self.ships.projectiles_position

        # Also calculate previous ship positions for swept collision
        prev_ship_positions = self.ships.position - self.ships.velocity * self.target_dt
        current_ship_positions = self.ships.position

        # Map back to original projectile indices for ownership check
        active_indices = torch.nonzero(active_mask)  # (n_active_proj, 2) - [ship_idx, proj_idx]

        if len(active_indices) == 0:
            return

        # Check each active projectile against all ships
        for proj_idx in range(len(active_indices)):
            owner_ship_idx = active_indices[proj_idx, 0].item()
            original_proj_idx = active_indices[proj_idx, 1].item()

            # Get projectile's path (ray)
            proj_p1 = prev_projectile_positions[owner_ship_idx, original_proj_idx]
            proj_p2 = current_projectile_positions[owner_ship_idx, original_proj_idx]

            # Skip zero-length rays
            if abs(proj_p2 - proj_p1) < 1e-6:
                # For stationary projectiles, use simple distance check
                for ship_idx in range(self.n_ships):
                    if (
                        ship_idx != owner_ship_idx
                        and ship_active_mask[ship_idx]
                    ):
                        ship_center = current_ship_positions[ship_idx]
                        ship_radius = self.ships.collision_radius[ship_idx]
                        
                        if abs(proj_p2 - ship_center) < ship_radius:
                            # Apply damage
                            self.ships.health[ship_idx] -= self.ships.projectile_damage[owner_ship_idx]
                            # Deactivate projectile
                            self.ships.projectiles_active[owner_ship_idx, original_proj_idx] = False
                            break
                continue

            # Check collision with each ship (except owner)
            for ship_idx in range(self.n_ships):
                if (
                    ship_idx != owner_ship_idx
                    and ship_active_mask[ship_idx]
                ):
                    # Get ship's swept path
                    ship_p1 = prev_ship_positions[ship_idx]
                    ship_p2 = current_ship_positions[ship_idx]
                    ship_radius = self.ships.collision_radius[ship_idx]

                    # Use swept collision detection (moving circle vs moving point)
                    if self._swept_collision_detection(proj_p1, proj_p2, ship_p1, ship_p2, ship_radius):
                        # Apply damage
                        self.ships.health[ship_idx] -= self.ships.projectile_damage[owner_ship_idx]

                        # Deactivate projectile
                        self.ships.projectiles_active[owner_ship_idx, original_proj_idx] = False
                        break  # Projectile can only hit one target

    def _swept_collision_detection(self, proj_start, proj_end, ship_start, ship_end, ship_radius):
        """Swept collision detection for moving projectile vs moving circular ship.
        
        This efficiently handles the case where both projectile and ship are moving
        by calculating their relative motion and checking if they intersect during the timestep.
        
        Args:
            proj_start: Complex number representing projectile start position
            proj_end: Complex number representing projectile end position
            ship_start: Complex number representing ship start position  
            ship_end: Complex number representing ship end position
            ship_radius: Float representing ship collision radius
            
        Returns:
            bool: True if collision occurs during the timestep
        """
        # Calculate relative motion (projectile relative to ship)
        rel_start = proj_start - ship_start
        rel_end = proj_end - ship_end
        
        # If relative motion is tiny, use simple distance check
        if abs(rel_end - rel_start) < 1e-6:
            return min(abs(rel_start), abs(rel_end)) < ship_radius
        
        # Convert to 2D coordinates for easier calculation
        p1 = torch.tensor([rel_start.real.item(), rel_start.imag.item()])
        p2 = torch.tensor([rel_end.real.item(), rel_end.imag.item()])
        
        # Ray from relative start to relative end
        ray_dir = p2 - p1
        ray_length = torch.norm(ray_dir)
        
        if ray_length < 1e-6:
            return torch.norm(p1) < ship_radius
        
        # Normalize ray direction
        ray_dir_norm = ray_dir / ray_length
        
        # Find closest approach to origin (ship center in relative space)
        # Project origin onto ray
        t = -torch.dot(p1, ray_dir_norm)
        t = torch.clamp(t, 0, ray_length)  # Clamp to ray segment
        
        # Closest point on ray to origin
        closest_point = p1 + t * ray_dir_norm
        closest_distance = torch.norm(closest_point)
        
        return closest_distance < ship_radius
    
    def _ray_circle_intersection(self, ray_start, ray_end, circle_center, circle_radius):
        """Check if a ray intersects with a circle (efficient version).
        
        Args:
            ray_start: Complex number representing ray start point
            ray_end: Complex number representing ray end point  
            circle_center: Complex number representing circle center
            circle_radius: Float representing circle radius
            
        Returns:
            bool: True if intersection occurs
        """
        # Use swept collision with stationary circle
        return self._swept_collision_detection(ray_start, ray_end, circle_center, circle_center, circle_radius)
    
    def _check_discrete_projectile_collisions(self, active_mask, ship_active_mask):
        """Original discrete collision detection method (for comparison)."""
        # Get positions of all active projectiles
        active_proj_positions = self.ships.projectiles_position[
            active_mask
        ]  # (n_active_proj,)

        if len(active_proj_positions) == 0:
            return

        # Get ship positions and create distance matrix
        ship_positions = self.ships.position  # (n_ships,)

        # Convert to 2D for cdist
        proj_pos_2d = torch.stack(
            [active_proj_positions.real, active_proj_positions.imag], dim=-1
        )
        ship_pos_2d = torch.stack([ship_positions.real, ship_positions.imag], dim=-1)

        # Compute all distances at once
        distances = torch.cdist(
            proj_pos_2d, ship_pos_2d, p=2
        )  # (n_active_proj, n_ships)

        # Get collision radii for ships
        ship_radii = self.ships.collision_radius.unsqueeze(0)  # (1, n_ships)

        # Find collisions
        collision_mask = distances < ship_radii  # (n_active_proj, n_ships)

        # Map back to original projectile indices to check ownership
        active_indices = torch.nonzero(
            active_mask
        )  # (n_active_proj, 2) - [ship_idx, proj_idx]

        for proj_idx in range(len(active_indices)):
            owner_ship_idx = active_indices[proj_idx, 0].item()
            original_proj_idx = active_indices[proj_idx, 1].item()

            # Check collisions with all ships except owner
            for ship_idx in range(self.n_ships):
                if (
                    ship_idx != owner_ship_idx
                    and collision_mask[proj_idx, ship_idx]
                    and ship_active_mask[ship_idx]  # Only damage active ships
                ):

                    # Apply damage
                    self.ships.health[ship_idx] -= self.ships.projectile_damage[
                        owner_ship_idx
                    ]

                    # Deactivate projectile
                    self.ships.projectiles_active[owner_ship_idx, original_proj_idx] = (
                        False
                    )

    def _check_ship_obstacle_collisions(self):
        """Optimized ship-obstacle collision detection."""
        if len(self.obstacles) == 0:
            return

        # Only check collisions for active ships
        ship_active_mask = self.ships.get_active_mask()
        if not torch.any(ship_active_mask):
            return

        ship_positions_2d = torch.stack(
            [self.ships.position.real, self.ships.position.imag], dim=-1
        )

        for ship_idx in range(self.n_ships):
            if not ship_active_mask[ship_idx]:
                continue

            ship_pos = ship_positions_2d[ship_idx]
            ship_rad = self.ships.collision_radius[ship_idx]

            for obs in self.obstacles:
                obs_min = obs["position"] - obs["size"] / 2
                obs_max = obs["position"] + obs["size"] / 2

                # Find closest point on obstacle to ship
                closest = torch.clamp(ship_pos, obs_min, obs_max)

                # Distance to closest point
                dist = torch.norm(ship_pos - closest)

                if dist < ship_rad:
                    # Simple collision response - bounce and damage
                    # self.ships.velocity[ship_idx] *= -0.5
                    self.ships.health[ship_idx] -= 100

    def get_observation(self):
        # For now, return all ship and projectile data
        # This will be refined in later stages
        return {
            "ships": self.ships,
            "ships_history": list(self.ships_history),  # Make available for transformer/memory systems
            "projectiles": self.projectiles,
            "obstacles": self.obstacles,
        }
