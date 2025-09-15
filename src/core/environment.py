from collections import deque
import copy
import torch

from core.ships import Ships
from core.ship_physics import ShipPhysics
from core.projectile import update_projectiles, fire_projectile
from utils.config import Actions


class Environment:
    def __init__(
        self, world_size=(1200, 800), n_ships=2, n_obstacles=5, memory_length=8
    ):
        self.world_size = torch.tensor(world_size)
        self.n_ships = n_ships
        self.n_obstacles = n_obstacles
        self.memory_length = memory_length
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
        fire_mask = (actions[:, Actions.shoot] > 0) & (
            self.ships.projectile_cooldown <= 0
        )

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
        """Vectorized bullet-ship collision detection."""
        if not hasattr(self.ships, "projectiles_active"):
            return

        # Get all active projectiles
        active_mask = self.ships.projectiles_active  # (n_ships, max_projectiles)

        if not torch.any(active_mask):
            return

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
                    and self.ships.health[ship_idx] > 0
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

        ship_positions_2d = torch.stack(
            [self.ships.position.real, self.ships.position.imag], dim=-1
        )

        for ship_idx in range(self.n_ships):
            if self.ships.health[ship_idx] <= 0:
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
