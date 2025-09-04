import torch

from core.ship import Ships
from core.ship_physics import ShipPhysics
from core.projectile import update_projectiles, fire_projectile
from utils.config import Actions


class Environment:
    def __init__(self, world_size=(1200, 800), n_ships=2, n_obstacles=5):
        self.world_size = torch.tensor(world_size)
        self.n_ships = n_ships
        self.n_obstacles = n_obstacles
        self.ships = None
        self.projectiles = {}
        self.obstacles = self._generate_obstacles()
        self.physics_engine = ShipPhysics()
        self.target_dt = (
            self.physics_engine.target_timestep
        )  # Use same timestep as physics engine

    def _generate_obstacles(self):
        obstacles = []
        for _ in range(self.n_obstacles):
            pos = torch.rand(2) * self.world_size
            size = torch.rand(2) * 50 + 30  # Random size between 30-80
            obstacles.append({"position": pos, "size": size})
        return obstacles

    def reset(self):
        self.ships = Ships.from_scalars(
            n_ships=self.n_ships, world_size=self.world_size
        )
        self.projectiles = {}
        return self.get_observation()

    def step(self, actions):
        # Update ships
        self.physics_engine(self.ships, actions)

        # Handle ship firing
        for i in range(self.n_ships):
            if actions[i, Actions.shoot] and self.ships.projectile_cooldown[i] <= 0:
                fire_projectile(self.ships, i)
            self.ships.projectile_cooldown[i] = max(
                0, self.ships.projectile_cooldown[i] - self.target_dt
            )

        # Update projectiles
        update_projectiles(self.ships, self.target_dt)

        # Check for bullet collisions
        self._check_bullet_collisions()

        # Apply wrap-around for ship positions
        self.ships.position.real = torch.fmod(
            self.ships.position.real, self.world_size[0]
        )
        self.ships.position.imag = torch.fmod(
            self.ships.position.imag, self.world_size[1]
        )
        # Handle negative positions (fmod can return negative values)
        self.ships.position.real = torch.where(
            self.ships.position.real < 0,
            self.ships.position.real + self.world_size[0],
            self.ships.position.real,
        )
        self.ships.position.imag = torch.where(
            self.ships.position.imag < 0,
            self.ships.position.imag + self.world_size[1],
            self.ships.position.imag,
        )

        # Apply wrap-around for projectile positions
        self.ships.projectiles_position.real = torch.fmod(
            self.ships.projectiles_position.real, self.world_size[0]
        )
        self.ships.projectiles_position.imag = torch.fmod(
            self.ships.projectiles_position.imag, self.world_size[1]
        )
        self.ships.projectiles_position.real = torch.where(
            self.ships.projectiles_position.real < 0,
            self.ships.projectiles_position.real + self.world_size[0],
            self.ships.projectiles_position.real,
        )
        self.ships.projectiles_position.imag = torch.where(
            self.ships.projectiles_position.imag < 0,
            self.ships.projectiles_position.imag + self.world_size[1],
            self.ships.projectiles_position.imag,
        )

        # Check collisions
        self._check_collisions()

        # Get observation
        observation = self.get_observation()

        # For now, use dummy rewards
        rewards = torch.zeros(self.n_ships)

        # Check if episode is done
        done = torch.all(self.ships.health <= 0)

        return observation, rewards, done

    def _check_collisions(self):
        # Check ship-projectile collisions
        for proj in self.projectiles:
            for i in range(self.n_ships):
                if i != proj["owner"]:  # Don't collide with owner
                    dist = torch.abs(self.ships.position[i] - proj["position"])
                    if dist < self.ships.collision_radius[i]:
                        self.ships.health[i] -= 10  # Damage
                        proj["lifetime"] = 0  # Remove projectile

        # Check ship-obstacle collisions (simple AABB)
        for ship_idx in range(self.n_ships):
            ship_pos = self.ships.position[ship_idx]
            ship_rad = self.ships.collision_radius[ship_idx]

            for obs in self.obstacles:
                # Extract real/imaginary parts for AABB collision
                ship_x = ship_pos.real
                ship_y = ship_pos.imag

                obs_min = obs["position"] - obs["size"] / 2
                obs_max = obs["position"] + obs["size"] / 2

                # Find closest point on obstacle to ship
                closest_x = torch.clamp(ship_x, obs_min[0], obs_max[0])
                closest_y = torch.clamp(ship_y, obs_min[1], obs_max[1])

                # Distance using complex number
                closest_point = torch.complex(closest_x, closest_y)
                dist = torch.abs(ship_pos - closest_point)

                if dist < ship_rad:
                    # Simple collision response - bounce
                    self.ships.velocity[ship_idx] *= -0.5
                    self.ships.health[ship_idx] -= 1

        # Check bullet-ship collisions
        self._check_bullet_collisions()

    def _check_bullet_collisions(self):
        """Check for collisions between bullets and ships and apply damage."""
        for ship_idx in range(self.n_ships):
            for proj_idx in range(self.ships.max_projectiles):
                if self.ships.projectiles_active[ship_idx, proj_idx]:
                    proj_pos = self.ships.projectiles_position[ship_idx, proj_idx]
                    for target_idx in range(self.n_ships):
                        if target_idx != ship_idx and self.ships.health[target_idx] > 0:
                            dist = torch.abs(proj_pos - self.ships.position[target_idx])
                            if dist < self.ships.collision_radius[target_idx]:
                                # Collision detected
                                self.ships.health[target_idx] -= 10

    def get_observation(self):
        # For now, return all ship and projectile data
        # This will be refined in later stages
        return {
            "ships": self.ships,
            "projectiles": self.projectiles,
            "obstacles": self.obstacles,
        }
