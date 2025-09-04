import pygame
import torch
import math


class PygameRenderer:
    def __init__(self, world_size=(1200, 800)):
        pygame.init()
        self.screen = pygame.display.set_mode(world_size)
        pygame.display.set_caption("Ship Simulation")
        self.clock = pygame.time.Clock()
        self.world_size = world_size
        self.font = pygame.font.SysFont(None, 24)

        # Create sprite surfaces
        self.ship_surface = self._create_ship_surface()
        self.projectile_surface = self._create_projectile_surface()

        # Cache for debug text surfaces
        self.debug_surfaces = {}

    def _create_ship_surface(self):
        """Create a ship sprite surface in Asteroids style"""
        size = 32  # Size of the sprite surface
        surface = pygame.Surface((size, size), pygame.SRCALPHA)

        # Calculate points for Asteroids-style ship
        # The ship points to the right (0 degrees) as the base orientation
        center_x = size // 2
        center_y = size // 2
        ship_length = size * 0.8  # Length of the ship
        ship_width = size * 0.5  # Width of the ship

        points = [
            (center_x + ship_length // 2, center_y),  # nose
            (center_x - ship_length // 2, center_y - ship_width // 2),  # top back
            (center_x - ship_length // 3, center_y),  # back indent
            (center_x - ship_length // 2, center_y + ship_width // 2),  # bottom back
        ]

        # Draw the ship outline in white for classic Asteroids look
        pygame.draw.polygon(surface, (255, 255, 255), points, 2)
        return surface

    def _create_projectile_surface(self):
        """Create a projectile sprite surface"""
        size = 6  # Size of the projectile sprite
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(surface, (255, 255, 0), (size // 2, size // 2), size // 2)
        return surface

    def convert_ships_to_dict(self, ships):
        """Convert Ships object to dictionary for rendering"""
        return {
            "id": ships.id.detach().cpu().tolist(),
            "position": torch.stack([ships.position.real, ships.position.imag], dim=-1)
            .detach()
            .cpu()
            .tolist(),
            "attitude": torch.stack([ships.attitude.real, ships.attitude.imag], dim=-1)
            .detach()
            .cpu()
            .tolist(),
            "collision_radius": ships.collision_radius.detach().cpu().tolist(),
            "health": ships.health.detach().cpu().tolist(),
        }

    @staticmethod
    def tensors_to_numpy(nested_dict: dict) -> dict:
        if isinstance(nested_dict, list):
            return [PygameRenderer.tensors_to_numpy(v) for v in nested_dict]

        result = {}
        for key, value in nested_dict.items():
            if isinstance(value, torch.Tensor):
                if value.dtype.is_complex:
                    # Convert complex tensor to [real, imag] pairs
                    result[key] = (
                        torch.stack([value.real, value.imag], dim=-1)
                        .detach()
                        .cpu()
                        .tolist()
                    )
                else:
                    result[key] = value.detach().cpu().tolist()
            elif isinstance(value, dict):
                result[key] = PygameRenderer.tensors_to_numpy(value)
            else:
                result[key] = value

        return result

    def render(self, ships_object, projectiles_tensor, obstacles_tensor):
        self.screen.fill((0, 0, 0))  # Clear screen with black

        # Convert Ships object to dictionary
        ships = self.convert_ships_to_dict(ships_object)
        projectiles = self.tensors_to_numpy(projectiles_tensor)
        obstacles = self.tensors_to_numpy(obstacles_tensor)

        # Draw obstacles
        for obs in obstacles:
            rect = pygame.Rect(
                obs["position"][0] - obs["size"][0] / 2,
                obs["position"][1] - obs["size"][1] / 2,
                obs["size"][0],
                obs["size"][1],
            )
            pygame.draw.rect(self.screen, (100, 100, 100), rect)

        # Draw ships
        for i, ship_id in enumerate(ships["id"]):
            pos = ships["position"][i]  # [x, y] from complex conversion
            attitude = ships["attitude"][i]  # [cos, sin] from complex conversion
            radius = ships["collision_radius"][i]
            health = ships["health"][i]

            # Get rotation angle from complex unit vector
            # Add 90 degrees (π/2) to rotate from right-facing (0°) to up-facing (90°)
            angle = math.atan2(attitude[1], attitude[0])

            # Draw ship sprite
            # Create a copy of the surface for rotation and alpha
            rotated_surface = pygame.transform.rotate(
                self.ship_surface, -math.degrees(angle)
            )
            # Set alpha based on health
            rotated_surface.set_alpha(int(255 * health / 100) if health > 0 else 100)

            # Get the new rect for the rotated surface
            rect = rotated_surface.get_rect(center=(pos[0], pos[1]))
            self.screen.blit(rotated_surface, rect)

            # Draw health bar
            bar_width = 30
            bar_height = 5
            # Draw background (red)
            pygame.draw.rect(
                self.screen,
                (255, 0, 0),
                (pos[0] - bar_width / 2, pos[1] - radius - 10, bar_width, bar_height),
            )
            # Draw health (green)
            if health > 0:
                pygame.draw.rect(
                    self.screen,
                    (0, 255, 0),
                    (
                        pos[0] - bar_width / 2,
                        pos[1] - radius - 10,
                        bar_width * health / 100,
                        bar_height,
                    ),
                )

        # Draw projectiles using sprite batching
        projectiles_pos = (
            torch.stack(
                [
                    ships_object.projectiles_position.real,
                    ships_object.projectiles_position.imag,
                ],
                dim=-1,
            )
            .detach()
            .cpu()
        )
        projectiles_active = ships_object.projectiles_active.detach().cpu()

        # Pre-calculate projectile rectangles for batch blitting
        proj_rects = []
        for ship_idx in range(ships_object.n_ships):
            for proj_idx in range(ships_object.max_projectiles):
                if projectiles_active[ship_idx, proj_idx]:
                    pos = projectiles_pos[ship_idx, proj_idx]
                    rect = self.projectile_surface.get_rect(
                        center=(int(pos[0]), int(pos[1]))
                    )
                    proj_rects.append(rect)

        # Batch blit all projectiles
        if proj_rects:
            self.screen.blits([(self.projectile_surface, rect) for rect in proj_rects])

        # Draw debug info for ship 0
        if len(ships["id"]) > 0:
            pos = ships["position"][0]
            vel = ships_object.velocity[0].detach().cpu()
            attitude = ships["attitude"][0]
            turn_offset = ships_object.turn_offset[0].detach().cpu().item()
            boost = ships_object.boost[0].detach().cpu().item()
            health = ships["health"][0]
            speed = ships_object.velocity[0].abs().detach().cpu().item()
            ammo_count = ships_object.ammo_count[0].detach().cpu().item()

            # Prepare debug info
            fps = self.clock.get_fps()
            debug_info = [
                f"Ships: {len(ships['id'])} | Projectiles: {projectiles_active[0].sum().item()}",
                f"Position: ({pos[0]:.1f}, {pos[1]:.1f})",
                f"Velocity: ({vel.real:.1f}, {vel.imag:.1f})",
                f"Attitude: ({attitude[0]:.2f}, {attitude[1]:.2f})",
                f"Turn Offset: {turn_offset:.1f}°",
                f"Boost: {boost:.1f}",
                f"Health: {health}",
                f"Speed: {speed:.1f}",
                f"Ammo: {ammo_count:.1f}",
                f"FPS: {fps:.1f}",
            ]

            # Cache and render debug text
            debug_surfaces = []
            for i, text in enumerate(debug_info):
                # Only render if text changed or not in cache
                if text not in self.debug_surfaces:
                    self.debug_surfaces[text] = self.font.render(
                        text, True, (255, 255, 255)
                    )
                debug_surfaces.append((self.debug_surfaces[text], (10, 10 + i * 25)))

            # Batch blit debug text
            self.screen.blits(debug_surfaces)

        pygame.display.flip()
        self.clock.tick(50)  # Match the environment's 50 FPS

    def close(self):
        pygame.quit()
