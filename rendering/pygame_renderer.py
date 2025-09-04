import pygame
import torch
import math


class PygameRenderer:
    def __init__(self, world_size=(800, 600)):
        pygame.init()
        self.screen = pygame.display.set_mode(world_size)
        pygame.display.set_caption("Ship Simulation")
        self.clock = pygame.time.Clock()
        self.world_size = world_size
        self.font = pygame.font.SysFont(None, 24)

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
            angle = math.atan2(attitude[1], attitude[0])  # atan2(sin, cos)

            # Draw ship as a triangle
            points = []
            for j in range(3):
                point_angle = angle + j * 2 * math.pi / 3
                x = pos[0] + radius * math.cos(point_angle)
                y = pos[1] + radius * math.sin(point_angle)
                points.append((x, y))

            # Determine color based on health
            color = (0, int(255 * health / 100), 0) if health > 0 else (100, 100, 100)
            pygame.draw.polygon(self.screen, color, points)

            # Draw health bar
            bar_width = 30
            bar_height = 5
            pygame.draw.rect(
                self.screen,
                (255, 0, 0),
                (pos[0] - bar_width / 2, pos[1] - radius - 10, bar_width, bar_height),
            )
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

        # # Draw projectiles
        # for proj in projectiles:
        #     pos = proj['position']  # [x, y] from complex conversion
        #     pygame.draw.circle(self.screen, (255, 255, 0), (int(pos[0]), int(pos[1])), 3)

        # Draw debug info
        debug_text = f"Ships: {len(ships['id'])} | Projectiles: {len(projectiles)}"
        text_surface = self.font.render(debug_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 10))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()
