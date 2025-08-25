import torch


class PhysicsEngine:
    def __init__(self, dt=0.016):  # 60 FPS default
        self.dt = dt

    def update_ships(self, ships, actions):
        # This is now handled by the ShipPhysics forward pass
        # Keep for backward compatibility but delegate to the ships' physics
        return ships

    def _handle_shooting(self, ships, shoot_action):
        return {}

    def update_projectiles(self, projectiles):
        if not projectiles:
            return projectiles

        projectiles["position"] += projectiles["velocity"] * self.dt
        projectiles["lifetime"] -= self.dt
        projectiles["active"] = projectiles["lifetime"] > 0

        return projectiles
