"""
Deathmatch game mode for Aves Horizons.

In deathmatch mode, teams fight until only one team remains standing.
Each team starts with a configurable number of ships, and the game ends
when all ships from all but one team have been eliminated.
"""

import torch
import numpy as np
import copy
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from core.ships import Ships
from core.environment import Environment


@dataclass
class DeathmatchConfig:
    """Configuration for deathmatch game mode."""
    n_teams: int = 2
    ships_per_team: int = 4
    world_size: Tuple[float, float] = (1200.0, 800.0)
    spawn_radius: float = 100.0  # Radius around spawn points
    min_team_separation: float = 300.0  # Minimum distance between team spawn areas
    

class DeathmatchEnvironment(Environment):
    """
    Deathmatch-specific environment that handles team-based spawning and win conditions.
    
    Features:
    - Team-based ship spawning with separation
    - Win condition checking (last team standing)
    - Team-aware observations and rewards
    """
    
    def __init__(self, config: DeathmatchConfig, **kwargs):
        self.config = config
        self.total_ships = config.n_teams * config.ships_per_team
        
        # Initialize base environment
        super().__init__(
            n_ships=self.total_ships,
            world_size=config.world_size,
            **kwargs
        )
        
        # Generate team spawn points
        self.team_spawn_points = self._generate_team_spawn_points()
        
    def _generate_team_spawn_points(self) -> List[complex]:
        """Generate spawn points for each team, ensuring minimum separation."""
        spawn_points = []
        
        if self.config.n_teams == 2:
            # For 2 teams, place them on opposite sides
            spawn_points = [
                complex(self.config.spawn_radius, self.world_size[1] / 2),
                complex(self.world_size[0] - self.config.spawn_radius, self.world_size[1] / 2)
            ]
        else:
            # For more teams, arrange in a circle
            center = complex(self.world_size[0] / 2, self.world_size[1] / 2)
            radius = min(self.world_size) / 3
            
            for i in range(self.config.n_teams):
                angle = 2 * np.pi * i / self.config.n_teams
                spawn_point = center + radius * complex(np.cos(angle), np.sin(angle))
                spawn_points.append(spawn_point)
                
        return spawn_points
    
    def _create_team_ships(self) -> Ships:
        """Create ships with proper team assignments and spawn positions."""
        # Create team assignments
        team_ids = []
        positions = []
        
        for team_id in range(self.config.n_teams):
            spawn_center = self.team_spawn_points[team_id]
            
            for ship_in_team in range(self.config.ships_per_team):
                team_ids.append(team_id)
                
                # Random position around spawn point
                angle = np.random.uniform(0, 2 * np.pi)
                distance = np.random.uniform(0, self.config.spawn_radius)
                offset = distance * complex(np.cos(angle), np.sin(angle))
                
                ship_pos = spawn_center + offset
                # Ensure ship is within world bounds
                ship_pos = complex(
                    max(50, min(self.world_size[0] - 50, ship_pos.real)),
                    max(50, min(self.world_size[1] - 50, ship_pos.imag))
                )
                positions.append(ship_pos)
        
        # Create ships with team assignments
        ships = Ships.from_scalars(
            n_ships=self.total_ships,
            world_size=self.world_size,
            random_positions=False,
            initial_position=0+0j,  # Will be overridden
            team_ids=team_ids
        )
        
        # Set custom positions
        for i, pos in enumerate(positions):
            ships.position[i] = pos
            
        return ships
    
    def reset(self):
        """Reset environment with team-based spawning."""
        self.ships = self._create_team_ships()
        # Add to history
        self.ships_history.append(copy.deepcopy(self.ships))
        self.projectiles = {}
        return self.get_observation()
    
    def check_win_condition(self) -> Tuple[bool, Optional[int]]:
        """
        Check if the game is over and which team won.
        
        Returns:
            (game_over, winning_team_id)
        """
        if not hasattr(self.ships, 'active'):
            return False, None
            
        # Count active ships per team
        team_counts = {}
        active_mask = self.ships.get_active_mask()
        
        for team_id in range(self.config.n_teams):
            team_mask = (self.ships.team_id == team_id) & active_mask
            team_counts[team_id] = torch.sum(team_mask).item()
        
        # Check how many teams have ships remaining
        teams_with_ships = [team_id for team_id, count in team_counts.items() if count > 0]
        
        if len(teams_with_ships) <= 1:
            # Game over
            winning_team = teams_with_ships[0] if teams_with_ships else None
            return True, winning_team
        else:
            # Game continues
            return False, None
    
    def calculate_team_rewards(self) -> torch.Tensor:
        """
        Calculate rewards for each ship based on team performance.
        
        Returns:
            rewards: [n_ships] tensor of rewards
        """
        rewards = torch.zeros(self.total_ships)
        
        # Basic survival reward for active ships
        active_mask = self.ships.get_active_mask()
        rewards[active_mask] += 1.0
        
        # Team-based rewards could be added here
        # e.g., bonus for eliminating enemy ships, penalty for losing teammates
        
        return rewards
    
    def step(self, actions):
        """Enhanced step function with team-aware rewards and win condition checking."""
        # Standard environment step
        observation, _, _ = super().step(actions)
        
        # Calculate team-aware rewards
        rewards = self.calculate_team_rewards()
        
        # Check win condition
        game_over, winning_team = self.check_win_condition()
        
        # Enhanced observation with game state
        observation['game_over'] = game_over
        observation['winning_team'] = winning_team
        observation['team_counts'] = self.get_team_ship_counts()
        
        return observation, rewards, game_over
    
    def get_team_ship_counts(self) -> Dict[int, int]:
        """Get the number of active ships per team."""
        team_counts = {}
        active_mask = self.ships.get_active_mask()
        
        for team_id in range(self.config.n_teams):
            team_mask = (self.ships.team_id == team_id) & active_mask
            team_counts[team_id] = torch.sum(team_mask).item()
            
        return team_counts
    
    def get_teams_alive(self) -> List[int]:
        """Get list of team IDs that still have active ships."""
        team_counts = self.get_team_ship_counts()
        return [team_id for team_id, count in team_counts.items() if count > 0]


def create_deathmatch_game(
    n_teams: int = 2,
    ships_per_team: int = 4,
    world_size: Tuple[float, float] = (1200.0, 800.0),
    use_continuous_collision: bool = True,  # Enable continuous collision by default
    **kwargs
) -> DeathmatchEnvironment:
    """
    Convenience function to create a deathmatch game with standard settings.
    
    Args:
        n_teams: Number of teams
        ships_per_team: Ships per team
        world_size: World dimensions
        use_continuous_collision: Whether to use continuous collision detection
        **kwargs: Additional arguments passed to environment
        
    Returns:
        Configured DeathmatchEnvironment
    """
    config = DeathmatchConfig(
        n_teams=n_teams,
        ships_per_team=ships_per_team,
        world_size=world_size
    )
    
    return DeathmatchEnvironment(
        config, 
        use_continuous_collision=use_continuous_collision,
        **kwargs
    )
