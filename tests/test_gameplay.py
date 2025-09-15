"""
Comprehensive gameplay tests for Aves Horizons.

Tests various gameplay scenarios including:
- Ship combat and health reduction
- Death handling and inactive ships
- Team-based gameplay
- Deathmatch game mode
"""

import torch
import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.environment import Environment
from core.ships import Ships
from game_modes.deathmatch import DeathmatchEnvironment, DeathmatchConfig, create_deathmatch_game
from utils.config import Actions


class TestBasicGameplay:
    """Test basic gameplay mechanics like combat, health, and death."""
    
    def test_ship_combat_health_reduction(self):
        """Test that ships lose health when hit by projectiles."""
        # Create simple 2-ship scenario
        env = Environment(n_ships=2, n_obstacles=0)
        env.reset()
        
        # Position ships close together
        env.ships.position[0] = complex(100, 100)  # Ship 1
        env.ships.position[1] = complex(150, 100)  # Ship 2, nearby
        
        # Ship 1 aims at ship 2 and fires
        actions = torch.zeros((2, len(Actions)), dtype=torch.bool)
        actions[0, Actions.shoot] = True  # Ship 1 shoots
        
        # Record initial health
        initial_health = env.ships.health[1].item()
        
        # Simulate several steps to allow projectile to hit
        for _ in range(20):
            _, _, done = env.step(actions)
            if done:
                break
        
        # Check that ship 2's health has decreased
        final_health = env.ships.health[1].item()
        assert final_health < initial_health, f"Ship 2 health should have decreased: {initial_health} -> {final_health}"
    
    def test_ship_trailing_and_shooting(self):
        """Test scenario where one ship trails another and fires."""
        env = Environment(n_ships=2, n_obstacles=0)
        env.reset()
        
        # Set up trailing scenario
        # Ship 1 moving right
        env.ships.position[0] = complex(100, 100)
        env.ships.velocity[0] = complex(50, 0)  # Moving right
        env.ships.attitude[0] = complex(1, 0)   # Facing right
        
        # Ship 2 behind ship 1, also moving right and shooting
        env.ships.position[1] = complex(50, 100)  # Behind ship 1
        env.ships.velocity[1] = complex(50, 0)   # Same velocity
        env.ships.attitude[1] = complex(1, 0)    # Facing same direction
        
        # Set up actions: both ships move forward, ship 2 shoots
        actions = torch.zeros((2, len(Actions)), dtype=torch.bool)
        actions[:, Actions.forward] = True  # Both move forward
        actions[1, Actions.shoot] = True    # Ship 2 shoots
        
        initial_health = env.ships.health[0].item()
        
        # Simulate for several steps
        for step in range(50):
            _, _, done = env.step(actions)
            current_health = env.ships.health[0].item()
            
            # If ship 1's health decreased, the test succeeded
            if current_health < initial_health:
                print(f"Ship 1 took damage at step {step}: {initial_health} -> {current_health}")
                break
        else:
            pytest.fail("Ship 1 should have taken damage from trailing ship 2")
    
    def test_ship_death_and_active_status(self):
        """Test that ships become inactive when health reaches zero."""
        env = Environment(n_ships=2, n_obstacles=0)
        env.reset()
        
        # Manually set one ship's health to very low
        env.ships.health[1] = 1.0
        
        # Set up combat scenario
        env.ships.position[0] = complex(100, 100)
        env.ships.position[1] = complex(120, 100)  # Close to ship 1
        
        actions = torch.zeros((2, len(Actions)), dtype=torch.bool)
        actions[0, Actions.shoot] = True
        
        # Initial status
        assert env.ships.get_active_mask()[1] == True, "Ship 2 should start active"
        
        # Simulate until ship 2 dies
        max_steps = 50
        for step in range(max_steps):
            _, _, done = env.step(actions)
            
            if env.ships.health[1] <= 0:
                # Check that ship is now inactive
                assert env.ships.get_active_mask()[1] == False, "Ship 2 should be inactive when health <= 0"
                print(f"Ship 2 died at step {step}")
                break
        else:
            pytest.fail("Ship 2 should have died within the test timeframe")
    
    def test_dead_ships_excluded_from_observations(self):
        """Test that dead ships are excluded from token encoder observations."""
        from models.token_encoder import ShipTokenEncoder
        
        env = Environment(n_ships=3, n_obstacles=0)
        env.reset()
        
        # Kill one ship
        env.ships.health[1] = 0
        env.ships.update_active_status()
        
        encoder = ShipTokenEncoder(max_ships=3)
        tokens = encoder.encode_ships_to_tokens(env.ships)
        
        # Should only have tokens for 2 active ships (ships 0 and 2)
        assert tokens.shape[0] == 2, f"Should have 2 tokens for active ships, got {tokens.shape[0]}"
        
        # Verify the active ship IDs in the tokens match expectations
        active_indices = torch.where(env.ships.get_active_mask())[0]
        assert len(active_indices) == 2
        assert 0 in active_indices and 2 in active_indices
    
    def test_physics_skip_dead_ships(self):
        """Test that physics calculations skip dead ships."""
        env = Environment(n_ships=2, n_obstacles=0)
        env.reset()
        
        # Kill ship 1
        env.ships.health[1] = 0
        env.ships.update_active_status()
        
        # Record initial state of dead ship
        initial_pos = env.ships.position[1].clone()
        initial_vel = env.ships.velocity[1].clone()
        
        # Apply actions to both ships
        actions = torch.zeros((2, len(Actions)), dtype=torch.bool)
        actions[:, Actions.forward] = True
        
        # Step physics
        env.step(actions)
        
        # Dead ship's position and velocity should be unchanged
        assert torch.allclose(env.ships.position[1], initial_pos, atol=1e-6), "Dead ship position should not change"
        assert torch.allclose(env.ships.velocity[1], initial_vel, atol=1e-6), "Dead ship velocity should not change"


class TestDeathmatchGameMode:
    """Test the deathmatch game mode functionality."""
    
    def test_deathmatch_creation(self):
        """Test creating a deathmatch game."""
        game = create_deathmatch_game(n_teams=2, ships_per_team=3)
        
        assert game.config.n_teams == 2
        assert game.config.ships_per_team == 3
        assert game.total_ships == 6
        
        # Check that team spawn points are generated
        assert len(game.team_spawn_points) == 2
        
        # Spawn points should be separated
        distance = abs(game.team_spawn_points[0] - game.team_spawn_points[1])
        assert distance > game.config.min_team_separation
    
    def test_team_ship_spawning(self):
        """Test that ships are assigned to teams correctly."""
        game = create_deathmatch_game(n_teams=2, ships_per_team=2)
        game.reset()
        
        # Check team assignments
        team_0_ships = torch.sum(game.ships.team_id == 0).item()
        team_1_ships = torch.sum(game.ships.team_id == 1).item()
        
        assert team_0_ships == 2, f"Team 0 should have 2 ships, got {team_0_ships}"
        assert team_1_ships == 2, f"Team 1 should have 2 ships, got {team_1_ships}"
        
        # Check that teams are spatially separated
        team_0_positions = game.ships.position[game.ships.team_id == 0]
        team_1_positions = game.ships.position[game.ships.team_id == 1]
        
        # Calculate average team positions
        team_0_center = torch.mean(team_0_positions)
        team_1_center = torch.mean(team_1_positions)
        
        team_separation = abs(team_0_center - team_1_center)
        assert team_separation > 100, f"Teams should be spatially separated, distance: {team_separation}"
    
    def test_win_condition_detection(self):
        """Test that win conditions are detected correctly."""
        game = create_deathmatch_game(n_teams=2, ships_per_team=2)
        game.reset()
        
        # Initially, no team should have won
        game_over, winning_team = game.check_win_condition()
        assert not game_over, "Game should not be over initially"
        assert winning_team is None, "No team should have won initially"
        
        # Kill all ships from team 1
        team_1_mask = game.ships.team_id == 1
        game.ships.health[team_1_mask] = 0
        game.ships.update_active_status()  # Update active mask after health change

        # Now team 0 should win
        game_over, winning_team = game.check_win_condition()
        assert game_over, "Game should be over when only one team remains"
        assert winning_team == 0, f"Team 0 should have won, got {winning_team}"
    
    def test_team_ship_counts(self):
        """Test tracking of ships per team."""
        game = create_deathmatch_game(n_teams=2, ships_per_team=3)
        game.reset()
        
        # Initial counts
        team_counts = game.get_team_ship_counts()
        assert team_counts[0] == 3, f"Team 0 should start with 3 ships, got {team_counts[0]}"
        assert team_counts[1] == 3, f"Team 1 should start with 3 ships, got {team_counts[1]}"
        
        # Kill 2 ships from team 0
        team_0_indices = torch.where(game.ships.team_id == 0)[0][:2]
        game.ships.health[team_0_indices] = 0
        game.ships.update_active_status()  # Update active mask after health change

        # Updated counts
        team_counts = game.get_team_ship_counts()
        assert team_counts[0] == 1, f"Team 0 should have 1 ship remaining, got {team_counts[0]}"
        assert team_counts[1] == 3, f"Team 1 should still have 3 ships, got {team_counts[1]}"
    
    def test_deathmatch_step_integration(self):
        """Test the full deathmatch step integration."""
        game = create_deathmatch_game(n_teams=2, ships_per_team=2)
        game.reset()
        
        actions = torch.zeros((4, len(Actions)), dtype=torch.bool)
        
        # Test a normal step
        observation, rewards, done = game.step(actions)
        
        # Check enhanced observation
        assert 'game_over' in observation
        assert 'winning_team' in observation
        assert 'team_counts' in observation
        
        assert observation['game_over'] == False
        assert observation['winning_team'] is None
        assert observation['team_counts'][0] == 2
        assert observation['team_counts'][1] == 2
        
        # Test rewards (should be positive for surviving ships)
        assert torch.all(rewards >= 0), "All surviving ships should have non-negative rewards"


class TestCombatMechanics:
    """Test detailed combat mechanics."""
    
    def test_projectile_collision_damage(self):
        """Test that projectiles deal correct damage on collision."""
        env = Environment(n_ships=2, n_obstacles=0)
        env.reset()
        
        # Record initial damage value
        expected_damage = env.ships.projectile_damage[0].item()
        
        # Set up close combat
        env.ships.position[0] = complex(100, 100)
        env.ships.position[1] = complex(110, 100)  # Very close
        
        initial_health = env.ships.health[1].item()
        
        actions = torch.zeros((2, len(Actions)), dtype=torch.bool)
        actions[0, Actions.shoot] = True
        
        # Simulate until hit
        for _ in range(30):
            _, _, _ = env.step(actions)
            current_health = env.ships.health[1].item()
            
            if current_health < initial_health:
                damage_dealt = initial_health - current_health
                # Allow for some tolerance due to multiple hits
                assert damage_dealt >= expected_damage * 0.8, f"Damage dealt ({damage_dealt}) should be close to expected ({expected_damage})"
                break
        else:
            pytest.fail("No damage was dealt within the test timeframe")
    
    def test_friendly_fire_prevention(self):
        """Test that ships don't damage teammates (if implemented)."""
        game = create_deathmatch_game(n_teams=2, ships_per_team=2)
        game.reset()
        
        # Find two ships on the same team
        team_0_indices = torch.where(game.ships.team_id == 0)[0]
        ship_1_idx = team_0_indices[0].item()
        ship_2_idx = team_0_indices[1].item()
        
        # Position them close together
        game.ships.position[ship_1_idx] = complex(100, 100)
        game.ships.position[ship_2_idx] = complex(110, 100)
        
        initial_health = game.ships.health[ship_2_idx].item()
        
        actions = torch.zeros((game.total_ships, len(Actions)), dtype=torch.bool)
        actions[ship_1_idx, Actions.shoot] = True
        
        # Simulate for a while
        for _ in range(50):
            _, _, _ = game.step(actions)
        
        final_health = game.ships.health[ship_2_idx].item()
        
        # Note: The current implementation doesn't prevent friendly fire
        # This test documents the current behavior
        # If friendly fire prevention is added later, this assertion should be updated
        print(f"Teammate health change: {initial_health} -> {final_health}")
        # For now, we just verify the test runs without crashing


class TestTokenEncoding:
    """Test token encoding with teams and dead ships."""
    
    def test_team_id_in_tokens(self):
        """Test that team IDs are correctly encoded in tokens."""
        from models.token_encoder import ShipTokenEncoder
        
        game = create_deathmatch_game(n_teams=2, ships_per_team=2)
        game.reset()
        
        encoder = ShipTokenEncoder(max_ships=4)
        tokens = encoder.encode_ships_to_tokens(game.ships)
        
        # Tokens should be 13-dimensional (including team_id)
        assert tokens.shape[1] == 13, f"Tokens should be 13-dimensional, got {tokens.shape[1]}"
        
        # Check that team IDs are present in tokens (11th dimension, 0-indexed)
        team_ids_in_tokens = tokens[:, 11]  # Team ID is at index 11
        
        # Should have team IDs 0 and 1
        unique_teams = torch.unique(team_ids_in_tokens)
        assert 0.0 in unique_teams and 1.0 in unique_teams, f"Should have teams 0 and 1, got {unique_teams}"
    
    def test_variable_ship_encoding(self):
        """Test that token encoder handles variable numbers of active ships."""
        from models.token_encoder import ShipTokenEncoder
        
        env = Environment(n_ships=5, n_obstacles=0)
        env.reset()
        
        encoder = ShipTokenEncoder(max_ships=5)
        
        # Initially all ships active
        tokens = encoder.encode_ships_to_tokens(env.ships)
        assert tokens.shape[0] == 5, "Should encode all 5 ships initially"
        
        # Kill 2 ships
        env.ships.health[1] = 0
        env.ships.health[3] = 0
        env.ships.update_active_status()
        
        # Now should only encode 3 ships
        tokens = encoder.encode_ships_to_tokens(env.ships)
        assert tokens.shape[0] == 3, "Should encode only 3 active ships after deaths"