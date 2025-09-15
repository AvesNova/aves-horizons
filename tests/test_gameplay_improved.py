"""
Improved gameplay tests using fixtures and constants.

This demonstrates how fixtures make gameplay tests more readable
and maintainable by providing pre-configured test scenarios.
"""

import pytest
import torch
from .test_constants import *


class TestImprovedGameplay:
    """Improved gameplay tests using fixtures."""
    
    def test_combat_health_reduction(self, combat_scenario_ships):
        """Test that ships lose health when hit by projectiles."""
        # Create environment with correct ship count for the scenario
        from core.environment import Environment
        env = Environment(n_ships=2, n_obstacles=0)
        env.reset()
        
        # Set up environment with combat scenario ships
        env.ships = combat_scenario_ships
        
        # Ship 1 aims at ship 2 and fires
        actions = torch.zeros((2, ACTION_DIM), dtype=torch.bool)
        actions[0, 5] = True  # Ship 1 shoots (action index 5 is shoot)
        
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
    
    def test_trailing_and_shooting(self, trailing_scenario_ships):
        """Test scenario where one ship trails another and fires."""
        # Create environment with correct ship count for the scenario
        from core.environment import Environment
        env = Environment(n_ships=2, n_obstacles=0)
        env.reset()
        
        # Set up environment with trailing scenario
        env.ships = trailing_scenario_ships
        
        # Set up actions: both ships move forward, ship 2 shoots
        actions = torch.zeros((2, ACTION_DIM), dtype=torch.bool)
        actions[:, 0] = True  # Both move forward (action index 0)
        actions[1, 5] = True  # Ship 2 shoots (action index 5)
        
        initial_health = env.ships.health[0].item()
        
        # Simulate for several steps
        for step in range(TEST_MAX_STEPS):
            _, _, done = env.step(actions)
            current_health = env.ships.health[0].item()
            
            # If ship 1's health decreased, the test succeeded
            if current_health < initial_health:
                print(f"Ship 1 took damage at step {step}: {initial_health} -> {current_health}")
                break
        else:
            pytest.fail("Ship 1 should have taken damage from trailing ship 2")
    
    def test_ship_death_and_active_status(self, test_environment):
        """Test that ships become inactive when health reaches zero."""
        test_environment.reset()
        
        # Manually set one ship's health to very low
        test_environment.ships.health[1] = 1.0
        
        # Set up combat scenario
        test_environment.ships.position[0] = complex(100, 100)
        test_environment.ships.position[1] = complex(120, 100)  # Close to ship 1
        
        actions = torch.zeros((DEFAULT_N_SHIPS, ACTION_DIM), dtype=torch.bool)
        actions[0, 5] = True  # Ship 0 shoots
        
        # Initial status
        assert test_environment.ships.get_active_mask()[1] == True, "Ship 2 should start active"
        
        # Simulate until ship 2 dies
        for step in range(TEST_MAX_STEPS):
            _, _, done = test_environment.step(actions)
            
            if test_environment.ships.health[1] <= 0:
                # Check that ship is now inactive
                assert test_environment.ships.get_active_mask()[1] == False, "Ship 2 should be inactive when health <= 0"
                print(f"Ship 2 died at step {step}")
                break
        else:
            pytest.fail("Ship 2 should have died within the test timeframe")
    
    def test_dead_ships_excluded_from_observations(self, test_environment, test_token_encoder):
        """Test that dead ships are excluded from token encoder observations."""
        test_environment.reset()
        
        # Kill one ship
        test_environment.ships.health[1] = 0
        test_environment.ships.update_active_status()
        
        tokens = test_token_encoder.encode_ships_to_tokens(test_environment.ships)
        
        # Should only have tokens for active ships (in this case, 3 out of 4)
        expected_active = DEFAULT_N_SHIPS - 1  # One ship dead
        assert tokens.shape[0] == expected_active, f"Should have {expected_active} tokens for active ships, got {tokens.shape[0]}"
        
        # Verify the active ship IDs
        active_indices = torch.where(test_environment.ships.get_active_mask())[0]
        assert len(active_indices) == expected_active
    
    def test_physics_skip_dead_ships(self, test_environment):
        """Test that physics calculations skip dead ships."""
        test_environment.reset()
        
        # Kill ship 1
        test_environment.ships.health[1] = 0
        test_environment.ships.update_active_status()
        
        # Record initial state of dead ship
        initial_pos = test_environment.ships.position[1].clone()
        initial_vel = test_environment.ships.velocity[1].clone()
        
        # Apply actions to all ships
        actions = torch.zeros((DEFAULT_N_SHIPS, ACTION_DIM), dtype=torch.bool)
        actions[:, 0] = True  # All ships try to move forward
        
        # Step physics
        test_environment.step(actions)
        
        # Dead ship's position and velocity should be unchanged
        tolerance = FLOAT_TOLERANCE
        assert torch.allclose(test_environment.ships.position[1], initial_pos, atol=tolerance), "Dead ship position should not change"
        assert torch.allclose(test_environment.ships.velocity[1], initial_vel, atol=tolerance), "Dead ship velocity should not change"


class TestImprovedDeathmatch:
    """Improved deathmatch tests using fixtures."""
    
    def test_deathmatch_creation(self, deathmatch_game):
        """Test creating a deathmatch game using fixture."""
        assert deathmatch_game.config.n_teams == DEFAULT_N_TEAMS
        assert deathmatch_game.config.ships_per_team == DEFAULT_SHIPS_PER_TEAM
        assert deathmatch_game.total_ships == DEFAULT_N_TEAMS * DEFAULT_SHIPS_PER_TEAM
        
        # Check that team spawn points are generated
        assert len(deathmatch_game.team_spawn_points) == DEFAULT_N_TEAMS
        
        # Spawn points should be separated
        distance = abs(deathmatch_game.team_spawn_points[0] - deathmatch_game.team_spawn_points[1])
        assert distance > deathmatch_game.config.min_team_separation
    
    def test_team_ship_spawning(self, deathmatch_game):
        """Test that ships are assigned to teams correctly."""
        deathmatch_game.reset()
        
        # Check team assignments
        for team_id in range(DEFAULT_N_TEAMS):
            team_ships = torch.sum(deathmatch_game.ships.team_id == team_id).item()
            assert team_ships == DEFAULT_SHIPS_PER_TEAM, f"Team {team_id} should have {DEFAULT_SHIPS_PER_TEAM} ships, got {team_ships}"
        
        # Check spatial separation between teams
        for team_id in range(DEFAULT_N_TEAMS):
            team_positions = deathmatch_game.ships.position[deathmatch_game.ships.team_id == team_id]
            team_center = torch.mean(team_positions)
            
            # Compare with other teams
            for other_team_id in range(team_id + 1, DEFAULT_N_TEAMS):
                other_positions = deathmatch_game.ships.position[deathmatch_game.ships.team_id == other_team_id]
                other_center = torch.mean(other_positions)
                
                separation = abs(team_center - other_center)
                assert separation > 100, f"Teams {team_id} and {other_team_id} should be spatially separated, distance: {separation}"
    
    def test_win_condition_detection(self, deathmatch_game):
        """Test that win conditions are detected correctly."""
        deathmatch_game.reset()
        
        # Initially, no team should have won
        game_over, winning_team = deathmatch_game.check_win_condition()
        assert not game_over, "Game should not be over initially"
        assert winning_team is None, "No team should have won initially"
        
        # Kill all ships from team 1
        team_1_mask = deathmatch_game.ships.team_id == 1
        deathmatch_game.ships.health[team_1_mask] = 0
        deathmatch_game.ships.update_active_status()  # Update active mask after health change
        
        # Now team 0 should win
        game_over, winning_team = deathmatch_game.check_win_condition()
        assert game_over, "Game should be over when only one team remains"
        assert winning_team == 0, f"Team 0 should have won, got {winning_team}"
    
    def test_team_rewards(self, deathmatch_game):
        """Test team-based reward calculation."""
        deathmatch_game.reset()
        
        rewards = deathmatch_game.calculate_team_rewards()
        
        # All ships should have positive survival rewards initially
        assert torch.all(rewards >= 0), "All surviving ships should have non-negative rewards"
        
        # Kill some ships and check rewards change appropriately
        deathmatch_game.ships.health[0] = 0
        deathmatch_game.ships.update_active_status()
        
        new_rewards = deathmatch_game.calculate_team_rewards()
        
        # Dead ship should have 0 reward
        assert new_rewards[0] == 0, "Dead ship should have 0 reward"
        # Living ships should still have positive rewards
        active_mask = deathmatch_game.ships.get_active_mask()
        assert torch.all(new_rewards[active_mask] > 0), "Active ships should have positive rewards"