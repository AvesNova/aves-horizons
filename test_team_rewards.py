#!/usr/bin/env python3
"""
Test script to verify the new team-based reward system works correctly.
Tests both death rewards and damage rewards.
"""

import sys
import os
sys.path.append('src')

import torch
import numpy as np
from env import Environment
from constants import Actions

def test_team_death_rewards():
    """Test team-based death rewards."""
    
    print("Testing team-based death reward system...")
    
    # Create environment with custom state to test reward system directly
    from state import State
    from ship import Ship, default_ship_config
    
    env = Environment(memory_size=3, max_ships=2)
    
    # Create two states manually: one with both alive, one with ship 1 dead
    ship_0_alive = Ship(
        ship_id=0, team_id=0, ship_config=default_ship_config,
        initial_x=100, initial_y=100, initial_vx=50, initial_vy=0,
        world_size=env.world_size
    )
    ship_1_alive = Ship(
        ship_id=1, team_id=1, ship_config=default_ship_config,
        initial_x=200, initial_y=200, initial_vx=50, initial_vy=0,
        world_size=env.world_size
    )
    
    # State 1: Both ships alive
    state_1 = State(ships={0: ship_0_alive, 1: ship_1_alive})
    
    # State 2: Ship 1 is dead
    ship_0_alive_2 = Ship(
        ship_id=0, team_id=0, ship_config=default_ship_config,
        initial_x=110, initial_y=110, initial_vx=50, initial_vy=0,
        world_size=env.world_size
    )
    ship_1_dead = Ship(
        ship_id=1, team_id=1, ship_config=default_ship_config,
        initial_x=210, initial_y=210, initial_vx=50, initial_vy=0,
        world_size=env.world_size
    )
    ship_1_dead.damage_ship(1000)  # Kill it
    
    state_2 = State(ships={0: ship_0_alive_2, 1: ship_1_dead})
    
    # Set up environment state history
    env.state.clear()
    env.state.append(state_1)  # Previous state
    env.state.append(state_2)  # Current state
    
    print(f"State 1: Ship 0 alive={state_1.ships[0].alive}, Ship 1 alive={state_1.ships[1].alive}")
    print(f"State 2: Ship 0 alive={state_2.ships[0].alive}, Ship 1 alive={state_2.ships[1].alive}")
    
    # Test team reward calculation directly
    team_0_reward = env._calculate_team_reward(state_2, team_id=0)
    team_1_reward = env._calculate_team_reward(state_2, team_id=1)
    
    print(f"Team 0 reward: {team_0_reward}")
    print(f"Team 1 reward: {team_1_reward}")
    
    # Expected: Team 0 should get +1 (enemy died), Team 1 should get -1 (ally died)
    expected_team_0_reward = 1.0  # Enemy died
    expected_team_1_reward = -1.0  # Our ship died
    
    print(f"Expected: Team 0 = {expected_team_0_reward}, Team 1 = {expected_team_1_reward}")
    print(f"Actual:   Team 0 = {team_0_reward}, Team 1 = {team_1_reward}")
    
    # Test passed if rewards match expectations
    success = (abs(team_0_reward - expected_team_0_reward) < 0.01 and 
               abs(team_1_reward - expected_team_1_reward) < 0.01)
    
    print(f"Test {'PASSED' if success else 'FAILED'}!")
    
    return success

def test_team_damage_rewards():
    """Test team-based damage rewards."""
    
    print("\n" + "="*50)
    print("Testing team-based damage reward system...")
    
    from state import State
    from ship import Ship, default_ship_config
    
    env = Environment(memory_size=3, max_ships=2)
    
    # State 1: Both ships at full health
    ship_0_full = Ship(
        ship_id=0, team_id=0, ship_config=default_ship_config,
        initial_x=100, initial_y=100, initial_vx=50, initial_vy=0,
        world_size=env.world_size
    )
    ship_1_full = Ship(
        ship_id=1, team_id=1, ship_config=default_ship_config,
        initial_x=200, initial_y=200, initial_vx=50, initial_vy=0,
        world_size=env.world_size
    )
    
    state_1 = State(ships={0: ship_0_full, 1: ship_1_full})
    
    # State 2: Ship 1 takes 10 damage
    ship_0_full_2 = Ship(
        ship_id=0, team_id=0, ship_config=default_ship_config,
        initial_x=110, initial_y=110, initial_vx=50, initial_vy=0,
        world_size=env.world_size
    )
    ship_1_damaged = Ship(
        ship_id=1, team_id=1, ship_config=default_ship_config,
        initial_x=210, initial_y=210, initial_vx=50, initial_vy=0,
        world_size=env.world_size
    )
    ship_1_damaged.damage_ship(10)  # 10 damage
    
    state_2 = State(ships={0: ship_0_full_2, 1: ship_1_damaged})
    
    # Set up environment state history
    env.state.clear()
    env.state.append(state_1)  # Previous state
    env.state.append(state_2)  # Current state
    
    print(f"State 1: Ship 0 health={state_1.ships[0].health}, Ship 1 health={state_1.ships[1].health}")
    print(f"State 2: Ship 0 health={state_2.ships[0].health}, Ship 1 health={state_2.ships[1].health}")
    
    # Test team reward calculation directly
    team_0_reward = env._calculate_team_reward(state_2, team_id=0)
    team_1_reward = env._calculate_team_reward(state_2, team_id=1)
    
    print(f"Team 0 reward: {team_0_reward}")
    print(f"Team 1 reward: {team_1_reward}")
    
    # Expected: Team 0 should get +0.1 (enemy took 10 damage * 0.01), Team 1 should get -0.1 (ally took damage)
    expected_team_0_reward = 0.1  # Enemy took 10 damage * 0.01 = 0.1
    expected_team_1_reward = -0.1  # Our ship took 10 damage * 0.01 = -0.1
    
    print(f"Expected: Team 0 = {expected_team_0_reward}, Team 1 = {expected_team_1_reward}")
    print(f"Actual:   Team 0 = {team_0_reward}, Team 1 = {team_1_reward}")
    
    # Test passed if rewards match expectations
    success = (abs(team_0_reward - expected_team_0_reward) < 0.01 and 
               abs(team_1_reward - expected_team_1_reward) < 0.01)
    
    print(f"Damage reward test {'PASSED' if success else 'FAILED'}!")
    
    return success

def test_combined_death_and_damage():
    """Test combined death and damage rewards."""
    
    print("\n" + "="*50)
    print("Testing combined death and damage rewards...")
    
    from state import State
    from ship import Ship, default_ship_config
    
    env = Environment(memory_size=3, max_ships=4)
    
    # State 1: All ships alive, full health (2v2)
    ships_state1 = {}
    for i in range(4):
        team_id = 0 if i < 2 else 1  # Ships 0,1 on team 0; ships 2,3 on team 1
        ships_state1[i] = Ship(
            ship_id=i, team_id=team_id, ship_config=default_ship_config,
            initial_x=100 + i * 100, initial_y=100, initial_vx=50, initial_vy=0,
            world_size=env.world_size
        )
    state_1 = State(ships=ships_state1)
    
    # State 2: Ship 1 dies, Ship 2 takes damage
    ships_state2 = {}
    for i in range(4):
        team_id = 0 if i < 2 else 1
        ships_state2[i] = Ship(
            ship_id=i, team_id=team_id, ship_config=default_ship_config,
            initial_x=110 + i * 100, initial_y=110, initial_vx=50, initial_vy=0,
            world_size=env.world_size
        )
        if i == 1:  # Kill ship 1 (team 0)
            ships_state2[i].damage_ship(1000)
        elif i == 2:  # Damage ship 2 (team 1)
            ships_state2[i].damage_ship(20)
    
    state_2 = State(ships=ships_state2)
    
    # Set up environment state history
    env.state.clear()
    env.state.append(state_1)  # Previous state
    env.state.append(state_2)  # Current state
    
    print("Team assignments:")
    print("  Ships 0,1: Team 0")
    print("  Ships 2,3: Team 1")
    print(f"State 2 events:")
    print(f"  Ship 1 (team 0): died")
    print(f"  Ship 2 (team 1): took 20 damage")
    
    # Test team reward calculation
    team_0_reward = env._calculate_team_reward(state_2, team_id=0)
    team_1_reward = env._calculate_team_reward(state_2, team_id=1)
    
    print(f"Team 0 reward: {team_0_reward}")
    print(f"Team 1 reward: {team_1_reward}")
    
    # Expected:
    # Team 0: -1.0 (ally died) + 0.2 (enemy took 20 damage * 0.01) = -0.8
    # Team 1: +1.0 (enemy died) - 0.2 (ally took 20 damage * 0.01) = +0.8
    expected_team_0_reward = -0.8
    expected_team_1_reward = 0.8
    
    print(f"Expected: Team 0 = {expected_team_0_reward}, Team 1 = {expected_team_1_reward}")
    print(f"Actual:   Team 0 = {team_0_reward}, Team 1 = {team_1_reward}")
    
    success = (abs(team_0_reward - expected_team_0_reward) < 0.01 and 
               abs(team_1_reward - expected_team_1_reward) < 0.01)
    
    print(f"Combined test {'PASSED' if success else 'FAILED'}!")
    
    return success

if __name__ == "__main__":
    test1_passed = test_team_death_rewards()
    test2_passed = test_team_damage_rewards()
    test3_passed = test_combined_death_and_damage()
    
    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"Death rewards test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Damage rewards test: {'PASSED' if test2_passed else 'FAILED'}")
    print(f"Combined rewards test: {'PASSED' if test3_passed else 'FAILED'}")
    
    if test1_passed and test2_passed and test3_passed:
        print("All tests PASSED! Team-based reward system is working correctly.")
    else:
        print("Some tests FAILED. Please check the implementation.")