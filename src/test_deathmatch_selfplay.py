"""
Test script for the deathmatch self-play environment setup.
"""

import sys
sys.path.append('.')

import torch
from game_modes.deathmatch import create_deathmatch_game


def test_deathmatch_creation():
    """Test basic deathmatch environment creation."""
    print("Testing deathmatch environment creation...")
    
    # Create a 2v2 deathmatch game
    env = create_deathmatch_game(
        n_teams=2,
        ships_per_team=2,
        world_size=(1200.0, 800.0)
    )
    
    print(f"Environment created successfully!")
    print(f"Total ships: {env.total_ships}")
    print(f"Teams: {env.config.n_teams}")
    print(f"Ships per team: {env.config.ships_per_team}")
    
    # Test reset
    env.reset()
    print(f"Ships after reset: {len(env.ships.health)}")
    print(f"Ship team IDs: {env.ships.team_id.tolist()}")
    print(f"Ship health: {env.ships.health.tolist()}")
    
    # Test a few steps
    print("\nTesting environment steps...")
    for step in range(5):
        # Create random actions for all ships
        actions = torch.randint(0, 2, (env.total_ships, 6), dtype=torch.bool)
        
        observation, rewards, done = env.step(actions)
        
        print(f"Step {step + 1}:")
        print(f"  Alive ships: {(env.ships.health > 0).sum().item()}")
        print(f"  Done: {done}")
        print(f"  Team 0 alive: {(env.ships.health[:env.config.ships_per_team] > 0).sum().item()}")
        print(f"  Team 1 alive: {(env.ships.health[env.config.ships_per_team:] > 0).sum().item()}")
        
        if done:
            print("  Game ended!")
            break
    
    print("\nDeathmatch environment test completed successfully!")
    return True


def test_team_colors():
    """Test the team color rendering functionality."""
    print("\nTesting team color rendering...")
    
    # Create environment
    env = create_deathmatch_game(n_teams=2, ships_per_team=2)
    env.reset()
    
    # Test the renderer's ship conversion with team IDs
    from rendering.pygame_renderer import PygameRenderer
    
    renderer = PygameRenderer(world_size=[1200, 800])
    ships_dict = renderer.convert_ships_to_dict(env.ships)
    
    print(f"Ships dict keys: {ships_dict.keys()}")
    
    if 'team_id' in ships_dict:
        print(f"Team IDs found: {ships_dict['team_id']}")
        print("Team color rendering should work!")
    else:
        print("Warning: No team_id found in ships dict")
        
    renderer.close()
    return True


if __name__ == "__main__":
    try:
        test_deathmatch_creation()
        test_team_colors()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()