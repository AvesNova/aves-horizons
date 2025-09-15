"""
Test that main.py pipeline still works with our transformer implementation.
"""
import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Test imports that main.py uses
def test_main_imports():
    """Test that all main.py imports work correctly."""
    # Test pygame import (might not work in headless environment)
    try:
        import pygame
        pygame_available = True
    except ImportError:
        pygame_available = False
        
    # Test core imports
    from core.environment import Environment
    from utils.config import Actions
    
    # Test renderer import
    try:
        from rendering.pygame_renderer import PygameRenderer  
        renderer_available = True
    except ImportError:
        renderer_available = False
        
    assert Environment is not None
    assert Actions is not None
    
    # If pygame is available, renderer should be too
    if pygame_available:
        assert renderer_available, "pygame available but renderer not importable"
        
def test_environment_basic_functionality():
    """Test that Environment works as expected by main.py."""
    from core.environment import Environment
    from utils.config import Actions
    
    # Create environment like main.py does
    env = Environment(n_ships=8, n_obstacles=0)
    
    # Reset environment
    env.reset()
    
    # Check basic properties
    assert env.n_ships == 8
    assert hasattr(env, 'world_size')
    assert hasattr(env, 'ships')
    assert hasattr(env, 'projectiles') 
    assert hasattr(env, 'obstacles')
    
    # Test step functionality like main.py uses
    actions = torch.zeros((env.n_ships, len(Actions)), dtype=torch.bool)
    
    # Set some actions like keyboard input would
    actions[0, Actions.forward] = True
    actions[0, Actions.shoot] = True
    
    # Random actions for other ships like main.py does
    for i in range(1, env.n_ships):
        actions[i] = torch.randint(0, 2, (len(Actions),), dtype=torch.bool)
    
    # Step environment
    observation, rewards, done = env.step(actions)
    
    # Verify outputs
    assert observation is not None
    assert rewards is not None
    assert isinstance(done, bool)
    
def test_renderer_creation():
    """Test that PygameRenderer can be created if pygame is available."""
    try:
        import pygame
        from rendering.pygame_renderer import PygameRenderer
        from core.environment import Environment
        
        # Create environment to get world size
        env = Environment(n_ships=4, n_obstacles=0)
        
        # Create renderer like main.py does
        renderer = PygameRenderer(world_size=env.world_size.tolist())
        
        # Basic checks
        assert renderer is not None
        assert hasattr(renderer, 'render')
        assert hasattr(renderer, 'close')
        
        # Clean up
        renderer.close()
        
    except ImportError:
        pytest.skip("pygame not available, skipping renderer test")
        
def test_main_pipeline_simulation():
    """Simulate what main.py does without the pygame event loop."""
    from core.environment import Environment
    from utils.config import Actions
    
    # Setup like main.py
    env = Environment(n_ships=8, n_obstacles=0)
    env.reset()
    
    controlled_ship = 0
    
    # Simulate several game steps
    for step in range(10):
        # Create actions like keyboard input
        actions = torch.zeros((env.n_ships, len(Actions)), dtype=torch.bool)
        
        # Simulate some keyboard input
        if step % 3 == 0:
            actions[controlled_ship, Actions.forward] = True
        if step % 4 == 0:
            actions[controlled_ship, Actions.shoot] = True
        if step % 5 == 0:
            actions[controlled_ship, Actions.left] = True
            
        # Other ships take random actions like main.py
        for i in range(env.n_ships):
            if i != controlled_ship:
                actions[i] = torch.randint(0, 2, (len(Actions),), dtype=torch.bool)
        
        # Step environment
        observation, rewards, done = env.step(actions)
        
        # Verify step worked
        assert observation is not None
        assert rewards is not None
        assert isinstance(done, bool)
        
        # Reset if done (like main.py)
        if done:
            env.reset()
            
    print("Main pipeline simulation completed successfully")

def test_main_with_transformer_integration():
    """Test that main.py environment works with our new transformer components."""
    from core.environment import Environment
    from utils.config import Actions
    from models.state_history import StateHistory
    from models.ship_transformer import ShipTransformerMVP
    
    # Setup environment like main.py
    env = Environment(n_ships=8, n_obstacles=0)
    env.reset()
    
    # Add transformer components for AI control
    history = StateHistory(sequence_length=3, max_ships=8)
    model = ShipTransformerMVP(d_model=32, nhead=2, num_layers=1)
    
    controlled_ship = 0  # Human controlled
    ai_ships = list(range(1, 8))  # AI controlled
    
    # Run simulation with AI + human control
    for step in range(5):
        actions = torch.zeros((env.n_ships, len(Actions)), dtype=torch.bool)
        
        # Simulate human control of ship 0
        if step % 2 == 0:
            actions[controlled_ship, Actions.forward] = True
        if step % 3 == 0:
            actions[controlled_ship, Actions.shoot] = True
            
        # AI control for other ships using transformer
        history.add_state(env.ships, actions)
        
        if history.is_ready():
            # Get AI actions from transformer
            tokens, ship_ids = history.get_token_sequence()
            with torch.no_grad():
                ai_actions_logits = model(tokens.unsqueeze(0), ship_ids.unsqueeze(0))
                ai_actions_probs = torch.sigmoid(ai_actions_logits[0])  # Remove batch dim
                ai_actions = torch.bernoulli(ai_actions_probs)
                
                # Apply AI actions to AI-controlled ships
                for i, ship_idx in enumerate(ai_ships):
                    if i < ai_actions.shape[0]:
                        actions[ship_idx] = ai_actions[i].bool()
        else:
            # Random actions until AI is ready
            for ship_idx in ai_ships:
                actions[ship_idx] = torch.randint(0, 2, (len(Actions),), dtype=torch.bool)
        
        # Step environment
        observation, rewards, done = env.step(actions)
        
        assert observation is not None
        assert rewards is not None
        
        if done:
            env.reset()
            history.reset()
            
    print("Main pipeline with transformer AI integration successful")