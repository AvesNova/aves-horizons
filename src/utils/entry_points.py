"""
Unified entry point system for Aves Horizons.

This module provides a common interface for all entry points (main.py, training scripts, tests)
to reduce duplicate code and improve consistency.
"""

import argparse
import sys
from typing import Dict, Any, Optional, Callable
from pathlib import Path

from utils.config import ModelConfig, Actions


class EntryPointConfig:
    """Configuration manager for entry points."""
    
    def __init__(self):
        self.config = {}
        self._parsers = {}
    
    def create_base_parser(self, description: str) -> argparse.ArgumentParser:
        """Create a base argument parser with common options."""
        parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Game configuration
        parser.add_argument('--game-mode', choices=['standard', 'deathmatch'], 
                          default='standard', help='Game mode to play')
        parser.add_argument('--n-ships', type=int, default=ModelConfig.DEFAULT_N_SHIPS,
                          help='Total number of ships (for standard mode)')
        parser.add_argument('--ships-per-team', type=int, default=None,
                          help='Ships per team (for deathmatch mode)')
        parser.add_argument('--n-obstacles', type=int, default=ModelConfig.DEFAULT_N_OBSTACLES,
                          help='Number of obstacles in the environment')
        parser.add_argument('--world-size', type=float, nargs=2, 
                          default=ModelConfig.DEFAULT_WORLD_SIZE,
                          help='World dimensions (width height)')
        
        # Model configuration
        parser.add_argument('--sequence-length', type=int, default=ModelConfig.DEFAULT_SEQUENCE_LENGTH,
                          help='Sequence length for temporal models')
        parser.add_argument('--controlled-team-size', type=int, default=ModelConfig.DEFAULT_CONTROLLED_TEAM_SIZE,
                          help='Number of ships controlled by the agent')
        
        # Physics configuration
        parser.add_argument('--disable-continuous-collision', action='store_true',
                          help='Disable continuous collision detection')
        
        # Training configuration (optional)
        parser.add_argument('--batch-size', type=int, default=ModelConfig.DEFAULT_BATCH_SIZE,
                          help='Training batch size')
        parser.add_argument('--max-episode-steps', type=int, default=ModelConfig.DEFAULT_MAX_EPISODE_STEPS,
                          help='Maximum steps per episode')
        
        # Output configuration
        parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
        parser.add_argument('--output-dir', type=Path, help='Output directory for results')
        
        return parser
    
    def add_training_args(self, parser: argparse.ArgumentParser):
        """Add training-specific arguments to parser."""
        training_group = parser.add_argument_group('Training Options')
        training_group.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
        training_group.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
        training_group.add_argument('--save-frequency', type=int, default=100, help='Model save frequency')
        training_group.add_argument('--eval-frequency', type=int, default=50, help='Evaluation frequency')
        training_group.add_argument('--model-dir', type=Path, help='Directory to save/load models')
        
        # Transformer-specific
        training_group.add_argument('--d-model', type=int, default=ModelConfig.DEFAULT_D_MODEL,
                                  help='Transformer model dimension')
        training_group.add_argument('--n-head', type=int, default=ModelConfig.DEFAULT_N_HEAD,
                                  help='Number of attention heads')
        training_group.add_argument('--num-layers', type=int, default=ModelConfig.DEFAULT_NUM_LAYERS,
                                  help='Number of transformer layers')
    
    def add_rendering_args(self, parser: argparse.ArgumentParser):
        """Add rendering-specific arguments to parser."""
        render_group = parser.add_argument_group('Rendering Options')
        render_group.add_argument('--controlled-ship', type=int, default=0,
                                help='Which ship to control with keyboard (0-indexed)')
        render_group.add_argument('--render-fps', type=int, default=50, help='Rendering FPS')
        render_group.add_argument('--no-render', action='store_true', help='Disable rendering')
    
    def validate_args(self, args) -> Dict[str, Any]:
        """Validate and process parsed arguments."""
        config = vars(args).copy()
        
        # Validate game mode configuration
        if args.game_mode == 'deathmatch':
            if args.ships_per_team is None:
                config['ships_per_team'] = args.n_ships // ModelConfig.DEFAULT_N_TEAMS
            total_ships = ModelConfig.DEFAULT_N_TEAMS * config['ships_per_team']
        else:
            total_ships = args.n_ships
            
        # Validate controlled ship
        if hasattr(args, 'controlled_ship') and args.controlled_ship >= total_ships:
            raise ValueError(f"controlled-ship ({args.controlled_ship}) must be less than total ships ({total_ships})")
        
        # Convert world size to tuple
        config['world_size'] = tuple(args.world_size)
        
        # Set derived parameters
        config['total_ships'] = total_ships
        config['use_continuous_collision'] = not args.disable_continuous_collision
        
        return config


def setup_environment_from_config(config: Dict[str, Any]):
    """Create environment based on configuration."""
    from core.environment import Environment
    from game_modes.deathmatch import create_deathmatch_game
    
    if config['game_mode'] == 'deathmatch':
        env = create_deathmatch_game(
            n_teams=ModelConfig.DEFAULT_N_TEAMS,
            ships_per_team=config['ships_per_team'],
            world_size=config['world_size'],
            use_continuous_collision=config['use_continuous_collision']
        )
    else:  # 'standard' or default
        env = Environment(
            n_ships=config['n_ships'],
            n_obstacles=config['n_obstacles'],
            world_size=config['world_size'],
            use_continuous_collision=config['use_continuous_collision']
        )
    
    return env


def setup_transformer_env_from_config(config: Dict[str, Any]):
    """Create ShipTransformerEnv based on configuration."""
    from gym_env.ship_transformer_env import ShipTransformerEnv
    
    return ShipTransformerEnv(
        n_ships=config['n_ships'],
        n_obstacles=config['n_obstacles'],
        controlled_team_size=config['controlled_team_size'],
        sequence_length=config['sequence_length'],
        world_size=config['world_size'],
        normalize_coordinates=True,
        opponent_policy="random"
    )


def print_config(config: Dict[str, Any], title: str = "Configuration"):
    """Pretty print configuration."""
    print(f"\n{title}:")
    print("=" * len(title))
    for key, value in sorted(config.items()):
        if not key.startswith('_'):  # Skip private keys
            print(f"  {key}: {value}")
    print()


def handle_common_errors(func: Callable) -> Callable:
    """Decorator to handle common errors across entry points."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}")
            if '--verbose' in sys.argv:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    return wrapper


# Global entry point manager instance
entry_point_manager = EntryPointConfig()