"""
Training configuration for Aves Horizons.

Centralized configuration system for all training parameters.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
from pathlib import Path

from utils.config import ModelConfig


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    
    # ShipNN Model parameters
    hidden_dim: int = 128  # ShipNN hidden dimension
    encoder_layers: int = 2  # Number of encoder layers
    transformer_layers: int = 3  # Number of transformer layers
    decoder_layers: int = 2  # Number of decoder layers
    n_heads: int = 4  # Number of attention heads
    dim_feedforward: int = 256  # Transformer feedforward dimension
    
    # Legacy parameters for compatibility (mapped to new ones)
    d_model: int = None  # Will be set to hidden_dim in post_init
    n_head: int = None   # Will be set to n_heads in post_init
    num_layers: int = None  # Will be set to transformer_layers in post_init
    
    # Game parameters
    n_teams: int = ModelConfig.DEFAULT_N_TEAMS
    ships_per_team: int = ModelConfig.DEFAULT_CONTROLLED_TEAM_SIZE
    sequence_length: int = ModelConfig.DEFAULT_SEQUENCE_LENGTH
    world_size: Tuple[float, float] = ModelConfig.DEFAULT_WORLD_SIZE
    
    # PPO parameters
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_steps: int = 2048  # Steps per environment per update
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float = None  # If None, uses same as clip_range
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None
    
    # Self-play parameters
    selfplay_update_freq: int = 50000  # Steps between adding models to opponent pool
    max_opponent_pool_size: int = 10
    opponent_selection_probs: Dict[str, float] = None  # Will be set in post_init
    
    # Environment parameters
    max_episode_steps: int = ModelConfig.DEFAULT_MAX_EPISODE_STEPS
    
    # Training control
    total_timesteps: int = 1000000
    eval_freq: int = 10000
    eval_episodes: int = 10
    save_freq: int = 20000
    log_interval: int = 1
    
    # Directories
    model_dir: Path = Path("./models")
    log_dir: Path = Path("./logs")
    tensorboard_log: Path = Path("./tensorboard_logs")
    
    # Device
    device: str = "auto"  # "auto", "cpu", "cuda"
    
    def __post_init__(self):
        # Handle legacy parameter compatibility
        if self.d_model is None:
            self.d_model = self.hidden_dim
        if self.n_head is None:
            self.n_head = self.n_heads
        if self.num_layers is None:
            self.num_layers = self.transformer_layers
        
        if self.opponent_selection_probs is None:
            self.opponent_selection_probs = {
                "random": 0.1,
                "heuristic": 0.2,
                "selfplay": 0.7
            }
        
        # Ensure directories exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_log.mkdir(parents=True, exist_ok=True)
        
        # If clip_range_vf is None, use same as clip_range
        if self.clip_range_vf is None:
            self.clip_range_vf = self.clip_range
    
    @property
    def total_ships(self) -> int:
        """Total number of ships in the game."""
        return self.n_teams * self.ships_per_team
    
    @property
    def controlled_ships(self) -> int:
        """Number of controlled ships (assumes we control team 0)."""
        return self.ships_per_team