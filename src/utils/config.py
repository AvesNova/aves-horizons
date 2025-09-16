from enum import IntEnum, auto
from typing import Tuple
import numpy as np


class Actions(IntEnum):
    """Action indices for ship control."""
    forward = 0
    backward = auto()
    left = auto()
    right = auto()
    sharp_turn = auto()
    shoot = auto()


class ModelConfig:
    """Central configuration for all model parameters and constants."""
    
    # Observation/Token dimensions
    TOKEN_DIM = 13  # Base token dimension (includes team_id)
    TOKEN_FEATURES = {
        'pos_x': 0,
        'pos_y': 1,
        'vel_x': 2,
        'vel_y': 3,
        'attitude_x': 4,  # cos(θ)
        'attitude_y': 5,  # sin(θ)
        'turn_offset': 6,
        'boost_norm': 7,
        'health_norm': 8,
        'ammo_norm': 9,
        'is_shooting': 10,
        'team_id': 11,
        'timestep_offset': 12
    }
    
    # Default game parameters
    DEFAULT_N_SHIPS = 8
    DEFAULT_CONTROLLED_TEAM_SIZE = 4
    DEFAULT_SEQUENCE_LENGTH = 6
    DEFAULT_WORLD_SIZE = (1200.0, 800.0)
    
    # Physics parameters
    DEFAULT_MAX_SPEED = 300.0
    DEFAULT_COLLISION_RADIUS = 10.0
    DEFAULT_MAX_BOOST = 100.0
    DEFAULT_MAX_HEALTH = 100.0
    DEFAULT_MAX_AMMO = 32.0
    
    # Transformer parameters
    DEFAULT_D_MODEL = 64
    DEFAULT_N_HEAD = 4
    DEFAULT_NUM_LAYERS = 3
    
    # Training parameters
    DEFAULT_MAX_EPISODE_STEPS = 2000
    DEFAULT_BATCH_SIZE = 32
    
    # Environment parameters
    DEFAULT_N_OBSTACLES = 0
    DEFAULT_N_TEAMS = 2
    
    @classmethod
    def get_observation_dim(cls, sequence_length: int = None, n_ships: int = None) -> int:
        """Calculate observation dimension for gym environments."""
        seq_len = sequence_length or cls.DEFAULT_SEQUENCE_LENGTH
        n_ships = n_ships or cls.DEFAULT_N_SHIPS
        return seq_len * n_ships * cls.TOKEN_DIM
    
    @classmethod
    def get_action_dim(cls, controlled_team_size: int = None) -> int:
        """Calculate action dimension for gym environments."""
        team_size = controlled_team_size or cls.DEFAULT_CONTROLLED_TEAM_SIZE
        return team_size * len(Actions)


class PhysicsConfig:
    """Physics simulation parameters."""
    
    # Thrust system defaults
    DEFAULT_THRUST = 10.0
    DEFAULT_FORWARD_BOOST = 8.0
    DEFAULT_BACKWARD_BOOST = 0.0
    DEFAULT_BASE_ENERGY_COST = -10.0
    DEFAULT_FORWARD_ENERGY_COST = 40.0
    DEFAULT_BACKWARD_ENERGY_COST = -20.0
    
    # Projectile system defaults
    DEFAULT_MAX_PROJECTILES = 16
    DEFAULT_PROJECTILE_SPEED = 500.0
    DEFAULT_PROJECTILE_DAMAGE = 20.0
    DEFAULT_PROJECTILE_LIFETIME = 1.0
    DEFAULT_FIRING_COOLDOWN = 0.04
    DEFAULT_AMMO_REGEN_RATE = 4.0
    DEFAULT_PROJECTILE_SPREAD = np.deg2rad(3.0)
    
    # Aerodynamic defaults
    DEFAULT_NO_TURN_DRAG = 8e-4
    DEFAULT_NORMAL_TURN_ANGLE = np.deg2rad(5.0)
    DEFAULT_NORMAL_TURN_LIFT_COEF = 15e-3
    DEFAULT_NORMAL_TURN_DRAG_COEF = 1e-3
    DEFAULT_SHARP_TURN_ANGLE = np.deg2rad(15.0)
    DEFAULT_SHARP_TURN_LIFT_COEF = 30e-3
    DEFAULT_SHARP_TURN_DRAG_COEF = 3e-3
