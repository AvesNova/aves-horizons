"""
Aves Horizons Training Module

Modular training system using StableBaselines3 PPO with self-play.
"""

from .config import TrainingConfig
from .environment import DeathmatchSelfPlayEnv
from .policy import TransformerActorCriticPolicy, create_policy_class
from .selfplay import OpponentPool, SelfPlayCallback
from .trainer import AvesHorizonsTrainer, create_trainer_from_config_file
from .callbacks import (
    SelfPlayManagementCallback, 
    TrainingMetricsCallback, 
    PerformanceMonitoringCallback
)

__all__ = [
    "TrainingConfig",
    "DeathmatchSelfPlayEnv", 
    "TransformerActorCriticPolicy",
    "create_policy_class",
    "OpponentPool",
    "SelfPlayCallback",
    "AvesHorizonsTrainer",
    "create_trainer_from_config_file",
    "SelfPlayManagementCallback",
    "TrainingMetricsCallback",
    "PerformanceMonitoringCallback"
]