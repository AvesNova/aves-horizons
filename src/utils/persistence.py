"""
Model Checkpointing and State Persistence for ShipTransformer.

This module provides utilities for saving and loading model weights,
training state, and temporal history for training continuity.
"""

import torch
import pickle
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import asdict
import numpy as np
from datetime import datetime

from models.ship_transformer import ShipTransformer, ShipTransformerMVP
from models.state_history import StateHistory, BatchedStateHistory
from models.token_encoder import ShipTokenEncoder


class ModelCheckpoint:
    """
    Comprehensive model checkpoint that includes all training state.
    
    Saves:
    - Model weights and architecture parameters
    - Training metadata (epoch, step, etc.)
    - Optimizer state
    - Learning rate scheduler state
    - Random states for reproducibility
    - Custom training configuration
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        step: int = 0,
        best_reward: float = -float('inf'),
        training_config: Optional[Dict[str, Any]] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = epoch
        self.step = step
        self.best_reward = best_reward
        self.training_config = training_config or {}
        
        # Capture random states for reproducibility
        self.random_states = {
            'python': np.random.get_state(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
        }
        
        if torch.cuda.is_available():
            self.random_states['torch_cuda'] = torch.cuda.get_rng_state()
    
    def save(self, filepath: str):
        """Save checkpoint to file."""
        checkpoint_data = {
            'model_state_dict': self.model.state_dict(),
            'model_class': self.model.__class__.__name__,
            'model_config': self._get_model_config(),
            'epoch': self.epoch,
            'step': self.step,
            'best_reward': self.best_reward,
            'training_config': self.training_config,
            'random_states': self.random_states,
            'timestamp': datetime.now().isoformat(),
        }
        
        if self.optimizer is not None:
            checkpoint_data['optimizer_state_dict'] = self.optimizer.state_dict()
            checkpoint_data['optimizer_class'] = self.optimizer.__class__.__name__
        
        if self.scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = self.scheduler.state_dict()
            checkpoint_data['scheduler_class'] = self.scheduler.__class__.__name__
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint_data, filepath)
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Extract model configuration for reconstruction."""
        if hasattr(self.model, '__dict__'):
            config = {}
            for key, value in self.model.__dict__.items():
                if isinstance(value, (int, float, str, bool, list, tuple)):
                    config[key] = value
            return config
        return {}
    
    @classmethod
    def load(
        cls, 
        filepath: str, 
        model_class: Optional[type] = None,
        map_location: Optional[str] = None
    ) -> 'ModelCheckpoint':
        """
        Load checkpoint from file.
        
        Args:
            filepath: Path to checkpoint file
            model_class: Model class to instantiate (auto-detected if None)
            map_location: Device to load to
            
        Returns:
            Loaded ModelCheckpoint instance
        """
        checkpoint_data = torch.load(filepath, map_location=map_location)
        
        # Determine model class
        if model_class is None:
            model_class_name = checkpoint_data.get('model_class', 'ShipTransformerMVP')
            if model_class_name == 'ShipTransformer':
                model_class = ShipTransformer
            elif model_class_name == 'ShipTransformerMVP':
                model_class = ShipTransformerMVP
            else:
                raise ValueError(f"Unknown model class: {model_class_name}")
        
        # Create model instance
        model_config = checkpoint_data.get('model_config', {})
        model = model_class(**model_config)
        model.load_state_dict(checkpoint_data['model_state_dict'])
        
        # Create checkpoint instance
        checkpoint = cls(
            model=model,
            epoch=checkpoint_data.get('epoch', 0),
            step=checkpoint_data.get('step', 0),
            best_reward=checkpoint_data.get('best_reward', -float('inf')),
            training_config=checkpoint_data.get('training_config', {})
        )
        
        # Restore random states
        if 'random_states' in checkpoint_data:
            checkpoint.random_states = checkpoint_data['random_states']
        
        return checkpoint
    
    def restore_random_states(self):
        """Restore random states for reproducibility."""
        if hasattr(self, 'random_states'):
            if 'python' in self.random_states:
                np.random.set_state(self.random_states['python'])
            if 'numpy' in self.random_states:
                np.random.set_state(self.random_states['numpy'])
            if 'torch' in self.random_states:
                torch.set_rng_state(self.random_states['torch'])
            if 'torch_cuda' in self.random_states and torch.cuda.is_available():
                torch.cuda.set_rng_state(self.random_states['torch_cuda'])


class StateHistoryPersistence:
    """
    Persistence utilities for StateHistory objects.
    
    Allows saving and loading state history for training continuity,
    evaluation replay, and debugging purposes.
    """
    
    @staticmethod
    def save_state_history(state_history: StateHistory, filepath: str):
        """Save StateHistory to file."""
        # Convert deque to list for serialization
        buffer_list = list(state_history.state_buffer)
        
        # Convert tensors to numpy for serialization
        serializable_buffer = []
        for state in buffer_list:
            serializable_state = {}
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    serializable_state[key] = value.cpu().numpy()
                else:
                    serializable_state[key] = value
            serializable_buffer.append(serializable_state)
        
        data = {
            'sequence_length': state_history.sequence_length,
            'max_ships': state_history.max_ships,
            'world_size': state_history.world_size,
            'normalize_coordinates': state_history.normalize_coordinates,
            'current_timestep': state_history.current_timestep,
            'active_ships': list(state_history.active_ships),
            'buffer': serializable_buffer,
            'timestamp': datetime.now().isoformat()
        }
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @staticmethod
    def load_state_history(filepath: str) -> StateHistory:
        """Load StateHistory from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Create new StateHistory instance
        state_history = StateHistory(
            sequence_length=data['sequence_length'],
            max_ships=data['max_ships'],
            world_size=tuple(data['world_size']),
            normalize_coordinates=data['normalize_coordinates']
        )
        
        # Restore state
        state_history.current_timestep = data['current_timestep']
        state_history.active_ships = set(data['active_ships'])
        
        # Restore buffer
        state_history.state_buffer.clear()
        for serializable_state in data['buffer']:
            state = {}
            for key, value in serializable_state.items():
                if isinstance(value, np.ndarray):
                    if value.dtype == np.complex64 or value.dtype == np.complex128:
                        state[key] = torch.from_numpy(value)
                    elif 'bool' in str(value.dtype):
                        state[key] = torch.from_numpy(value).bool()
                    else:
                        state[key] = torch.from_numpy(value).float()
                else:
                    state[key] = value
            state_history.state_buffer.append(state)
        
        return state_history
    
    @staticmethod
    def save_batched_state_history(batched_history: BatchedStateHistory, filepath: str):
        """Save BatchedStateHistory to file."""
        data = {
            'batch_size': batched_history.batch_size,
            'histories': []
        }
        
        # Save each individual history
        for i, history in enumerate(batched_history.histories):
            # Convert to serializable format (same as single history)
            buffer_list = list(history.state_buffer)
            serializable_buffer = []
            for state in buffer_list:
                serializable_state = {}
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        serializable_state[key] = value.cpu().numpy()
                    else:
                        serializable_state[key] = value
                serializable_buffer.append(serializable_state)
            
            history_data = {
                'sequence_length': history.sequence_length,
                'max_ships': history.max_ships,
                'world_size': history.world_size,
                'normalize_coordinates': history.normalize_coordinates,
                'current_timestep': history.current_timestep,
                'active_ships': list(history.active_ships),
                'buffer': serializable_buffer,
            }
            data['histories'].append(history_data)
        
        data['timestamp'] = datetime.now().isoformat()
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @staticmethod
    def load_batched_state_history(filepath: str) -> BatchedStateHistory:
        """Load BatchedStateHistory from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Get configuration from first history
        first_history = data['histories'][0]
        
        # Create new BatchedStateHistory instance
        batched_history = BatchedStateHistory(
            batch_size=data['batch_size'],
            sequence_length=first_history['sequence_length'],
            max_ships=first_history['max_ships'],
            world_size=tuple(first_history['world_size']),
            normalize_coordinates=first_history['normalize_coordinates']
        )
        
        # Restore each individual history
        for i, history_data in enumerate(data['histories']):
            history = batched_history.histories[i]
            
            # Restore state
            history.current_timestep = history_data['current_timestep']
            history.active_ships = set(history_data['active_ships'])
            
            # Restore buffer
            history.state_buffer.clear()
            for serializable_state in history_data['buffer']:
                state = {}
                for key, value in serializable_state.items():
                    if isinstance(value, np.ndarray):
                        if value.dtype == np.complex64 or value.dtype == np.complex128:
                            state[key] = torch.from_numpy(value)
                        elif 'bool' in str(value.dtype):
                            state[key] = torch.from_numpy(value).bool()
                        else:
                            state[key] = torch.from_numpy(value).float()
                    else:
                        state[key] = value
                history.state_buffer.append(state)
        
        return batched_history


class TrainingSession:
    """
    Manages a complete training session with automatic checkpointing.
    
    Features:
    - Automatic checkpoint saving at intervals
    - Best model tracking
    - Resume from checkpoint capability
    - Training statistics logging
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        save_dir: str,
        save_freq: int = 1000,
        keep_n_checkpoints: int = 5,
        auto_save_best: bool = True
    ):
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_freq = save_freq
        self.keep_n_checkpoints = keep_n_checkpoints
        self.auto_save_best = auto_save_best
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_reward = -float('inf')
        self.training_stats = []
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize checkpoint list
        self.checkpoint_files = []
    
    def save_checkpoint(
        self,
        epoch: int,
        step: int,
        reward: Optional[float] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        training_config: Optional[Dict[str, Any]] = None,
        force_save: bool = False
    ):
        """Save a checkpoint."""
        # Update training state
        self.current_epoch = epoch
        self.current_step = step
        
        # Check if we should save
        should_save = force_save or (step % self.save_freq == 0)
        is_best = reward is not None and reward > self.best_reward
        
        if reward is not None:
            self.training_stats.append({
                'epoch': epoch,
                'step': step,
                'reward': reward,
                'timestamp': datetime.now().isoformat()
            })
        
        if should_save or (is_best and self.auto_save_best):
            # Create checkpoint
            checkpoint = ModelCheckpoint(
                model=self.model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=step,
                best_reward=self.best_reward,
                training_config=training_config
            )
            
            # Determine filename
            if is_best and self.auto_save_best:
                filename = f"best_model_step_{step}.pt"
                self.best_reward = reward
            else:
                filename = f"checkpoint_step_{step}.pt"
            
            filepath = self.save_dir / filename
            checkpoint.save(str(filepath))
            
            # Track checkpoint files
            if not is_best:
                self.checkpoint_files.append(filepath)
                self._cleanup_old_checkpoints()
            
            print(f"Saved checkpoint: {filename}")
            if is_best:
                print(f"New best reward: {reward:.4f}")
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoint files to save disk space."""
        if len(self.checkpoint_files) > self.keep_n_checkpoints:
            # Remove oldest checkpoints
            files_to_remove = self.checkpoint_files[:-self.keep_n_checkpoints]
            for filepath in files_to_remove:
                if filepath.exists():
                    filepath.unlink()
            self.checkpoint_files = self.checkpoint_files[-self.keep_n_checkpoints:]
    
    def load_latest_checkpoint(self) -> Optional[ModelCheckpoint]:
        """Load the most recent checkpoint."""
        checkpoint_files = list(self.save_dir.glob("checkpoint_step_*.pt"))
        if not checkpoint_files:
            return None
        
        # Sort by step number (extracted from filename)
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        latest_file = checkpoint_files[-1]
        
        checkpoint = ModelCheckpoint.load(str(latest_file))
        
        # Update training state
        self.current_epoch = checkpoint.epoch
        self.current_step = checkpoint.step
        self.best_reward = checkpoint.best_reward
        
        print(f"Loaded checkpoint from step {checkpoint.step}")
        return checkpoint
    
    def load_best_checkpoint(self) -> Optional[ModelCheckpoint]:
        """Load the best performing checkpoint."""
        best_files = list(self.save_dir.glob("best_model_step_*.pt"))
        if not best_files:
            return None
        
        # Sort by step number and take the latest best
        best_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        best_file = best_files[-1]
        
        checkpoint = ModelCheckpoint.load(str(best_file))
        print(f"Loaded best checkpoint from step {checkpoint.step} (reward: {checkpoint.best_reward:.4f})")
        return checkpoint
    
    def save_training_stats(self):
        """Save training statistics to JSON."""
        stats_file = self.save_dir / "training_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
    
    def can_resume(self) -> bool:
        """Check if training can be resumed from a checkpoint."""
        return len(list(self.save_dir.glob("checkpoint_step_*.pt"))) > 0


def create_training_session(
    model: torch.nn.Module,
    save_dir: str,
    **kwargs
) -> TrainingSession:
    """
    Create a new training session or resume from existing checkpoints.
    
    Args:
        model: Model to train
        save_dir: Directory to save checkpoints
        **kwargs: Additional arguments for TrainingSession
        
    Returns:
        TrainingSession instance
    """
    session = TrainingSession(model, save_dir, **kwargs)
    
    # Try to resume from checkpoint
    if session.can_resume():
        print(f"Found existing checkpoints in {save_dir}")
        response = input("Resume from latest checkpoint? (y/n): ").lower().strip()
        if response == 'y':
            checkpoint = session.load_latest_checkpoint()
            if checkpoint:
                # Update model weights
                model.load_state_dict(checkpoint.model.state_dict())
                checkpoint.restore_random_states()
                print("Resumed training from checkpoint")
            else:
                print("Failed to load checkpoint, starting fresh")
    
    return session