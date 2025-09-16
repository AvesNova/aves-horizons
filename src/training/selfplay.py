"""
Self-play opponent pool management for Aves Horizons training.

Manages a pool of opponent models for diverse self-play training.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional
import torch


class OpponentPool:
    """Manages opponent models for self-play training."""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.models: List[Path] = []
        self.model_metadata: Dict[Path, Dict] = {}
        
    def add_model(self, model_path: Path, metadata: Dict = None):
        """Add a model to the pool."""
        if model_path not in self.models:
            self.models.append(model_path)
            self.model_metadata[model_path] = metadata or {}
            
            print(f"Added model to opponent pool: {model_path.name}")
            
            # Remove oldest if at capacity
            if len(self.models) > self.max_size:
                old_model = self.models.pop(0)
                del self.model_metadata[old_model]
                # Optionally delete old model file to save space
                try:
                    old_model.unlink(missing_ok=True)
                    print(f"Removed old model from pool: {old_model.name}")
                except Exception as e:
                    print(f"Warning: Could not delete old model {old_model}: {e}")
    
    def sample_opponent(self, exclude_latest: bool = True) -> Optional[Path]:
        """Sample an opponent model from the pool."""
        if not self.models:
            return None
        
        # Optionally exclude the latest model to avoid self-play against identical models
        candidates = self.models[:-1] if exclude_latest and len(self.models) > 1 else self.models
        return random.choice(candidates) if candidates else None
    
    def get_latest(self) -> Optional[Path]:
        """Get the latest model in the pool."""
        return self.models[-1] if self.models else None
    
    def get_model_info(self, model_path: Path) -> Dict:
        """Get metadata for a specific model."""
        return self.model_metadata.get(model_path, {})
    
    def list_models(self) -> List[Dict]:
        """List all models with their metadata."""
        return [
            {
                "path": model_path,
                "metadata": self.model_metadata[model_path]
            }
            for model_path in self.models
        ]
    
    def clear(self):
        """Clear the opponent pool."""
        for model_path in self.models:
            try:
                model_path.unlink(missing_ok=True)
            except Exception as e:
                print(f"Warning: Could not delete model {model_path}: {e}")
        
        self.models.clear()
        self.model_metadata.clear()
        print("Cleared opponent pool")
    
    def __len__(self) -> int:
        """Get number of models in pool."""
        return len(self.models)
    
    def __str__(self) -> str:
        """String representation of the pool."""
        return f"OpponentPool(size={len(self.models)}/{self.max_size}, models={[p.name for p in self.models]})"


class SelfPlayCallback:
    """Callback to manage self-play model pool during training."""
    
    def __init__(
        self,
        opponent_pool: OpponentPool,
        save_freq: int = 50000,
        model_save_path: Path = Path("./models/selfplay/"),
        verbose: bool = True
    ):
        self.opponent_pool = opponent_pool
        self.save_freq = save_freq
        self.model_save_path = model_save_path
        self.verbose = verbose
        
        # Ensure save directory exists
        self.model_save_path.mkdir(parents=True, exist_ok=True)
    
    def should_save_model(self, timestep: int) -> bool:
        """Check if we should save the current model to the pool."""
        return timestep > 0 and timestep % self.save_freq == 0
    
    def save_model_to_pool(self, model, timestep: int, additional_metadata: Dict = None):
        """Save current model to the opponent pool."""
        model_name = f"selfplay_model_step_{timestep}"
        model_path = self.model_save_path / f"{model_name}.zip"
        
        # Save the model
        model.save(str(model_path))
        
        # Prepare metadata
        metadata = {
            "timestep": timestep,
            "model_name": model_name,
            "save_path": str(model_path)
        }
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # Add to opponent pool
        self.opponent_pool.add_model(model_path, metadata)
        
        if self.verbose:
            print(f"Saved model to opponent pool at timestep {timestep}: {model_name}")
        
        return model_path