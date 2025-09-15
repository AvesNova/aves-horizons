"""
State History Tracking System for ShipTransformer.

Maintains a rolling window of ship states over time to create temporal sequences
for the transformer model. Efficiently stores and manages state transitions.
"""

import torch
from typing import Dict, List, Tuple, Optional
from collections import deque
import numpy as np

from core.ships import Ships


class StateHistory:
    """
    Maintains a rolling window of ship states for temporal sequence generation.
    
    Features:
    - Efficient circular buffer for state storage
    - Automatic timestep tracking
    - Token sequence generation for transformer
    - Support for variable number of ships
    - Batch processing capabilities
    """
    
    def __init__(
        self,
        sequence_length: int = 6,
        max_ships: int = 8,
        world_size: Tuple[float, float] = (1200.0, 800.0),
        normalize_coordinates: bool = True
    ):
        self.sequence_length = sequence_length
        self.max_ships = max_ships
        self.world_size = world_size
        self.normalize_coordinates = normalize_coordinates
        
        # Rolling buffer for storing ship states
        # Each element is a dict containing state information for all ships at one timestep
        self.state_buffer: deque = deque(maxlen=sequence_length)
        
        # Track current timestep for temporal encoding
        self.current_timestep = 0
        
        # Track which ships are active/alive
        self.active_ships = set(range(max_ships))
        
        # Track whether we have any real states (not just initialization)
        self.has_real_states = False
        
        # Initialize with empty states
        self._initialize_buffer()
    
    def _initialize_buffer(self):
        """Initialize buffer with zero states."""
        for _ in range(self.sequence_length):
            empty_state = self._create_empty_state()
            self.state_buffer.append(empty_state)
    
    def _create_empty_state(self) -> Dict[str, torch.Tensor]:
        """Create an empty state dict with zeros."""
        return {
            'position': torch.zeros(self.max_ships, dtype=torch.complex64),
            'velocity': torch.zeros(self.max_ships, dtype=torch.complex64),
            'attitude': torch.ones(self.max_ships, dtype=torch.complex64),  # Default facing right
            'turn_offset': torch.zeros(self.max_ships),
            'boost_norm': torch.ones(self.max_ships),  # Start with full boost
            'health_norm': torch.ones(self.max_ships),  # Start with full health
            'ammo_norm': torch.ones(self.max_ships),    # Start with full ammo
            'team_id': torch.zeros(self.max_ships, dtype=torch.long),  # Default team 0
            'is_shooting': torch.zeros(self.max_ships, dtype=torch.bool),
            'timestep': self.current_timestep,
            'active_mask': torch.ones(self.max_ships, dtype=torch.bool)
        }
    
    def add_state(self, ships: Ships, actions: Optional[torch.Tensor] = None):
        """
        Add a new ship state to the history buffer.
        
        Args:
            ships: Current Ships dataclass instance
            actions: Optional actions tensor [n_ships, 6] for tracking shooting state
        """
        # Extract state information from ships
        state = self._extract_ship_state(ships, actions)
        
        # Add to rolling buffer (automatically removes oldest if at capacity)
        self.state_buffer.append(state)
        
        # Mark that we have real states now
        self.has_real_states = True
        
        # Increment timestep counter
        self.current_timestep += 1
        
        # Update active ships based on health
        self._update_active_ships(ships)
    
    def _extract_ship_state(
        self, 
        ships: Ships, 
        actions: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Extract relevant state information from Ships instance."""
        # Normalize coordinates if requested
        if self.normalize_coordinates:
            pos_x = ships.position.real / self.world_size[0]
            pos_y = ships.position.imag / self.world_size[1]
            position = torch.complex(pos_x, pos_y)
            
            # Normalize velocities by a reasonable maximum speed
            max_speed = 300.0  # Reasonable max speed for normalization
            velocity = ships.velocity / max_speed
        else:
            position = ships.position.clone()
            velocity = ships.velocity.clone()
        
        # Normalize other values
        boost_norm = ships.boost / ships.max_boost
        health_norm = ships.health / ships.max_health
        ammo_norm = ships.ammo_count / ships.max_ammo
        
        # Extract team IDs
        team_ids = ships.team_id if hasattr(ships, 'team_id') else torch.zeros(self.max_ships, dtype=torch.long)
        
        # Determine shooting state from actions if provided
        if actions is not None:
            # Assuming action index 5 is shoot (from config.py)
            is_shooting = actions[:, 5] if actions.shape[0] >= self.max_ships else torch.zeros(self.max_ships, dtype=torch.bool)
        else:
            is_shooting = torch.zeros(self.max_ships, dtype=torch.bool)
        
        # Create active mask (ships with health > 0)
        active_mask = ships.health > 0
        
        return {
            'position': position[:self.max_ships],
            'velocity': velocity[:self.max_ships],
            'attitude': ships.attitude[:self.max_ships].clone(),
            'turn_offset': ships.turn_offset[:self.max_ships].clone(),
            'boost_norm': boost_norm[:self.max_ships],
            'health_norm': health_norm[:self.max_ships],
            'ammo_norm': ammo_norm[:self.max_ships],
            'team_id': team_ids[:self.max_ships],
            'is_shooting': is_shooting[:self.max_ships],
            'timestep': self.current_timestep,
            'active_mask': active_mask[:self.max_ships]
        }
    
    def _update_active_ships(self, ships: Ships):
        """Update the set of active ships based on health."""
        self.active_ships = set()
        for i in range(min(len(ships.health), self.max_ships)):
            if ships.health[i] > 0:
                self.active_ships.add(i)
    
    def get_token_sequence(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate token sequence for transformer input.
        
        Returns:
            tokens: [seq_len, 13] Token features
            ship_ids: [seq_len] Ship ID for each token
            
        Token sequence format (time-major):
        [ship0_t-5, ship1_t-5, ..., ship0_t-4, ship1_t-4, ..., ship0_t-0, ship1_t-0]
        """
        tokens_list = []
        ship_ids_list = []
        
        # Process each timestep in chronological order
        states_list = list(self.state_buffer)
        
        for timestep_idx, state in enumerate(states_list):
            # Calculate timestep offset (negative for past, 0 for current)
            timestep_offset = timestep_idx - (self.sequence_length - 1)
            
            # Create tokens for all ships at this timestep, but only up to max_ships
            # and only for ships that exist in the state
            actual_ships = state['position'].shape[0]
            for ship_id in range(self.max_ships):
                if ship_id < actual_ships:
                    # Extract base token features (13-dimensional as per model spec)
                    token = self._create_base_token(state, ship_id, timestep_offset)
                else:
                    # Create a zero/empty token for non-existent ships
                    token = self._create_empty_token(timestep_offset)
                tokens_list.append(token)
                ship_ids_list.append(ship_id)
        
        # Convert to tensors
        tokens = torch.stack(tokens_list, dim=0)  # [seq_len, 13]
        ship_ids = torch.tensor(ship_ids_list, dtype=torch.long)  # [seq_len]
        
        return tokens, ship_ids
    
    def _create_base_token(
        self, 
        state: Dict[str, torch.Tensor], 
        ship_id: int, 
        timestep_offset: int
    ) -> torch.Tensor:
        """
        Create 13-dimensional base token for a single ship at one timestep.
        
        Base token format:
        [pos_x, pos_y, vel_x, vel_y, attitude_x, attitude_y, turn_offset,
         boost_norm, health_norm, ammo_norm, is_shooting, team_id, timestep_offset]
        """
        position = state['position'][ship_id]
        velocity = state['velocity'][ship_id]
        attitude = state['attitude'][ship_id]
        
        token = torch.tensor([
            position.real.item(),                    # pos_x
            position.imag.item(),                    # pos_y
            velocity.real.item(),                    # vel_x
            velocity.imag.item(),                    # vel_y
            attitude.real.item(),                    # attitude_x (cos θ)
            attitude.imag.item(),                    # attitude_y (sin θ)
            state['turn_offset'][ship_id].item(),    # turn_offset
            state['boost_norm'][ship_id].item(),     # boost_norm
            state['health_norm'][ship_id].item(),    # health_norm
            state['ammo_norm'][ship_id].item(),      # ammo_norm
            float(state['is_shooting'][ship_id]),    # is_shooting (0 or 1)
            float(state['team_id'][ship_id].item()), # team_id
            float(timestep_offset),                  # timestep_offset
        ], dtype=torch.float32)
        
        return token
    
    def _create_empty_token(self, timestep_offset: int) -> torch.Tensor:
        """
        Create an empty 13-dimensional token for non-existent ships.
        
        Base token format:
        [pos_x, pos_y, vel_x, vel_y, attitude_x, attitude_y, turn_offset,
         boost_norm, health_norm, ammo_norm, is_shooting, team_id, timestep_offset]
        """
        token = torch.tensor([
            0.0,                       # pos_x
            0.0,                       # pos_y
            0.0,                       # vel_x
            0.0,                       # vel_y
            1.0,                       # attitude_x (facing right)
            0.0,                       # attitude_y
            0.0,                       # turn_offset
            0.0,                       # boost_norm (empty)
            0.0,                       # health_norm (dead)
            0.0,                       # ammo_norm (empty)
            0.0,                       # is_shooting
            0.0,                       # team_id (default team 0)
            float(timestep_offset),    # timestep_offset
        ], dtype=torch.float32)
        
        return token
    
    def get_batch_sequences(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate batched token sequences for training.
        
        Args:
            batch_size: Number of sequences in batch
            
        Returns:
            tokens: [batch_size, seq_len, base_token_dim] Batched token sequences
            ship_ids: [batch_size, seq_len] Batched ship IDs
        """
        tokens, ship_ids = self.get_token_sequence()
        
        # Repeat the sequence for batch processing
        # In practice, this would come from multiple different game states
        tokens_batch = tokens.unsqueeze(0).repeat(batch_size, 1, 1)
        ship_ids_batch = ship_ids.unsqueeze(0).repeat(batch_size, 1)
        
        return tokens_batch, ship_ids_batch
    
    def reset(self):
        """Reset the state history buffer."""
        self.state_buffer.clear()
        self.current_timestep = 0
        self.active_ships = set(range(self.max_ships))
        self.has_real_states = False
        self._initialize_buffer()
    
    def get_sequence_length(self) -> int:
        """Get the current sequence length."""
        return len(self.state_buffer)
    
    def is_ready(self) -> bool:
        """Check if buffer has enough history for sequence generation."""
        return (len(self.state_buffer) >= self.sequence_length and 
                self.has_real_states)
    
    def get_current_state(self) -> Dict[str, torch.Tensor]:
        """Get the most recent state from the buffer."""
        if len(self.state_buffer) > 0:
            return self.state_buffer[-1].copy()
        return self._create_empty_state()
    
    def set_world_size(self, world_size: Tuple[float, float]):
        """Update world size for coordinate normalization."""
        self.world_size = world_size
    
    def get_active_ships(self) -> set:
        """Get the set of currently active ship IDs."""
        return self.active_ships.copy()


class BatchedStateHistory:
    """
    Manages multiple StateHistory instances for batch training.
    
    This class handles multiple game instances running in parallel,
    each with their own state history.
    """
    
    def __init__(
        self,
        batch_size: int,
        sequence_length: int = 6,
        max_ships: int = 8,
        world_size: Tuple[float, float] = (1200.0, 800.0),
        normalize_coordinates: bool = True
    ):
        self.batch_size = batch_size
        
        # Create individual state history for each batch element
        self.histories = [
            StateHistory(
                sequence_length=sequence_length,
                max_ships=max_ships,
                world_size=world_size,
                normalize_coordinates=normalize_coordinates
            )
            for _ in range(batch_size)
        ]
    
    def add_states(self, ships_batch: List[Ships], actions_batch: Optional[List[torch.Tensor]] = None):
        """
        Add states for all batch elements.
        
        Args:
            ships_batch: List of Ships instances for each batch element
            actions_batch: Optional list of action tensors
        """
        for i, ships in enumerate(ships_batch):
            actions = actions_batch[i] if actions_batch else None
            self.histories[i].add_state(ships, actions)
    
    def get_batch_sequences(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get batched token sequences from all histories.
        
        Returns:
            tokens: [batch_size, seq_len, base_token_dim] Batched sequences
            ship_ids: [batch_size, seq_len] Batched ship IDs
        """
        tokens_list = []
        ship_ids_list = []
        
        for history in self.histories:
            tokens, ship_ids = history.get_token_sequence()
            tokens_list.append(tokens)
            ship_ids_list.append(ship_ids)
        
        # Stack into batch tensors
        tokens_batch = torch.stack(tokens_list, dim=0)
        ship_ids_batch = torch.stack(ship_ids_list, dim=0)
        
        return tokens_batch, ship_ids_batch
    
    def reset_all(self):
        """Reset all state histories in the batch."""
        for history in self.histories:
            history.reset()
    
    def reset_single(self, batch_idx: int):
        """Reset a single state history in the batch."""
        if 0 <= batch_idx < self.batch_size:
            self.histories[batch_idx].reset()
    
    def are_ready(self) -> List[bool]:
        """Check which histories have enough data for sequence generation."""
        return [history.is_ready() for history in self.histories]
    
    def all_ready(self) -> bool:
        """Check if all histories are ready for sequence generation."""
        return all(history.is_ready() for history in self.histories)