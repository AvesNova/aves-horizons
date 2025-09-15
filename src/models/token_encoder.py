"""
Ship Token Encoder for ShipTransformer.

Handles conversion between ship states and transformer tokens according to the
model specification in docs/model.md. Supports both individual and batch encoding.
"""

import torch
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

from core.ships import Ships


class ShipTokenEncoder:
    """
    Encodes ship states into transformer tokens following the model specification.
    
    Token Format (12-dimensional base tokens):
    [pos_x, pos_y, vel_x, vel_y, attitude_x, attitude_y, turn_offset,
     boost_norm, health_norm, ammo_norm, is_shooting, timestep_offset]
     
    Features:
    - Coordinate normalization
    - Temporal encoding
    - Ship identity handling
    - Batch processing support
    - Flexible input formats
    """
    
    def __init__(
        self,
        world_size: Tuple[float, float] = (1200.0, 800.0),
        max_speed: float = 300.0,
        normalize_coordinates: bool = True,
        max_ships: int = 8
    ):
        self.world_size = world_size
        self.max_speed = max_speed
        self.normalize_coordinates = normalize_coordinates
        self.max_ships = max_ships
    
    def encode_ships_to_tokens(
        self, 
        ships: Ships, 
        timestep_offset: float = 0.0,
        actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode a Ships instance to base tokens.
        
        Args:
            ships: Ships dataclass instance
            timestep_offset: Temporal offset for this state (0 for current, negative for past)
            actions: Optional actions tensor [n_ships, 6] for shooting state
            
        Returns:
            tokens: [n_ships, 12] Base token tensor
        """
        n_ships = min(ships.n_ships, self.max_ships)
        
        # Extract and normalize coordinates
        if self.normalize_coordinates:
            pos_x = ships.position.real[:n_ships] / self.world_size[0]
            pos_y = ships.position.imag[:n_ships] / self.world_size[1]
            vel_x = ships.velocity.real[:n_ships] / self.max_speed
            vel_y = ships.velocity.imag[:n_ships] / self.max_speed
        else:
            pos_x = ships.position.real[:n_ships]
            pos_y = ships.position.imag[:n_ships]
            vel_x = ships.velocity.real[:n_ships]
            vel_y = ships.velocity.imag[:n_ships]
        
        # Extract attitude components
        attitude_x = ships.attitude.real[:n_ships]  # cos(θ)
        attitude_y = ships.attitude.imag[:n_ships]  # sin(θ)
        
        # Extract normalized state values
        turn_offset = ships.turn_offset[:n_ships]
        boost_norm = ships.boost[:n_ships] / ships.max_boost[:n_ships]
        health_norm = ships.health[:n_ships] / ships.max_health[:n_ships]
        ammo_norm = ships.ammo_count[:n_ships] / ships.max_ammo[:n_ships]
        
        # Determine shooting state
        if actions is not None and actions.shape[0] >= n_ships:
            # Assuming action index 5 is shoot
            is_shooting = actions[:n_ships, 5].float()
        else:
            is_shooting = torch.zeros(n_ships)
        
        # Create timestep offset tensor
        timestep_offset_tensor = torch.full((n_ships,), timestep_offset)
        
        # Stack all components into tokens [n_ships, 12]
        tokens = torch.stack([
            pos_x, pos_y,                    # Position (2)
            vel_x, vel_y,                    # Velocity (2)
            attitude_x, attitude_y,          # Attitude (2)
            turn_offset,                     # Turn offset (1)
            boost_norm,                      # Boost normalized (1)
            health_norm,                     # Health normalized (1)
            ammo_norm,                       # Ammo normalized (1)
            is_shooting,                     # Shooting state (1)
            timestep_offset_tensor           # Temporal offset (1)
        ], dim=1)
        
        return tokens
    
    def encode_state_dict_to_tokens(
        self,
        state: Dict[str, torch.Tensor],
        ship_id: int,
        timestep_offset: float = 0.0
    ) -> torch.Tensor:
        """
        Encode a single ship's state from a state dictionary to a token.
        
        Args:
            state: State dictionary containing ship information
            ship_id: ID of the ship to encode
            timestep_offset: Temporal offset for this state
            
        Returns:
            token: [12] Single token tensor
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
            float(timestep_offset),                  # timestep_offset
        ], dtype=torch.float32)
        
        return token
    
    def create_temporal_sequence(
        self,
        ships_history: List[Ships],
        actions_history: Optional[List[torch.Tensor]] = None,
        sequence_length: int = 6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create a temporal sequence of tokens from a history of ship states.
        
        Args:
            ships_history: List of Ships instances over time (oldest first)
            actions_history: Optional list of action tensors corresponding to ships_history
            sequence_length: Number of timesteps to include
            
        Returns:
            tokens: [seq_len, 12] Token sequence
            ship_ids: [seq_len] Ship ID for each token
        """
        # Trim history to sequence length
        ships_history = ships_history[-sequence_length:]
        if actions_history:
            actions_history = actions_history[-sequence_length:]
        
        tokens_list = []
        ship_ids_list = []
        
        for timestep_idx, ships in enumerate(ships_history):
            # Calculate timestep offset (negative for past, 0 for current)
            timestep_offset = timestep_idx - (len(ships_history) - 1)
            
            # Get actions for this timestep
            actions = actions_history[timestep_idx] if actions_history else None
            
            # Encode all ships at this timestep
            ship_tokens = self.encode_ships_to_tokens(ships, timestep_offset, actions)
            
            # Add to sequence (time-major order: all ships at t, then all ships at t+1)
            for ship_id in range(min(ships.n_ships, self.max_ships)):
                tokens_list.append(ship_tokens[ship_id])
                ship_ids_list.append(ship_id)
        
        # Convert to tensors
        tokens = torch.stack(tokens_list, dim=0)  # [seq_len, 12]
        ship_ids = torch.tensor(ship_ids_list, dtype=torch.long)  # [seq_len]
        
        return tokens, ship_ids
    
    def create_batch_sequences(
        self,
        ships_batch_history: List[List[Ships]],
        actions_batch_history: Optional[List[List[torch.Tensor]]] = None,
        sequence_length: int = 6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create batched temporal sequences from multiple game histories.
        
        Args:
            ships_batch_history: List of ship histories, one per batch element
            actions_batch_history: Optional list of action histories
            sequence_length: Number of timesteps to include
            
        Returns:
            tokens: [batch_size, seq_len, 12] Batched token sequences
            ship_ids: [batch_size, seq_len] Batched ship IDs
        """
        batch_size = len(ships_batch_history)
        tokens_list = []
        ship_ids_list = []
        
        for batch_idx in range(batch_size):
            ships_history = ships_batch_history[batch_idx]
            actions_history = actions_batch_history[batch_idx] if actions_batch_history else None
            
            tokens, ship_ids = self.create_temporal_sequence(
                ships_history, actions_history, sequence_length
            )
            
            tokens_list.append(tokens)
            ship_ids_list.append(ship_ids)
        
        # Stack into batch tensors
        tokens_batch = torch.stack(tokens_list, dim=0)  # [batch_size, seq_len, 12]
        ship_ids_batch = torch.stack(ship_ids_list, dim=0)  # [batch_size, seq_len]
        
        return tokens_batch, ship_ids_batch
    
    def decode_tokens_to_state(
        self, 
        tokens: torch.Tensor,
        ship_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Decode tokens back to state representation (for debugging/analysis).
        
        Args:
            tokens: [seq_len, 12] Token tensor
            ship_ids: [seq_len] Ship ID for each token
            
        Returns:
            state: Dictionary containing decoded state information
        """
        seq_len = tokens.shape[0]
        
        # Initialize state tensors
        positions = torch.zeros(self.max_ships, dtype=torch.complex64)
        velocities = torch.zeros(self.max_ships, dtype=torch.complex64)
        attitudes = torch.zeros(self.max_ships, dtype=torch.complex64)
        turn_offsets = torch.zeros(self.max_ships)
        boost_norms = torch.zeros(self.max_ships)
        health_norms = torch.zeros(self.max_ships)
        ammo_norms = torch.zeros(self.max_ships)
        is_shooting = torch.zeros(self.max_ships, dtype=torch.bool)
        
        # Decode each token
        for i in range(seq_len):
            ship_id = ship_ids[i].item()
            token = tokens[i]
            
            if ship_id < self.max_ships:
                # Denormalize coordinates if they were normalized
                if self.normalize_coordinates:
                    pos_x = token[0] * self.world_size[0]
                    pos_y = token[1] * self.world_size[1]
                    vel_x = token[2] * self.max_speed
                    vel_y = token[3] * self.max_speed
                else:
                    pos_x = token[0]
                    pos_y = token[1]
                    vel_x = token[2]
                    vel_y = token[3]
                
                positions[ship_id] = torch.complex(pos_x, pos_y)
                velocities[ship_id] = torch.complex(vel_x, vel_y)
                attitudes[ship_id] = torch.complex(token[4], token[5])  # attitude_x, attitude_y
                turn_offsets[ship_id] = token[6]
                boost_norms[ship_id] = token[7]
                health_norms[ship_id] = token[8]
                ammo_norms[ship_id] = token[9]
                is_shooting[ship_id] = token[10] > 0.5
        
        return {
            'position': positions,
            'velocity': velocities,
            'attitude': attitudes,
            'turn_offset': turn_offsets,
            'boost_norm': boost_norms,
            'health_norm': health_norms,
            'ammo_norm': ammo_norms,
            'is_shooting': is_shooting,
        }
    
    def get_token_dim(self) -> int:
        """Get the dimensionality of base tokens."""
        return 12
    
    def set_world_size(self, world_size: Tuple[float, float]):
        """Update world size for coordinate normalization."""
        self.world_size = world_size
    
    def set_max_speed(self, max_speed: float):
        """Update maximum speed for velocity normalization."""
        self.max_speed = max_speed


class BatchTokenEncoder:
    """
    Batch-optimized version of ShipTokenEncoder for high-throughput training.
    
    Processes multiple game states simultaneously for better GPU utilization.
    """
    
    def __init__(
        self,
        world_size: Tuple[float, float] = (1200.0, 800.0),
        max_speed: float = 300.0,
        normalize_coordinates: bool = True,
        max_ships: int = 8,
        sequence_length: int = 6
    ):
        self.world_size = world_size
        self.max_speed = max_speed
        self.normalize_coordinates = normalize_coordinates
        self.max_ships = max_ships
        self.sequence_length = sequence_length
        
        # Create base encoder for individual operations
        self.base_encoder = ShipTokenEncoder(
            world_size=world_size,
            max_speed=max_speed,
            normalize_coordinates=normalize_coordinates,
            max_ships=max_ships
        )
    
    def encode_batch_ships(
        self,
        ships_batch: List[Ships],
        timestep_offsets: List[float],
        actions_batch: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of Ships instances to tokens.
        
        Args:
            ships_batch: List of Ships instances
            timestep_offsets: List of timestep offsets for each ships instance
            actions_batch: Optional list of action tensors
            
        Returns:
            tokens: [batch_size, max_ships, 12] Batched tokens
            ship_ids: [batch_size, max_ships] Ship IDs
        """
        batch_size = len(ships_batch)
        tokens_batch = torch.zeros(batch_size, self.max_ships, 12)
        ship_ids_batch = torch.zeros(batch_size, self.max_ships, dtype=torch.long)
        
        for i, ships in enumerate(ships_batch):
            timestep_offset = timestep_offsets[i]
            actions = actions_batch[i] if actions_batch else None
            
            # Encode this ships instance
            tokens = self.base_encoder.encode_ships_to_tokens(ships, timestep_offset, actions)
            
            # Pad to max_ships if necessary
            n_ships = min(ships.n_ships, self.max_ships)
            tokens_batch[i, :n_ships, :] = tokens[:n_ships]
            
            # Create ship IDs
            ship_ids_batch[i, :n_ships] = torch.arange(n_ships)
        
        return tokens_batch, ship_ids_batch
    
    def create_temporal_batch_sequences(
        self,
        ships_history_batch: List[List[Ships]],
        actions_history_batch: Optional[List[List[torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create temporal sequences for a batch of game histories.
        
        Args:
            ships_history_batch: List of ship histories (batch_size x sequence_length)
            actions_history_batch: Optional actions histories
            
        Returns:
            tokens: [batch_size, seq_len, 12] Temporal token sequences
            ship_ids: [batch_size, seq_len] Ship IDs for each token
        """
        return self.base_encoder.create_batch_sequences(
            ships_history_batch, actions_history_batch, self.sequence_length
        )