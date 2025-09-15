"""
ShipTransformer: Transformer-based neural network for ship combat AI.

Implements the architecture from docs/model.md:
- Temporal sequences of ship states as tokens
- Multi-agent coordination through single model
- Ship identity embeddings for multi-ship support
- End-to-end training with reinforcement learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer input."""
    
    def __init__(self, d_model: int, max_seq_length: int = 100):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tokens."""
        return x + self.pe[:x.size(0), :].transpose(0, 1)


class ShipTransformer(nn.Module):
    """
    Transformer model for ship combat AI.
    
    Architecture:
    - Input: Temporal sequence of ship states (tokens)
    - Ship embeddings: Learnable identity embeddings for each ship
    - Transformer: Multi-head attention across temporal sequences
    - Output: Actions for all ships (multi-agent coordination)
    
    Token sequence format:
    [ship0_t-5, ship1_t-5, ..., ship0_t-4, ship1_t-4, ..., ship0_t-0, ship1_t-0]
    
    Each base token is 12-dimensional:
    [pos_x, pos_y, vel_x, vel_y, attitude_x, attitude_y, turn_offset,
     boost_norm, health_norm, ammo_norm, is_shooting, timestep_offset]
    """
    
    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_ships: int = 8,
        sequence_length: int = 6,
        base_token_dim: int = 12,
        action_dim: int = 6,
        use_positional_encoding: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_ships = max_ships
        self.sequence_length = sequence_length
        self.base_token_dim = base_token_dim
        self.action_dim = action_dim
        
        # Ship identity embeddings (learnable embeddings for ship slots 0-7)
        self.ship_embeddings = nn.Embedding(max_ships, d_model)
        
        # Project base tokens to model dimension
        self.input_projection = nn.Linear(base_token_dim, d_model)
        
        # Optional positional encoding
        if use_positional_encoding:
            max_seq_len = sequence_length * max_ships
            self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        else:
            self.positional_encoding = None
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=F.relu,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Action prediction head
        self.action_head = nn.Linear(d_model, action_dim)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters."""
        # Initialize ship embeddings with small random values
        nn.init.normal_(self.ship_embeddings.weight, mean=0.0, std=0.02)
        
        # Initialize input projection with smaller scale
        nn.init.xavier_uniform_(self.input_projection.weight, gain=0.1)
        nn.init.zeros_(self.input_projection.bias)
        
        # Initialize action head with smaller scale to prevent large gradients
        nn.init.xavier_uniform_(self.action_head.weight, gain=0.1)
        nn.init.zeros_(self.action_head.bias)
        
        # Initialize transformer layers with smaller scale
        for layer in self.transformer.layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param, gain=0.1)
    
    def create_ship_sequence(
        self, 
        tokens: torch.Tensor, 
        ship_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create temporal sequence organized by time-major order.
        
        Args:
            tokens: [batch, seq_len, base_token_dim] Base token features
            ship_ids: [batch, seq_len] Ship ID for each token
            
        Returns:
            embedded_tokens: [batch, seq_len, d_model] Token embeddings
            attention_mask: [batch, seq_len] Mask for valid tokens
        """
        batch_size, seq_len = tokens.shape[:2]
        
        # Project base tokens to model dimension
        projected = self.input_projection(tokens)  # [batch, seq_len, d_model]
        
        # Add ship identity embeddings
        ship_embeds = self.ship_embeddings(ship_ids)  # [batch, seq_len, d_model]
        embedded_tokens = projected + ship_embeds
        
        # Add positional encoding if enabled
        if self.positional_encoding is not None:
            # Transpose for positional encoding (expects [seq_len, batch, d_model])
            embedded_tokens = embedded_tokens.transpose(0, 1)
            embedded_tokens = self.positional_encoding(embedded_tokens)
            embedded_tokens = embedded_tokens.transpose(0, 1)
        
        # Create attention mask (all tokens are valid in this implementation)
        attention_mask = torch.ones(batch_size, seq_len, device=tokens.device)
        
        return embedded_tokens, attention_mask
    
    def forward(
        self, 
        tokens: torch.Tensor, 
        ship_ids: torch.Tensor,
        controlled_ships: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer.
        
        Args:
            tokens: [batch, seq_len, base_token_dim] Temporal sequence of ship states
            ship_ids: [batch, seq_len] Ship ID for each token (0-7)
            controlled_ships: [batch, max_ships] Binary mask for controlled ships
            
        Returns:
            actions: [batch, max_ships, action_dim] Predicted actions for all ships
        """
        batch_size = tokens.shape[0]
        
        # Create embedded token sequence
        embedded_tokens, attention_mask = self.create_ship_sequence(tokens, ship_ids)
        
        # Pass through transformer
        # Note: PyTorch's TransformerEncoder doesn't use the attention mask the same way
        # We'll implement padding mask for variable length sequences if needed later
        transformer_output = self.transformer(embedded_tokens)  # [batch, seq_len, d_model]
        
        # Extract final timestep tokens for each ship
        # Assumes tokens are organized as [ship0_t-N, ..., ship0_t-0, ship1_t-N, ..., ship1_t-0, ...]
        # We need the last occurrence of each ship (t=0 tokens)
        
        # Reshape to extract ship-specific representations
        # For now, we'll use the mean of all tokens for each ship as a simple approach
        # TODO: Implement proper extraction of t=0 tokens for each ship
        ship_representations = torch.zeros(
            batch_size, self.max_ships, self.d_model, 
            device=tokens.device
        )
        
        # Group tokens by ship and take the mean
        # This is a simplified approach - in practice, we'd want to extract
        # the final timestep token for each ship
        for ship_id in range(self.max_ships):
            ship_mask = (ship_ids == ship_id).float().unsqueeze(-1)  # [batch, seq_len, 1]
            if ship_mask.sum() > 0:
                ship_tokens = transformer_output * ship_mask  # [batch, seq_len, d_model]
                ship_sum = ship_tokens.sum(dim=1)  # [batch, d_model]
                ship_count = ship_mask.sum(dim=1)  # [batch, 1]
                ship_count = torch.clamp(ship_count, min=1.0)  # Avoid division by zero
                ship_representations[:, ship_id, :] = ship_sum / ship_count
        
        # Predict actions for all ships
        actions = self.action_head(ship_representations)  # [batch, max_ships, action_dim]
        
        # Apply mask to controlled ships if provided
        if controlled_ships is not None:
            # Set actions for uncontrolled ships to zero (or some neutral value)
            controlled_mask = controlled_ships.unsqueeze(-1)  # [batch, max_ships, 1]
            actions = actions * controlled_mask
        
        return actions
    
    def extract_current_timestep_tokens(
        self, 
        transformer_output: torch.Tensor,
        ship_ids: torch.Tensor,
        sequence_length: int
    ) -> torch.Tensor:
        """
        Extract tokens from the current timestep (t=0) for each ship.
        
        This assumes the token sequence follows the time-major organization:
        [ship0_t-5, ship1_t-5, ..., ship0_t-4, ship1_t-4, ..., ship0_t-0, ship1_t-0]
        
        Args:
            transformer_output: [batch, seq_len, d_model] Transformer output
            ship_ids: [batch, seq_len] Ship IDs for each token
            sequence_length: Number of timesteps in sequence
            
        Returns:
            current_tokens: [batch, max_ships, d_model] Current timestep representations
        """
        batch_size = transformer_output.shape[0]
        
        # Calculate the starting index of the current timestep (t=0)
        current_timestep_start = (sequence_length - 1) * self.max_ships
        current_timestep_end = current_timestep_start + self.max_ships
        
        # Extract current timestep tokens
        current_tokens = torch.zeros(
            batch_size, self.max_ships, self.d_model,
            device=transformer_output.device
        )
        
        if current_timestep_end <= transformer_output.shape[1]:
            current_timestep_tokens = transformer_output[:, current_timestep_start:current_timestep_end, :]
            current_ship_ids = ship_ids[:, current_timestep_start:current_timestep_end]
            
            # Organize tokens by ship ID
            for batch_idx in range(batch_size):
                for token_idx, ship_id in enumerate(current_ship_ids[batch_idx]):
                    if ship_id < self.max_ships:
                        current_tokens[batch_idx, ship_id, :] = current_timestep_tokens[batch_idx, token_idx, :]
        
        return current_tokens
    
    def get_action_probabilities(self, action_logits: torch.Tensor) -> torch.Tensor:
        """
        Convert action logits to probabilities using sigmoid for multi-binary actions.
        
        Args:
            action_logits: [batch, max_ships, action_dim] Raw action logits
            
        Returns:
            action_probs: [batch, max_ships, action_dim] Action probabilities
        """
        return torch.sigmoid(action_logits)
    
    def sample_actions(self, action_logits: torch.Tensor) -> torch.Tensor:
        """
        Sample binary actions from logits.
        
        Args:
            action_logits: [batch, max_ships, action_dim] Raw action logits
            
        Returns:
            actions: [batch, max_ships, action_dim] Sampled binary actions
        """
        action_probs = self.get_action_probabilities(action_logits)
        return torch.bernoulli(action_probs)


class ShipTransformerMVP(ShipTransformer):
    """
    Simplified MVP version of ShipTransformer with fixed 4v4 constraints.
    
    This implementation follows the MVP specification from docs/model.md:
    - Fixed 4v4 maximum (8 ships total)
    - Fixed map size
    - 6 timestep history
    - No complex encodings
    - Simple architecture for proof of concept
    """
    
    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        **kwargs
    ):
        # Override defaults for MVP, but allow kwargs to override them
        mvp_defaults = {
            'dim_feedforward': 256,
            'dropout': 0.1,
            'max_ships': 8,
            'sequence_length': 6,
            'base_token_dim': 12,
            'action_dim': 6,
            'use_positional_encoding': False,  # No complex encodings for MVP
        }
        
        # Merge defaults with kwargs, kwargs take precedence
        merged_kwargs = {**mvp_defaults, **kwargs}
        
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            **merged_kwargs
        )
    
    def forward(
        self, 
        tokens: torch.Tensor, 
        ship_ids: torch.Tensor,
        controlled_ships: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Simplified forward pass for MVP.
        
        Extracts the final 8 tokens (t=0 for all ships) and predicts actions.
        """
        batch_size = tokens.shape[0]
        
        # Create embedded token sequence
        embedded_tokens, _ = self.create_ship_sequence(tokens, ship_ids)
        
        # Pass through transformer
        transformer_output = self.transformer(embedded_tokens)
        
        # Extract final 8 tokens (t=0 for all ships)
        # Assumes token sequence ends with [ship0_t0, ship1_t0, ..., ship7_t0]
        final_tokens = transformer_output[:, -self.max_ships:, :]  # [batch, 8, d_model]
        
        # Predict actions for all ships
        actions = self.action_head(final_tokens)  # [batch, 8, 6]
        
        return actions