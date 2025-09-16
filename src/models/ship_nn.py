"""
ShipNN: Three-stage neural network for ship combat AI.

Architecture:
1. Encoder: Feed-forward layers to project ship tokens to transformer dimension
2. Transformer: Multi-layer transformer core for temporal reasoning
3. Decoder: Feed-forward layers to convert final hidden states to actions

Features:
- Configurable layer counts for all three stages
- Configurable hidden dimensions
- LeakyReLU activation throughout
- Ship identity embeddings
- Temporal positional encoding
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
        """Add positional encoding to input tokens.
        
        Args:
            x: [seq_len, batch, d_model] Input tensor
        """
        seq_len = x.size(0)
        return x + self.pe[:seq_len, :]


class ShipEncoder(nn.Module):
    """
    Encoder stage: Projects ship tokens to transformer dimension.
    
    Uses feed-forward layers with LeakyReLU activation.
    """
    
    def __init__(
        self,
        input_dim: int = 13,  # Base token dimension
        hidden_dim: int = 128,  # Transformer dimension
        num_layers: int = 2,
        dropout: float = 0.1,
        negative_slope: float = 0.01
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        if num_layers == 1:
            # Single layer: direct projection
            self.layers = nn.ModuleList([
                nn.Linear(input_dim, hidden_dim)
            ])
        else:
            # Multi-layer: gradually increase dimension
            layers = []
            
            # First layer: input -> intermediate
            intermediate_dim = max(hidden_dim // 2, input_dim * 2)
            layers.append(nn.Linear(input_dim, intermediate_dim))
            
            # Hidden layers: gradually increase to target dimension
            for i in range(num_layers - 2):
                next_dim = intermediate_dim + (hidden_dim - intermediate_dim) * (i + 1) // (num_layers - 1)
                layers.append(nn.Linear(intermediate_dim, next_dim))
                intermediate_dim = next_dim
            
            # Final layer: -> target dimension
            layers.append(nn.Linear(intermediate_dim, hidden_dim))
            
            self.layers = nn.ModuleList(layers)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize encoder weights."""
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=0.1)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            x: [batch, seq_len, input_dim] Input tokens
            
        Returns:
            encoded: [batch, seq_len, hidden_dim] Encoded tokens
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # No activation after final layer
                x = self.activation(x)
                x = self.dropout(x)
        
        return x


class ShipDecoder(nn.Module):
    """
    Decoder stage: Converts transformer outputs to action predictions.
    
    Uses feed-forward layers with LeakyReLU activation.
    """
    
    def __init__(
        self,
        input_dim: int = 128,  # Transformer dimension
        output_dim: int = 6,   # Action dimension
        num_layers: int = 2,
        dropout: float = 0.1,
        negative_slope: float = 0.01
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        if num_layers == 1:
            # Single layer: direct projection
            self.layers = nn.ModuleList([
                nn.Linear(input_dim, output_dim)
            ])
        else:
            # Multi-layer: gradually decrease dimension
            layers = []
            
            # First layer: input -> intermediate
            intermediate_dim = max(output_dim * 4, input_dim // 2)
            layers.append(nn.Linear(input_dim, intermediate_dim))
            
            # Hidden layers: gradually decrease to target dimension
            for i in range(num_layers - 2):
                next_dim = intermediate_dim - (intermediate_dim - output_dim) * (i + 1) // (num_layers - 1)
                layers.append(nn.Linear(intermediate_dim, next_dim))
                intermediate_dim = next_dim
            
            # Final layer: -> output dimension
            layers.append(nn.Linear(intermediate_dim, output_dim))
            
            self.layers = nn.ModuleList(layers)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize decoder weights."""
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=0.1)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            x: [batch, max_ships, input_dim] Transformer outputs
            
        Returns:
            actions: [batch, max_ships, output_dim] Action predictions
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # No activation after final layer
                x = self.activation(x)
                x = self.dropout(x)
        
        return x


class ShipNN(nn.Module):
    """
    Complete three-stage neural network for ship combat AI.
    
    Architecture:
    1. Encoder: FF layers (ship tokens -> transformer dimension)
    2. Transformer: Multi-layer transformer core
    3. Decoder: FF layers (transformer outputs -> actions)
    
    Features:
    - Configurable layer counts for all stages
    - Configurable hidden dimensions
    - LeakyReLU activation
    - Ship identity embeddings
    - Temporal positional encoding
    """
    
    def __init__(
        self,
        # Model dimensions
        input_dim: int = 13,      # Base token dimension
        hidden_dim: int = 128,    # Transformer dimension
        output_dim: int = 6,      # Action dimension
        max_ships: int = 8,       # Maximum number of ships
        sequence_length: int = 6, # Temporal sequence length
        
        # Layer counts
        encoder_layers: int = 2,      # Encoder FF layers
        transformer_layers: int = 3,  # Transformer layers
        decoder_layers: int = 2,      # Decoder FF layers
        
        # Transformer parameters
        n_heads: int = 4,             # Attention heads
        dim_feedforward: int = 256,   # Transformer FF dimension
        
        # Regularization
        dropout: float = 0.1,
        negative_slope: float = 0.01,  # LeakyReLU slope
        
        # Features
        use_positional_encoding: bool = True,
        use_ship_embeddings: bool = True
    ):
        super().__init__()
        
        # Store configuration
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_ships = max_ships
        self.sequence_length = sequence_length
        self.encoder_layers = encoder_layers
        self.transformer_layers = transformer_layers
        self.decoder_layers = decoder_layers
        self.use_positional_encoding = use_positional_encoding
        self.use_ship_embeddings = use_ship_embeddings
        
        # 1. Encoder: Ship tokens -> Transformer dimension
        self.encoder = ShipEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=encoder_layers,
            dropout=dropout,
            negative_slope=negative_slope
        )
        
        # Ship identity embeddings
        if use_ship_embeddings:
            self.ship_embeddings = nn.Embedding(max_ships, hidden_dim)
            nn.init.normal_(self.ship_embeddings.weight, mean=0.0, std=0.02)
        else:
            self.ship_embeddings = None
        
        # Positional encoding
        if use_positional_encoding:
            max_seq_len = sequence_length * max_ships
            self.positional_encoding = PositionalEncoding(hidden_dim, max_seq_len)
        else:
            self.positional_encoding = None
        
        # 2. Transformer: Multi-layer attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=F.leaky_relu,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, transformer_layers)
        
        # 3. Decoder: Transformer outputs -> Actions
        self.decoder = ShipDecoder(
            input_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=decoder_layers,
            dropout=dropout,
            negative_slope=negative_slope
        )
        
        # Initialize transformer weights
        self._init_transformer_weights()
    
    def _init_transformer_weights(self):
        """Initialize transformer weights with smaller scale."""
        for layer in self.transformer.layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param, gain=0.1)
    
    def create_embedded_sequence(
        self, 
        tokens: torch.Tensor, 
        ship_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Create embedded token sequence ready for transformer.
        
        Args:
            tokens: [batch, seq_len, input_dim] Raw ship tokens
            ship_ids: [batch, seq_len] Ship ID for each token
            
        Returns:
            embedded_tokens: [batch, seq_len, hidden_dim] Ready for transformer
        """
        batch_size, seq_len = tokens.shape[:2]
        
        # 1. Project through encoder
        encoded_tokens = self.encoder(tokens)  # [batch, seq_len, hidden_dim]
        
        # 2. Add ship identity embeddings
        if self.use_ship_embeddings:
            ship_embeds = self.ship_embeddings(ship_ids)  # [batch, seq_len, hidden_dim]
            encoded_tokens = encoded_tokens + ship_embeds
        
        # 3. Add positional encoding
        if self.use_positional_encoding:
            # Transpose for positional encoding (expects [seq_len, batch, hidden_dim])
            encoded_tokens = encoded_tokens.transpose(0, 1)
            encoded_tokens = self.positional_encoding(encoded_tokens)
            encoded_tokens = encoded_tokens.transpose(0, 1)
        
        return encoded_tokens
    
    def extract_ship_representations(
        self, 
        transformer_output: torch.Tensor,
        ship_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract final representations for each ship from transformer output.
        
        For now, uses the last occurrence of each ship (t=0 tokens).
        
        Args:
            transformer_output: [batch, seq_len, hidden_dim] Transformer output
            ship_ids: [batch, seq_len] Ship IDs
            
        Returns:
            ship_representations: [batch, max_ships, hidden_dim] Per-ship features
        """
        batch_size = transformer_output.shape[0]
        
        # Extract final max_ships tokens (assuming they correspond to t=0 for each ship)
        final_tokens = transformer_output[:, -self.max_ships:, :]  # [batch, max_ships, hidden_dim]
        
        return final_tokens
    
    def forward(
        self, 
        tokens: torch.Tensor, 
        ship_ids: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through complete ShipNN.
        
        Args:
            tokens: [batch, seq_len, input_dim] Temporal sequence of ship tokens
            ship_ids: [batch, seq_len] Ship ID for each token (0 to max_ships-1)
            return_features: If True, return features instead of actions
            
        Returns:
            If return_features=False: actions [batch, max_ships, output_dim]
            If return_features=True: features [batch, max_ships, hidden_dim]
        """
        # 1. Encoder stage: Create embedded sequence
        embedded_tokens = self.create_embedded_sequence(tokens, ship_ids)
        
        # 2. Transformer stage: Multi-layer attention
        transformer_output = self.transformer(embedded_tokens)  # [batch, seq_len, hidden_dim]
        
        # 3. Extract per-ship representations
        ship_features = self.extract_ship_representations(transformer_output, ship_ids)
        
        # Return features if requested (for features extraction)
        if return_features:
            return ship_features
        
        # 4. Decoder stage: Convert to actions
        actions = self.decoder(ship_features)  # [batch, max_ships, output_dim]
        
        return actions
    
    def get_action_probabilities(self, action_logits: torch.Tensor) -> torch.Tensor:
        """
        Convert action logits to probabilities using sigmoid for multi-binary actions.
        
        Args:
            action_logits: [batch, max_ships, output_dim] Raw action logits
            
        Returns:
            action_probs: [batch, max_ships, output_dim] Action probabilities
        """
        return torch.sigmoid(action_logits)
    
    def sample_actions(self, action_logits: torch.Tensor) -> torch.Tensor:
        """
        Sample binary actions from logits.
        
        Args:
            action_logits: [batch, max_ships, output_dim] Raw action logits
            
        Returns:
            actions: [batch, max_ships, output_dim] Sampled binary actions
        """
        action_probs = self.get_action_probabilities(action_logits)
        return torch.bernoulli(action_probs)


def create_ship_nn(
    # Quick presets for common configurations
    preset: str = "default",
    **kwargs
) -> ShipNN:
    """
    Create ShipNN with preset configurations.
    
    Available presets:
    - "default": Balanced configuration for general use
    - "small": Lightweight model for fast training
    - "large": High-capacity model for complex behaviors
    - "deep": Deep architecture with many layers
    - "wide": Wide architecture with large hidden dimensions
    """
    presets = {
        "default": {
            "hidden_dim": 128,
            "encoder_layers": 2,
            "transformer_layers": 3,
            "decoder_layers": 2,
            "n_heads": 4,
            "dim_feedforward": 256,
        },
        "small": {
            "hidden_dim": 64,
            "encoder_layers": 1,
            "transformer_layers": 2,
            "decoder_layers": 1,
            "n_heads": 2,
            "dim_feedforward": 128,
        },
        "large": {
            "hidden_dim": 256,
            "encoder_layers": 3,
            "transformer_layers": 6,
            "decoder_layers": 3,
            "n_heads": 8,
            "dim_feedforward": 512,
        },
        "deep": {
            "hidden_dim": 128,
            "encoder_layers": 4,
            "transformer_layers": 8,
            "decoder_layers": 4,
            "n_heads": 4,
            "dim_feedforward": 256,
        },
        "wide": {
            "hidden_dim": 512,
            "encoder_layers": 2,
            "transformer_layers": 3,
            "decoder_layers": 2,
            "n_heads": 16,
            "dim_feedforward": 1024,
        }
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(presets.keys())}")
    
    # Merge preset with kwargs (kwargs override preset)
    config = {**presets[preset], **kwargs}
    
    return ShipNN(**config)