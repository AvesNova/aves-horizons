# Ship Game Transformer Architecture Design

## Overview

This document outlines the design of a transformer-based neural network architecture for the physics-based ship combat game. The architecture uses temporal sequences of ship states to predict optimal actions, enabling sophisticated multi-agent coordination and opponent modeling.

## Core Concept

The model treats ship states as tokens in a temporal sequence, using transformer attention mechanisms to model complex interactions between ships across time. Rather than directly mapping current observations to actions, the model learns to understand temporal dynamics and multi-agent interactions.

### Key Innovation
- **Temporal Token Representation**: Each ship's state at each timestep becomes a token
- **Multi-Agent Awareness**: Single model predicts actions for all ships simultaneously
- **Unified Team Control**: One forward pass generates coordinated actions for entire team

## Architecture Philosophy

### Two-Part Conceptual Framework

**Part 1: Temporal Dynamics Understanding**
- Input: Sequence of ship states across multiple timesteps
- Process: Transformer layers learn temporal patterns and multi-agent interactions
- Goal: Understand "what should happen next"

**Part 2: Action Translation**
- Input: Rich contextual understanding from transformer
- Process: Linear layers map to discrete game actions
- Goal: Determine "how to make it happen"

### Implementation Strategy
While conceptually two-part, the actual implementation uses end-to-end training with RL, allowing the model to learn optimal representations without enforcing hard separation between temporal understanding and action selection.

## Token Design

### Base Token Representation
Each token represents one ship's state at one timestep:

```python
base_token = [
    pos_x,           # World X coordinate
    pos_y,           # World Y coordinate  
    vel_x,           # Velocity X component
    vel_y,           # Velocity Y component
    attitude_x,      # Facing direction X (cos θ)
    attitude_y,      # Facing direction Y (sin θ)
    turn_offset,     # Turn angle offset from velocity
    boost_norm,      # Boost energy (normalized 0-1)
    health_norm,     # Health points (normalized 0-1)
    ammo_norm,       # Ammunition count (normalized 0-1)
    is_shooting,     # Binary shooting state
    timestep_offset  # Temporal position in sequence
]
# Total: 12 dimensions
```

### Encoding Strategy

**Ship Identity Encoding**
- Fixed learnable embeddings for ship slots 0-7
- `final_token = base_token + ship_embedding[ship_id]`
- Supports up to 8 ships (4v4 maximum)
- Simple addition operation for easy implementation

**Temporal Encoding**
- Include `timestep_offset` as scalar in base token
- Alternative: Add learned `time_embedding[timestep_offset]`
- Timestep sequence: typically 6 steps covering recent history

**Spatial Encoding**
- Use raw continuous coordinates initially
- Avoid complex sinusoidal encodings for MVP
- Transformer can learn spatial relationships from raw positions

## Multi-Ship Token Organization

### Sequence Structure
For multi-ship scenarios, tokens are organized time-major:

```
Timesteps: [t-5, t-4, t-3, t-2, t-1, t-0]
Ships:     [A, B, C, D] (example 2v2)

Token sequence:
[A_t-5, B_t-5, C_t-5, D_t-5, A_t-4, B_t-4, C_t-4, D_t-4, ..., A_t-0, B_t-0, C_t-0, D_t-0]
```

**Benefits:**
- Natural data collection order
- Simple implementation
- Clear temporal progression

**Example Dimensions:**
- 4 ships × 6 timesteps = 24 tokens total
- Each token: 12D base + embedding dimension

## Training Strategy

### Reinforcement Learning Approach
- End-to-end training using standard RL algorithms (PPO recommended)
- Model predicts actions for all ships, gradients applied only to controllable team
- Auxiliary losses possible for temporal prediction (future consideration)

### Multi-Agent Output Strategy
The model outputs actions for all ships but only executes actions for the controlled team:

```python
# Model outputs: [ship0_actions, ship1_actions, ..., ship7_actions]
# Each ship_actions: [forward, backward, left, right, sharp_turn, shoot]

# For team control, extract relevant ship actions:
team_actions = model_output[team_ship_indices]
```

**Benefits:**
- Single forward pass for team coordination
- Opponent behavior modeling
- Potential for counter-strategy development

**Considerations:**
- Mask gradients for non-controlled ships
- Monitor for training instability from unused outputs

## Implementation Roadmap

### Phase 1: MVP (Minimal Viable Product)

**Goal:** Prove basic concept works reliably

**Constraints:**
- Fixed 4v4 maximum (8 ships total)
- Fixed map size
- 6 timestep history
- No complex encodings

**Architecture:**
```python
class ShipTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=3):
        self.ship_embeddings = nn.Embedding(8, d_model)  # 8 ship slots
        self.input_projection = nn.Linear(12, d_model)   # Base token to d_model
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), 
            num_layers
        )
        self.action_head = nn.Linear(d_model, 6)  # 6 actions per ship
    
    def forward(self, tokens, ship_ids):
        # tokens: [batch, seq_len, 12]
        # ship_ids: [batch, seq_len] 
        
        x = self.input_projection(tokens)
        x += self.ship_embeddings(ship_ids)
        
        x = self.transformer(x)
        
        # Extract final timestep tokens for each ship
        actions = self.action_head(x[:, -8:])  # Last 8 tokens (t=0 for all ships)
        return actions  # [batch, 8, 6]
```

**Training Process:**
1. Start with 1v1 scenarios
2. Gradually increase to 2v2, then 4v4
3. Simple reward shaping (survival, damage dealt, energy efficiency)

**Success Metrics:**
- Consistent training convergence
- Basic combat behaviors emerge
- Model generalizes across different ship counts

### Phase 2: Enhanced Features

**Additions after MVP success:**

**Improved Encodings:**
- Sinusoidal positional encoding for spatial coordinates
- Learned temporal embeddings with better frequency patterns

**Advanced Training:**
- Opponent modeling auxiliary losses
- Self-play training protocols
- Curriculum learning from simple to complex scenarios

**Architecture Improvements:**
- Attention masking for strategic information hiding
- Multi-head attention for different interaction types
- Hierarchical attention (local ship interactions vs global strategy)

### Phase 3: Full Vision (Long-term)

**Advanced Multi-Agent Capabilities:**

**Team Coordination:**
- Implicit communication through shared model
- Role specialization emergence
- Formation and strategy learning

**Opponent Modeling:**
- Predict enemy ship actions with high accuracy
- Adapt strategies based on opponent behavior patterns
- Counter-strategy development

**Diverse AI Personalities:**
- Single model generates different "AI styles" for different ships
- Behavioral variation through conditioning or role embeddings
- Maintained consistency within individual ship "characters"

**Map and Environmental Awareness:**
- Integration of static obstacles and map features
- Line-of-sight considerations for information hiding
- Dynamic environmental hazards

**Variable Scenarios:**
- Support for arbitrary team sizes and compositions
- Asymmetric scenarios (2v3, 1v4, etc.)
- Mixed human-AI teams

## Technical Considerations

### Memory and Computational Efficiency
- Token sequence length scales as `num_ships × timesteps`
- Attention complexity: O(n²) where n = sequence length
- Consider gradient checkpointing for longer sequences
- Batch processing for multiple games simultaneously

### Training Stability
- Careful initialization of ship embeddings
- Gradient clipping to prevent instability from multi-ship outputs
- Regularization to prevent overfitting to specific ship configurations
- Monitor for mode collapse in multi-agent scenarios

### Evaluation Metrics
- Win rate across different team compositions
- Individual ship survival times
- Resource efficiency (ammo, energy usage)
- Emergent coordination behaviors
- Adaptation to different opponent styles

## Alternative Architectures Considered

### Hard Separation Approach
**Rejected for MVP:** Two-stage training with explicit state prediction
- **Stage 1:** Train transformer to predict future states
- **Stage 2:** Train action network to achieve predicted states
- **Issues:** Training complexity, error propagation, limited flexibility

### Single-Ship Focus
**Rejected for scope:** Individual ship control only
- Simpler implementation but misses multi-agent coordination potential
- Difficult to scale to team scenarios
- Less interesting emergent behaviors

### Direct State-Action Mapping
**Rejected for capability:** Traditional RL without temporal modeling
- Misses temporal patterns and multi-step planning
- Limited ability to model opponent behavior
- No natural team coordination mechanism

## Success Criteria

### MVP Success Indicators
- [ ] Model trains consistently without divergence
- [ ] Basic combat behaviors emerge (shooting, evasion, pursuit)
- [ ] Performance improves with training time
- [ ] Generalizes across 1v1, 2v2, 4v4 scenarios
- [ ] Outperforms random baseline by significant margin

### Full Vision Success Indicators
- [ ] Sophisticated team coordination without explicit communication
- [ ] Accurate opponent behavior prediction and counter-strategies
- [ ] Diverse, consistent AI personalities per ship
- [ ] Human-level or superhuman combat performance
- [ ] Emergent strategic behaviors not explicitly programmed

## Conclusion

This transformer-based architecture represents a novel approach to multi-agent combat AI, combining temporal sequence modeling with unified team control. The phased implementation strategy balances ambition with practical engineering concerns, providing clear milestones for validation while maintaining the vision for sophisticated multi-agent behaviors.

The key innovation lies in treating multi-agent combat as a temporal sequence prediction problem, allowing the model to naturally learn coordination, opponent modeling, and strategic planning through self-supervised temporal dynamics and reinforcement learning signals.