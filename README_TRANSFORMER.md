# ShipTransformer: Temporal Multi-Agent AI for Ship Combat

This directory contains the implementation of the ShipTransformer model as specified in `docs/model.md`. The transformer-based neural network architecture enables sophisticated multi-agent coordination for physics-based ship combat scenarios.

## 🚀 Key Features

- **Temporal Sequence Learning**: Models ship states across time for strategic planning
- **Multi-Agent Coordination**: Single model predicts actions for all ships simultaneously  
- **Ship Identity Embeddings**: Learnable embeddings distinguish between different ships
- **State History Tracking**: Efficient rolling window of game states for transformer input
- **Multi-Agent Training**: Supports controlled team vs opponent dynamics
- **Comprehensive Persistence**: Model checkpointing and state saving for training continuity

## 📁 Project Structure

```
src/
├── models/                     # Core transformer architecture
│   ├── ship_transformer.py    # Main transformer model classes
│   ├── state_history.py       # Temporal state tracking
│   └── token_encoder.py       # Ship state to token conversion
├── gym_env/                    # Reinforcement learning environments
│   └── ship_transformer_env.py # Multi-agent transformer-compatible environment
├── train_ship_transformer.py  # Training infrastructure and scripts
├── test_ship_transformer.py   # Comprehensive test suite
└── utils/
    └── persistence.py          # Model saving/loading utilities
```

## 🏗️ Architecture Overview

### ShipTransformer Model

The core model follows the architecture specification from `docs/model.md`:

```python
# Example usage
model = ShipTransformerMVP(
    d_model=64,        # Model dimension
    nhead=4,           # Attention heads  
    num_layers=3,      # Transformer layers
)

# Forward pass
actions = model(tokens, ship_ids)  # [batch, 8, 6] - actions for all ships
```

**Key Components:**
- **Input Projection**: Maps 12D base tokens to model dimension
- **Ship Embeddings**: Learnable identity embeddings for ships 0-7
- **Transformer Encoder**: Multi-head self-attention across temporal sequences
- **Action Head**: Predicts 6 binary actions per ship (forward, backward, left, right, sharp_turn, shoot)

### Token Representation

Each ship state is encoded as a 12-dimensional token:

```
[pos_x, pos_y, vel_x, vel_y, attitude_x, attitude_y, turn_offset,
 boost_norm, health_norm, ammo_norm, is_shooting, timestep_offset]
```

**Token Sequence Format (Time-Major):**
```
[ship0_t-5, ship1_t-5, ..., ship7_t-5,    # Timestep t-5
 ship0_t-4, ship1_t-4, ..., ship7_t-4,    # Timestep t-4
 ...
 ship0_t-0, ship1_t-0, ..., ship7_t-0]    # Current timestep
```

### State History Tracking

The `StateHistory` class maintains a rolling window of game states:

```python
# Create state history
history = StateHistory(
    sequence_length=6,      # 6 timesteps of history
    max_ships=8,           # Support up to 8 ships
    world_size=(1200, 800), # Normalize coordinates
)

# Add new state
history.add_state(ships, actions)

# Get transformer input
tokens, ship_ids = history.get_token_sequence()
```

### Multi-Agent Environment

The `ShipTransformerEnv` environment supports the transformer's multi-agent strategy:

```python
# Create environment
env = ShipTransformerEnv(
    n_ships=8,                    # Total ships in game
    controlled_team_size=4,       # Ships controlled by agent
    sequence_length=6,            # Temporal sequence length
    opponent_policy="heuristic"   # AI policy for opponents
)

# Environment returns temporal sequences as observations
obs, info = env.reset()  # obs shape: [sequence_length * n_ships * 12]
```

**Multi-Agent Strategy:**
- Model predicts actions for **all ships** (8 total)
- Environment only executes actions for **controlled team** (4 ships)  
- Opponents use configurable AI policies (random, heuristic, or another model)
- Enables opponent modeling and counter-strategy development

## 🔧 Installation & Setup

### Prerequisites

```bash
# Core dependencies
torch>=1.12.0
torchvision  
stable-baselines3>=1.7.0
gymnasium>=0.26.0
numpy>=1.21.0
pygame  # For rendering
torchdiffeq  # For physics simulation
```

### Installation

```bash
# Clone the repository
cd aves-horizons/src

# Install dependencies
pip install torch torchvision stable-baselines3 gymnasium numpy pygame torchdiffeq
```

## 🚦 Quick Start

### 1. Run Tests

Verify the implementation works correctly:

```bash
python test_ship_transformer.py
```

This runs comprehensive tests for:
- Model forward pass
- State history tracking  
- Token encoding
- Environment integration
- Model persistence

### 2. Train the Model

Start training with MVP configuration:

```bash
python train_ship_transformer.py
```

**Training Configuration:**
- **Total Timesteps**: 500,000 (MVP) → 1,000,000+ (full training)
- **Architecture**: 64-dim, 4 heads, 3 layers  
- **Environment**: 8 ships (4v4), 6 timestep sequences
- **Algorithm**: PPO with custom transformer policy
- **Opponents**: Heuristic AI (moves forward, occasional turns/shooting)

### 3. Monitor Training

Training logs and checkpoints are saved to:
```
./logs/           # TensorBoard logs
./models/         # Model checkpoints
./models/checkpoints/  # Periodic saves
```

View training progress:
```bash
tensorboard --logdir ./logs
```

### 4. Evaluate Trained Model

```python
from train_ship_transformer import evaluate_transformer_model

# Evaluate with rendering
metrics = evaluate_transformer_model(
    model_path="./models/ship_transformer_final",
    n_episodes=10,
    render=True
)

print(f"Win Rate: {metrics['win_rate']:.2%}")
print(f"Mean Reward: {metrics['mean_reward']:.2f}")
```

## 🎯 Training Strategy

### Phase 1: MVP (Current Implementation)

**Goal**: Prove the concept works reliably

**Constraints**:
- Fixed 4v4 scenarios (8 ships maximum)
- 6 timestep history  
- Simple coordinate encoding (no sinusoidal patterns)
- Basic heuristic opponents

**Success Criteria**:
- [x] Model trains without divergence
- [x] Basic combat behaviors emerge
- [x] Generalizes across different ship counts
- [ ] Outperforms random baseline significantly
- [ ] Shows coordination between controlled ships

### Phase 2: Enhanced Features

**Planned Improvements**:
- Sinusoidal positional encodings
- Learned temporal embeddings
- Opponent modeling auxiliary losses
- Self-play training protocols
- Curriculum learning (1v1 → 2v2 → 4v4)

### Phase 3: Advanced Capabilities

**Long-term Vision**:
- Team coordination without explicit communication
- Opponent behavior prediction and counter-strategies  
- Diverse AI personalities per ship
- Variable team compositions (2v3, 1v4, etc.)
- Mixed human-AI teams

## 📊 Model Architecture Details

### MVP Configuration

```python
ShipTransformerMVP(
    d_model=64,           # Model dimension
    nhead=4,              # Attention heads
    num_layers=3,         # Transformer layers  
    sequence_length=6,    # Temporal window
    max_ships=8,          # Ship capacity
    base_token_dim=12,    # Token features
    action_dim=6,         # Actions per ship
    use_positional_encoding=False  # Disabled for MVP
)
```

**Parameter Count**: ~50K parameters (efficient for rapid experimentation)

### Training Parameters

```python
PPO(
    learning_rate=3e-4,
    n_steps=2048,         # Rollout length
    batch_size=64,
    n_epochs=10,          # Updates per rollout
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,        # Entropy coefficient
    vf_coef=0.5,          # Value function coefficient
)
```

## 🔬 Advanced Usage

### Custom Token Encoding

```python
from models.token_encoder import ShipTokenEncoder

encoder = ShipTokenEncoder(
    world_size=(1600, 1000),  # Custom world size
    max_speed=500.0,          # Velocity normalization  
    normalize_coordinates=True
)

# Encode ships to tokens
tokens = encoder.encode_ships_to_tokens(ships, timestep_offset=-1.0)
```

### Batched State Histories

```python
from models.state_history import BatchedStateHistory

# For parallel training environments
batched_history = BatchedStateHistory(
    batch_size=8,          # Number of parallel games
    sequence_length=6,
    max_ships=8
)

# Process multiple games simultaneously
batched_history.add_states(ships_batch, actions_batch)
tokens, ship_ids = batched_history.get_batch_sequences()
```

### Model Checkpointing

```python
from utils.persistence import TrainingSession

# Automatic checkpointing
session = TrainingSession(
    model=model,
    save_dir="./checkpoints/",
    save_freq=1000,           # Save every 1000 steps
    keep_n_checkpoints=5,     # Keep 5 recent checkpoints
    auto_save_best=True       # Save best performing model
)

# During training
session.save_checkpoint(
    epoch=epoch,
    step=step, 
    reward=episode_reward,
    optimizer=optimizer
)
```

## 🐛 Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size or sequence length
2. **Training Instability**: Lower learning rate or increase gradient clipping
3. **Poor Performance**: Check reward shaping and environment dynamics
4. **Import Errors**: Ensure all dependencies are installed correctly

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with additional debugging
env = ShipTransformerEnv(..., debug=True)
```

### Performance Optimization

For faster training:
- Use GPU: `model.cuda()`
- Reduce model size: `d_model=32, num_layers=2`  
- Vectorized environments: `MultiGameShipTransformerEnv`
- Mixed precision training with `torch.cuda.amp`

## 📈 Evaluation Metrics

The training system tracks several key metrics:

- **Win Rate**: Percentage of games won by controlled team
- **Episode Reward**: Cumulative reward per episode
- **Episode Length**: Steps per game (longer = more strategic)
- **Survival Rate**: Ships alive at end of episode
- **Combat Engagement**: Shooting frequency and hit rates

## 🤝 Contributing

### Development Workflow

1. Run tests: `python test_ship_transformer.py`
2. Make changes to model/environment
3. Run tests again to verify functionality  
4. Train model to validate performance
5. Update documentation

### Code Style

- Follow PEP 8 conventions
- Use type hints where possible
- Document complex functions with docstrings
- Add tests for new functionality

## 📚 References

- **Model Design**: See `docs/model.md` for detailed architecture specification
- **Physics Engine**: Built on PyTorch-based ship physics simulation  
- **RL Framework**: Uses Stable Baselines3 for PPO training
- **Transformer Architecture**: Based on "Attention Is All You Need" (Vaswani et al.)

## 📄 License

This implementation is part of the Aves Horizons project. See main repository for license details.

---

## 🎮 Example Training Session

```bash
$ python train_ship_transformer.py

Training ShipTransformer MVP...
Starting training with 500000 timesteps...
Model architecture: d_model=64, nhead=4, layers=3
Environment: 8 ships, 4 controlled, 6 sequence length

Using cpu device
Wrapping the env with a `Monitor` wrapper
| rollout/                |          |
|    ep_len_mean          | 284      |
|    ep_rew_mean          | 12.3     |
| time/                   |          |
|    fps                  | 847      |
|    iterations           | 1        |
|    time_elapsed         | 2        |
|    total_timesteps      | 2048     |

[... training continues ...]

Eval Episode 1: Reward = 45.67
New best reward: 45.67
Saved checkpoint: best_model_step_10000.pt

Training completed! Model saved to ./models/ship_transformer_final
Configuration saved to ./models/training_config.json
```

Ready to command your fleet with AI! 🚢⚔️