# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

# Aves Horizons: Ship Combat AI with ShipNN

This is a PyTorch-based AI system for physics-based ship combat using the ShipNN three-stage neural network architecture. The project implements multi-agent coordination through temporal sequence learning, enabling teams of ships to fight cooperatively using sophisticated AI strategies.

## Essential Commands

### Testing
```bash
# Run all tests (44+ tests covering imports, models, training)
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_imports.py -v          # Dependency verification
python -m pytest tests/test_ship_nn.py -v          # ShipNN model architecture tests
python -m pytest tests/test_functional.py -v       # Training capability tests
python -m pytest tests/test_integration.py -v      # End-to-end pipeline tests

# Quick test run
python -m pytest tests/test_imports.py
```

### Training
```bash
# Basic training with default settings (1M timesteps)
python src/train.py --total-timesteps 1000000

# Quick test training run
python src/train.py --total-timesteps 1000 --hidden-dim 32 --batch-size 16

# Advanced training with custom parameters
python src/train.py \
    --total-timesteps 5000000 \
    --hidden-dim 128 \
    --n-heads 8 \
    --encoder-layers 3 \
    --transformer-layers 4 \
    --decoder-layers 3 \
    --learning-rate 0.0001 \
    --batch-size 256

# Training with configuration file
python src/train.py --config-file example_training_config.json --total-timesteps 10000000

# Evaluate existing model
python src/train.py --eval-only --load-model models/ship_transformer_final.zip
```

### Game Simulation
```bash
# Interactive game with human controls
python src/main.py --game-mode deathmatch --ships-per-team 4

# Standard game mode with multiple ships
python src/main.py --game-mode standard --n-ships 10

# Control different ship
python src/main.py --game-mode deathmatch --ships-per-team 3 --controlled-ship 2
```

### Testing Individual Components
```bash
# Test specific training functionality
python src/test_train.py

# Test ShipNN model in isolation
python -c "from src.models.ship_nn import ShipNN; print('ShipNN loads successfully')"
```

## Architecture Overview

### Core Components

1. **ShipNN** (`src/models/ship_nn.py`)
   - Three-stage neural network: Encoder → Transformer → Decoder
   - Configurable architecture with presets (small, default, large, deep, wide)
   - Processes temporal sequences of 13-dimensional ship state tokens
   - Outputs actions for all 8 ships simultaneously (multi-agent approach)
   - Features learnable ship identity embeddings and positional encoding

3. **Training System** (`src/train.py`, `src/training/`)
   - Unified training infrastructure using StableBaselines3 PPO
   - Self-play training with opponent pool management
   - Automatic model checkpointing and evaluation

3. **Game Environments**
   - `src/core/environment.py` - Physics-based ship combat simulation
   - `src/game_modes/deathmatch.py` - Team-based combat scenarios
   - `src/gym_env/ship_env.py` - RL-compatible environment wrapper

4. **Configuration System** (`src/utils/config.py`, `src/training/config.py`)
   - Centralized configuration management with ShipNN-specific parameters
   - Support for JSON config files and command-line arguments
   - Legacy parameter compatibility for smooth transitions
   - Model, training, and game parameters

### Data Flow

1. **State Collection**: Ships move and interact in physics simulation
2. **Token Encoding**: Ship states converted to 13D tokens with team information
3. **ShipNN Processing**: Three-stage processing (Encoder → Transformer → Decoder)
4. **Temporal Understanding**: Model processes time-major token sequences with attention
5. **Action Generation**: Model outputs coordinated actions for entire team
6. **Environment Step**: Actions executed in physics simulation
7. **Training**: PPO updates model based on rewards and self-play opponents

### Token Representation

Each ship state becomes a 13-dimensional token:
```
[pos_x, pos_y, vel_x, vel_y, attitude_x, attitude_y, turn_offset,
 boost_norm, health_norm, ammo_norm, is_shooting, team_id, timestep_offset]
```

Temporal sequences organized time-major for 6 timesteps with up to 8 ships:
```
Token sequence (48 tokens total, 13D each):
[ship0_t-5, ship1_t-5, ..., ship7_t-5,    # Timestep t-5
 ship0_t-4, ship1_t-4, ..., ship7_t-4,    # Timestep t-4
 ...
 ship0_t-0, ship1_t-0, ..., ship7_t-0]    # Current timestep
```

## Development Patterns

### Configuration Management
- Use JSON config files for complex training setups (see `example_training_config.json`)
- Command-line arguments override config file values
- Default configurations in `src/utils/config.py` and `src/training/config.py`

### Multi-Agent Strategy
- Model predicts actions for ALL ships (8 total) in single forward pass
- Environment only executes actions for controlled team (typically 4 ships)
- Enables opponent modeling and team coordination
- Self-play training rotates opponent models automatically

### Testing Strategy
- Comprehensive test suite with 44+ tests validates all functionality
- Import tests verify dependencies work correctly
- Model tests validate transformer architecture and forward passes
- Functional tests demonstrate learning capability (99.95% accuracy on simple tasks)
- Integration tests verify end-to-end training pipeline

### Model Persistence
- Automatic checkpointing every 20,000 steps (configurable)
- Best model saving based on evaluation performance
- State history and training progress preserved for resumption
- Models saved in `./models/`, logs in `./logs/`, tensorboard logs in `./tensorboard_logs/`

## Project Structure Insights

```
src/
├── core/               # Physics simulation and game mechanics
├── models/             # Transformer architecture and token processing
├── game_modes/         # Game types (deathmatch, etc.)
├── gym_env/           # RL environment wrappers
├── training/          # Training infrastructure and configuration
├── rendering/         # Pygame-based visualization
└── utils/            # Configuration management and utilities

tests/                 # Comprehensive test suite (44+ tests)
docs/                 # Architecture documentation and specifications
```

Key files:
- `src/train.py` - Main training entry point
- `src/main.py` - Interactive game simulation
- `src/models/ship_nn.py` - Core ShipNN model with three-stage architecture
- `src/training/trainer.py` - Training orchestration
- `tests/test_ship_nn.py` - ShipNN model tests
- `tests/test_functional.py` - Training capability validation

## Training Configuration Examples

### Quick Test (Debug)
```json
{
  "hidden_dim": 32,
  "encoder_layers": 1,
  "transformer_layers": 2,
  "decoder_layers": 1,
  "n_heads": 2,
  "total_timesteps": 1000,
  "batch_size": 16
}
```

### Production Training
```json
{
  "hidden_dim": 256,
  "encoder_layers": 3,
  "transformer_layers": 6,
  "decoder_layers": 3,
  "n_heads": 8,
  "dim_feedforward": 512,
  "learning_rate": 0.0001,
  "total_timesteps": 5000000,
  "batch_size": 256,
  "selfplay_update_freq": 100000
}
```

## Important Implementation Details

### Self-Play Training
- Opponent pool automatically managed (default: max 10 models)
- Mixed opponent strategy: 70% self-play, 20% heuristic, 10% random
- Models added to pool every 50,000 timesteps (configurable)
- Prevents overfitting to single opponent strategy

### Model Architecture Choices
- Three-stage design: Encoder (2 layers) → Transformer (3 layers) → Decoder (2 layers)
- Default: 128-dim hidden, 4 attention heads (~100K parameters)
- Configurable presets: small (64d), large (256d), deep (8 layers), wide (512d)
- Sinusoidal positional encoding and ship identity embeddings
- LeakyReLU activation throughout encoder/decoder stages

### Memory and Performance
- Training speed: ~1000 steps/minute on modern hardware
- Model checkpoints: ~10-50MB each
- Automatic disk space management for opponent pool
- GPU acceleration supported (set `device: "cuda"` in config)

### Reward Shaping
- Survival bonuses for keeping ships alive
- Damage dealing rewards for combat effectiveness
- Team coordination incentives
- Episode length penalties to encourage decisive action

This system represents a complete, tested implementation of ShipNN-based multi-agent ship combat AI with self-play training capabilities. The three-stage neural network architecture provides superior flexibility and performance compared to traditional transformer-only approaches.
