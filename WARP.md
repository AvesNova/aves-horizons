# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

Aves Horizons is a physics-based space combat simulation featuring transformer-based AI agents. The project combines reinforcement learning, behavior cloning, and sophisticated physics simulation to create intelligent multi-agent combat scenarios.

## Development Commands

### Testing
```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_environment_basics.py
pytest tests/test_environment_physics.py
pytest tests/test_environment_combat.py
pytest tests/test_ship_physics.py
pytest tests/test_ship_combat.py

# Run tests with verbose output
pytest -v

# Run tests with coverage
pytest --tb=short -v
```

### Training Pipeline

**Complete training pipeline (recommended):**
```bash
# 1. Collect BC training data
python src/collect_data.py collect_bc --config src/unified_training.yaml

# 2. Run full pipeline: BC pretraining → RL training → evaluation
python src/unified_train.py full --config src/unified_training.yaml

# 3. Evaluate the final model
python src/collect_data.py evaluate_model --model checkpoints/unified_full_*/final_rl_model.zip
```

**Individual training phases:**
```bash
# Behavior cloning only
python src/unified_train.py bc --config src/unified_training.yaml

# RL training from scratch
python src/unified_train.py rl --config src/unified_training.yaml

# RL training from BC model
python src/unified_train.py rl --config src/unified_training.yaml \
    --bc-model checkpoints/unified_bc_*/best_bc_model.pt
```

**Data collection:**
```bash
# Collect BC data (scripted vs scripted)
python src/collect_data.py collect_bc --config src/unified_training.yaml

# Collect self-play data
python src/collect_data.py collect_selfplay --config src/unified_training.yaml
```

### Evaluation and Testing
```bash
# Evaluate a trained model
python src/collect_data.py evaluate_model --model path/to/model.zip

# Human play testing
python src/collect_data.py human_play

# Compare multiple models
for model in checkpoints/*/best_*.pt; do
    echo "Evaluating $model"
    python src/collect_data.py evaluate_model --model "$model"
done
```

### Development Workflow
```bash
# Run from project root (required for imports)
cd /path/to/aves-horizons
python src/[script].py

# Add src to Python path if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

## High-Level Architecture

### Core System Components

**Environment System (`env.py`)**
- OpenAI Gym-compatible multi-agent environment
- Physics simulation using implicit Euler
- Supports 1v1, 2v2, 3v3, 4v4, and variable n-vs-n scenarios
- Toroidal world boundaries with collision detection
- Configurable timesteps: agent_dt (action frequency) and physics_dt (simulation accuracy)

**Agent System (`agents.py`)**
- Pluggable architecture supporting multiple agent types:
  - **ScriptedAgentProvider**: Sophisticated predictive targeting agents
  - **RLAgentProvider**: Supports both transformer and PPO models
  - **HumanAgentProvider**: Real-time human control via keyboard/mouse
  - **SelfPlayAgentProvider**: Manages model memory for self-play training
- Unified action interface across all agent types

**Training Pipeline (`unified_train.py`)**
- Two-phase training: Behavior Cloning → Reinforcement Learning
- Configurable opponent types: scripted, self-play, or mixed
- Support for model initialization from BC pretraining
- Integrated evaluation and checkpointing

### Physics and Simulation

**Ship Dynamics (`ship.py`)**
- Complex physics model with realistic aerodynamics
- Attitude-based control system with velocity-relative turning
- Energy management (boost/thrust consumption and regeneration)
- Combat system with projectiles and health management

**State Representation (`state.py`)**
- Centralized game state management
- Token-based observations for transformer models
- Normalized observations for consistent ML training

**Physics Integration**
- Uses simple implicit Euler integration
- Force-based dynamics with thrust, drag, and lift forces

### AI and Learning Architecture

**Transformer-Based Models (`team_transformer_model.py`)**
- Multi-agent transformer architecture treating ships as temporal tokens
- Unified team control: single model predicts actions for entire team
- Ship identity embeddings and temporal encoding
- Supports both policy and value heads for RL training

**Training Approaches**
- **Behavior Cloning**: Learn from scripted agent demonstrations
- **Reinforcement Learning**: PPO training with configurable opponents
- **Self-Play**: Dynamic opponent memory with model versioning
- **Mixed Training**: Combines scripted opponents and self-play

### Key Design Patterns

**Multi-Agent Coordination**
- Single forward pass generates coordinated actions for entire team
- Opponent modeling through shared attention mechanisms
- Implicit communication via shared model representations

**Modular Configuration**
- YAML-based configuration system (`unified_training.yaml`)
- Derived physics parameters (`derived_ship_parameters.yaml`)
- Flexible training regimes and hyperparameter management

**Scalable Architecture**
- Supports variable team sizes (1v1 to 4v4)
- Fractals-based ship positioning for balanced starts
- Efficient batched physics simulation across multiple agents

## Important Implementation Details

**Coordinate System**
- Uses complex numbers for 2D positions and velocities
- Attitude represented as unit complex numbers (direction vectors)
- Turn offsets maintain persistent state for orientation control

**Action Space**
- MultiBinary(6) actions: [forward, backward, left, right, sharp_turn, shoot]
- Turn state lookup table using bit patterns for efficient processing
- Instant response turning with velocity-relative attitude calculations

**Training Data Flow**
- BC training uses Monte Carlo returns for value function learning
- RL training supports configurable opponent curricula
- Self-play maintains rolling memory of previous model versions

**Performance Optimizations**
- GPU-accelerated physics simulation
- Vectorized operations across ship batches
- Lazy-loaded rendering for headless training
- Compressed data storage for large training datasets

## Configuration Management

The project uses a hierarchical YAML configuration system:

- **`src/unified_training.yaml`**: Main training configuration
- **`src/derived_ship_parameters.yaml`**: Physics parameters derived from simulation
- **`pytest.ini`**: Test configuration and filtering

Key configuration sections:
- `environment`: World size, timesteps, ship limits
- `model`: Transformer architecture and training hyperparameters  
- `training`: RL configuration, opponent types, evaluation frequency
- `data_collection`: Episode counts, game modes, output directories

## Testing Strategy

The test suite emphasizes physics accuracy and multi-agent correctness:

- **Environment tests**: Initialization, reset behavior, observation structure
- **Physics tests**: Force calculations, integration accuracy, collision detection
- **Combat tests**: Projectile mechanics, damage systems, health management
- **Agent tests**: Action validation, team coordination, opponent interaction

Tests use deterministic fixtures and configurable tolerances for reliable CI/CD integration.