# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

**Boost and Broadside** is a physics-based ship combat simulation with transformer-based AI agents. This is a reinforcement learning (RL) research project implementing a custom Gymnasium environment where ships battle in 2D space using realistic physics (thrust, drag, lift). The core ML component is a transformer model that learns team coordination via behavior cloning (BC) pretraining followed by PPO RL training.

**Package Name:** `aves-horizons`

## Development Commands

### Setup and Installation
```powershell
# Create virtual environment (project uses ./venv/)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install package in development mode with dev dependencies
pip install -e ".[dev]"
```

### Running Tests
```powershell
# Run all tests with pytest (preferred method)
pytest tests

# Run a specific test file
pytest tests/test_environment_basics.py

# Run a specific test function
pytest tests/test_environment_basics.py::test_reset

# Run tests with verbose output
pytest -v tests

# Run with short traceback (default configuration)
pytest --tb=short tests
```

### Training and Data Collection

#### Collect Behavior Cloning Data
```powershell
# Collect BC training data (scripted vs scripted)
python src/collect_data.py collect_bc --config src/unified_training.yaml
```

#### Train Models
```powershell
# Full pipeline: BC pretraining + RL training
python src/unified_train.py full --config src/unified_training.yaml

# BC pretraining only
python src/unified_train.py bc --config src/unified_training.yaml

# RL training only (from scratch)
python src/unified_train.py rl --config src/unified_training.yaml

# RL training from BC checkpoint
python src/unified_train.py rl --config src/unified_training.yaml --bc-model checkpoints/bc_model.pt
```

#### Human Play & Playback
```powershell
# Play against scripted opponent (requires pygame)
python src/collect_data.py human_play

# Replay recorded episode
python src/collect_data.py playback --episode-file data/bc_pretraining/1v1_episodes.pkl.gz
```

### Code Quality
```powershell
# Format code with black
black src tests

# Check formatting without changes
black --check src tests
```

## Architecture Overview

### Core System Components

**Environment System (`env.py`, `state.py`)**
- `Environment`: Main Gymnasium-compatible environment implementing the simulation loop
- `State`: Immutable snapshot of game state (ships, bullets) at a given time
- Physics runs at `physics_dt` (default 0.02s), agents act at `agent_dt` (default 0.04s)
- Supports multiple game modes: 1v1, 2v2, 3v3, 4v4, nvn (variable team sizes)
- Ships positioned using fractal pattern generation for multi-ship scenarios

**Ship Physics (`ship.py`)**
- Complex aerodynamic model with thrust, drag, and lift forces
- Lookup tables for action combinations (forward/backward, left/right, sharp turn)
- Power management system (boosting drains power, idling recharges)
- Bullet firing with cooldown and energy cost
- All ships must have non-zero initial velocity (enforced by assertion)

**Agent System (`agents.py`)**
- Unified `Agent` interface: `get_actions(obs_dict, ship_ids) -> dict[int, Tensor]`
- Multiple agent types:
  - `ScriptedAgentProvider`: Rule-based targeting and maneuvering
  - `RLAgentProvider`: Wraps trained transformer or PPO models
  - `HumanAgentProvider`: Keyboard controls via pygame renderer
  - `RandomAgentProvider`: Random action sampling
  - `SelfPlayAgent`: Maintains memory of past model versions for self-play
  - `PlaybackAgent`: Replays recorded episode actions

### Machine Learning Pipeline

**Transformer Model (`team_transformer_model.py`)**
- `TeamTransformerModel`: Core neural network architecture
  - Input: Ship tokens (10-dim vectors: health, power, position, velocity, attitude)
  - Processing: Token projection → Multi-head self-attention → Action head
  - Output: Action logits for all ships (6 binary actions per ship: forward, backward, left, right, sharp_turn, shoot)
  - Uses attention masking to handle variable team sizes and dead ships
- `TeamController`: Maps ship IDs to team assignments for multi-team control

**Training Pipeline (`unified_train.py`, `bc_training.py`)**
1. **Phase 1: Behavior Cloning (BC)**
   - Collect expert demonstrations (scripted vs scripted battles)
   - Train transformer to imitate scripted agent via supervised learning
   - Outputs: BC model checkpoint + training metrics
   
2. **Phase 2: Reinforcement Learning (PPO)**
   - Initialize from BC model (optional but recommended)
   - Train using Stable-Baselines3 PPO with custom transformer policy
   - Opponent types: scripted, self-play, or mixed
   - Self-play memory maintains pool of past model versions

**RL Wrapper (`rl_wrapper.py`)**
- `UnifiedRLWrapper`: Bridges Environment with SB3's PPO
  - Observation space: (max_ships, token_dim) token matrix
  - Action space: MultiBinary for controlled ships' actions
  - Handles team assignments and opponent switching
  - Tracks win/loss statistics and episode types

**Custom SB3 Policy (`transformer_policy.py`)**
- `TransformerFeaturesExtractor`: Extracts ship embeddings via transformer
- `TransformerActorCriticPolicy`: Actor-critic with transformer backbone
- Integrates with PPO for policy gradient training

### Data Collection & Playback

**Game Runner (`game_runner.py`)**
- `UnifiedGameRunner`: Central orchestrator for all game modes
  - Data collection for BC training
  - Human play sessions
  - Episode playback with speed control
  - RL evaluation
- Manages team assignments and agent coordination

**Data Collection (`collect_data.py`)**
- Collects episodes with observations, actions, rewards, and outcomes
- Computes Monte Carlo returns for BC training
- Saves compressed episode files (`.pkl.gz`)
- Supports checkpointing per game mode

## Project Structure

```
src/
├── env.py                      # Gymnasium environment
├── state.py                    # Game state representation
├── ship.py                     # Ship physics and configuration
├── bullets.py                  # Bullet system
├── constants.py                # Action indices, rewards, etc.
├── agents.py                   # Unified agent interface
├── scripted_agent.py           # Rule-based AI opponent
├── team_transformer_model.py   # Core transformer architecture
├── transformer_policy.py       # SB3 custom policy
├── rl_wrapper.py               # RL training wrapper
├── unified_train.py            # Training pipeline entry point
├── bc_training.py              # Behavior cloning implementation
├── collect_data.py             # Data collection entry point
├── game_runner.py              # Game orchestration
├── playback_agent.py           # Episode replay
├── renderer.py                 # Pygame visualization
├── callbacks.py                # Training callbacks (self-play, etc.)
└── unified_training.yaml       # Default configuration

tests/
├── conftest.py                 # Shared fixtures
├── test_environment_*.py       # Environment tests
├── test_ship_*.py              # Ship physics tests
├── test_bullets.py             # Bullet system tests
├── test_agents_integration.py  # Agent system tests
├── test_gym_compliance.py      # Gymnasium API compliance
└── test_*.py                   # Additional test modules
```

## Type Hinting Standards

- Use modern union syntax: `dict | None` instead of `Optional[Dict]`
- Type hint all function parameters and return values
- Complex types: `dict[int, torch.Tensor]`, `list[int]`, `tuple[float, float]`

## Key Observations & Domain Knowledge

### Observation Space Structure
The observation dictionary contains:
- `tokens`: (max_ships, 10) tensor of ship state vectors
- `alive`: (max_ships, 1) binary alive/dead indicators
- Additional keys for specific use cases

Ship token format (10 dimensions):
1. Health (0-100)
2. Power (0-100)
3-6. Position and velocity (x, y, vx, vy)
7-10. Attitude and other derived features

### Action Space
6 binary actions per ship:
- `forward` (0): Boost forward
- `backward` (1): Reverse thrust
- `left` (2): Turn left
- `right` (3): Turn right
- `sharp_turn` (4): Sharp turn modifier
- `shoot` (5): Fire bullet

### Team Assignments
Teams are represented as: `dict[int, list[int]]` where keys are team IDs and values are lists of ship IDs.
Example: `{0: [0, 1], 1: [2, 3]}` means team 0 controls ships 0-1, team 1 controls ships 2-3.

### Configuration System
The project uses YAML configuration files (default: `src/unified_training.yaml`) for all hyperparameters:
- Environment parameters (world size, time steps, max ships)
- Model architecture (transformer dimensions, heads, layers)
- Training parameters (learning rates, batch sizes, episodes)
- Opponent configuration (scripted/self-play mix ratios)

### Physics Simulation
- Ships cannot be stationary (enforced in `Ship.__init__`)
- Multiple physics substeps per agent decision for stability
- Drag and lift coefficients vary by turn state
- Power management creates strategic trade-offs (boost vs sustained fire)

### Self-Play System
- Maintains memory buffer of past model checkpoints
- Periodically updates opponent with current policy
- Mixed opponent mode combines scripted and self-play for curriculum learning
- `SelfPlayCallback` handles opponent updates during training

## Common Development Tasks

### Adding a New Agent Type
1. Create class inheriting from `Agent` in `agents.py`
2. Implement `get_actions(obs_dict, ship_ids)` method
3. Implement `get_agent_type()` method
4. Add factory function `create_*_agent()` at bottom of file

### Modifying Ship Physics
1. Update `ShipConfig` dataclass in `ship.py`
2. Modify force calculation in `Ship._calculate_forces()`
3. Update lookup tables in `Ship._build_lookup_tables()` if needed
4. Add tests in `tests/test_ship_physics.py`

### Adding New Reward Signals
1. Add reward constants to `constants.py`
2. Implement reward calculation in `Environment.step()`
3. Update reward tracking in game runners

### Debugging Training Issues
- Check tensorboard logs in `logs/` directory
- Review checkpoint files in `checkpoints/` directory
- Use human play mode to visualize agent behavior
- Use playback mode to analyze specific episodes
- Lower `agent_dt` or `physics_dt` if physics seems unstable

## VSCode Configuration

The project includes VSCode launch configurations (`.vscode/launch.json`):
- **Play**: Launch human play mode
- **Replay**: Playback recorded episodes
- **Collect BC**: Collect behavior cloning data
- **Train**: Run full training pipeline
- **Current File**: Debug currently open file

Default Python interpreter: `./venv/Scripts/python.exe`
