# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

Aves-Horizons is a physics-based space combat simulation game designed for AI research and reinforcement learning. Ships engage in combat using realistic physics with thrust, turning, and projectile systems. The game serves as a testbed for multi-agent AI training, particularly for a novel transformer-based architecture that treats combat as temporal sequence prediction.

## Architecture Overview

### Core Components

**Environment System (`src/env.py`)**
- OpenAI Gym-compatible environment for RL training
- Multi-agent support with independent action/observation spaces
- Physics substeps at 50Hz with agent decisions at 10Hz
- Toroidal (wrap-around) world boundaries
- Memory system storing temporal snapshots for transformer training

**Ship System (`src/ship.py`)**
- Complex physics model with thrust, drag, and lift forces
- 6-action control: forward, backward, left, right, sharp_turn, shoot
- Turn offset system for velocity-relative attitude control
- Energy management with power consumption/regeneration
- Lookup tables for efficient force/coefficient calculations

**Projectile System (`src/bullets.py`)**
- High-performance bullet management with O(1) allocation
- Continuous collision detection for accurate hit registration at low FPS
- Vectorized position updates and collision checks

**Rendering System (`src/renderer.py`)**
- Pygame-based visualization with real-time human controls
- Multi-player support (Ship 0: WASD+Space, Ship 1: IJKL+O)
- Health/power visualization and team color coding

### Key Architectural Patterns

**Physics-First Design**: All game mechanics are physics-based rather than rule-based. Ships experience realistic thrust, drag, and lift forces with proper momentum conservation.

**Temporal Memory Architecture**: The environment maintains a deque of snapshots designed to support transformer models that learn from temporal sequences of ship states.

**Vectorized Operations**: Heavy use of NumPy for batch operations across multiple ships/bullets for performance.

## Development Commands

### Environment Setup
```bash
# Activate virtual environment (if not already active)
.\aves-horizons-env\Scripts\Activate.ps1

# Install dependencies (if needed)
pip install torch gymnasium numpy pygame
```

### Running the Game
```bash
# Human vs AI gameplay
python src/play.py

# Training mode (empty - needs implementation)
python src/train.py
```

### Testing
```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v
```

### Game Controls
- **Ship 0 (Human)**: Arrow Keys/WASD + Shift (sharp turn) + Space (shoot)
- **Ship 1 (Human)**: IJKL + U (sharp turn) + O (shoot)

## Critical Implementation Details

### Physics Integration
- Uses adaptive ODE integration for stability at variable framerates
- Physics runs at `physics_dt = 0.02s` (50 FPS) with agent decisions at `agent_dt = 0.04s` (25 FPS)
- Turn offset system allows ships to face different directions than velocity (aerodynamic turning)

### Action Space Architecture
Actions use MultiBinary(6) encoding where simultaneous button presses create emergent behaviors:
- Forward + Backward = Base thrust (cancels out)  
- Left + Right = Maintain current turn offset
- Sharp turn modifier changes turn angles and aerodynamic coefficients

### Memory and Observation Design
The environment maintains temporal history specifically for transformer training:
- `memory_size` parameter controls snapshot retention
- Observations include self state, enemy state, nearby bullets, and world info
- Designed to support the transformer architecture detailed in `docs/model.md`

## Transformer AI Architecture

This codebase is designed to support a novel transformer-based multi-agent AI system described in `docs/model.md`. Key aspects:

**Token Design**: Each ship's state at each timestep becomes a token with position, velocity, attitude, health, power, and temporal information.

**Multi-Agent Training**: Single model predicts actions for all ships simultaneously, enabling natural team coordination without explicit communication protocols.

**Temporal Understanding**: The model learns from sequences of game states to understand multi-step dynamics and opponent behavior patterns.

## Development Guidelines

### Code Style
- Use type hints throughout (established pattern in existing code)
- Use modern type hints such as `dict` and `None` instead of `Dict` and `Optional`
- Dataclasses for configuration objects (`ShipConfig`)
- Complex numbers for 2D vectors (position, velocity, attitude)
- Lookup tables for performance-critical calculations

### Performance Considerations
- All physics calculations are vectorized using NumPy
- Bullet system uses efficient memory management with free lists
- Collision detection optimized for batch processing
- GPU acceleration supported via PyTorch tensors

### Testing Strategy
- Pytest configuration in `pytest.ini` with verbose output and short tracebacks
- Tests should verify both discrete and continuous collision detection modes
- Performance benchmarks for different frame rates and ship counts

## Physics Configuration

Default ship parameters are defined in `ShipConfig` class:
- **Base Thrust**: 10.0 (continuous thrust)
- **Boost Thrust**: 80.0 (when forward pressed)
- **Turn Angles**: 5° (normal), 15° (sharp)
- **Collision Radius**: 10.0
- **Bullet Speed**: 500.0 with 12.0 spread
- **Energy Costs**: Forward costly, backward regenerative

## Key Research Features

### Multi-Agent Coordination  
The architecture supports emergent team coordination through shared model training rather than explicit communication protocols.

### Temporal Sequence Learning
Environment design specifically supports models that learn from temporal sequences of multi-agent interactions.

## Future Development

The transformer architecture roadmap includes:
1. **MVP**: Basic temporal sequence learning with fixed scenarios
2. **Enhanced**: Opponent modeling and self-play training  
3. **Full Vision**: Emergent team coordination and diverse AI personalities

Training infrastructure needs implementation in `src/train.py` to support this research direction.