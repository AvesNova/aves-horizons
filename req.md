# Ship Game Requirements Document

## Overview
A physics-based ship simulation game with OpenAI Gym compatibility for reinforcement learning. Ships are controlled through thrust and turning actions with realistic physics including aerodynamic forces. The simulation uses adaptive ODE integration via torchdiffeq for accurate physics modeling.

## Ship System

### Ship State Variables
Each ship maintains the following state:

- **Position**: Current position in world coordinates (complex tensor)
- **Velocity**: Current velocity vector (complex tensor)
- **Attitude**: Current facing direction (complex tensor, unit vector)
- **Turn Offset**: Current turn angle offset from velocity direction (real tensor, persistent state)
- **Boost**: Current boost energy remaining (real tensor)
- **Health**: Current health points (real tensor)

### Ship Parameters
Each ship has configurable physics parameters:

#### Thrust System
- **Base Thrust**: Base thrust force applied continuously (even when no input pressed)
- **Forward Boost**: Multiplier applied to base thrust when forward is pressed
- **Backward Boost**: Multiplier applied to base thrust when backward is pressed
- **Base Energy Cost**: Energy consumed per timestep when neither forward or backwards in pressed.
- **Forward Energy Cost**: Energy consumed per timestep when forward is pressed
- **Backward Energy Cost**: Energy consumed per timestep when backward is pressed (can be negative for regen)

#### Aerodynamic Parameters
- **No Turn Drag**: Drag coefficient when flying straight
- **Normal Turn Angle**: Angle offset for normal turns (default: 5°)
- **Normal Turn Lift Coefficient**: Lift force during normal turns
- **Normal Turn Drag Coefficient**: Additional drag during normal turns
- **Sharp Turn Angle**: Angle offset for sharp turns (default: 15°)
- **Sharp Turn Lift Coefficient**: Lift force during sharp turns
- **Sharp Turn Drag Coefficient**: Additional drag during sharp turns

#### Physical Properties
- **Collision Radius**: Radius for collision detection
- **Max Boost**: Maximum boost energy capacity
- **Max Health**: Maximum health points

## Control System

### Action Space (Gym Compatible)
Each ship is controlled by a MultiBinary(5) action vector:
- **Index 0 - Left (L)**: Turn left
- **Index 1 - Right (R)**: Turn right
- **Index 2 - Forward (F)**: Apply forward thrust multiplier
- **Index 3 - Backward (B)**: Apply backward thrust multiplier
- **Index 4 - Sharp Turn (S)**: Modifier to use sharp turn angles

### Orientation Control System

The ship's attitude is computed as velocity direction multiplied by the exponential of the turn offset angle. The turn offset state variable is updated based on button combinations using a lookup table:

| Bit Pattern (SLR) | Button Combination | Turn Offset | Description |
|-------------------|-------------------|-------------|-------------|
| 000 | None | 0° | Face velocity direction |
| 001 | R | -normal_angle | Normal right turn |
| 010 | L | +normal_angle | Normal left turn |
| 011 | LR | Previous | Maintain current offset |
| 100 | S | 0° | Face velocity direction |
| 101 | SR | -sharp_angle | Sharp right turn |
| 110 | SL | +sharp_angle | Sharp left turn |
| 111 | SLR | Previous | Maintain current offset |

#### Orientation Rules
1. **Instant Response**: Turn offset updates instantly when inputs change
2. **Velocity-Based**: Attitude is always relative to velocity direction
3. **State Persistence**: Turn offset is maintained as persistent state variable
4. **Speed Threshold**: Turning is ignored below minimum velocity (1e-6)
5. **Priority System**: When both L and R are pressed, maintain current turn offset

## Physics System

### Force Calculation

#### Thrust Forces
The thrust system applies base thrust continuously in the ship's attitude direction. When forward or backward inputs are pressed, the base thrust is multiplied by the corresponding boost multiplier. The effective thrust force is always in the direction of the ship's current attitude. Simultanious presses of forward and backward is the same as neither being pressed.

#### Energy Consumption
Energy is consumed or regenerated based on forward and backward actions. Each action has an associated energy cost per timestep. Forward actions typically consume energy while backward actions may provide regenerative braking (negative energy cost). There is also a base energy gain. Boost energy is clamped between zero and maximum capacity.

#### Aerodynamic Forces
Ships experience drag forces opposing their velocity and lift forces perpendicular to velocity when turning. Drag force magnitude is proportional to velocity squared and selected drag coefficient. Lift forces are generated during turns and are proportional to both velocity squared and the selected lift coefficient.

#### Coefficient Selection
Drag and lift coefficients are selected based on current turn state:
- **No Turn**: Uses no-turn drag coefficient, zero lift
- **Normal Turn**: Uses normal turn drag and lift coefficients
- **Sharp Turn**: Uses sharp turn drag and lift coefficients

### Integration via torchdiffeq

The physics simulation uses adaptive ODE integration for accurate and stable dynamics. The system state consists of position and velocity components for all ships. The dynamics function calculates force-based accelerations for integration. Adaptive timestep methods automatically adjust step size for optimal accuracy and performance.

#### Integration Configuration
- **Solver Methods**: Support for various ODE solvers (dopri5, euler, rk4, adaptive_heun)
- **Tolerance Settings**: Configurable relative and absolute tolerances
- **Step Control**: Maximum step count and initial step size limits
- **Batch Processing**: All ships integrated simultaneously for efficiency

## OpenAI Gym Environment

### Environment Interface
The game implements a standard Gym environment with multi-agent support. The environment manages multiple ships simultaneously with independent action spaces and shared observation space.

### Action Space
Multi-agent action space using MultiBinary for each ship. Total action dimension is 5 times the number of ships. Each ship's actions are decoded independently for physics simulation.

### Observation Space
Observations include per-ship state information and relative positions of other ships:

**Per Ship State (9 dimensions)**:
- Position coordinates (x, y)
- Velocity components (vx, vy) 
- Attitude direction (cos θ, sin θ)
- Turn offset angle
- Boost energy (normalized 0-1)
- Health points (normalized 0-1)

### Reward Function
Modular reward system supporting multiple objectives:
- **Survival Reward**: Positive reward for staying alive each timestep
- **Energy Efficiency**: Penalty proportional to energy consumption
- **Collision Penalty**: Large negative reward for taking damage
- **Task-Specific Rewards**: Racing checkpoints, combat elimination, exploration coverage

### Episode Management
Episodes terminate when ships are destroyed, time limits are reached, or task objectives are completed. Multi-agent environments support independent termination where ships can be eliminated while others continue. Truncation occurs at maximum episode length regardless of task completion.

## Technical Implementation

### Data Architecture
Ships are represented as dataclass structures containing both state variables and physics parameters. All calculations use PyTorch tensors for vectorized operations across multiple ships. Complex number representations efficiently handle 2D position and velocity vectors.

### Physics Pipeline
The physics system follows a clear pipeline: action extraction, turn offset updates, orientation calculation, force computation, and ODE integration. Each stage operates on batched tensor data for optimal performance.

### Performance Optimizations
- **Vectorized Operations**: All physics calculations batched across ships
- **GPU Acceleration**: Full PyTorch tensor operations support GPU execution
- **Lookup Tables**: Bit arithmetic for efficient turn state resolution
- **Sparse Updates**: Turn offset only updated when input states change
- **Memory Efficiency**: Reuse of tensor allocations and in-place operations where possible

## Default Configuration

### Physics Parameters
- **Thrust**: 300.0 (base continuous thrust)
- **Forward Boost**: 3.0 (multiplier when forward pressed)
- **Backward Boost**: 2.0 (multiplier when backward pressed)
- **Forward Energy Cost**: 5.0 (energy consumed per timestep)
- **Backward Energy Cost**: -1.0 (regenerative braking)
- **No Turn Drag**: 0.008 (baseline drag coefficient)
- **Normal Turn Angle**: 5° (standard turn offset)
- **Normal Turn Lift**: 1.0 (lift coefficient for normal turns)
- **Normal Turn Drag**: 0.01 (additional drag when turning)
- **Sharp Turn Angle**: 15° (enhanced turn offset)
- **Sharp Turn Lift**: 1.5 (increased lift for sharp turns)
- **Sharp Turn Drag**: 0.03 (higher drag penalty for sharp turns)
- **Collision Radius**: 10.0 (collision detection radius)
- **Max Boost**: 100.0 (maximum energy capacity)
- **Max Health**: 100.0 (maximum health points)

### Integration Settings
- **Target Timestep**: 0.016 seconds (60 FPS equivalent)
- **Default Solver**: dopri5 (adaptive Runge-Kutta)
- **Relative Tolerance**: 1e-7 (accuracy for adaptive methods)
- **Absolute Tolerance**: 1e-9 (minimum accuracy threshold)
- **Max Steps**: 1000 (integration step limit per timestep)

### Environment Settings
- **World Size**: 800x600 pixels (default arena dimensions)
- **Ship Count**: 4 (default multi-agent scenario)
- **Episode Length**: 1000 steps (maximum episode duration)
- **Render Modes**: human, rgb_array (visualization options)

## World System

### Boundaries
World boundaries can be configured as walls (collision), wrap-around (toroidal), or open space (unlimited). Collision boundaries instantly destroy the ship when hit.

### Multi-Ship Interactions
Ships interact through collision detection using configurable collision radii. Collisions can result in health damage, momentum transfer, or elimination depending on game mode settings.

### Environmental Effects
Support for environmental factors such as gravity wells, wind resistance, or area-specific physics modifications. These effects modify the base physics calculations during force computation.

## Game Modes

### Training Scenarios
- **Free Flight**: Open exploration with survival objectives
- **Racing**: Checkpoint-based navigation challenges
- **Combat**: Last-ship-standing elimination matches
- **Cooperative**: Multi-ship collaborative objectives
- **Obstacle Course**: Navigation through static hazards

### Evaluation Metrics
- **Performance**: Task completion time and efficiency
- **Robustness**: Success rate across various scenarios
- **Energy Management**: Optimal boost usage strategies
- **Multi-Agent Coordination**: Cooperative behavior emergence

## Future Extensions

### Advanced Features
- **Weapons System**: Projectiles with ballistic physics integration
- **Environmental Hazards**: Dynamic obstacles and area effects
- **Team Dynamics**: Alliance formation and competitive scenarios
- **Hierarchical Control**: High-level strategy combined with low-level piloting
- **Procedural Scenarios**: Automatically generated training environments

### Research Applications
- **Curriculum Learning**: Progressive difficulty training pipelines
- **Multi-Agent Learning**: Cooperative and competitive behavior study
- **Transfer Learning**: Skill transfer between different ship configurations
- **Real-time Strategy**: Higher-level planning combined with precise control
- **Human-AI Interaction**: Mixed human and AI pilot scenarios