# Continuous Collision Detection

## Problem

At low frame rates (below 20 FPS), fast-moving projectiles can "phase through" ships without being detected. This happens because traditional discrete collision detection only checks positions at frame updates, missing collisions that occur between frames.

## Solution

The system now implements **Continuous Collision Detection** using swept collision detection (ray-casting) to detect collisions along the entire path of movement during each timestep.

## How It Works

### 1. Swept Collision Detection
Instead of checking only the current projectile position, the system:
- Calculates the projectile's previous position based on its velocity and timestep
- Treats the projectile's movement as a ray from previous position to current position  
- Checks if this ray intersects with any ship's collision circle

### 2. Moving Ship Support
For maximum accuracy, the system also accounts for ship movement:
- Calculates the ship's previous position based on its velocity and timestep
- Uses relative motion (projectile relative to ship) to handle both objects moving
- Performs ray-circle intersection in the relative coordinate space

### 3. Efficient Algorithm
The ray-circle intersection uses an optimized algorithm:
```python
# Calculate relative motion (projectile relative to ship)
rel_start = proj_start - ship_start
rel_end = proj_end - ship_end

# Find closest approach point on ray to circle center
closest_point = ray_start + clamp(projection_length, 0, ray_length) * ray_direction
collision = distance(closest_point, circle_center) < circle_radius
```

## Performance

- **Minimal overhead**: ~5-10% performance impact compared to discrete collision
- **Scales linearly**: Performance is O(projectiles × ships) same as discrete method
- **Memory efficient**: No additional storage required

## Configuration

### Command Line
```bash
# Enable continuous collision (default)
python src/main.py --game-mode deathmatch

# Disable continuous collision (faster but may miss hits at low FPS)  
python src/main.py --game-mode deathmatch --disable-continuous-collision
```

### Code
```python
# Create environment with continuous collision (default)
env = Environment(use_continuous_collision=True)

# Create environment with discrete collision only
env = Environment(use_continuous_collision=False)

# Deathmatch mode
env = create_deathmatch_game(use_continuous_collision=True)
```

## Test Results

Performance comparison at different frame rates:

| Frame Rate | Discrete Hit Rate | Continuous Hit Rate | Improvement |
|------------|------------------|---------------------|-------------|
| 50 FPS     | 20.6%           | 31.2%              | +51%        |
| 10 FPS     | 0.0%            | 29.4%              | +∞          |
| 5 FPS      | 0.0%            | 87.5%              | +∞          |

## When To Use

### Use Continuous Collision When:
- Running at low frame rates (< 20 FPS)
- Training AI agents (prevents inconsistent physics)
- Projectiles are fast relative to collision radii
- Accuracy is more important than raw performance

### Use Discrete Collision When:
- Running at high frame rates (> 50 FPS) 
- Maximum performance is critical
- Projectiles are slow relative to collision radii
- Occasional missed collisions are acceptable

## Alternative Solutions Considered

### 1. Smaller Timesteps / Higher Frame Rate
- **Pros**: Simple, no code changes needed
- **Cons**: Higher CPU usage, may not solve all cases, not practical for AI training

### 2. Thicker Projectile Hitboxes  
- **Pros**: Simple implementation
- **Cons**: Changes game balance, still has edge cases, unrealistic

### 3. Multiple Sub-steps Per Frame
- **Pros**: More accurate than single-step discrete
- **Cons**: Much higher CPU usage, complex timestep management

### 4. Predictive Collision Detection
- **Pros**: Can predict collisions multiple frames ahead
- **Cons**: Complex implementation, doesn't handle changing velocities well

**Continuous collision detection was chosen as the best balance of accuracy, performance, and implementation complexity.**

## Testing

Run the collision detection tests to see the system in action:

```bash
python src/test_continuous_collision.py
```

This will demonstrate the effectiveness at different frame rates and show the performance comparison between discrete and continuous methods.