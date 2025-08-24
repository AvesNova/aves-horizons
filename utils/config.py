class Config:
    # Physics parameters
    DT = 0.1
    DRAG_COEFFICIENT = 0.05
    TURNING_PENALTY = 0.7
    
    # Ship parameters
    SHIP_ACCELERATION = 5.0
    SHIP_ROTATION_SPEED = 3.0
    SHIP_RADIUS = 10.0
    SHIP_MAX_HEALTH = 100.0
    SHIP_FIRE_COOLDOWN = 0.5
    PROJECTILE_SPEED = 300.0
    PROJECTILE_LIFETIME = 2.0
    
    # Environment parameters
    WORLD_SIZE = (800, 600)
    DEFAULT_N_SHIPS = 2
    DEFAULT_N_OBSTACLES = 5