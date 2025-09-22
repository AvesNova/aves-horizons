from enum import IntEnum, auto


class Actions(IntEnum):
    """Action indices for ship control."""

    forward = 0
    backward = auto()
    left = auto()
    right = auto()
    sharp_turn = auto()
    shoot = auto()


class RewardConstants:
    VICTORY_REWARD = 1.0
    DEFEAT_REWARD = -1.0
    DRAW_REWARD = 0.0

    ALLY_DEATH_PENALTY = -0.1
    ENEMY_DEATH_BONUS = 0.1

    DAMAGE_REWARD_SCALE = 0.001  # Per damage point

    # Test-specific constants
    TEST_DAMAGE_AMOUNT = 25.0
    TEST_LARGE_DAMAGE = 50.0
    TEST_SMALL_DAMAGE = 5.0
