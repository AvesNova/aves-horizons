from enum import IntEnum, auto


class Actions(IntEnum):
    forward = 0
    backward = auto()
    left = auto()
    right = auto()
    sharp_turn = auto()
    shoot = auto()
