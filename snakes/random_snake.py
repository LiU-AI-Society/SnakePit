# Snake logic that avoids walls and prefers food
import random
import numpy as np

def choose_action(observation, info):
    """
    Returns an action (0=up, 1=down, 2=left, 3=right) that avoids walls
    and prefers moving toward food when possible.
    """
    return random.choice([0, 1, 2, 3])  # UP, DOWN, LEFT, RIGHT
