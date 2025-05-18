import numpy as np
import random

def choose_action(game_state, index):
    """
    Heuristic opponent that avoids walls, snake bodies (including itself),
    avoids reverse moves, and prefers food when visible.

    Returns:
        action (int): 0=up, 1=down, 2=left, 3=right
    """
    snakes = game_state['board']['snakes']
    my_snake = snakes[index]
    if not my_snake['body']:
        return random.randint(0, 3)  # Dead snake fallback

    head = my_snake['body'][0]
    board_w = game_state['board']['width']
    board_h = game_state['board']['height']

    directions = {
        0: (0, -1),  # up
        1: (0, 1),   # down
        2: (-1, 0),  # left
        3: (1, 0),   # right
    }

    # --- Identify reverse direction to avoid going backward ---
    reverse_dir = None
    if len(my_snake['body']) >= 2:
        neck = my_snake['body'][1]
        dx = neck['x'] - head['x']
        dy = neck['y'] - head['y']
        if dx == 1: reverse_dir = 2  # left
        elif dx == -1: reverse_dir = 3  # right
        elif dy == 1: reverse_dir = 0  # up
        elif dy == -1: reverse_dir = 1  # down

    # --- Check each move for safety ---
    safe_moves = []
    for action, (dx, dy) in directions.items():
        if action == reverse_dir:
            continue  # don’t reverse into own neck

        nx, ny = head['x'] + dx, head['y'] + dy

        # Check wall collision
        if not (0 <= nx < board_w and 0 <= ny < board_h):
            continue

        # Check collision with any snake body segment
        is_safe = True
        for snake in snakes:
            for segment in snake['body']:
                if segment['x'] == nx and segment['y'] == ny:
                    is_safe = False
                    break
            if not is_safe:
                break

        if is_safe:
            safe_moves.append(action)

    if not safe_moves:
        return random.randint(0, 3)  # No safe moves, panic randomly

    # --- Try to move toward food if safe ---
    food = game_state['board']['food']
    if food:
        target = food[0]  # Pick the first food for simplicity
        dx = target['x'] - head['x']
        dy = target['y'] - head['y']
        food_moves = []
        if dy < 0 and 0 in safe_moves: food_moves.append(0)
        if dy > 0 and 1 in safe_moves: food_moves.append(1)
        if dx < 0 and 2 in safe_moves: food_moves.append(2)
        if dx > 0 and 3 in safe_moves: food_moves.append(3)
        if food_moves and random.random() < 0.8:
            return random.choice(food_moves)

    # --- Pick a random safe move if no preferred food move ---
    return random.choice(safe_moves)
