# Snake logic that avoids walls and prefers food
import random
import numpy as np

def choose_action(observation, info):
    """
    Returns an action (0=up, 1=down, 2=left, 3=right) that avoids walls
    and prefers moving toward food when possible.
    """
    # Use the observation matrix to get game state
    try:
        # Extract board dimensions and position info from observation
        height, width, _ = observation.shape
        
        # Find snake head position for our snake
        head_y, head_x = np.where(observation[:, :, 1] == 5)
        if len(head_y) == 0:  # If snake is dead, return random move
            return random.randint(0, 3)
            
        head_y, head_x = head_y[0], head_x[0]
        
        # Possible moves: up, down, left, right
        possible_moves = []
        
        # Check which moves are safe (not hitting walls)
        # UP = 0
        if head_y > 0:
            possible_moves.append(0)
        # DOWN = 1
        if head_y < height - 1:
            possible_moves.append(1)
        # LEFT = 2
        if head_x > 0:
            possible_moves.append(2)
        # RIGHT = 3
        if head_x < width - 1:
            possible_moves.append(3)
            
        # If we're in a corner with no safe moves, pick any direction
        if not possible_moves:
            return random.randint(0, 3)
            
        # Also avoid hitting snake bodies (both our snake and other snakes)
        safe_moves = []
        for move in possible_moves:
            new_y, new_x = head_y, head_x
            
            if move == 0:  # UP
                new_y -= 1
            elif move == 1:  # DOWN
                new_y += 1
            elif move == 2:  # LEFT
                new_x -= 1
            elif move == 3:  # RIGHT
                new_x += 1
                
            # Check for snake bodies (any channel except food channel)
            is_safe = True
            for i in range(1, observation.shape[2]):
                if observation[new_y, new_x, i] in [1, 5]:  # Body or head
                    is_safe = False
                    break
                    
            if is_safe:
                safe_moves.append(move)
                
        if safe_moves:
            possible_moves = safe_moves
            
        # Check for food location to prefer food-seeking moves
        food_y, food_x = np.where(observation[:, :, 0] == 1)
        if len(food_y) > 0:
            # Find closest food
            min_dist = float('inf')
            closest_food = None
            
            for fy, fx in zip(food_y, food_x):
                dist = abs(fy - head_y) + abs(fx - head_x)  # Manhattan distance
                if dist < min_dist:
                    min_dist = dist
                    closest_food = (fy, fx)
                    
            if closest_food:
                food_y, food_x = closest_food
                
                # Prefer moves that get us closer to food
                food_moves = []
                if head_y > food_y and 0 in possible_moves:  # Food is above
                    food_moves.append(0)  # UP
                elif head_y < food_y and 1 in possible_moves:  # Food is below
                    food_moves.append(1)  # DOWN
                if head_x > food_x and 2 in possible_moves:  # Food is to the left
                    food_moves.append(2)  # LEFT
                elif head_x < food_x and 3 in possible_moves:  # Food is to the right
                    food_moves.append(3)  # RIGHT
                    
                if food_moves:
                    # 80% chance to move towards food
                    if random.random() < 0.8:
                        return random.choice(food_moves)
        
        # Otherwise, pick a random safe move
        return random.choice(possible_moves)
        
    except Exception as e:
        # Fallback to random moves if any error occurs
        return random.randint(0, 3)
