\
import random
import traceback

SNAKE_NAME = "JakeTheSnake" # Added for better identification

# --- Constants ---
DEBUG = False  # Set to True to enable debug prints
HUNGRY = 60
FOOD_MIN = 1
STARVING = 40
FOOD_MAX = 40

# YOU MAY NOT USE THIS CODE FOR YOUR OWN SNAKE SINCE

# --- Global State ---
# This will be shared if multiple "jake_snake" instances are used in the same game,
# which might be an issue. A class-based approach or external state management
# would be needed for true instance-specific state.
ate_food_last_turn = False

# --- Action Mapping ---
# Project actions: 0=up, 1=down, 2=left, 3=right
# Jake's internal string moves: 'up', 'down', 'left', 'right'
ACTION_TO_STRING = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
STRING_TO_ACTION = {'up': 0, 'down': 1, 'left': 2, 'right': 3}

def debug_print(*args, **kwargs):
    if DEBUG:
        print("JAKE_SNAKE:", *args, **kwargs)

# --- Helper Functions (Adapted from battleJake2019) ---

def _convert_to_jake_coords(project_coords):
    """Converts project {'x': x, 'y': y} to Jake's (x, y) tuple."""
    return (project_coords['x'], project_coords['y'])

def _convert_body_to_jake_format(project_body):
    """Converts project body (list of dicts) to Jake's body (list of tuples)."""
    return [_convert_to_jake_coords(segment) for segment in project_body]

def _get_current_snake_data(game_state, index):
    """Prepares Jake's 'you' dictionary and other relevant data."""
    project_snakes = game_state['board']['snakes']
    my_project_snake = project_snakes[index]

    you = {}
    if not my_project_snake.get('body'): # Snake is dead or body is empty
        you['body'] = []
        you['head'] = None
        you['size'] = 0
        you['health'] = 0
        you['id'] = my_project_snake.get('id', f'snake_{index}')
        you['name'] = my_project_snake.get('name', f'Jake_{index}')
        return you, [], [], (0,0), 0 # Return empty/default values

    you['body'] = _convert_body_to_jake_format(my_project_snake['body'])
    you['head'] = you['body'][0]
    you['size'] = len(you['body'])
    you['health'] = my_project_snake.get('health', 100) # Default health if not present
    you['id'] = my_project_snake.get('id', f'snake_{index}')
    you['name'] = my_project_snake.get('name', f'Jake_{index}')


    walls = (game_state['board']['width'], game_state['board']['height'])
    
    all_snakes_jake_format = []
    for s_idx, s_data in enumerate(project_snakes):
        if not s_data.get('body'): continue # Skip dead or empty snakes
        snake_jake = {
            'body': _convert_body_to_jake_format(s_data['body']),
            'size': len(s_data['body']),
            'health': s_data.get('health', 100),
            'id': s_data.get('id', f'snake_{s_idx}'),
            'name': s_data.get('name', f'snake_{s_idx}')
        }
        snake_jake['head'] = snake_jake['body'][0]
        snake_jake['tail'] = snake_jake['body'][-1]
        all_snakes_jake_format.append(snake_jake)

    food_jake_format = [_convert_to_jake_coords(f) for f in game_state['board']['food']]
    turn = game_state.get('turn', 0)

    return you, all_snakes_jake_format, food_jake_format, walls, turn

def have_choice(move, moves):
    if move is not None:
        return False
    if len(moves) <= 1:
        return False
    return True

def get_space(space_tuple, move_str):
    """ space_tuple is (x,y), move_str is 'left', 'right', 'up', 'down' """
    x, y = space_tuple
    if move_str == 'left':
        return (x - 1, y)
    elif move_str == 'right':
        return (x + 1, y)
    elif move_str == 'up': # In Battlesnake, Y typically decreases for up
        return (x, y - 1)
    elif move_str == 'down': # Y typically increases for down
        return (x, y + 1)
    return space_tuple # Should not happen

def get_previous_move(head_tuple, second_segment_tuple):
    if head_tuple[0] == second_segment_tuple[0]: # Same X, vertical move
        if head_tuple[1] > second_segment_tuple[1]: # Head Y > Neck Y
            return 'down' # Moved down
        else:
            return 'up'   # Moved up
    else: # Same Y, horizontal move
        if head_tuple[0] > second_segment_tuple[0]: # Head X > Neck X
            return 'right' # Moved right
        else:
            return 'left'  # Moved left

def dont_hit_wall(current_moves_str, head_tuple, walls_tuple):
    """walls_tuple is (width, height)"""
    valid_moves = list(current_moves_str)
    width, height = walls_tuple
    head_x, head_y = head_tuple

    if head_x == width - 1 and 'right' in valid_moves:
        valid_moves.remove('right')
    elif head_x == 0 and 'left' in valid_moves:
        valid_moves.remove('left')

    if head_y == height - 1 and 'down' in valid_moves: # Max Y is height-1 for down
        valid_moves.remove('down')
    elif head_y == 0 and 'up' in valid_moves: # Min Y is 0 for up
        valid_moves.remove('up')
    return valid_moves

def dont_hit_snakes(current_moves_str, head_tuple, snakes_bodies_tuples, ignore_tuples):
    """
    snakes_bodies_tuples: list of (x,y) tuples representing all snake segments.
    ignore_tuples: list of (x,y) tuples to ignore (e.g., own tail).
    """
    valid_moves = list(current_moves_str)
    
    # Create a set of dangerous locations for faster lookup
    # Filter out ignored positions from snakes_bodies_tuples
    dangerous_positions = set(snakes_bodies_tuples) - set(ignore_tuples)

    for move_str in list(valid_moves): # Iterate over a copy
        next_head_tuple = get_space(head_tuple, move_str)
        if next_head_tuple in dangerous_positions:
            valid_moves.remove(move_str)
    return valid_moves

def dont_get_eaten(current_moves_str, you_dict, all_snakes_list_jake_format, sameSize=True):
    # This function is complex and relies on predicting opponent moves.
    # For a simpler heuristic, we can check immediate threats around potential next heads.
    # The original logic is quite involved. This is a simplified version.
    
    # Simplified: Avoid moving next to a larger or equal (if sameSize) snake's head if that move is one of their direct options.
    # This is still tricky. A more basic version: avoid squares adjacent to larger snakes' heads.
    
    candidate_moves = list(current_moves_str)
    
    for move_str in list(candidate_moves): # Iterate over a copy
        next_head_tuple = get_space(you_dict['head'], move_str)
        
        for other_snake in all_snakes_list_jake_format:
            if other_snake['id'] == you_dict['id']:
                continue

            can_be_eaten = (other_snake['size'] >= you_dict['size']) if sameSize else (other_snake['size'] > you_dict['size'])
            if not can_be_eaten:
                continue

            other_head_tuple = other_snake['head']
            
            # Check if our next_head_tuple is adjacent to other_snake's head
            # This means other_snake could potentially move into our next_head_tuple
            # Manhattan distance of 1 from other_head_tuple to our next_head_tuple
            if abs(next_head_tuple[0] - other_head_tuple[0]) + abs(next_head_tuple[1] - other_head_tuple[1]) == 1:
                # This is a potential collision spot if they move towards it.
                # A more robust check would see if *they* have a valid move to that spot.
                # For now, if we move there, and they are next to it, it's risky.
                if move_str in candidate_moves:
                     # This simplification might be too aggressive, removing too many moves.
                     # The original logic is more nuanced.
                     # Let's consider if their head is 2 units away, and could move to where we are going.
                     pass # Keeping it simple for now, original logic is very specific.


            # Original logic for `dont_get_eaten` checks if moving to a square
            # would put `you` in a position where an opponent snake could eat `you` on *their* next turn.
            # This involves checking squares around `next_head_tuple`.
            # Example: if `next_head_tuple` is (hx, hy), check (hx+1,hy), (hx-1,hy), (hx,hy+1), (hx,hy-1)
            # to see if an opponent head is there AND they are larger/equal.

            # Simplified: if any square adjacent to `next_head_tuple` is an opponent's head that can eat us.
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]: # Adjacent cells to our potential next head
                check_x, check_y = next_head_tuple[0] + dx, next_head_tuple[1] + dy
                if (check_x, check_y) == other_head_tuple:
                    if move_str in candidate_moves:
                        candidate_moves.remove(move_str)
                        break # Found a reason to remove this move_str
            if move_str not in candidate_moves:
                break # Move already removed by inner loop
    return candidate_moves


def get_food(current_moves_str, head_tuple, food_tuples, dist):
    valid_moves = [] # Prioritize food moves
    
    # Check if any current safe moves lead towards food within dist
    for move_str in current_moves_str:
        next_head_tuple = get_space(head_tuple, move_str)
        for f_tuple in food_tuples:
            # Manhattan distance from next_head_tuple to food
            manhattan_dist = abs(f_tuple[0] - next_head_tuple[0]) + abs(f_tuple[1] - next_head_tuple[1])
            
            # Original logic: if (abs(xdist) + abs(ydist)) <= dist:
            # Here, xdist, ydist are from head_tuple to food.
            # We should check if moving *towards* food is an option.
            
            # Simpler: if any food is reachable by one of the current_moves_str
            # and is closer or equally close than other options.
            
            # Let's use the original logic's spirit: find moves that take you to a square
            # from which food is within `dist` (or directly to food).
            # The original `get_food` returns moves that directly step towards food if food is within `dist`.

            food_x_dist = f_tuple[0] - head_tuple[0]
            food_y_dist = f_tuple[1] - head_tuple[1]

            if (abs(food_x_dist) + abs(food_y_dist)) <= dist: # Food is within range
                if food_x_dist > 0 and 'right' == move_str:
                    valid_moves.append('right')
                elif food_x_dist < 0 and 'left' == move_str:
                    valid_moves.append('left')
                elif food_y_dist > 0 and 'down' == move_str: # food is "down" (larger Y)
                    valid_moves.append('down')
                elif food_y_dist < 0 and 'up' == move_str: # food is "up" (smaller Y)
                    valid_moves.append('up')

    if not valid_moves:
        return current_moves_str # No specific food move found, return original safe moves
    return list(set(valid_moves)) # Return unique food-seeking moves among safe moves


def go_straight(current_moves_str, head_tuple, body_tuples):
    if len(body_tuples) > 1:
        second_segment_tuple = body_tuples[1]
        prev_move_str = get_previous_move(head_tuple, second_segment_tuple)
        if prev_move_str in current_moves_str:
            return prev_move_str # Return the string move
    return None # No straight move possible or preferred

def flee_wall(current_moves_str, walls_tuple, head_tuple):
    # This is a simplified interpretation of the original flee_wall
    # Prioritize moves away from walls if too close.
    width, height = walls_tuple
    head_x, head_y = head_tuple
    
    preferred_moves = []

    # If against a wall, the only move in that direction is to flee
    if head_x == width - 1 and 'left' in current_moves_str: return ['left']
    if head_x == 0 and 'right' in current_moves_str: return ['right']
    if head_y == height - 1 and 'up' in current_moves_str: return ['up']
    if head_y == 0 and 'down' in current_moves_str: return ['down']

    # Try to maintain a buffer if not directly against the wall
    # Original logic is more complex with buffer of 2.
    # Simplified: if near wall, prefer moves away from it among current_moves_str.
    
    temp_moves = list(current_moves_str)

    if head_x >= width - 2 and 'right' in temp_moves: # Near right wall
        temp_moves.remove('right')
    if head_x <= 1 and 'left' in temp_moves: # Near left wall
        temp_moves.remove('left')
    if head_y >= height - 2 and 'down' in temp_moves: # Near bottom wall
        temp_moves.remove('down')
    if head_y <= 1 and 'up' in temp_moves: # Near top wall
        temp_moves.remove('up')
        
    return temp_moves if temp_moves else current_moves_str


def eat_others(current_moves_str, head_tuple, my_size, all_snakes_list_jake_format):
    # Simplified: move towards a smaller snake's head if it's very close.
    # Original logic is more aggressive.
    valid_moves = []
    for move_str in current_moves_str:
        next_head_tuple = get_space(head_tuple, move_str)
        for other_snake in all_snakes_list_jake_format:
            if other_snake['size'] < my_size -1: # Original condition
                # If moving to next_head_tuple lands on their head (or very close)
                # This is a direct attack move.
                # Check if next_head_tuple is where other_snake's head is, or adjacent to it.
                # Original logic checks for various proximities (1 or 2 steps)
                
                other_head_x, other_head_y = other_snake['head']
                
                # If our next move is onto their head
                if next_head_tuple == other_snake['head']:
                    valid_moves.append(move_str)
                    continue

                # If our next move is adjacent to their head (setting up for next turn)
                # This is part of the original logic (dist == 1 for x and y, or dist == 2 for one axis)
                x_dist_to_other_head = other_head_x - next_head_tuple[0]
                y_dist_to_other_head = other_head_y - next_head_tuple[1]

                if (abs(x_dist_to_other_head) == 1 and y_dist_to_other_head == 0) or \
                   (abs(y_dist_to_other_head) == 1 and x_dist_to_other_head == 0):
                    # If we move, their head is one step away from our new head
                    valid_moves.append(move_str)

    if not valid_moves:
        return current_moves_str
    return list(set(valid_moves))


def ate_food(head_tuple, food_tuples, chosen_move_str):
    if not chosen_move_str: return False
    next_head_tuple = get_space(head_tuple, chosen_move_str)
    return next_head_tuple in food_tuples

# --- Main Action Function ---
def choose_action(game_state, index):
    global ate_food_last_turn # Declare usage of global

    you, all_snakes_jake, food_jake, walls_jake, turn_jake = _get_current_snake_data(game_state, index)

    if not you.get('body'): # Our snake is dead or not properly initialized
        debug_print("Jake: My snake data is invalid/empty. Random move.")
        return random.randint(0, 3)
    
    debug_print(f"--- Turn {turn_jake} ---")
    debug_print(f"My Health: {you['health']}, My Size: {you['size']}")
    debug_print(f"Ate food last turn: {ate_food_last_turn}")

    possible_moves_str = ['up', 'down', 'left', 'right']
    
    # Flatten all snake bodies for collision checks (excluding our own tail if not eaten food)
    snakes_together_tuples = []
    for s in all_snakes_jake:
        snakes_together_tuples.extend(s['body'])
    
    ignore_collision_tuples = []
    if not ate_food_last_turn and len(you['body']) > 1:
        ignore_collision_tuples.append(you['body'][-1]) # Own tail

    # Initial safety checks
    safe_moves_str = dont_hit_wall(list(possible_moves_str), you['head'], walls_jake)
    debug_print(f"After wall check: {safe_moves_str}")
    
    safe_moves_str = dont_hit_snakes(safe_moves_str, you['head'], snakes_together_tuples, ignore_collision_tuples)
    debug_print(f"After snake collision check: {safe_moves_str}")
    
    # dont_get_eaten is complex, using a simplified version or relying on subsequent checks
    safe_moves_str = dont_get_eaten(safe_moves_str, you, all_snakes_jake)
    debug_print(f"After 'don't get eaten' check: {safe_moves_str}")

    # Lookahead (simplified: ensure next move isn't an immediate dead end)
    if len(safe_moves_str) > 1:
        survivable_moves_str = []
        for m_str in safe_moves_str:
            next_head_candidate = get_space(you['head'], m_str)
            # What would be safe moves from *that* position?
            # For this check, assume we've moved, so our actual head is now part of snakes_together
            future_snakes_together = list(snakes_together_tuples)
            if you['head'] not in future_snakes_together: # Add current head as part of body
                 future_snakes_together.append(you['head'])
            
            # For lookahead, the tail to ignore would be our *current* second segment if we move.
            future_ignore = []
            if len(you['body']) > 0: # Current head becomes neck
                future_ignore.append(you['head'])


            next_potential_safe_moves = dont_hit_wall(list(possible_moves_str), next_head_candidate, walls_jake)
            next_potential_safe_moves = dont_hit_snakes(next_potential_safe_moves, next_head_candidate, future_snakes_together, future_ignore)
            
            if next_potential_safe_moves:
                survivable_moves_str.append(m_str)
        
        if survivable_moves_str:
            safe_moves_str = survivable_moves_str
            debug_print(f"After lookahead (survivable): {safe_moves_str}")
        else: # All current safe moves lead to a trap next, stick with original safe_moves
            debug_print(f"Lookahead found no better options, sticking with: {safe_moves_str}")


    chosen_move_str = None # Final decision

    # Food preference
    # Original: if you['size'] < 6: you["health"] = you["health"]/2 # This artificially makes snake hungry
    # For now, let's not modify health directly, just use the HUNGRY logic
    if you['health'] < HUNGRY and safe_moves_str:
        # maxFood = round( (1 - ((you["health"]-STARVING) / (HUNGRY-STARVING))) * (FOOD_MAX-FOOD_MIN) )
        # The loop for `maxFood` implies searching in increasing radius.
        # Simplified: try to get food if it's reasonably close using `get_food`
        # The `dist` parameter in `get_food` can simulate this search radius.
        # Let's try a moderate fixed distance or one based on hunger.
        food_search_radius = FOOD_MAX # Max radius
        if you['health'] > STARVING: # if not starving, reduce radius
             food_search_radius = int(FOOD_MIN + ( (HUNGRY - you['health']) / (HUNGRY - STARVING) ) * (FOOD_MAX - FOOD_MIN) )
        food_search_radius = max(FOOD_MIN, min(FOOD_MAX, food_search_radius))


        food_seeking_moves = get_food(list(safe_moves_str), you['head'], food_jake, food_search_radius)
        debug_print(f"Food seeking (radius {food_search_radius}): {food_seeking_moves} from safe {safe_moves_str}")
        if food_seeking_moves and any(m in safe_moves_str for m in food_seeking_moves):
            # Prefer food moves that are also safe
            # Choose one of them, perhaps randomly if multiple
            preferred_food_moves = [m for m in food_seeking_moves if m in safe_moves_str]
            if preferred_food_moves:
                chosen_move_str = random.choice(preferred_food_moves)
                debug_print(f"Chose food move: {chosen_move_str}")


    # Flee wall as preference (if no food move chosen and multiple options exist)
    if have_choice(chosen_move_str, safe_moves_str):
        flee_wall_moves = flee_wall(list(safe_moves_str), walls_jake, you['head'])
        if flee_wall_moves and len(flee_wall_moves) < len(safe_moves_str): # If it restricted choices
            # Potentially choose from these, or just update safe_moves_str
            # The original seems to update `moves` (our `safe_moves_str`)
            safe_moves_str = flee_wall_moves
            debug_print(f"After flee wall preference: {safe_moves_str}")


    # Eat others as preference
    if have_choice(chosen_move_str, safe_moves_str):
        aggressive_moves = eat_others(list(safe_moves_str), you['head'], you['size'], all_snakes_jake)
        if aggressive_moves and any(m in safe_moves_str for m in aggressive_moves):
            preferred_aggressive_moves = [m for m in aggressive_moves if m in safe_moves_str]
            if preferred_aggressive_moves and len(preferred_aggressive_moves) < len(safe_moves_str): # if it narrowed choices
                 # chosen_move_str = random.choice(preferred_aggressive_moves) # Original takes it if found
                 # For now, let's refine safe_moves_str
                 safe_moves_str = preferred_aggressive_moves
                 debug_print(f"After eat others preference: {safe_moves_str}")


    # Go straight as preference
    if have_choice(chosen_move_str, safe_moves_str):
        straight_move = go_straight(list(safe_moves_str), you['head'], you['body'])
        if straight_move:
            chosen_move_str = straight_move
            debug_print(f"Chose straight move: {chosen_move_str}")

    # If still no decision, and there are safe moves, pick one
    if chosen_move_str is None and safe_moves_str:
        chosen_move_str = random.choice(safe_moves_str)
        debug_print(f"Chose random from safe moves: {chosen_move_str}")
    
    # Fallback if no safe moves or no decision yet
    if chosen_move_str is None:
        debug_print("No preferred or safe move found. Desperate measures.")
        # Original has more complex fallbacks like `eat_tail`.
        # For now, if no safe_moves_str, pick any original possible_moves_str that's not an immediate wall hit.
        # Or, even simpler, a truly random move from the base 4.
        
        # Try to make a move that's not into a wall, even if it's into a snake
        desperate_moves = dont_hit_wall(list(possible_moves_str), you['head'], walls_jake)
        if desperate_moves:
            chosen_move_str = random.choice(desperate_moves)
            debug_print(f"Desperate non-wall move: {chosen_move_str}")
        else: # Trapped by walls on all sides
            chosen_move_str = random.choice(possible_moves_str) # Will hit a wall
            debug_print(f"Desperate random (will hit wall): {chosen_move_str}")

    # Update ate_food_last_turn for next call
    if chosen_move_str and ate_food(you['head'], food_jake, chosen_move_str):
        ate_food_last_turn = True
        debug_print("ATE FOOD THIS TURN!")
    else:
        ate_food_last_turn = False

    debug_print(f"Final chosen move (string): {chosen_move_str}")

    # Convert string move to integer action
    if chosen_move_str in STRING_TO_ACTION:
        final_action = STRING_TO_ACTION[chosen_move_str]
    else: # Should not happen if logic is correct
        debug_print(f"Error: chosen_move_str '{chosen_move_str}' not in mapping. Defaulting to random.")
        final_action = random.randint(0, 3)
        
    debug_print(f"Final action (int): {final_action}")
    return final_action

# Example of how to test (not part of the module to be loaded by train.py)
if __name__ == '__main__':
    DEBUG = True
    # A mock game_state
    mock_game_state = {
        "game": {"id": "test-game"},
        "turn": 5,
        "board": {
            "height": 11,
            "width": 11,
            "food": [
                {"x": 5, "y": 5},
                {"x": 1, "y": 8}
            ],
            "snakes": [
                {
                    "id": "jake",
                    "name": "JakeTheSnake",
                    "health": 80,
                    "body": [{"x": 2, "y": 2}, {"x": 2, "y": 3}, {"x": 2, "y": 4}],
                    "latency": "123",
                    "shout": "hungry!"
                },
                {
                    "id": "opponent",
                    "name": "OtherSnake",
                    "health": 90,
                    "body": [{"x": 8, "y": 8}, {"x": 8, "y": 7}, {"x": 8, "y": 6}],
                    "latency": "100",
                    "shout": "hiss"
                }
            ]
        },
        "you": { # This 'you' part is usually constructed by the game engine for the specific snake
            "id": "jake",
             # ... rest of 'you' would mirror the entry in board.snakes
        }
    }
    my_snake_index = 0 # Jake is the first snake in this mock
    
    # Simulate a few turns
    for i in range(5):
        print(f"\\nSIMULATING TURN {mock_game_state['turn'] + i}")
        action = choose_action(mock_game_state, my_snake_index)
        print(f"Jake chose action: {action} ({ACTION_TO_STRING.get(action)})")
        
        # Basic mock update (doesn't really move snake or change much)
        mock_game_state['turn'] += 1
        if mock_game_state['board']['snakes'][my_snake_index]['health'] > 0 :
             mock_game_state['board']['snakes'][my_snake_index]['health'] -=1
        if ate_food_last_turn: # If Jake "ate"
            mock_game_state['board']['snakes'][my_snake_index]['health'] = 100
            # Remove food (simplistic: remove first food)
            if mock_game_state['board']['food']:
                # This part is tricky: need to know which food was at the 'next_head'
                # For this test, let's just clear ate_food_last_turn if no food is where head would be
                # The actual ate_food logic inside choose_action handles this.
                pass 
        
        # Move head (very crudely for testing print)
        # This is not a real game update, just for demo
        head = mock_game_state['board']['snakes'][my_snake_index]['body'][0]
        if action == 0: head['y'] -=1 # up
        elif action == 1: head['y'] +=1 # down
        elif action == 2: head['x'] -=1 # left
        elif action == 3: head['x'] +=1 # right
        head['x'] = max(0, min(mock_game_state['board']['width']-1, head['x']))
        head['y'] = max(0, min(mock_game_state['board']['height']-1, head['y']))


