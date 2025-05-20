import pygame
import json
import os
import pygame.mixer  # Ensure pygame is imported for mixer

CELL_SIZE = 40
MARGIN = 2
LOG_DIR = "logs"
SNAKE_COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 0, 128),  # Purple
]

eat_sound = None
collide_sound = None  # New sound for collisions
_last_game_state_for_sound_check = None
_last_game_state_for_collision_check = None  # Tracker for collision detection

def init_sound(mute=False):
    global eat_sound, collide_sound
    if mute:
        eat_sound = None
        collide_sound = None
        return
    try:
        pygame.mixer.init()
        mixer_status = pygame.mixer.get_init()
        if not mixer_status:
            print("Warning: pygame.mixer.init() called, but pygame.mixer.get_init() returned None. Mixer might not be functional.")
            eat_sound = None
            collide_sound = None
            return

        script_dir = os.path.dirname(os.path.abspath(__file__))
        sound_file_path_visualize_relative = os.path.join(script_dir, "assets", "sounds", "eat_apple.mp3")
        sound_file_path_cwd_relative = os.path.join("assets", "sounds", "eat_apple.mp3")
        sound_file_path_parent_relative = os.path.join(script_dir, "..", "assets", "sounds", "eat_apple.mp3")

        final_sound_path_to_try = None
        if os.path.exists(sound_file_path_visualize_relative):
            final_sound_path_to_try = sound_file_path_visualize_relative
        elif os.path.exists(sound_file_path_cwd_relative):
            final_sound_path_to_try = sound_file_path_cwd_relative
        elif os.path.exists(sound_file_path_parent_relative):
            final_sound_path_to_try = sound_file_path_parent_relative
        else:
            print(f"Warning: Sound file 'eat_apple.mp3' not found. Searched paths based on script location and CWD.")
            eat_sound = None
            return
        
        eat_sound = pygame.mixer.Sound(final_sound_path_to_try)

        # Load collide sound
        collide_sound_file_path_visualize_relative = os.path.join(script_dir, "assets", "sounds", "collide.mp3")
        collide_sound_file_path_cwd_relative = os.path.join("assets", "sounds", "collide.mp3")
        collide_sound_file_path_parent_relative = os.path.join(script_dir, "..", "assets", "sounds", "collide.mp3")

        final_collide_sound_path_to_try = None
        if os.path.exists(collide_sound_file_path_visualize_relative):
            final_collide_sound_path_to_try = collide_sound_file_path_visualize_relative
        elif os.path.exists(collide_sound_file_path_cwd_relative):
            final_collide_sound_path_to_try = collide_sound_file_path_cwd_relative
        elif os.path.exists(collide_sound_file_path_parent_relative):
            final_collide_sound_path_to_try = collide_sound_file_path_parent_relative
        else:
            print(f"Warning: Sound file 'collide.mp3' not found. Searched paths based on script location and CWD.")
            collide_sound = None
        
        if final_collide_sound_path_to_try:
            collide_sound = pygame.mixer.Sound(final_collide_sound_path_to_try)

        # --- NEW: load & loop background music from game_loops ---
        music_dir = os.path.join(script_dir, "assets", "sounds", "game_loops")
        if os.path.isdir(music_dir):
            music_files = [f for f in os.listdir(music_dir) if f.lower().endswith(".mp3")]
            if music_files:
                music_path = os.path.join(music_dir, music_files[0])
                try:
                    pygame.mixer.music.load(music_path)
                    pygame.mixer.music.set_volume(0.5)
                    pygame.mixer.music.play(-1)  # loop indefinitely
                except pygame.error as e:
                    print(f"Warning: Failed to load/play background music: {e}")
        else:
            print(f"Warning: Background music directory not found: {music_dir}")

    except pygame.error as e:
        print(f"Warning: Pygame error during sound initialization: {e}")
        eat_sound = None
        collide_sound = None
    except Exception as e:
        print(f"Warning: Generic error during sound initialization: {e}")
        eat_sound = None
        collide_sound = None

def play_eat_sound(mute=False):
    if mute:
        return
    if eat_sound:
        try:
            channel = eat_sound.play()
            if channel is None:
                print("Warning: eat_sound.play() returned None. Sound may not have played (e.g., no available channels or mixer issue).")
        except pygame.error as e:
            print(f"Warning: Pygame error during eat_sound.play(): {e}")

def play_collide_sound(mute=False):
    if mute:
        return
    if collide_sound:
        try:
            channel = collide_sound.play()
            if channel is None:
                print("Warning: collide_sound.play() returned None. Sound may not have played.")
        except pygame.error as e:
            print(f"Warning: Pygame error during collide_sound.play(): {e}")

def reset_sound_state_tracker():
    global _last_game_state_for_sound_check, _last_game_state_for_collision_check
    _last_game_state_for_sound_check = None
    _last_game_state_for_collision_check = None

def check_for_eat_and_play_sound(current_game_state_json, mute=False):
    global _last_game_state_for_sound_check

    if not current_game_state_json or 'board' not in current_game_state_json:
        _last_game_state_for_sound_check = current_game_state_json
        return

    curr_board = current_game_state_json['board']
    curr_food = curr_board.get('food', [])

    if _last_game_state_for_sound_check is None or 'board' not in _last_game_state_for_sound_check:
        _last_game_state_for_sound_check = current_game_state_json
        return

    prev_board = _last_game_state_for_sound_check['board']
    prev_food = prev_board.get('food', [])

    if isinstance(prev_food, list) and isinstance(curr_food, list) and len(curr_food) < len(prev_food):
        play_eat_sound(mute=mute)

    _last_game_state_for_sound_check = current_game_state_json

def check_for_collision_and_play_sound(current_game_state_json, mute=False):
    global _last_game_state_for_collision_check

    # Ensure basic structure and 'turn' key are present for reliable episode start detection
    if not current_game_state_json or \
       'board' not in current_game_state_json or \
       'snakes' not in current_game_state_json['board'] or \
       'turn' not in current_game_state_json: # Check for 'turn' key
        _last_game_state_for_collision_check = current_game_state_json # Update to avoid stale state
        return

    current_turn = current_game_state_json['turn'] # 'turn' key is now confirmed to exist
    curr_snakes = current_game_state_json['board']['snakes']

    # Reset comparison if it's the start of a new game/episode (turn 0),
    # or if no previous state exists, or if the previous state was invalid.
    is_new_episode_or_first_run = (current_turn == 0)

    if _last_game_state_for_collision_check is None or \
       is_new_episode_or_first_run or \
       'board' not in _last_game_state_for_collision_check or \
       'snakes' not in _last_game_state_for_collision_check['board']:
        _last_game_state_for_collision_check = current_game_state_json
        return

    prev_snakes = _last_game_state_for_collision_check['board']['snakes']

    # If number of snakes decreased, a snake was eliminated
    if isinstance(prev_snakes, list) and \
       isinstance(curr_snakes, list) and \
       len(curr_snakes) < len(prev_snakes):
        play_collide_sound(mute=mute)
    
    _last_game_state_for_collision_check = current_game_state_json

def get_winner(game_states):
    last_state = game_states[-1]
    snakes = last_state['board']['snakes']
    if len(snakes) == 1:
        return f"Winner: {snakes[0]['name']}"
    elif len(snakes) == 0:
        return "Winner: None (all eliminated)"
    else:
        return "Winner: Undecided"

def draw_board(screen, game_state, snake_colors, mute=False):
    board_width = game_state['board'].get('width', 11)
    board_height = game_state['board'].get('height', 11)
    screen.fill((30, 30, 30))
    check_for_eat_and_play_sound(game_state, mute=mute)
    check_for_collision_and_play_sound(game_state, mute=mute)

    for x in range(board_width):
        for y in range(board_height):
            rect = pygame.Rect(x*CELL_SIZE+MARGIN, y*CELL_SIZE+MARGIN, CELL_SIZE-MARGIN*2, CELL_SIZE-MARGIN*2)
            pygame.draw.rect(screen, (50, 50, 50), rect, 1)
    
    for food in game_state['board']['food']:
        center = (food['x']*CELL_SIZE + CELL_SIZE//2, food['y']*CELL_SIZE + CELL_SIZE//2)
        pygame.draw.circle(screen, (0,255,0), center, CELL_SIZE//6)
    
    for idx, snake in enumerate(game_state['board']['snakes']):
        color = snake_colors.get(snake['name'], SNAKE_COLORS[idx % len(SNAKE_COLORS)])
        for i, segment in enumerate(snake['body']):
            if i == 0:
                head_center = (segment['x'] * CELL_SIZE + CELL_SIZE // 2, segment['y'] * CELL_SIZE + CELL_SIZE // 2)
                if len(snake['body']) > 1:
                    next_seg = snake['body'][1]
                    dx = segment['x'] - next_seg['x']
                    dy = segment['y'] - next_seg['y']
                    if abs(dx) > abs(dy):
                        head_w, head_h = int(CELL_SIZE * 0.8), int(CELL_SIZE * 0.6)
                    else:
                        head_w, head_h = int(CELL_SIZE * 0.6), int(CELL_SIZE * 0.8)
                else:
                    head_w = head_h = int(CELL_SIZE * 0.7)
                head_rect = pygame.Rect(0, 0, head_w, head_h)
                head_rect.center = head_center
                pygame.draw.ellipse(screen, color, head_rect)
                eye_radius = 3
                eye_offset = head_w // 3
                if len(snake['body']) > 1:
                    if abs(dx) > abs(dy):
                        if dx < 0:
                            eye1 = (head_center[0] - eye_offset, head_center[1] - eye_radius)
                            eye2 = (head_center[0] - eye_offset, head_center[1] + eye_radius)
                        else:
                            eye1 = (head_center[0] + eye_offset, head_center[1] - eye_radius)
                            eye2 = (head_center[0] + eye_offset, head_center[1] + eye_radius)
                    else:
                        if dy < 0:
                            eye1 = (head_center[0] - eye_radius, head_center[1] + eye_offset)
                            eye2 = (head_center[0] + eye_radius, head_center[1] + eye_offset)
                        else:
                            eye1 = (head_center[0] - eye_radius, head_center[1] - eye_offset)
                            eye2 = (head_center[0] + eye_radius, head_center[1] - eye_offset)
                    pygame.draw.circle(screen, (0, 0, 0), eye1, eye_radius)
                    pygame.draw.circle(screen, (0, 0, 0), eye2, eye_radius)
            else:
                prev_seg = snake['body'][i-1]
                prev_center = (prev_seg['x'] * CELL_SIZE + CELL_SIZE // 2,
                               prev_seg['y']*CELL_SIZE + CELL_SIZE // 2)
                curr_center = (segment['x'] * CELL_SIZE + CELL_SIZE // 2,
                               segment['y']*CELL_SIZE + CELL_SIZE // 2)
                thickness = CELL_SIZE // 3
                if prev_center[0] == curr_center[0]:
                    x = prev_center[0] - thickness // 2
                    y = min(prev_center[1], curr_center[1])
                    height = abs(curr_center[1] - prev_center[1])
                    rect = pygame.Rect(x, y, thickness, height)
                elif prev_center[1] == curr_center[1]:
                    y = prev_center[1] - thickness // 2
                    x = min(prev_center[0], curr_center[0])
                    width = abs(curr_center[0] - prev_center[0])
                    rect = pygame.Rect(x, y, width, thickness)
                else:
                    rect = pygame.Rect(segment['x']*CELL_SIZE, segment['y']*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, color, rect)

def load_games():
    if not os.path.exists(LOG_DIR):
        return []
    files = [f for f in os.listdir(LOG_DIR) if f.endswith(".jsonl")]
    # sort by modification time (newest first)
    files.sort(key=lambda f: os.path.getmtime(os.path.join(LOG_DIR, f)), reverse=True)
    games = []
    for fname in files:
        with open(os.path.join(LOG_DIR, fname)) as f:
            states = []
            for line in f:
                try:
                    data = json.loads(line)
                    if "board" in data:
                        states.append(data)
                except Exception:
                    continue
            if states:
                games.append((fname, states))
    return games

def render_info_lines(game_state, game_idx, num_games, fname, turn_idx, num_turns, winner, font, smallfont, live_mode, snake_colors):
    live_text = "Live Mode: ON" if live_mode else "Live Mode: OFF"
    lines = [
        f"Game {game_idx+1}/{num_games}: {fname}",
        f"Turn {turn_idx+1}/{num_turns}",
        winner,
        live_text,
        "Snakes:"
    ]
    for idx, snake in enumerate(game_state['board']['snakes']):
        color = snake_colors.get(snake['name'], SNAKE_COLORS[idx % len(SNAKE_COLORS)])
        lines.append(f"  {snake['name']} (Len: {snake['length']}, HP: {snake['health']})")
    return lines

# Visualization helper functions for training

def init_visualization(map_size, fps=10, mute=False):
    """Initialize pygame, mixer, screen, and clock for training visualization."""
    pygame.init()
    # Enable key repeat so holding arrow keys registers multiple events
    pygame.key.set_repeat(200, 50)
    init_sound(mute=mute)
    reset_sound_state_tracker() # Reset tracker for new training visualization
    screen = pygame.display.set_mode((map_size[1] * CELL_SIZE, map_size[0] * CELL_SIZE))
    pygame.display.set_caption("PPO Training Visualization")
    clock = pygame.time.Clock()
    return screen, clock


def visualize_step(game_state_json, snake_colors, screen, clock, mute=False, visualization_fps_value=10):
    """Handle one frame of training visualization: events, drawing, flip, tick."""
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            return False
    draw_board(screen, game_state_json, snake_colors, mute=mute)
    pygame.display.flip()
    if visualization_fps_value > 0:
        clock.tick(visualization_fps_value)
    return True


def close_visualization():
    pygame.quit()

def wait_for_debug_input_pygame(screen, clock, visualization_fps_value=10):
    """
    Waits for RIGHT ARROW key press to continue, or Q to quit.
    Keeps the pygame window responsive.
    Returns:
        bool: True to continue training, False to quit.
    """
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    waiting = False
                elif event.key == pygame.K_q:
                    pygame.quit() # Ensure pygame quits cleanly
                    return False
        if screen:
            pygame.display.flip()
        if clock and visualization_fps_value > 0:
            clock.tick(visualization_fps_value)
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Battlesnake Log Visualizer")
    parser.add_argument("--fast", action="store_true", help="Run visualization without FPS limit (disables clock.tick())")
    args = parser.parse_args()

    pygame.init()
    pygame.key.set_repeat(200, 50)
    font = pygame.font.SysFont("consolas", 18)
    smallfont = pygame.font.SysFont("consolas", 14)
    
    init_sound()

    games = load_games()
    
    game_idx = 0
    turn_idx = 0
    fname = ""
    game_states = []  # Initialize to empty list to prevent UnboundLocalError

    if not games:
        print("No games found in logs/")
        pygame.quit()
        return

    fname, game_states = games[game_idx]
    reset_sound_state_tracker()

    if not game_states:
        print(f"Game '{fname}' is empty (no turns). Exiting.")
        pygame.quit()
        return

    sample_state = game_states[0]
    board_width = sample_state['board'].get('width', 11)
    board_height = sample_state['board'].get('height', 11)

    slider_margin_top = 10
    slider_height = 20

    max_snakes = max(len(gs[1][0]['board']['snakes']) for gs in games if gs[1])
    info_lines_count = 7 + max_snakes
    info_height = info_lines_count * 18 + 10
    win_width = CELL_SIZE * board_width
    win_height = CELL_SIZE * board_height + slider_margin_top + slider_height + info_height

    screen = pygame.display.set_mode((win_width, win_height))
    pygame.display.set_caption("Battlesnake Log Visualizer")

    # --- REPLACE pygame_gui slider and label with custom slider ---
    slider_rect = pygame.Rect(10, CELL_SIZE * board_height + slider_margin_top, win_width - 120, slider_height)
    slider_knob_width = 16
    slider_knob_height = slider_height + 4
    slider_color = (180, 180, 180)
    slider_knob_color = (80, 80, 220)
    slider_bg_color = (60, 60, 60)
    slider_border_color = (100, 100, 100)
    slider_active = False

    def get_slider_knob_x(turn_idx, num_turns):
        if num_turns <= 0:
            return slider_rect.x
        frac = turn_idx / num_turns
        usable_width = slider_rect.width - slider_knob_width
        return int(slider_rect.x + frac * usable_width)

    def get_turn_idx_from_mouse(mx, num_turns):
        usable_width = slider_rect.width - slider_knob_width
        rel_x = mx - slider_rect.x
        rel_x = max(0, min(rel_x, usable_width))
        frac = rel_x / usable_width if usable_width > 0 else 0
        return int(round(frac * num_turns))

    live_mode = False
    running = True
    clock = pygame.time.Clock()
    
    while running:
        if not args.fast:
            time_delta = clock.tick(30) / 1000.0
        mouse_down_on_slider = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                new_game_selected = False
                if event.key == pygame.K_RIGHT:
                    if turn_idx < len(game_states) - 1:
                        turn_idx += 1
                elif event.key == pygame.K_LEFT:
                    if turn_idx > 0:
                        turn_idx -= 1
                elif event.key == pygame.K_UP:
                    if game_idx > 0:
                        game_idx -= 1
                        fname, game_states = games[game_idx]
                        turn_idx = 0
                        new_game_selected = True
                elif event.key == pygame.K_DOWN:
                    if game_idx < len(games) - 1:
                        game_idx += 1
                        fname, game_states = games[game_idx]
                        turn_idx = 0
                        new_game_selected = True
                elif event.key == pygame.K_l:
                    live_mode = not live_mode
                    print("Live mode:", "ON" if live_mode else "OFF")
                    if live_mode:
                        reset_sound_state_tracker()
                
                if new_game_selected:
                    reset_sound_state_tracker()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = event.pos
                    knob_x = get_slider_knob_x(turn_idx, len(game_states) - 1)
                    knob_rect = pygame.Rect(knob_x, slider_rect.y - 2, slider_knob_width, slider_knob_height)
                    if knob_rect.collidepoint(mx, my) or slider_rect.collidepoint(mx, my):
                        slider_active = True
                        turn_idx = get_turn_idx_from_mouse(mx, len(game_states) - 1)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    slider_active = False
            elif event.type == pygame.MOUSEMOTION:
                if slider_active:
                    mx, my = event.pos
                    turn_idx = get_turn_idx_from_mouse(mx, len(game_states) - 1)

        if live_mode:
            current_fname = fname
            loaded_games = load_games()
            if loaded_games:
                games = loaded_games
                if not games or (games[-1][0] != current_fname and len(games) - 1 != game_idx):
                    game_idx = len(games) - 1
                    fname, game_states = games[game_idx]
                    turn_idx = len(game_states) - 1 if game_states else 0
                    reset_sound_state_tracker()
                else:
                    game_idx = len(games) - 1
                    fname, game_states = games[game_idx]
                    turn_idx = len(game_states) - 1 if game_states else 0
                
            else:
                games = []
                fname = ""
                game_states = []
                screen.fill((30,30,30))
                no_games_text = font.render("No games in logs. Waiting for live data...", True, (220,220,220))
                text_rect = no_games_text.get_rect(center=(win_width // 2, win_height // 2))
                screen.blit(no_games_text, text_rect)
                pygame.display.flip()
                if not args.fast:
                    clock.tick(5)
                continue

        if not games:
            screen.fill((30,30,30))
            error_text = font.render("Error: No games loaded.", True, (255,0,0))
            text_rect = error_text.get_rect(center=(win_width // 2, win_height // 2))
            screen.blit(error_text, text_rect)
            pygame.display.flip()
            if not args.fast:
                clock.tick(30)
            continue

        if not game_states:
            screen.fill((30,30,30))
            no_turns_text = font.render(f"Game '{fname}' has no turns. Select another game.", True, (220,220,220))
            text_rect = no_turns_text.get_rect(center=(win_width // 2, win_height // 2))
            screen.blit(no_turns_text, text_rect)
            pygame.display.flip()
            if not args.fast:
                clock.tick(30)
            continue
            
        turn_idx = max(0, min(turn_idx, len(game_states) - 1))
        current_game_state_data = game_states[turn_idx]
        winner = get_winner(game_states)

        new_board_width = current_game_state_data['board'].get('width', 11)
        new_board_height = current_game_state_data['board'].get('height', 11)
        if new_board_width != board_width or new_board_height != board_height:
            board_width = new_board_width
            board_height = new_board_height
            win_width = CELL_SIZE * board_width
            win_height = CELL_SIZE * board_height + slider_margin_top + slider_height + info_height
            screen = pygame.display.set_mode((win_width, win_height))
            slider_rect = pygame.Rect(10, CELL_SIZE * board_height + slider_margin_top, win_width - 120, slider_height)

        initial_snakes = game_states[0]['board']['snakes']
        snake_colors = { snake['name']: SNAKE_COLORS[i % len(SNAKE_COLORS)] for i, snake in enumerate(initial_snakes) }

        draw_board(screen, current_game_state_data, snake_colors)

        # --- Draw custom slider ---
        pygame.draw.rect(screen, slider_bg_color, slider_rect)
        pygame.draw.rect(screen, slider_border_color, slider_rect, 2)
        # Draw filled portion
        if len(game_states) > 1:
            frac = turn_idx / (len(game_states) - 1)
        else:
            frac = 0
        filled_rect = pygame.Rect(slider_rect.x, slider_rect.y, int(slider_rect.width * frac), slider_rect.height)
        pygame.draw.rect(screen, slider_color, filled_rect)
        # Draw knob
        knob_x = get_slider_knob_x(turn_idx, len(game_states) - 1)
        knob_rect = pygame.Rect(knob_x, slider_rect.y - 2, slider_knob_width, slider_knob_height)
        pygame.draw.rect(screen, slider_knob_color, knob_rect, border_radius=6)
        pygame.draw.rect(screen, (30, 30, 30), knob_rect, 2, border_radius=6)
        # Draw turn label
        label_text = f"Turn: {turn_idx+1}"
        label_surf = font.render(label_text, True, (220, 220, 220))
        label_rect = label_surf.get_rect()
        label_rect.topleft = (slider_rect.right + 10, slider_rect.y)
        screen.blit(label_surf, label_rect)

        info_y = CELL_SIZE * board_height + slider_margin_top + slider_height + 10
        lines = render_info_lines(
            current_game_state_data, game_idx, len(games), fname, turn_idx, len(game_states),
            winner, font, smallfont, live_mode, snake_colors
        )
        for idx, line in enumerate(lines):
            if idx == 2 and line.startswith("Winner: "):
                winner_name = line[len("Winner: "):].strip()
                color = snake_colors.get(winner_name, (220, 220, 220))
            elif line.startswith("  "):
                snake_line = line.strip()
                name = snake_line.split(" (")[0]
                color = snake_colors.get(name, SNAKE_COLORS[0])
            else:
                color = (220, 220, 220)
            text = smallfont.render(line, True, color)
            screen.blit(text, (10, info_y))
            info_y += 18

        pygame.display.flip()
    pygame.quit()

if __name__ == "__main__":
    main()