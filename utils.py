import numpy as np
import importlib
import os
import matplotlib.pyplot as plt
from Gym.snake import Snake # For action definitions
from collections import defaultdict, Counter
import onnxruntime as ort
import torch
from scipy.signal.windows import gaussian  # fixed import
from visualize import visualize_step, wait_for_debug_input_pygame  # for debug visualization
from heuristic import heuristic  # for heuristic analysis


MODEL_SESSION = None
MODEL_PATH = None
MODEL = None


def moving_average(data, window=50):
    if len(data) < window:
        return np.convolve(data, np.ones(len(data)) / len(data), mode='valid')
    return np.convolve(data, np.ones(window) / window, mode='valid')

def gaussian_moving_average(data, window=50, std=None):
    """
    Compute Gaussian-weighted moving average.
    data: 1D sequence
    window: kernel size (will be truncated if data shorter)
    std: standard deviation of the Gaussian kernel (defaults to window/4)
    """
    if std is None:
        std = window / 4
    # adjust window if data shorter
    if len(data) < window:
        window = len(data)
    # build and normalize kernel
    weights = gaussian(window, std)
    weights /= weights.sum()
    # convolve and return valid region
    return np.convolve(data, weights, mode='valid')

def plot_training_stats_live(episode_rewards, episode_lengths, save_dir=None, show_plot=True, window=50):
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, label='Reward', alpha=0.5)
    if len(episode_rewards) >= window:
        plt.plot(range(window-1, len(episode_rewards)), moving_average(episode_rewards, window), label=f'MA{window}', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(episode_lengths, label='Length', color='orange', alpha=0.5)
    if len(episode_lengths) >= window:
        plt.plot(range(window-1, len(episode_lengths)), moving_average(episode_lengths, window), label=f'MA{window}', color='red', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Length')
    plt.legend()

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "training_stats.png"))
        plt.close()
    if show_plot:
        plt.pause(0.001)

def plot_death_reasons(death_reasons, save_dir=None, show_plot=True):
    counts = Counter(death_reasons)
    labels, values = zip(*counts.items()) if counts else ([], [])

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Snake Death Reasons")
    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir, "death_reasons.png"))
        plt.close()
    if show_plot:
        plt.show()

def plot_hyperparameters_and_rewards(run_dir):
    """Create a table visualization of training hyperparameters and rewards"""
    from train import (TOTAL_TIMESTEPS, SAVE_INTERVAL, N_STEPS_COLLECT, 
                      PPO_BATCH_SIZE, N_EPOCHS_PPO, LR, GAMMA, GAE_LAMBDA, 
                      PPO_CLIP, MAP_SIZE)
    from Gym.rewards import SimpleRewards

    # Create figure
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    ax.axis('off')

    # Get rewards
    rewards = SimpleRewards().reward_dict

    # Organize configuration in sections
    config = {
        "Training Parameters": {
            "Total Timesteps": f"{TOTAL_TIMESTEPS:,}",
            "Save Interval": SAVE_INTERVAL,
            "Steps per Collection": N_STEPS_COLLECT,
            "Map Size": f"{MAP_SIZE[0]}x{MAP_SIZE[1]}"
        },
        "PPO Parameters": {
            "Batch Size": PPO_BATCH_SIZE,
            "Number of Epochs": N_EPOCHS_PPO,
            "Learning Rate": f"{LR:.0e}",
            "Gamma (Discount)": GAMMA,
            "GAE Lambda": GAE_LAMBDA,
            "PPO Clip": PPO_CLIP
        },
        "Reward Structure": {
            k: f"{v:+.2f}" for k, v in sorted(rewards.items())
        }
    }

    # Format as a table string
    table_text = "Training Configuration\n" + "="*22 + "\n\n"
    
    for category, params in config.items():
        table_text += f"{category}\n" + "-"*len(category) + "\n"
        max_key_length = max(len(key) for key in params.keys())
        for key, value in params.items():
            table_text += f"{key:<{max_key_length}} : {value}\n"
        table_text += "\n"

    # Add text with styling
    ax.text(0.05, 0.95, table_text,
            fontfamily='monospace',
            fontsize=11,
            verticalalignment='top',
            bbox=dict(
                facecolor='white',
                edgecolor='gray',
                alpha=0.9,
                pad=10,
                boxstyle='round'
            ))

    # Save configuration table
    fig.savefig(f"{run_dir}/training_config.png", 
                bbox_inches='tight',
                dpi=150)
    plt.close(fig)

def log_and_plot_training(
    episode_rewards,
    episode_lengths,
    death_reasons,
    episode_wins_all,  # list of lists, one per snake
    episode_apples,    # new: apples eaten per episode (final snake length)
    run_dir,
    show_plot=False,
    ma_window=100,
    snake_names=None
):
    import matplotlib.pyplot as plt
    import datetime
    # Create figure with GridSpec to add reward dict
    fig = plt.figure(figsize=(15, 28))  # Increased height for larger plots
    gs = plt.GridSpec(6, 1, height_ratios=[5, 5, 5, 5, 5, 2], figure=fig)  # Each plot gets more height
    axs = [fig.add_subplot(gs[i]) for i in range(5)]
    ax_rewards = fig.add_subplot(gs[5])
    ax_rewards.axis('off')
    x = np.arange(len(episode_rewards))
    # Rewards
    axs[0].plot(x, episode_rewards, color='blue', alpha=0.4, linewidth=1, label='Raw Reward')
    if len(episode_rewards) >= ma_window:
        avg_rewards = moving_average(episode_rewards, ma_window)
        axs[0].plot(
            np.arange(ma_window-1, len(episode_rewards)),
            avg_rewards,
            color='C0',
            linewidth=2.5,
            label=f'Mean ({ma_window})'
        )
    axs[0].set_title('Episode Reward')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Reward')
    axs[0].legend()
    axs[0].grid(True)
    # Lengths (steps per episode)
    axs[1].plot(x, episode_lengths, color='blue', alpha=0.4, linewidth=1, label='Raw Length (steps)')
    if len(episode_lengths) >= ma_window:
        avg_lengths = moving_average(episode_lengths, ma_window)
        axs[1].plot(
            np.arange(ma_window-1, len(episode_lengths)),
            avg_lengths,
            color='C1',
            linewidth=2.5,
            label=f'Mean ({ma_window})'
        )
    axs[1].set_title('Episode Length (Steps)')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Steps')
    axs[1].legend()
    axs[1].grid(True)
    # Death reasons
    if death_reasons:
        from collections import Counter
        reason_counts = Counter(death_reasons)
        axs[2].bar(reason_counts.keys(), reason_counts.values(), color='C2', alpha=0.7)
        axs[2].set_title('Death Reasons')
        axs[2].set_ylabel('Count')
        axs[2].tick_params(axis='x', labelsize=8, rotation=45)
    else:
        axs[2].set_visible(False)
    axs[2].grid(True, axis='y')
    # Apples eaten (final snake length)
    if episode_apples:
        x_apples = np.arange(len(episode_apples))
        axs[3].plot(x_apples, episode_apples, color='C4', alpha=0.4, linewidth=1, label='Raw Apples Eaten')
        if len(episode_apples) >= ma_window:
            avg_apples = moving_average(episode_apples, ma_window)
            axs[3].plot(
                np.arange(ma_window-1, len(episode_apples)),
                avg_apples,
                color='C4',
                linewidth=2.5,
                label=f'Mean ({ma_window})'
            )
        mean_apples = np.mean(episode_apples) if episode_apples else 0
        axs[3].set_title(f'Apples Eaten (Final Length) | Avg: {mean_apples:.2f}')
        axs[3].set_xlabel('Episode')
        axs[3].set_ylabel('Apples')
        axs[3].legend()
        axs[3].grid(True)
    else:
        axs[3].set_visible(False)
    # Win rate for all snakes (only if there are opponents)
    if len(episode_wins_all) > 1:
        colors = ['C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C2']
        for idx, wins in enumerate(episode_wins_all):
            if not wins:
                continue
            label = snake_names[idx] if snake_names else ("Agent" if idx == 0 else f"Snake {idx}")
            color = colors[idx % len(colors)]
            axs[4].plot(
                np.arange(len(wins)),
                wins,
                color=color,
                alpha=0.15,
                linewidth=1,
                label=f'{label} Raw'
            )
            if len(wins) >= ma_window:
                avg_wins = moving_average(wins, ma_window)
                axs[4].plot(
                    np.arange(ma_window-1, len(wins)),
                    avg_wins,
                    color=color,
                    linewidth=2.5,
                    label=f'{label} Win Rate ({ma_window})'
                )
        for idx, wins in enumerate(episode_wins_all):
            if not wins:
                continue
            win_percentage = 100 * np.mean(wins)
            label = snake_names[idx] if snake_names else ("Agent" if idx == 0 else f"Snake {idx}")
            axs[4].annotate(
                f'{label} win: {win_percentage:.1f}%',
                xy=(0.02, 0.98 - idx*0.05),
                xycoords='axes fraction',
                fontsize=9,
                color=colors[idx % len(colors)]
            )
        axs[4].set_title('Win Rate')
        axs[4].set_xlabel('Episode')
        axs[4].set_ylabel('Win (1=win)')
        axs[4].legend(fontsize='small')
        axs[4].grid(True)
    else:
        axs[4].set_visible(False)
    # Add run title with time started (hh:mm) based on run_dir creation time
    ctime = os.path.getctime(run_dir)
    time_str = datetime.datetime.fromtimestamp(ctime).strftime("%H:%M")
    fig.suptitle(f"Training Stats: {os.path.basename(run_dir)} | Started: {time_str}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{run_dir}/training_stats.png", bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close(fig)

def load_opponent_modules(opponent_identifiers):
    """
    Loads opponent action functions.
    Can load from heuristic modules in the 'snakes' directory (e.g., "hungry_snake")
    or from .onnx model files (e.g., "path/to/model.onnx").

    Args:
        opponent_identifiers (list[str]): A list of opponent names or paths to .onnx files.

    Returns:
        list[dict]: A list of opponent configurations, where each dict has:
                    "func": the callable action function,
                    "is_model": boolean, True if it's an ONNX model opponent,
                    "id": the original identifier string,
                    "name": display name for logging/UI.
    """
    opponents_config = []

    for identifier in opponent_identifiers:
        if isinstance(identifier, str) and (identifier.endswith(".onnx") or identifier == "model_snake"):
            try:
                from snakes.model_snake import get_model_opponent_action_function
                
                # Now returns (action_fn, display_name)
                action_fn, display_name = get_model_opponent_action_function(identifier)
                
                if action_fn:  
                    opponents_config.append({
                        "func": action_fn, 
                        "is_model": True, 
                        "id": identifier, # Original identifier for reference
                        "name": display_name # Name for logging/UI
                    })
                    print(f"Successfully prepared ONNX model opponent: {display_name} (from {identifier})")
                else:
                    print(f"Could not get action function for ONNX model opponent: {identifier} (resolved to: {display_name}). It will be skipped.")
            except ImportError:
                print(f"Could not import 'get_model_opponent_action_function' from 'snakes.model_snake'. "
                      f"Ensure 'snakes/model_snake.py' exists. Model opponent '{identifier}' will be skipped.")
            except Exception as e:
                print(f"Error setting up ONNX model opponent {identifier}: {e}. It will be skipped.")
        else:  # Assume it's a heuristic module name like "hungry_snake"
            try:
                module = importlib.import_module(f"snakes.{identifier}")
                if hasattr(module, 'choose_action') and callable(module.choose_action):
                    opponents_config.append({
                        "func": module.choose_action, 
                        "is_model": False, 
                        "id": identifier,
                        "name": identifier # Use the module name as the display name for heuristics
                    })
                    print(f"Successfully loaded heuristic opponent: snakes.{identifier}")
                else:
                    print(f"'choose_action' function not found or not callable in module 'snakes.{identifier}'. "
                          f"Opponent '{identifier}' will be skipped.")
            except ImportError:
                print(f"Could not import heuristic opponent module 'snakes.{identifier}'. "
                      f"Ensure 'snakes/{identifier}.py' exists. Opponent '{identifier}' will be skipped.")
            except Exception as e: # Catch other potential errors during heuristic loading
                print(f"Error loading heuristic opponent 'snakes.{identifier}': {e}. It will be skipped.")
    return opponents_config

def game_state_to_observation(array_obs, your_index=0):
    """Convert multi-channel observation array to 3-channel format:
    - Channel 0: Food
    - Channel 1: Your snake
    - Channel 2: All other snakes merged
    """
    height, width, _ = array_obs.shape
    observation = np.zeros((height, width, 3), dtype=np.int8)

    # Food is assumed to be in channel 0
    observation[:, :, 0] = array_obs[:, :, 0]

    # Your snake is in channel 1 (usually the next channel)
    if array_obs.shape[2] > your_index + 1:
        observation[:, :, 1] = array_obs[:, :, your_index + 1]

    # Other snakes: add any other channels into channel 2
    for ch in range(1, array_obs.shape[2]):
        if ch == your_index + 1:
            continue
        observation[:, :, 2] = np.maximum(observation[:, :, 2], array_obs[:, :, ch])

    return observation

def game_state_to_matrix(game_state):
    """
    Convert Battlesnake game_state dictionary to observation matrix.
    """
    height = game_state['board']['height']
    width = game_state['board']['width']
    snakes = game_state['board']['snakes']
    
    num_snakes = len(snakes)
    matrix = np.zeros((height, width, 1 + num_snakes), dtype=np.int8)

    # Fill food
    for food in game_state['board']['food']:
        x = int(food['x'])
        y = int(food['y'])
        if 0 <= x < width and 0 <= y < height:
            matrix[y, x, 0] = 1  # Channel 0 = food

    # Fill snakes
    for idx, snake in enumerate(snakes):
        body = snake['body']
        for i, part in enumerate(body):
            x = int(part['x'])
            y = int(part['y'])
            if 0 <= x < width and 0 <= y < height:
                if i == 0:
                    matrix[y, x, 1 + idx] = 5  # Head
                else:
                    matrix[y, x, 1 + idx] = 1  # Body
    return matrix

def matrix_to_game_state(matrix, you_snake_index=0):
    """
    Convert observation matrix back to game_state dictionary.
    (Assumes simple situation with no health data etc.)
    """
    height, width, channels = matrix.shape
    food_list = []
    snakes_list = []

    # Food
    y_coords, x_coords = np.where(matrix[:, :, 0] == 1)
    for y, x in zip(y_coords, x_coords):
        food_list.append({"x": int(x), "y": int(y)})

    # Snakes
    for snake_idx in range(1, channels):
        snake_parts = []
        y_coords, x_coords = np.where((matrix[:, :, snake_idx] == 1) | (matrix[:, :, snake_idx] == 5))
        coords = list(zip(x_coords, y_coords))

        # Sort head first (head == 5)
        head_coords = np.where(matrix[:, :, snake_idx] == 5)
        if len(head_coords[0]) > 0:
            head = {"x": int(head_coords[1][0]), "y": int(head_coords[0][0])}
            coords.remove((head["x"], head["y"]))
            snake_body = [head] + [{"x": int(x), "y": int(y)} for x, y in coords]
        else:
            # No head found, just treat all parts as body
            snake_body = [{"x": int(x), "y": int(y)} for x, y in coords]

        if snake_body:
            snakes_list.append({
                "health": 100,  # Placeholder, because health info isn't stored in matrix
                "body": snake_body,
                "id": str(snake_idx),
                "name": f"Snake {snake_idx}"
            })

    game_state = {
        "game": {"id": "example"},
        "turn": 0,
        "board": {
            "height": height,
            "width": width,
            "food": food_list,
            "snakes": snakes_list,
        },
        "you": snakes_list[you_snake_index] if snakes_list else {}
    }
    return game_state



def get_next_run_dir(base_dir="runs"):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith("run") and os.path.isdir(os.path.join(base_dir, d))]
    nums = [int(d[3:]) for d in existing if d[3:].isdigit()]
    next_num = max(nums) + 1 if nums else 1
    run_dir = os.path.join(base_dir, f"run{next_num}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def moving_average(data, window=1000):
    if len(data) < window:
        return np.convolve(data, np.ones(len(data))/len(data), mode='valid')
    return np.convolve(data, np.ones(window)/window, mode='valid')

def plot_training_stats_live(episode_rewards, episode_lengths, save_dir=None, show_plot=True, window=10):
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, label='Reward', alpha=0.5)
    if len(episode_rewards) >= window:
        plt.plot(range(window-1, len(episode_rewards)), moving_average(episode_rewards, window), label=f'MA{window}', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(episode_lengths, label='Length', color='orange', alpha=0.5)
    if len(episode_lengths) >= window:
        plt.plot(range(window-1, len(episode_lengths)), moving_average(episode_lengths, window), label=f'MA{window}', color='red', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Length')
    plt.title('Episode Lengths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if show_plot:
        plt.pause(0.001)
    if save_dir:
        plt.savefig(os.path.join(save_dir, "training_stats.png"))



def get_valid_actions(observation, agent_idx=0):
    """
    Determine valid actions (non-suicidal moves) based on current state
    """
    return [0, 1, 2, 3]
    height, width, channels = observation.shape
    
    # Find the snake head (marked as 5 in the representation)
    head_y, head_x = np.where(observation[:, :, agent_idx + 1] == 5) # +1 to skip food channel
    if len(head_y) == 0:  # Snake is already dead
        return [0, 1, 2, 3]
        
    head_y, head_x = head_y[0], head_x[0]
    
    # Check which moves would be valid (not immediately hitting a wall or snake body)
    valid_actions = []
    
    # UP = 0
    if head_y > 0 and observation[head_y-1, head_x, agent_idx+1] != 1:
        valid_actions.append(0)
    # DOWN = 1
    if head_y < height-1 and observation[head_y+1, head_x, agent_idx+1] != 1:
        valid_actions.append(1)
    # LEFT = 2
    if head_x > 0 and observation[head_y, head_x-1, agent_idx+1] != 1:
        valid_actions.append(2)
    # RIGHT = 3
    if head_x < width-1 and observation[head_y, head_x+1, agent_idx+1] != 1:
        valid_actions.append(3)
    
    # If no valid actions, return all actions (snake will die anyway)
    if not valid_actions:
        return [0, 1, 2, 3]
        
    return valid_actions



def  battle_snake_game_state_to_observation(game_state):
    """Convert Battlesnake game state to DQN observation format (3-channel format).
    Channel 0 = food, Channel 1 = 'you', Channel 2 = other snakes.
    """
    board = game_state['board']
    width = board['width']
    height = board['height']

    # Always create a 3-channel observation matrix
    observation = np.zeros((height, width, 3), dtype=np.int8)

    # Add food to channel 0
    for food in board['food']:
        observation[food['y'], food['x'], 0] = 1

    # Add our snake to channel 1 (body=1, head=5)
    my_snake = game_state['you']
    for i, segment in enumerate(my_snake['body']):
        if i == 0:
            observation[segment['y'], segment['x'], 1] = 5
        else:
            observation[segment['y'], segment['x'], 1] = 1

    # Ensure other snakes are considered even if none exist
    other_snakes = [s for s in board['snakes'] if s['id'] != my_snake['id']]
    for snake in other_snakes:
        for i, segment in enumerate(snake['body']):
            if i == 0:
                observation[segment['y'], segment['x'], 2] = 5
            else:
                observation[segment['y'], segment['x'], 2] = 1
    observation = np.flip(observation, axis=0).copy()
    return observation

def load_model(model_path="models/dqn_final.pt", map_size=(11, 11), channels=3):
    """Load the DQN model for inference"""
    global MODEL
    
    # Create model instance
    MODEL = EasyNet(map_size, 4, channels).to(DEVICE)
    
    try:
        # Load saved weights
        MODEL.load_state_dict(torch.load(model_path, map_location=DEVICE))
        MODEL.eval()
        print(f"Successfully loaded model from {model_path}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
def load_model_onnx(model_path=MODEL_PATH):
    """Load the ONNX model for inference with GPU support if available"""
    global MODEL_SESSION
    # select execution providers: prefer CUDA, then MPS, else CPU
    providers = []
    if torch.cuda.is_available():
        providers.append('CUDAExecutionProvider')
    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        providers.append('MPSExecutionProvider')
    else:
        providers.append('CPUExecutionProvider')

    try:
        # create session with preferred providers
        session = ort.InferenceSession(model_path, providers=providers)
        MODEL_SESSION = session
        print(f"Successfully loaded ONNX model from {model_path} using {providers}")
        return True, MODEL_SESSION
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return False, MODEL_SESSION

from Gym.snake import Snake # For action definitions
import numpy as np
def get_valid_moves(game_state):
    """Return list of valid moves that won't immediately kill the snake"""

    my_head = game_state['you']['body'][0]
    my_neck = game_state['you']['body'][1] if len(game_state['you']['body']) > 1 else None
    board_width = game_state['board']['width']
    board_height = game_state['board']['height']
    
    # Default all moves to valid
    moves = {"up": True, "down": True, "left": True, "right": True}

    # Don't move backwards into your neck
    if my_neck:
        if my_neck['x'] < my_head['x']:  # Neck is left of head, don't move left
            moves['left'] = False
        elif my_neck['x'] > my_head['x']:  # Neck is right of head, don't move right
            moves['right'] = False
        elif my_neck['y'] < my_head['y']:  # Neck is below head, don't move down
            moves['down'] = False
        elif my_neck['y'] > my_head['y']:  # Neck is above head, don't move up
            moves['up'] = False
   
    # Don't hit walls
    if my_head['x'] == 0:
        moves['left'] = False
    if my_head['x'] == board_width - 1:
        moves['right'] = False
    if my_head['y'] == 0:
        moves['down'] = False
    if my_head['y'] == board_height - 1:
        moves['up'] = False
    valid_moves = [move for move, valid in moves.items() if valid]
    return valid_moves, moves
    # Don't hit snake bodies (including self)
    for snake in game_state['board']['snakes']:
        for segment in snake['body'][:-1]:  # exclude tail which will move
            # Check each potential move for collision
            if moves['up'] and my_head['x'] == segment['x'] and my_head['y'] + 1 == segment['y']:
                moves['up'] = False
            if moves['down'] and my_head['x'] == segment['x'] and my_head['y'] - 1 == segment['y']:
                moves['down'] = False
            if moves['left'] and my_head['x'] - 1 == segment['x'] and my_head['y'] == segment['y']:
                moves['left'] = False
            if moves['right'] and my_head['x'] + 1 == segment['x'] and my_head['y'] == segment['y']:
                moves['right'] = False
    
    # Convert dict to list of valid moves
    valid_moves = [move for move, valid in moves.items() if valid]
    
    return valid_moves, moves




def get_neck_collision_mask(snake_object, n_actions=4):
    """
    Generates a mask for actions that would lead to collision with the snake's own neck.
    Returns a list of booleans [UP, DOWN, LEFT, RIGHT], True if action is invalid.
    """
    mask = [False] * n_actions # UP, DOWN, LEFT, RIGHT
    if not snake_object.is_alive() or len(snake_object.locations) < 2:
        return mask # No neck or dead snake

    head = snake_object.get_head()
    neck = snake_object.locations[-2] # Second segment from the head

    action_map = {
        Snake.UP: 0,
        Snake.DOWN: 1,
        Snake.LEFT: 2,
        Snake.RIGHT: 3
    }

    for action_val in [Snake.UP, Snake.DOWN, Snake.LEFT, Snake.RIGHT]:
        # Simulate where the head would be if this action is taken
        potential_next_head = snake_object._translate_coordinate_in_direction(np.copy(head), action_val)
        
        if np.array_equal(potential_next_head, neck):
            mask_index = action_map.get(action_val)
            if mask_index is not None:
                mask[mask_index] = True
    return mask

def determine_winner(info, num_snakes, our_snake_name):
    """
    Returns (win_flags, winner_string) for the episode.
    win_flags: list of 0/1 for each snake, 1 if that snake won.
    winner_string: human-readable winner description.
    """
    alive_dict = info.get('alive', {})
    health_dict = info.get('snake_health', {})
    win_flags = [0] * num_snakes
    alive_snakes = [idx for idx, alive in alive_dict.items() if alive]

    # Case 1: Standard win - one snake is alive and others are dead
    if len(alive_snakes) == 1:
        winner_idx = alive_snakes[0]
        win_flags[winner_idx] = 1
        winner = our_snake_name if winner_idx == 0 else f"Snake {winner_idx}"
    # Case 2: Draw - determine winner by health
    elif len(alive_snakes) == 0:
        if health_dict:
            max_health = -1
            max_health_idx = -1
            for idx, health in health_dict.items():
                if health > max_health:
                    max_health = health
                    max_health_idx = idx
            if max_health_idx >= 0:
                win_flags[max_health_idx] = 1
                winner = our_snake_name if max_health_idx == 0 else f"Snake {max_health_idx}"
                winner += f" (most health at draw: {max_health})"
            else:
                winner = "Draw (all died)"
        else:
            winner = "Draw (all died)"
    # Case 3: Multiple snakes alive (shouldn't happen with done=True)
    else:
        if health_dict:
            max_health = -1
            max_health_idx = -1
            for idx in alive_snakes:
                health = health_dict.get(idx, 0)
                if health > max_health:
                    max_health = health
                    max_health_idx = idx
            if max_health_idx >= 0:
                win_flags[max_health_idx] = 1
                winner = our_snake_name if max_health_idx == 0 else f"Snake {max_health_idx}"
                winner += f" (still alive with most health: {max_health})"
            else:
                winner = "Unknown winner (multiple alive)"
        else:
            winner = "Unknown winner (multiple alive)"
    return win_flags, winner

def debug_pause_step(
    episode_num,
    batch_step,
    total_steps,
    json_obj,
    observation,
    obs_tensor,
    agent,
    screen,
    clock,
    snake_colors,
    snake_index=0,
    mute=False,
    fast_visualization=False # Add fast_visualization flag
):
    """
    Pause training for debugging: visualize and print debugging info until RIGHT arrow press or Q quits.
    Returns False if user quits, True to continue.
    """
    # Separator for clarity
    print("\n" + "=" * 60)
    print(f"DEBUG STEP | Episode: {episode_num} | BatchStep: {batch_step} | TotalSteps: {total_steps}")
    print("-" * 60)
    # Visualize current state
    visualize_step(json_obj, snake_colors, screen, clock, fps=10, mute=mute)
    # Print observation matrix channels side-by-side
    try:
        print("Observation matrix (channels side-by-side):")
        rows, cols, channels = observation.shape
        # Header row with channel labels
        header = '   '.join(f"Ch{ch}".center(cols*3) for ch in range(channels))
        print(header)
        # Print each row across channels
        for i in range(rows):
            row_segments = []
            for ch in range(channels):
                # Format each value as 2-digit int
                vals = [f"{int(v):2d}" for v in observation[i, :, ch]]
                row_segments.append(' '.join(vals))
            print(' | '.join(row_segments))
    except Exception as e:
        print(f"Warning: could not print observation matrix: {e}")
    # Heuristic analysis
    h_valid, _, h_mask = heuristic(observation)
    print("Heuristic Analysis:")
    print(f"  Valid Moves: {h_valid}")
    print(f"  Mask Blocks (True=blocked): {h_mask.tolist()}")
    # Game rules valid actions
    gr_valid = get_valid_actions(json_obj['board'], snake_index)
    print("Game Rules Valid Actions:")
    print(f"  {gr_valid}")
    # Agent policy probabilities
    with torch.no_grad():
        logits, _ = agent.actor_critic(obs_tensor.to(agent.device))
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
    probs_str = ", ".join(f"{action_names[i]}={p:.3f}" for i, p in enumerate(probs))
    print(f"Agent Policy Probabilities: {probs_str}")
    print("=" * 60)
    # Wait for user to continue or quit
    cont = wait_for_debug_input_pygame(screen, clock, fps=10, fast_visualization=fast_visualization) # Pass the flag
    return cont


