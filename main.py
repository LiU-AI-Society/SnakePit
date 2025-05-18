import torch
import numpy as np
import typing
import time
import onnxruntime as ort
import argparse
from heuristic import heuristic
from utils import load_model, load_model_onnx, battle_snake_game_state_to_observation

# Global model for inference
MODEL_SESSION = None
MODEL_PATH = None
MODEL_INPUT_SHAPE = None  # <-- Track model input shape
DEVICE = torch.device("metal" if torch.cuda.is_available() else "cpu")

def action_index_to_move(action_idx):
    """Convert action index to move string"""
    action_map = {
        0: "up",
        1: "down",
        2: "left",
        3: "right"
    }
    return action_map.get(action_idx, "up")  # Default to up if invalid index

def move_to_action_index(move):
    """Convert move string to action index"""
    action_map = {
        "up": 0,
        "down": 1,
        "left": 2,
        "right": 3
    }
    return action_map.get(move, 0)  # Default to up if invalid move

def check_model_input_shape(session, obs):
    # Ensure the model input shape matches the observation shape
    input_shape = session.get_inputs()[0].shape
    obs_shape = obs.transpose(2,0,1).shape  # (C,H,W)
    # ONNX shape: (batch, C, H, W)
    if len(input_shape) == 4:
        _, c, h, w = input_shape
        if (c, h, w) != obs_shape:
            raise ValueError(
                f"Model expects input shape (C,H,W)=({c},{h},{w}), "
                f"but got ({obs_shape[0]},{obs_shape[1]},{obs_shape[2]}). "
                "Check your model export and board size!"
            )

# API Functions

def info() -> typing.Dict:
    """
    Return snake customization options
    """
    return {
        "apiversion": "1",
        "author": "DQNSnake",
        "color": "#FF9D00",  # orange
        "head": "smart",
        "tail": "bolt",
    }

def start(game_state: typing.Dict):
    """Called when game starts"""
    global MODEL_SESSION
    board = game_state['board']
    map_size = (board['height'], board['width'])
    channels = 3  # food + our snake + other snakes
    
    # Number of snakes to determine input channels
    num_snakes = len(game_state['board']['snakes'])
    
    suc, MODEL_SESSION = load_model_onnx(model_path=MODEL_PATH)
    opponents = [
        ("hungry bot" if s['id']=="gs_FjqHmRDthjQDpGdwbXSdWV69" else s['id'])
        for s in game_state['board']['snakes'] 
        if s['id'] != game_state['you']['id']
    ]
    print(
        f"[Start] "
        f"Mode={'ONNX':<4} | "
        f"Model={MODEL_PATH:<30} | "
        f"Opp={(','.join(opponents) or 'None'):<20} | "
        f"Size={map_size[0]:>2}x{map_size[1]:<2}"
    )

def end(game_state: typing.Dict):
    """Called when game ends"""
    alive = {s['id'] for s in game_state['board']['snakes']}
    you = game_state['you']['id']
    result = "won" if you in alive else "lost"
    healths = ", ".join(
        f"{('hungry bot' if s['id']=='gs_FjqHmRDthjQDpGdwbXSdWV69' else s['id'])}={s.get('health', '?')}"
        for s in game_state['board']['snakes']
    )
    print(f"[End  ] You {result:<4} | Health={healths:<40}")
    print("-" * 60)

def move(game_state: typing.Dict) -> typing.Dict:
    """
    Choose a move for the snake using the trained PPO ONNX model
    """
    global MODEL_SESSION
    if MODEL_SESSION is None:
        suc, MODEL_SESSION = load_model_onnx(MODEL_PATH)
        print("[Move ] ONNX model loaded")

    obs = battle_snake_game_state_to_observation(game_state)
    # --- Check model input shape matches observation shape ---
    try:
        check_model_input_shape(MODEL_SESSION, obs)
    except Exception as e:
        print(f"[ERROR] {e}")
        # Fallback: random move
        return {"move": np.random.choice(["up","down","left","right"])}

    valid, _, _ = heuristic(obs)
    if not valid:
        print("[Move ] No valid — random")
        return {"move": np.random.choice(["up","down","left","right"])}
    mask = np.zeros(4, bool)
    for m in valid: mask[move_to_action_index(m)] = True

    inp = torch.tensor(obs.transpose(2,0,1), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    logits, _ = MODEL_SESSION.run(None, {MODEL_SESSION.get_inputs()[0].name: inp.cpu().numpy()})
    lg = logits[0]
    lg_masked = np.where(mask, lg, -np.inf)
    probs = np.exp(lg_masked - lg_masked.max())
    probs /= probs.sum()
    dirs = ["up","down","left","right"]
    prob_str = ", ".join(f"{d}={probs[i]:.2f}" for i,d in enumerate(dirs))
    choice = dirs[int(np.argmax(probs))]
    print(
        f"[Move ] "
        f"Valid={','.join(valid):<20} | "
        f"Probs={prob_str:<25} | "
        f"Chosen={choice:<5}"
    )
    return {"move": choice}

if __name__ == "__main__":
    from server import run_server
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model file")
    args = parser.parse_args()
    MODEL_PATH = args.model
    if MODEL_PATH.endswith(".pt"):
        corrected = MODEL_PATH.replace(".pt", ".onnx")
        print(f"[Info  ] .pt->.onnx: {corrected:<30}")
        MODEL_PATH = corrected
    run_server({"info": info, "start": start, "move": move, "end": end})