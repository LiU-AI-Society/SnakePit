import numpy as np
import onnxruntime
import torch
import os
# Assuming heuristic.py is in the parent directory or accessible via Python path
from heuristic import heuristic

# Global cache for loaded models to avoid reloading
LOADED_MODELS = {}
DEFAULT_MODEL_PATH = "good_snakes/bad_snake/ppo_ep1.onnx"  # Relative path to default model

def get_default_model_path():
    # Return the default model path as is (relative to current working directory)
    return os.path.join(os.getcwd(), DEFAULT_MODEL_PATH)

def load_onnx_model_for_opponent(model_path):
    if model_path in LOADED_MODELS:
        return LOADED_MODELS[model_path]
    try:
        session = onnxruntime.InferenceSession(model_path)
        LOADED_MODELS[model_path] = session
        print(f"Successfully loaded opponent ONNX model: {model_path}")
        return session
    except Exception as e:
        print(f"Error loading ONNX model {model_path}: {e}")
        return None

class ModelOpponent:
    def __init__(self, model_path, device_str="cpu"):
        self.session = load_onnx_model_for_opponent(model_path)
        self.model_path = model_path
        self.device = torch.device(device_str) 

        if self.session:
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
        else:
            print(f"Failed to initialize ModelOpponent due to load failure: {model_path}")

    def choose_action(self, game_matrix_observation, snake_index_in_json_irrelevant):
        """
        Chooses an action based on the ONNX model and heuristic.
        game_matrix_observation: Numpy array (H, W, 3), the 3-channel observation for this snake.
        snake_index_in_json_irrelevant: The index of this snake (not used by model directly).
        """
        if not self.session:
            print(f"ModelOpponent ({self.model_path}) not loaded, returning random action.")
            return np.random.randint(0, 4) 

        _, _, action_mask_tensor = heuristic(game_matrix_observation)
        
        obs_tensor_model = torch.tensor(
            game_matrix_observation.transpose(2, 0, 1),
            dtype=torch.float32
        ).unsqueeze(0)

        obs_numpy = obs_tensor_model.cpu().numpy()
        inputs = {self.input_name: obs_numpy}

        try:
            model_outputs = self.session.run(self.output_names, inputs)
            action_logits = torch.tensor(model_outputs[0]) 

            masked_logits = action_logits.masked_fill(action_mask_tensor.unsqueeze(0), -float('inf'))
            
            action = torch.argmax(masked_logits, dim=1).item()
            return action

        except Exception as e:
            print(f"Error during model inference for {self.model_path}: {e}")
            return np.random.randint(0, 4)

def get_model_opponent_action_function(model_path):
    """Factory to create a bound method for the opponent."""
    # Handle the special "model_snake" identifier to use the default path
    actual_model_path = model_path
    display_name = os.path.splitext(os.path.basename(actual_model_path))[0]

    if model_path == "model_snake": # Special keyword for default model
        actual_model_path = get_default_model_path()
        # Ensure the default path exists
        if not os.path.exists(actual_model_path):
            print(f"Default model path {actual_model_path} does not exist. Cannot load default model opponent.")
            return None, "model_snake_default_not_found" # Return None and a name indicating failure
        display_name = os.path.splitext(os.path.basename(actual_model_path))[0] # e.g., "right_side_snake"
        print(f"Using default model for 'model_snake': {actual_model_path}")

    opponent = ModelOpponent(actual_model_path)
    if not opponent.session: 
        return None, display_name # Return None if session failed, but still return the name
    return opponent.choose_action, display_name
