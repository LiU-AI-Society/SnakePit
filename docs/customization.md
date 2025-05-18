# 🐍 Customize Your Snake

You can influence your snake’s behavior, learning process, and appearance by modifying a few key files. Below is a guide to what you can change and what it means.

---

## 🎯 1. Reward Function (`Gym/rewards.py`)

The reward system defines how your snake evaluates its actions. The default reward structure is:

```python
self.reward_dict = {
    "another_turn": +0.01,     # Small reward for surviving each turn
    "ate_food": 1,             # Reward for eating food
    "won": 10,                 # Large reward for winning the game
    "died": -1,                # Penalty for dying
    "ate_another_snake": 5,    # Big reward for eating other snakes
    "hit_wall": -10,           # Severe penalty for wall collision
    "hit_other_snake": 0,      # Neutral for hitting other snake
    "hit_self": -1,            # Penalty for self-collision
    "was_eaten": -1,           # Penalty for being eaten
    "other_snake_hit_body": -1,# Penalty when others hit your body
    "forbidden_move": -1,      # Penalty for invalid moves (hitting neck)
    "starved": -1              # Penalty for dying from starvation
}
```

### Reward Tuning Guidelines

1. **Survival Rewards**
   - `another_turn`: Small positive reward encourages survival
   - `died`, `starved`: Negative rewards discourage early death

2. **Food Strategy**
   - `ate_food`: Moderate reward encourages food gathering
   - Balance with survival rewards to prevent reckless food pursuit

3. **Combat Behavior**
   - `ate_another_snake`: High reward promotes aggressive play
   - `was_eaten`: Penalty encourages caution
   - `hit_other_snake`: Adjust to control aggression level

4. **Movement Penalties**
   - `hit_wall`: Large penalty prevents wall collisions
   - `hit_self`: Discourages self-intersection
   - `forbidden_move`: Prevents illegal moves

### Example Modifications

For an aggressive snake:
```python
self.reward_dict.update({
    "ate_another_snake": 10,    # Double reward for eliminations
    "hit_other_snake": 0.5,     # Small reward for aggressive moves
    "was_eaten": -2             # Higher penalty for being eliminated
})
```

For a cautious snake:
```python
self.reward_dict.update({
    "another_turn": 0.05,       # Higher survival reward
    "ate_food": 2,              # Focus on food gathering
    "ate_another_snake": 3,     # Reduced combat reward
    "hit_other_snake": -0.5     # Discourage risky encounters
})
```

For a balanced snake:
```python
self.reward_dict.update({
    "another_turn": 0.02,       # Moderate survival reward
    "ate_food": 1.5,            # Slightly increased food reward
    "ate_another_snake": 5,     # Keep original combat reward
    "hit_wall": -15             # Stronger wall avoidance
})
```

Modify these values to shape your snake's behavior, then test different combinations in training to find the optimal strategy for your goals.

## 🧠 2. Opponent Snake Behavior (`snakes/random_snake.py`)

Opponent behavior influences the environment your snake learns in. The default opponent chooses random actions:

```python
def choose_action(observation, info):
    return random.choice([0, 1, 2, 3])  # UP, DOWN, LEFT, RIGHT
```

Do so by adding a new file with the name of the snake you want and make sure the file contains the choose action function. Then when training use the --opponents flag to include those snakes you have designed. Add new opponents to folder "snakes".

```bash
python train.py --opponents hungry_snake
```

## 🛠️ 3. Training Hyperparameters in your training script

These affect how your snake learns during training:

```python
# ---- Config ----

TOTAL_TIMESTEPS = 1000_000
SAVE_INTERVAL = 1000
N_STEPS_COLLECT = 2000 
PPO_BATCH_SIZE = 64
N_EPOCHS_PPO = 4
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_CLIP = 0.2

YOUR_SNAKE_INDEX = 0
OBS_TYPE = "flat-51s"
MAP_SIZE = (11, 11)          # Size of the board

```


## 🎯 4. Heuristic Function (`heuristic.py`)

The heuristic function validates moves and provides action masking. You can customize this function to add sophisticated decision-making logic that works alongside your trained model.

### Default Implementation

```python
# filepath: /Users/isakknutsson/Desktop/AIS/SnakePit/heuristic.py
def heuristic(matrix):
    """
    Validates moves using the game state matrix.
    
    Args:
        matrix: Game state with shape (height, width, 3)
               channel 0: food (1)
               channel 1: your snake (head=5, body=1)
               channel 2: other snakes (head=5, body=1)
    
    Returns:
        valid_moves: List of safe directions
        moves: Dictionary of direction validity
        action_mask: Boolean tensor for [up, down, left, right]
    """
```

### Customization Possibilities

1. **Safety Enhancements**
- Snake collision prediction
- Trap detection
- Space analysis using flood fill
- Path blocking detection

2. **Strategic Elements**
- Food path weighting
- Territory control
- Opponent head tracking
- Safe retreat routes

3. **Model Integration**
- Override model on low confidence
- Risk-based decision weights
- Hybrid decision making
- Emergency fallback logic

4. **Advanced Features**
- Area control scoring
- Distance-based risk assessment
- Food path optimization
- Head-to-head conflict resolution

### How to Modify

1. Create a new function that wraps the base heuristic:
```python
def advanced_heuristic(matrix, critic_value=None):
    valid_moves, moves, action_mask = heuristic(matrix)
    
    # Add your custom logic here
    # Example: Override on low confidence
    if critic_value is not None and critic_value < 0.3:
        # Implement safety-first logic
        pass
        
    return valid_moves, moves, action_mask
```

2. Replace the standard heuristic in `train.py` and `main.py`:
```python
# In train.py and main.py, update the heuristic import
from heuristic import advanced_heuristic as heuristic
```

Remember:
- Keep modifications efficient
- Test thoroughly with different scenarios
- Balance safety vs opportunity
- Consider performance impact