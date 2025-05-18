from snake_gym import BattlesnakeGym
from test_utils import simulate_snake
from snake import Snake
import numpy as np

def clean_json(d):
    if isinstance(d, dict):
        return {k: clean_json(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [clean_json(v) for v in d]
    elif isinstance(d, np.integer):
        return int(d)
    else:
        return d

snake_locations = [(1, 1), (6, 6)]
map_size = (10, 10)
food_location = [(5, 5)]  # No food for now
num_snakes = 2

env = BattlesnakeGym(map_size=map_size, number_of_snakes=num_snakes,
                     snake_spawn_locations=snake_locations,
                     food_spawn_locations=food_location)

state = env._get_observation()
print("Before move:")
print(state[:, :, 2])

# Move RIGHT
action = [3, 3]
observation, _, _, info = env.step(action)

state = env._get_observation()
print("After move:")
print(state[:, :, 2])

js = clean_json(env.get_json())

print(js)
