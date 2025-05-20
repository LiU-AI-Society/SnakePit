# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# or in the "license" file accompanying this file. This file is distributed 
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either 
# express or implied. See the License for the specific language governing 
# permissions and limitations under the License.

import numpy as np

class Rewards:
    '''
    Base class to set up rewards for the battlesnake gym
    '''
    def get_reward(self, name, snake_id, episode):
        raise NotImplemented()

class SimpleRewards(Rewards):
    '''
    Simple class to handle a fixed reward scheme
    '''
    def __init__(self):
        self.reward_dict = {
            "another_turn": 0,  # Survived another turn
            "ate_food": 0,  # Ate a piece of food
            "won": 0,  # Won the game (e.g., last snake alive)
            "died": 0,  # Died for any reason (general death event)
            "ate_another_snake": 0,  # Eliminated another snake by eating its head
            "hit_wall": 1,  # Collided with a wall
            "hit_other_snake": 0,  # Own head collided with another snake's body
            "hit_self": 0,  # Own head collided with own body
            "was_eaten": 0,  # Was eaten by a larger snake
            "other_snake_hit_body": 0,  # Another snake collided with own snake's body (and died)
            "forbidden_move": 0,  # Attempted an illegal move (e.g., moving into own neck)
            "starved": 0  # Died due to health reaching zero
        }

    def get_reward(self, name, snake_id, episode):
        return self.reward_dict[name]
