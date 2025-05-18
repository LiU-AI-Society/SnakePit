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
import gym
import math
import onnx
import onnxruntime as ort
import torch


def save_onnx_model_dqn(model, dummy_input, save_path, input_names=["input"], output_names=["output"], opset_version=11):
    model.eval()
    
    if isinstance(dummy_input, (tuple, list)):
        inputs = dummy_input
    else:
        inputs = (dummy_input,)

    torch.onnx.export(
        model,
        inputs,
        save_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        dynamic_axes={
            input_names[0]: {0: 'batch_size'},
            input_names[1]: {0: 'batch_size'},
            output_names[0]: {0: 'batch_size'}
        }
    )
    print(f"Exported ONNX model to: {save_path}")


def save_onnx_model_ppo(model, dummy_input, save_path,
                    input_names=["input"], output_names=["logits", "value"], opset_version=11):
    model.eval()

    if isinstance(dummy_input, (tuple, list)):
        inputs = dummy_input
    else:
        inputs = (dummy_input,)

    torch.onnx.export(
        model,
        inputs,
        save_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        dynamic_axes={
            input_names[0]: {0: 'batch_size'},
            output_names[0]: {0: 'batch_size'},
            output_names[1]: {0: 'batch_size'},
        }
    )
    print(f"Exported ONNX model to: {save_path}")


def load_onnx_model(onnx_path):
    """
    Loads an ONNX model and returns an ONNX Runtime session.

    Args:
        onnx_path: Path to the .onnx model file.

    Returns:
        ort.InferenceSession instance.
    """
    # Optional: check if the model is well formed
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Create an ONNX Runtime session
    session = ort.InferenceSession(onnx_path)
    return session
def is_coord_in(coord, array):
    for a in array:
        if a[0] == coord[0] and a[1] == coord[1]:
            return True
    return False

def get_random_coordinates(map_size, n, excluding=[]):
    '''
    Helper function to get n number of random coordinates based on the map
    Parameters:
    ----------
    map_size, (int, int)
        Size of the map with possible coordinates
    
    n, int
        number of coordinates to get

    excluding: [(int, int)]
        A list of coordinates to not include in the randomly generated coordinates
    '''
    coordinates_indexes = []
    coordinates = []
    count = 0
    for i in range(map_size[0]):
        for j in range(map_size[1]):
            if is_coord_in(coord=(i, j), array=excluding):
                continue
            coordinates.append((i, j))
            coordinates_indexes.append(count)
            count += 1

    indexes = np.random.choice(coordinates_indexes, n, replace=False)
    random_coordinates = np.array(coordinates)[indexes]
    return random_coordinates

def generate_coordinate_list_from_binary_map(map_image):
    '''
    Helper function to convert binary maps into a list of coordinates
    '''
    coordinate_list = []
    for i in range(map_image.shape[0]):
        for j in range(map_image.shape[1]):
            if map_image[i][j] > 0:
                coordinate_list.append((i, j))
    return coordinate_list

class MultiAgentActionSpace(list):
    '''
    Code taken from https://github.com/koulanurag/ma-gym/blob/master/ma_gym/envs/utils/action_space.py
    '''
    def __init__(self, agents_action_space):
        for x in agents_action_space:
            assert isinstance(x, gym.spaces.space.Space)

        super(MultiAgentActionSpace, self).__init__(agents_action_space)
        self._agents_action_space = agents_action_space
        self.n = len(self._agents_action_space)

    def sample(self):
        """ samples action for each agent from uniform distribution"""
        return [agent_action_space.sample() for agent_action_space in self._agents_action_space]

def get_distance(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
