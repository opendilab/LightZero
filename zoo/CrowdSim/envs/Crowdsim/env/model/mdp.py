from collections import namedtuple
from itertools import product
import torch
import numpy as np


# State
class HumanState(object):
    def __init__(self, px, py, theta, aoi):
        self.px = px
        self.py = py
        self.theta = theta
        self.aoi = aoi
        self.position = (self.px, self.py)

    def __add__(self, other):
        return other + (self.px, self.py, self.theta, self.aoi)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.theta, self.aoi]])

    def to_tuple(self):
        return self.px, self.py, self.theta, self.aoi


class RobotState(object):
    def __init__(self, px, py, theta, energy):
        self.px = px
        self.py = py
        self.theta = theta
        self.energy = energy

        self.position = (self.px, self.py)

    def __add__(self, other):
        return other + (self.px, self.py, self.theta, self.energy)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.theta, self.energy]])

    def to_tuple(self):
        return self.px, self.py, self.theta, self.energy


class JointState(object):
    def __init__(self, robot_states, human_states):
        for robot_state in robot_states:
            assert isinstance(robot_state, RobotState)
        for human_state in human_states:
            assert isinstance(human_state, HumanState)

        self.robot_states = robot_states
        self.human_states = human_states

    def to_tensor(self, add_batch_size=False, device=None):
        robot_states_tensor = torch.tensor([robot_state.to_tuple() for robot_state in self.robot_states],
                                           dtype=torch.float32)
        human_states_tensor = torch.tensor([human_state.to_tuple() for human_state in self.human_states],
                                           dtype=torch.float32)

        if add_batch_size:  # True
            robot_states_tensor = robot_states_tensor.unsqueeze(0)
            human_states_tensor = human_states_tensor.unsqueeze(0)

        if device is not None:
            robot_states_tensor = robot_states_tensor.to(device)
            human_states_tensor = human_states_tensor.to(device)

        return robot_states_tensor, human_states_tensor
    
    def to_array(self):
        robot_states_array = np.array([robot_state.to_tuple() for robot_state in self.robot_states])
        human_states_array = np.array([human_state.to_tuple() for human_state in self.human_states])

        return robot_states_array, human_states_array


def build_action_space(config):
    robot_num = config.robot_num

    # dx, dy
    one_uav_action_space = config.one_uav_action_space
    action_space = list(product(one_uav_action_space, repeat=robot_num))

    return np.array(action_space)


if __name__ == "__main__":
    print(build_action_space())
