"""
The following code is adapted from https://github.com/YeWR/EfficientZero/blob/main/core/model.py
"""

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn


@dataclass
class NetworkOutput:
    # output format of the model
    value: float
    reward: float
    policy_logits: List[float]
    hidden_state: List[float]
    reward_hidden_state: object


def renormalize(input, first_dim=1):
    # normalize the input (states)
    if first_dim < 0:
        first_dim = len(input.shape) + first_dim
    flat_input = input.view(*input.shape[:first_dim], -1)
    max_val = torch.max(flat_input, first_dim, keepdim=True).values
    min_val = torch.min(flat_input, first_dim, keepdim=True).values
    flat_input = (flat_input - min_val) / (max_val - min_val)

    return flat_input.view(*input.shape)


class BaseNet(nn.Module):

    def __init__(self, lstm_hidden_size):
        """
        Overview:
            Base Network for EfficientZero.
        Arguments：
            - lstm_hidden_size (:obj:`int`): dim of lstm hidden
        """
        super(BaseNet, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size

    def prediction(self, state):
        raise NotImplementedError

    def representation(self, obs_history):
        raise NotImplementedError

    def dynamics(self, state, reward_hidden, action):
        raise NotImplementedError

    def initial_inference(self, obs) -> NetworkOutput:
        num = obs.size(0)
        hidden_state = self.representation(obs)
        policy_logits, value = self.prediction(hidden_state)
        # zero initialization for reward hidden states
        reward_hidden = (torch.zeros(1, num, self.lstm_hidden_size), torch.zeros(1, num, self.lstm_hidden_size))
        return NetworkOutput(value, [0. for _ in range(num)], policy_logits, hidden_state, reward_hidden)

    def recurrent_inference(
            self, hidden_state: torch.Tensor, reward_hidden: torch.Tensor, action: torch.Tensor
    ) -> NetworkOutput:
        hidden_state, reward_hidden, reward = self.dynamics(hidden_state, reward_hidden, action)
        policy_logits, value = self.prediction(hidden_state)
        return NetworkOutput(value, reward, policy_logits, hidden_state, reward_hidden)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients: torch.Tensor):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = g
