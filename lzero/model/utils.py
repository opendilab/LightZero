import torch
import numpy as np


def renormalize(input, first_dim=1):
    # normalize the input (states)
    if first_dim < 0:
        first_dim = len(input.shape) + first_dim
    flat_input = input.view(*input.shape[:first_dim], -1)
    max_val = torch.max(flat_input, first_dim, keepdim=True).values
    min_val = torch.min(flat_input, first_dim, keepdim=True).values
    flat_input = (flat_input - min_val) / (max_val - min_val)

    return flat_input.view(*input.shape)


def get_dynamic_mean(model):
    dynamic_mean = np.abs(model.conv.weight.detach().cpu().numpy().reshape(-1)).tolist()

    for block in model.resblocks:
        for name, param in block.named_parameters():
            dynamic_mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
    dynamic_mean = sum(dynamic_mean) / len(dynamic_mean)
    return dynamic_mean


def get_reward_mean(model):
    reward_w_dist = model.conv1x1_reward.weight.detach().cpu().numpy().reshape(-1)

    for name, param in model.fc.named_parameters():
        temp_weights = param.detach().cpu().numpy().reshape(-1)
        reward_w_dist = np.concatenate((reward_w_dist, temp_weights))
    reward_mean = np.abs(reward_w_dist).mean()
    return reward_w_dist, reward_mean


def get_params_mean(xzero_model):
    representation_mean = xzero_model.representation_network.get_param_mean()
    dynamic_mean = xzero_model.dynamics_network.get_dynamic_mean()
    reward_w_dist, reward_mean = xzero_model.dynamics_network.get_reward_mean()

    return reward_w_dist, representation_mean, dynamic_mean, reward_mean
