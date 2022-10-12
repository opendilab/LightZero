"""
The following code is adapted from https://github.com/YeWR/EfficientZero/core/utils.py
"""

import numpy as np
import torch
from scipy.stats import entropy


def get_augmented_data(board_size, play_data):
    """
    Overview:
        augment the data set by rotation and flipping
    Arguments:
        play_data: [(state, mcts_prob, winner_z), ..., ...]
    """
    extend_data = []
    for data in play_data:
        state = data['state']
        mcts_prob = data['mcts_prob']
        winner = data['winner']
        for i in [1, 2, 3, 4]:
            # rotate counterclockwise
            equi_state = np.array([np.rot90(s, i) for s in state])
            equi_mcts_prob = np.rot90(np.flipud(mcts_prob.reshape(board_size, board_size)), i)
            extend_data.append(
                {
                    'state': equi_state,
                    'mcts_prob': np.flipud(equi_mcts_prob).flatten(),
                    'winner': winner
                }
            )
            # flip horizontally
            equi_state = np.array([np.fliplr(s) for s in equi_state])
            equi_mcts_prob = np.fliplr(equi_mcts_prob)
            extend_data.append(
                {
                    'state': equi_state,
                    'mcts_prob': np.flipud(equi_mcts_prob).flatten(),
                    'winner': winner
                }
            )
    return extend_data


def select_action(visit_counts, temperature=1, deterministic=True):
    """
    Overview:
        select action from the root visit counts.
    Arguments:
        - temperature (:obj:`float`): the temperature for the distribution
        - deterministic (:obj:`bool`):  True -> select the argmax, False -> sample from the distribution
    """
    try:
        action_probs = [visit_count_i ** (1 / temperature) for visit_count_i in visit_counts]
    except Exception as error:
        print(error)
    action_probs = [x / sum(action_probs) for x in action_probs]

    if deterministic:
        action_pos = np.argmax([v for v in visit_counts])
    else:
        action_pos = np.random.choice(len(visit_counts), p=action_probs)

    visit_count_distribution_entropy = entropy(action_probs, base=2)
    return action_pos, visit_count_distribution_entropy


def prepare_observation_lst(observation_lst):
    """
    Overview:
        Prepare the observations to satisfy the input format of torch
        [B, S, W, H, C] -> [B, S x C, W, H]
        batch, stack num, width, height, channel
    """
    # B, S, W, H, C
    # observation_lst = np.array(observation_lst, dtype=np.uint8)
    observation_lst = np.array(observation_lst)
    observation_lst = np.moveaxis(observation_lst, -1, 2)

    shape = observation_lst.shape
    observation_lst = observation_lst.reshape((shape[0], -1, shape[-2], shape[-1]))

    return observation_lst


def concat_output_value(output_lst):
    # concat the values of the model output list
    value_lst = []
    for output in output_lst:
        value_lst.append(output.value)

    value_lst = np.concatenate(value_lst)

    return value_lst


def concat_output(output_lst):
    # concat the model output
    value_lst, reward_lst, policy_logits_lst, hidden_state_lst = [], [], [], []
    reward_hidden_state_c_lst, reward_hidden_state_h_lst = [], []
    for output in output_lst:
        value_lst.append(output.value)
        reward_lst.append(output.value_prefix)
        policy_logits_lst.append(output.policy_logits)
        hidden_state_lst.append(output.hidden_state)
        reward_hidden_state_c_lst.append(output.reward_hidden_state[0].squeeze(0))
        reward_hidden_state_h_lst.append(output.reward_hidden_state[1].squeeze(0))

    value_lst = np.concatenate(value_lst)
    reward_lst = np.concatenate(reward_lst)
    policy_logits_lst = np.concatenate(policy_logits_lst)
    hidden_state_lst = np.concatenate(hidden_state_lst)
    reward_hidden_state_c_lst = np.expand_dims(np.concatenate(reward_hidden_state_c_lst), axis=0)
    reward_hidden_state_h_lst = np.expand_dims(np.concatenate(reward_hidden_state_h_lst), axis=0)

    return value_lst, reward_lst, policy_logits_lst, hidden_state_lst, (
        reward_hidden_state_c_lst, reward_hidden_state_h_lst
    )


def mask_nan(x: torch.Tensor) -> torch.Tensor:
    nan_part = torch.isnan(x)
    x[nan_part] = 0.
    return x
