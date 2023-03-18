from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import entropy


def negative_cosine_similarity(x1, x2):
    """
    Overview:
        consistency loss function: the negative cosine similarity.
    Arguments:
        - x1 (:obj:`torch.Tensor`): shape (batch_size, dim), e.g. (256, 512)
        - x2 (:obj:`torch.Tensor`): shape (batch_size, dim), e.g. (256, 512)
    Returns:
        (x1 * x2).sum(dim=1) is the cosine similarity between vector x1 and x2.
        The cosine similarity always belongs to the interval [-1, 1].
        For example, two proportional vectors have a cosine similarity of 1,
        two orthogonal vectors have a similarity of 0,
        and two opposite vectors have a similarity of -1.
         -(x1 * x2).sum(dim=1) is consistency loss, i.e. the negative cosine similarity.
    Reference:
        https://en.wikipedia.org/wiki/Cosine_similarity
    """
    x1 = F.normalize(x1, p=2., dim=-1, eps=1e-5)
    x2 = F.normalize(x2, p=2., dim=-1, eps=1e-5)
    return -(x1 * x2).sum(dim=1)


def get_max_entropy(action_shape: int) -> None:
    p = 1.0 / action_shape
    return -action_shape * p * np.log2(p)


def select_action(visit_counts, temperature=1, deterministic=True):
    """
    Overview:
        select action from the root visit counts.
    Arguments:
        - temperature (:obj:`float`): the temperature for the distribution
        - deterministic (:obj:`bool`):  True -> select the argmax, False -> sample from the distribution
    """
    action_probs = [visit_count_i ** (1 / temperature) for visit_count_i in visit_counts]
    action_probs = [x / sum(action_probs) for x in action_probs]

    if deterministic:
        action_pos = np.argmax([v for v in visit_counts])
    else:
        action_pos = np.random.choice(len(visit_counts), p=action_probs)

    visit_count_distribution_entropy = entropy(action_probs, base=2)
    return action_pos, visit_count_distribution_entropy


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
        try:
            reward_lst.append(output.value_prefix)
        except:
            reward_lst.append(output.reward)

        policy_logits_lst.append(output.policy_logits)
        hidden_state_lst.append(output.hidden_state)
        try:
            reward_hidden_state_c_lst.append(output.reward_hidden_state[0].squeeze(0))
            reward_hidden_state_h_lst.append(output.reward_hidden_state[1].squeeze(0))
        except:
            pass

    value_lst = np.concatenate(value_lst)
    reward_lst = np.concatenate(reward_lst)
    policy_logits_lst = np.concatenate(policy_logits_lst)
    hidden_state_lst = np.concatenate(hidden_state_lst)
    try:
        reward_hidden_state_c_lst = np.expand_dims(np.concatenate(reward_hidden_state_c_lst), axis=0)
        reward_hidden_state_h_lst = np.expand_dims(np.concatenate(reward_hidden_state_h_lst), axis=0)
        return value_lst, reward_lst, policy_logits_lst, hidden_state_lst, (
            reward_hidden_state_c_lst, reward_hidden_state_h_lst
        )
    except:
        return value_lst, reward_lst, policy_logits_lst, hidden_state_lst


def to_torch_float_tensor(data_list: List, device):
    output_data_list = []
    for data in data_list:
        output_data_list.append(torch.from_numpy(data).to(device).float())
    return output_data_list


def to_detach_cpu_numpy(data_list: List):
    output_data_list = []
    for data in data_list:
        output_data_list.append(data.detach().cpu().numpy())
    return output_data_list


def ez_network_output_unpack(network_output):
    hidden_state = network_output.hidden_state  # shape:（batch_size, lstm_hidden_size, num_unroll_steps+1, num_unroll_steps+1）
    value_prefix = network_output.value_prefix  # shape: (batch_size, support_support_size)
    reward_hidden_state = network_output.reward_hidden_state  # shape: {tuple: 2} -> (1, batch_size, 512)
    value = network_output.value  # shape: (batch_size, support_support_size)
    policy_logits = network_output.policy_logits  # shape: (batch_size, action_space_size)
    return hidden_state, value_prefix, reward_hidden_state, value, policy_logits


def mz_network_output_unpack(network_output):
    hidden_state = network_output.hidden_state  # shape:（batch_size, lstm_hidden_size, num_unroll_steps+1, num_unroll_steps+1）
    reward = network_output.reward  # shape: (batch_size, support_support_size)
    value = network_output.value  # shape: (batch_size, support_support_size)
    policy_logits = network_output.policy_logits  # shape: (batch_size, action_space_size)
    return hidden_state, reward, value, policy_logits
