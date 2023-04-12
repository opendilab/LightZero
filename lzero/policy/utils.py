from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import entropy


def prepare_obs(obs_batch_ori, cfg):
    """
    Overview:
        Prepare the observations for the model, including:
        1. convert the obs to torch tensor
        2. stack the obs
        3. calculate the consistency loss
    Arguments:
        - obs_batch_ori (:obj:`np.ndarray`): the original observations in a batch style
        - cfg (:obj:`EasyDict`): the config dict
    Returns:
        - obs_batch (:obj:`torch.Tensor`): the stacked observations
        - obs_target_batch (:obj:`torch.Tensor`): the stacked observations for calculating consistency loss
    """
    obs_target_batch = None
    if cfg.model.model_type == 'conv':
        # for 3-dimensional image obs
        """
        ``obs_batch_ori`` is the original observations in a batch style, shape is:
        (batch_size, stack_num+num_unroll_steps, W, H, C) -> (batch_size, (stack_num+num_unroll_steps)*C, W, H )

        e.g. in pong: stack_num=4, num_unroll_steps=5
        (4, 9, 96, 96, 3) -> (4, 9*3, 96, 96) = (4, 27, 96, 96)

        the second dim of ``obs_batch_ori``:
        timestep t:     1,   2,   3,  4,    5,   6,   7,   8,     9
        channel_num:    3    3    3   3     3    3    3    3      3
                       ---, ---, ---, ---,  ---, ---, ---, ---,   ---
        """
        obs_batch_ori = torch.from_numpy(obs_batch_ori).to(cfg.device).float()
        # ``obs_batch`` is used in ``initial_inference()``, which is the first stacked obs at timestep t in
        # ``obs_batch_ori``. shape is (4, 4*3, 96, 96) = (4, 12, 96, 96)
        obs_batch = obs_batch_ori[:, 0:cfg.model.frame_stack_num * cfg.model.image_channel, :, :]

        if cfg.model.self_supervised_learning_loss:
            # ``obs_target_batch`` is only used for calculate consistency loss, which take the all obs other than
            # timestep t1, and is only performed in the last 8 timesteps in the second dim in ``obs_batch_ori``.
            obs_target_batch = obs_batch_ori[:, cfg.model.image_channel:, :, :]
    elif cfg.model.model_type == 'mlp':
        # for 1-dimensional vector obs
        """
        ``obs_batch_ori`` is the original observations in a batch style, shape is:
        (batch_size, stack_num+num_unroll_steps, obs_shape) -> (batch_size, (stack_num+num_unroll_steps)*obs_shape)

        e.g. in cartpole: stack_num=1, num_unroll_steps=5, obs_shape=4
        (4, 6, 4) -> (4, 6*4) = (4, 24)

        the second dim of ``obs_batch_ori``:
        timestep t:     1,   2,      3,     4,    5,   6,  
        obs_shape:      4    4       4      4     4    4
                       ----, ----,  ----, ----,  ----,  ----, 
        """
        obs_batch_ori = torch.from_numpy(obs_batch_ori).to(cfg.device).float()
        # ``obs_batch`` is used in ``initial_inference()``, which is the first stacked obs at timestep t1 in
        # ``obs_batch_ori``. shape is (4, 4*3) = (4, 12)
        obs_batch = obs_batch_ori[:, 0:cfg.model.frame_stack_num * cfg.model.observation_shape]

        if cfg.model.self_supervised_learning_loss:
            # ``obs_target_batch`` is only used for calculate consistency loss, which take the all obs other than
            # timestep t1, and is only performed in the last 8 timesteps in the second dim in ``obs_batch_ori``.
            obs_target_batch = obs_batch_ori[:, cfg.model.observation_shape:]

    return obs_batch, obs_target_batch


def negative_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
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
    """
    Overview:
        get the max entropy of the action space.
    Arguments:
        - action_shape (:obj:`int`): the shape of the action space
    Returns:
        - max_entropy (:obj:`float`): the max entropy of the action space
    """
    p = 1.0 / action_shape
    return -action_shape * p * np.log2(p)


def select_action(visit_counts: np.array, temperature: float = 1, deterministic: bool = True) -> Tuple:
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


def concat_output_value(output_lst: List) -> np.array:
    """
    Overview:
        concat the values of the model output list.
    Arguments:
        - output_lst (:obj:`List`): the model output list
    Returns:
        - value_lst (:obj:`np.array`): the values of the model output list
    """
    # concat the values of the model output list
    value_lst = []
    for output in output_lst:
        value_lst.append(output.value)

    value_lst = np.concatenate(value_lst)

    return value_lst


def concat_output(output_lst: List, data_type: str = 'muzero') -> Tuple:
    """
    Overview:
        concat the model output.
    Arguments:
        - output_lst (:obj:`List`): the model output list
        - data_type (:obj:`str`): the data type, should be 'muzero' or 'efficientzero'
    Returns:
        - value_lst (:obj:`np.array`): the values of the model output list
    """
    assert data_type in ['muzero', 'efficientzero'], "data_type should be 'muzero' or 'efficientzero'"
    # concat the model output
    value_lst, reward_lst, policy_logits_lst, hidden_state_lst = [], [], [], []
    reward_hidden_state_c_lst, reward_hidden_state_h_lst = [], []
    for output in output_lst:
        value_lst.append(output.value)
        if data_type == 'muzero':
            reward_lst.append(output.reward)
        elif data_type == 'efficientzero':
            reward_lst.append(output.value_prefix)

        policy_logits_lst.append(output.policy_logits)
        hidden_state_lst.append(output.hidden_state)
        if data_type == 'efficientzero':
            reward_hidden_state_c_lst.append(output.reward_hidden_state[0].squeeze(0))
            reward_hidden_state_h_lst.append(output.reward_hidden_state[1].squeeze(0))

    value_lst = np.concatenate(value_lst)
    reward_lst = np.concatenate(reward_lst)
    policy_logits_lst = np.concatenate(policy_logits_lst)
    hidden_state_lst = np.concatenate(hidden_state_lst)
    if data_type == 'muzero':
        return value_lst, reward_lst, policy_logits_lst, hidden_state_lst
    elif data_type == 'efficientzero':
        reward_hidden_state_c_lst = np.expand_dims(np.concatenate(reward_hidden_state_c_lst), axis=0)
        reward_hidden_state_h_lst = np.expand_dims(np.concatenate(reward_hidden_state_h_lst), axis=0)
        return value_lst, reward_lst, policy_logits_lst, hidden_state_lst, (
            reward_hidden_state_c_lst, reward_hidden_state_h_lst
        )


def to_torch_float_tensor(data_list: List, device) -> List:
    """
    Overview:
        convert the data list to torch float tensor
    Arguments:
        - data_list (:obj:`List`): the data list
        - device (:obj:`torch.device`): the device
    Returns:
        - output_data_list (:obj:`List`): the output data list
    """
    output_data_list = []
    for data in data_list:
        output_data_list.append(torch.from_numpy(data).to(device).float())
    return output_data_list


def to_detach_cpu_numpy(data_list: List) -> List:
    """
    Overview:
        convert the data list to detach cpu numpy
    Arguments:
        - data_list (:obj:`List`): the data list
    Returns:
        - output_data_list (:obj:`List`): the output data list
    """
    output_data_list = []
    for data in data_list:
        output_data_list.append(data.detach().cpu().numpy())
    return output_data_list


def ez_network_output_unpack(network_output: Dict) -> Tuple:
    """
    Overview:
        unpack the network output of efficientzero
    Arguments:
        - network_output (:obj:`Tuple`): the network output of efficientzero
    """
    hidden_state = network_output.latent_state  # shape:（batch_size, lstm_hidden_size, num_unroll_steps+1, num_unroll_steps+1）
    value_prefix = network_output.value_prefix  # shape: (batch_size, support_support_size)
    reward_hidden_state = network_output.reward_hidden_state  # shape: {tuple: 2} -> (1, batch_size, 512)
    value = network_output.value  # shape: (batch_size, support_support_size)
    policy_logits = network_output.policy_logits  # shape: (batch_size, action_space_size)
    return hidden_state, value_prefix, reward_hidden_state, value, policy_logits


def mz_network_output_unpack(network_output: Dict) -> Tuple:
    """
    Overview:
        unpack the network output of muzero
    Arguments:
        - network_output (:obj:`Tuple`): the network output of muzero
    """
    hidden_state = network_output.latent_state  # shape:（batch_size, lstm_hidden_size, num_unroll_steps+1, num_unroll_steps+1）
    reward = network_output.reward  # shape: (batch_size, support_support_size)
    value = network_output.value  # shape: (batch_size, support_support_size)
    policy_logits = network_output.policy_logits  # shape: (batch_size, action_space_size)
    return hidden_state, reward, value, policy_logits
