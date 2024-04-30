import inspect
import logging
from typing import List, Tuple, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict
from scipy.stats import entropy
from torch.nn import functional as F


def pad_and_get_lengths(inputs, num_of_sampled_actions):
    """
    Overview:
        Pad root_sampled_actions to make sure that the length of root_sampled_actions is equal to num_of_sampled_actions.
        Also record the true length of each sequence before padding.
    Arguments:
        - inputs (:obj:`List[dict]`): The input data.
        - num_of_sampled_actions (:obj:`int`): The number of sampled actions.
    Returns:
        - inputs (:obj:`List[dict]`): The input data after padding. Each dict also contains 'action_length' which indicates
                                      the true length of 'root_sampled_actions' before padding.
    Example:
        >>> inputs = [{'root_sampled_actions': torch.tensor([1, 2])}, {'root_sampled_actions': torch.tensor([3, 4, 5])}]
        >>> num_of_sampled_actions = 5
        >>> result = pad_and_get_lengths(inputs, num_of_sampled_actions)
        >>> print(result)  # Prints [{'root_sampled_actions': tensor([1, 2, 2, 2, 2]), 'action_length': 2},
                                       {'root_sampled_actions': tensor([3, 4, 5, 5, 5]), 'action_length': 3}]
    """
    for input_dict in inputs:
        root_sampled_actions = input_dict['root_sampled_actions']
        input_dict['action_length'] = len(root_sampled_actions)
        if len(root_sampled_actions) < num_of_sampled_actions:
            # Use the last element to pad root_sampled_actions
            padding = root_sampled_actions[-1].repeat(num_of_sampled_actions - len(root_sampled_actions))
            input_dict['root_sampled_actions'] = torch.cat((root_sampled_actions, padding))
    return inputs


def visualize_avg_softmax(logits):
    """
    Overview:
        Visualize the average softmax distribution across a minibatch.
    Arguments:
        logits (Tensor): The logits output from the model.
    """
    # Apply softmax to logits to get the probabilities.
    probabilities = F.softmax(logits, dim=1)

    # Compute the average probabilities across the minibatch.
    avg_probabilities = torch.mean(probabilities, dim=0)

    # Convert to numpy for visualization.
    avg_probabilities_np = avg_probabilities.detach().numpy()

    # Create a bar plot.
    plt.figure(figsize=(10, 8))
    plt.bar(np.arange(len(avg_probabilities_np)), avg_probabilities_np)

    plt.xlabel('Classes')
    plt.ylabel('Average Probability')
    plt.title('Average Softmax Probabilities Across the Minibatch')
    plt.savefig('avg_softmax_probabilities.png')
    plt.close()


def calculate_topk_accuracy(logits, true_one_hot, top_k):
    """
    Overview:
        Calculate the top-k accuracy.
    Arguments:
        logits (Tensor): The logits output from the model.
        true_one_hot (Tensor): The one-hot encoded true labels.
        top_k (int): The number of top predictions to consider for a match.
    Returns:
        match_percentage (float): The percentage of matches in top-k predictions.
    """
    # Apply softmax to logits to get the probabilities.
    probabilities = F.softmax(logits, dim=1)

    # Use topk to find the indices of the highest k probabilities.
    topk_indices = torch.topk(probabilities, top_k, dim=1)[1]

    # Get the true labels from the one-hot encoded tensor.
    true_labels = torch.argmax(true_one_hot, dim=1).unsqueeze(1)

    # Compare the predicted top-k labels with the true labels.
    matches = (topk_indices == true_labels).sum().item()

    # Calculate the percentage of matches.
    match_percentage = matches / logits.size(0) * 100

    return match_percentage


def plot_topk_accuracy(afterstate_policy_logits, true_chance_one_hot, top_k_values):
    """
    Overview:
        Plot the top_K accuracy based on the given afterstate_policy_logits and true_chance_one_hot tensors.
    Arguments:
        afterstate_policy_logits (torch.Tensor): Tensor of shape (batch_size, num_classes) representing the logits.
        true_chance_one_hot (torch.Tensor): Tensor of shape (batch_size, num_classes) representing the one-hot encoded true labels.
        top_k_values (range or list): Range or list of top_K values to calculate the accuracy for.
    Returns:
        None (plots the graph)
    """
    match_percentages = []
    for top_k in top_k_values:
        match_percentage = calculate_topk_accuracy(afterstate_policy_logits, true_chance_one_hot, top_k=top_k)
        match_percentages.append(match_percentage)

    plt.plot(top_k_values, match_percentages)
    plt.xlabel('top_K')
    plt.ylabel('Match Percentage')
    plt.title('Top_K Accuracy')
    plt.savefig('topk_accuracy.png')
    plt.close()


def compare_argmax(afterstate_policy_logits, chance_one_hot):
    """
    Overview:
        Compare the argmax of afterstate_policy_logits and chance_one_hot tensors.
    Arguments:
        afterstate_policy_logits (torch.Tensor): Tensor of shape (batch_size, num_classes) representing the logits.
        chance_one_hot (torch.Tensor): Tensor of shape (batch_size, num_classes) representing the one-hot encoded labels.
    Returns:
        None (plots the graph)
    Example usage:
        >>> afterstate_policy_logits = torch.randn(1024, 32)
        >>> chance_one_hot = torch.randn(1024, 32)
        >>> compare_argmax(afterstate_policy_logits, chance_one_hot)
    """

    # Calculate the argmax of afterstate_policy_logits and chance_one_hot tensors.
    argmax_afterstate = torch.argmax(afterstate_policy_logits, dim=1)
    argmax_chance = torch.argmax(chance_one_hot, dim=1)

    # Check if the argmax values are equal.
    matches = (argmax_afterstate == argmax_chance)

    # Create a list of sample indices.
    sample_indices = list(range(afterstate_policy_logits.size(0)))

    # Create a list to store the equality values (1 for equal, 0 for not equal).
    equality_values = [int(match) for match in matches]

    # Plot the equality values.
    plt.plot(sample_indices, equality_values)
    plt.xlabel('Sample Index')
    plt.ylabel('Equality')
    plt.title('Comparison of argmax')
    plt.savefig('compare_argmax.png')
    plt.close()


def plot_argmax_distribution(true_chance_one_hot):
    """
    Overview:
        Plot the distribution of possible values obtained from argmax(true_chance_one_hot).
    Arguments:
        true_chance_one_hot (torch.Tensor): Tensor of shape (batch_size, num_classes) representing the one-hot encoded true labels.
    Returns:
        None (plots the graph)
    """

    # Calculate the argmax of true_chance_one_hot tensor.
    argmax_values = torch.argmax(true_chance_one_hot, dim=1)

    # Calculate the count of each unique argmax value.
    unique_values, counts = torch.unique(argmax_values, return_counts=True)

    # Convert the tensor to a list for plotting.
    unique_values = unique_values.tolist()
    counts = counts.tolist()

    # Plot the distribution of argmax values.
    plt.bar(unique_values, counts)
    plt.xlabel('Argmax Values')
    plt.ylabel('Count')
    plt.title('Distribution of Argmax Values')
    plt.savefig('argmax_distribution.png')
    plt.close()


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


def configure_optimizers(
        model: nn.Module,
        weight_decay: float = 0,
        learning_rate: float = 3e-3,
        betas: tuple = (0.9, 0.999),
        device_type: str = "cuda"
):
    """
    Overview:
        This function is adapted from https://github.com/karpathy/nanoGPT/blob/master/model.py

        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, layernorm, embedding weights, and batchnorm).
        We are then returning the PyTorch optimizer object.
    Arguments:
        - model (:obj:`nn.Module`): The model to be optimized.
        - weight_decay (:obj:`float`): The weight decay factor.
        - learning_rate (:obj:`float`): The learning rate.
        - betas (:obj:`tuple`): The betas for Adam.
        - device_type (:obj:`str`): The device type.
    Returns:
        - optimizer (:obj:`torch.optim`): The optimizer.
    """
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.LSTM,  torch.nn.GRU, nn.Conv2d)
    blacklist_weight_modules = (
        torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d
    )
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
            # random note: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if pn.endswith('bias') or pn.endswith('lstm.bias_ih_l0') or pn.endswith('lstm.bias_hh_l0') or pn.endswith('gru.bias_ih_l0') or pn.endswith('gru.bias_hh_l0'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif (pn.endswith('weight_ih_l0') or pn.endswith('weight_hh_l0')) and isinstance(m,
                                                                                             whitelist_weight_modules):
                # some special weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
    try:
        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove('lm_head.weight')
    except KeyError:
        logging.info("lm_head.weight not found in decay set, so not removing it")

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert len(
        param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params),)

    # create the pytorch optimizer object
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": weight_decay
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0
        },
    ]
    # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
    use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
    print(f"using fused AdamW: {use_fused}")
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

    return optimizer

def prepare_obs_stack4_for_gpt_bkp(obs_batch_ori: np.ndarray, cfg: EasyDict) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    概述:
        为模型准备观测数据,包括:
        1. 将观测数据转换为torch张量
        2. 堆叠观测数据
        3. 计算一致性损失
    参数:
        - obs_batch_ori (:obj:`np.ndarray`): 原始的批量观测数据
        - cfg (:obj:`EasyDict`): 配置字典
    返回:
        - obs_batch (:obj:`torch.Tensor`): 堆叠后的观测数据
        - obs_target_batch (:obj:`torch.Tensor`): 用于计算一致性损失的堆叠观测数据
    """
    obs_target_batch = None
    obs_batch_ori = torch.from_numpy(obs_batch_ori).to(cfg.device).float()

    if cfg.model.model_type == 'conv':
        # 对于三维图像观测数据
        """
        ``obs_batch_ori`` 是原始的批量观测数据,形状为:
        (batch_size, stack_num+num_unroll_steps, C, W, H) -> (batch_size, (stack_num+num_unroll_steps)*C, W, H)

        例如,在pong游戏中: stack_num=4, num_unroll_steps=5
        (4, 9, 3, 96, 96) -> (4, 9*3, 96, 96) = (4, 27, 96, 96)

        ``obs_batch_ori`` 的第二维度:
        时间步 t:        1,   2,   3,  4,    5,   6,   7,   8,     9
        通道数:          3    3    3   3     3    3    3    3      3
                       ---, ---, ---, ---,  ---, ---, ---, ---,   ---
        """
        # ``obs_batch`` 在 ``initial_inference()`` 中使用,它是 ``obs_batch_ori`` 中时间步 t 的第一个堆叠观测数据
        # 形状为 (4, 4*3, 96, 96) = (4, 12, 96, 96)
        obs_batch = obs_batch_ori[:, :cfg.model.frame_stack_num * cfg.model.image_channel, :, :]

        if cfg.model.self_supervised_learning_loss:
            # ``obs_target_batch`` 仅用于计算一致性损失,它包含除时间步 t1 之外的所有观测数据
            # 并且只在 ``obs_batch_ori`` 第二维度的最后8个时间步中执行
            obs_target_batch = torch.empty(
                (obs_batch_ori.shape[0], cfg.model.frame_stack_num * (obs_batch_ori.shape[1]-cfg.model.frame_stack_num),
                 obs_batch_ori.shape[2], obs_batch_ori.shape[3]),
                device=cfg.device
            )

            # 用 obs_batch_ori 中的连续4帧填充 obs_target_batch
            for i in range(obs_batch_ori.shape[1] - cfg.model.frame_stack_num):
                obs_target_batch[:, i*cfg.model.frame_stack_num:(i+1)*cfg.model.frame_stack_num, :, :] = \
                    obs_batch_ori[:, i+1:i+1+cfg.model.frame_stack_num, :, :].reshape(
                        obs_batch_ori.shape[0], -1, obs_batch_ori.shape[2], obs_batch_ori.shape[3])

    elif cfg.model.model_type == 'mlp':
        # 对于一维向量观测数据  
        """
        ``obs_batch_ori`` 是原始的批量观测数据,形状为:
        (batch_size, stack_num+num_unroll_steps, obs_shape) -> (batch_size, (stack_num+num_unroll_steps)*obs_shape)

        例如,在cartpole环境中: stack_num=1, num_unroll_steps=5, obs_shape=4
        (4, 6, 4) -> (4, 6*4) = (4, 24)

        ``obs_batch_ori`` 的第二维度:
        时间步 t:     1,   2,      3,     4,    5,   6,  
        obs_shape:    4    4       4      4     4    4
                     ----, ----,  ----, ----,  ----,  ----,
        """
        # ``obs_batch`` 在 ``initial_inference()`` 中使用,它是 ``obs_batch_ori`` 中时间步 t1 的第一个堆叠观测数据
        # 形状为 (4, 4*3) = (4, 12)
        obs_batch = obs_batch_ori[:, :cfg.model.frame_stack_num * cfg.model.observation_shape]
        
        if cfg.model.self_supervised_learning_loss:
            # ``obs_target_batch`` 仅用于计算一致性损失,它包含除时间步 t1 之外的所有观测数据  
            # 并且只在 ``obs_batch_ori`` 第二维度的最后几个时间步中执行
            obs_target_batch = obs_batch_ori[:, cfg.model.observation_shape:]

    return obs_batch, obs_target_batch

def prepare_obs_stack4_for_gpt(obs_batch_ori: np.ndarray, cfg: EasyDict) -> Tuple[torch.Tensor, torch.Tensor]:
    obs_batch_ori = torch.from_numpy(obs_batch_ori).to(cfg.device).float()
    obs_batch = obs_batch_ori[:, :cfg.model.frame_stack_num * (cfg.model.image_channel if cfg.model.model_type == 'conv' else cfg.model.observation_shape), ...]

    obs_target_batch = None
    if cfg.model.self_supervised_learning_loss:
        if cfg.model.model_type == 'conv':
            obs_target_batch = obs_batch_ori[:, cfg.model.image_channel:, ...].unfold(1, cfg.model.frame_stack_num * cfg.model.image_channel, cfg.model.image_channel).reshape(
                obs_batch_ori.shape[0], -1, *obs_batch_ori.shape[2:])
        else:
            obs_target_batch = obs_batch_ori[:, cfg.model.observation_shape:]

    return obs_batch, obs_target_batch

def prepare_obs(obs_batch_ori: np.ndarray, cfg: EasyDict) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Overview:
        Prepare the observations for the model by converting the original batch of observations
        to a PyTorch tensor, and then slicing it to create the batches used for the initial inference
        and for calculating the consistency loss if required.

    Arguments:
        - obs_batch_ori (:obj:`np.ndarray`): The original observations in a batch style.
        - cfg (:obj:`EasyDict`): The configuration dictionary containing model settings.

    Returns:
        - obs_batch (:obj:`torch.Tensor`): The tensor containing the observations for the initial inference.
        - obs_target_batch (:obj:`torch.Tensor`): The tensor containing the observations for calculating
                                                   the consistency loss, if applicable.
    """
    # Convert the numpy array of original observations to a PyTorch tensor and transfer it to the specified device.
    # Also, ensure the tensor is of the correct floating-point type for the model.
    obs_batch_ori = torch.from_numpy(obs_batch_ori).to(cfg.device)

    # Calculate the dimension size to slice based on the model configuration.
    # For convolutional models ('conv'), use the number of frames to stack times the number of channels.
    # For multi-layer perceptron models ('mlp'), use the number of frames to stack times the size of the observation space.
    stack_dim = cfg.model.frame_stack_num * (
        cfg.model.image_channel if cfg.model.model_type == 'conv' else cfg.model.observation_shape)

    # Slice the original observation tensor to obtain the batch for the initial inference.
    obs_batch = obs_batch_ori[:, :stack_dim]

    # Initialize the target batch for consistency loss as `None`. It will only be set if consistency loss calculation is enabled.
    obs_target_batch = None
    # If the model configuration specifies the use of self-supervised learning loss, prepare the target batch for the consistency loss.
    if cfg.model.self_supervised_learning_loss:
        # Determine the starting dimension to exclude based on the model type.
        # For 'conv', exclude the first 'image_channel' dimensions.
        # For 'mlp', exclude the first 'observation_shape' dimensions.
        exclude_dim = cfg.model.image_channel if cfg.model.model_type == 'conv' else cfg.model.observation_shape

        # Slice the original observation tensor to obtain the batch for consistency loss calculation.
        obs_target_batch = obs_batch_ori[:, exclude_dim:]

    # Return the prepared batches: one for the initial inference and one for the consistency loss calculation (if applicable).
    return obs_batch, obs_target_batch


def prepare_obs_bkp(obs_batch_ori: np.ndarray, cfg: EasyDict) -> Tuple[torch.Tensor, torch.Tensor]:
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
        (batch_size, stack_num+num_unroll_steps, C, W, H) -> (batch_size, (stack_num+num_unroll_steps)*C, W, H )

        e.g. in pong: stack_num=4, num_unroll_steps=5
        (4, (4+5), 3, 96, 96) -> (4, 9, 3, 96, 96) -> (4, 9*3, 96, 96) = (4, 27, 96, 96)

        the second dim of ``obs_batch_ori``:
        timestep t:     1,   2,   3,  4,    5,   6,   7,   8,     9
        channel_num:    3    3    3   3     3    3    3    3      3
                       ---, ---, ---, ---,  ---, ---, ---, ---,   ---
        """
        # obs_batch_ori = torch.from_numpy(obs_batch_ori).to(cfg.device).float()
        obs_batch_ori = torch.from_numpy(obs_batch_ori).to(cfg.device)
        # ``obs_batch`` is used in ``initial_inference()``, which is the first stacked obs at timestep t in
        # ``obs_batch_ori``. shape is (4, (4+5)*1, 96, 96) = (4, 9, 96, 96)
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


def compute_entropy(policy_probs: torch.Tensor) -> torch.Tensor:
    dist = torch.distributions.Categorical(probs=policy_probs)
    entropy = dist.entropy().mean()
    return entropy


def get_max_entropy(action_shape: int) -> np.float32:
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


def select_action(visit_counts: np.ndarray,
                  temperature: float = 1,
                  deterministic: bool = True) -> Tuple[np.int64, np.ndarray]:
    """
    Overview:
        Select action from visit counts of the root node.
    Arguments:
        - visit_counts (:obj:`np.ndarray`): The visit counts of the root node.
        - temperature (:obj:`float`): The temperature used to adjust the sampling distribution.
        - deterministic (:obj:`bool`):  Whether to enable deterministic mode in action selection. True means to \
            select the argmax result, False indicates to sample action from the distribution.
    Returns:
        - action_pos (:obj:`np.int64`): The selected action position (index).
        - visit_count_distribution_entropy (:obj:`np.ndarray`): The entropy of the visit count distribution.
    """
    action_probs = [visit_count_i ** (1 / temperature) for visit_count_i in visit_counts]
    action_probs = [x / sum(action_probs) for x in action_probs]

    if deterministic:
        action_pos = np.argmax([v for v in visit_counts])
    else:
        action_pos = np.random.choice(len(visit_counts), p=action_probs)

    visit_count_distribution_entropy = entropy(action_probs, base=2)
    return action_pos, visit_count_distribution_entropy


def concat_output_value(output_lst: List) -> np.ndarray:
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
        - output_lst (:obj:`List`): The model output list.
        - data_type (:obj:`str`): The data type, should be 'muzero' or 'efficientzero'.
    Returns:
        - value_lst (:obj:`np.array`): the values of the model output list
    """
    assert data_type in ['muzero', 'efficientzero'], "data_type should be 'muzero' or 'efficientzero'"
    # concat the model output
    value_lst, reward_lst, policy_logits_lst, latent_state_lst = [], [], [], []
    reward_hidden_state_c_lst, reward_hidden_state_h_lst = [], []
    for output in output_lst:
        value_lst.append(output.value)
        if data_type == 'muzero':
            reward_lst.append(output.reward)
        elif data_type == 'efficientzero':
            reward_lst.append(output.value_prefix)

        policy_logits_lst.append(output.policy_logits)
        latent_state_lst.append(output.latent_state)
        if data_type == 'efficientzero':
            reward_hidden_state_c_lst.append(output.reward_hidden_state[0].squeeze(0))
            reward_hidden_state_h_lst.append(output.reward_hidden_state[1].squeeze(0))

    value_lst = np.concatenate(value_lst)
    reward_lst = np.concatenate(reward_lst)
    policy_logits_lst = np.concatenate(policy_logits_lst)
    latent_state_lst = np.concatenate(latent_state_lst)
    if data_type == 'muzero':
        return value_lst, reward_lst, policy_logits_lst, latent_state_lst
    elif data_type == 'efficientzero':
        reward_hidden_state_c_lst = np.expand_dims(np.concatenate(reward_hidden_state_c_lst), axis=0)
        reward_hidden_state_h_lst = np.expand_dims(np.concatenate(reward_hidden_state_h_lst), axis=0)
        return value_lst, reward_lst, policy_logits_lst, latent_state_lst, (
            reward_hidden_state_c_lst, reward_hidden_state_h_lst
        )


def to_torch_float_tensor(data_list: Union[np.ndarray, List[np.ndarray]], device: torch.device) -> Union[
    torch.Tensor, List[torch.Tensor]]:
    """
    Overview:
        convert the data or data list to torch float tensor
    Arguments:
        - data_list (:obj:`Union[np.ndarray, List[np.ndarray]]`): The data or data list.
        - device (:obj:`torch.device`): The device.
    Returns:
        - output_data_list (:obj:`Union[torch.Tensor, List[torch.Tensor]]`): The output data or data list.
    """
    if isinstance(data_list, np.ndarray):
        return (torch.from_numpy(data_list).to(device).float())
    elif isinstance(data_list, list) and all(isinstance(data, np.ndarray) for data in data_list):
        output_data_list = []
        for data in data_list:
            output_data_list.append(torch.from_numpy(data).to(device).float())
        return output_data_list
    else:
        raise TypeError("The type of input must be np.ndarray or List[np.ndarray]")


def to_detach_cpu_numpy(data_list: Union[torch.Tensor, List[torch.Tensor]]) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Overview:
        convert the data or data list to detach cpu numpy.
    Arguments:
        - data_list (:obj:`Union[torch.Tensor, List[torch.Tensor]]`): the data or data list
    Returns:
        - output_data_list (:obj:`Union[np.ndarray,List[np.ndarray]]`): the output data or data list
    """
    if isinstance(data_list, torch.Tensor):
        return data_list.detach().cpu().numpy()
    elif isinstance(data_list, list) and all(isinstance(data, torch.Tensor) for data in data_list):
        output_data_list = []
        for data in data_list:
            output_data_list.append(data.detach().cpu().numpy())
        return output_data_list
    else:
        raise TypeError("The type of input must be torch.Tensor or List[torch.Tensor]")


def ez_network_output_unpack(network_output: Dict) -> Tuple:
    """
    Overview:
        unpack the network output of efficientzero
    Arguments:
        - network_output (:obj:`Tuple`): the network output of efficientzero
    """
    latent_state = network_output.latent_state  # shape:（batch_size, lstm_hidden_size, num_unroll_steps+1, num_unroll_steps+1）
    value_prefix = network_output.value_prefix  # shape: (batch_size, support_support_size)
    reward_hidden_state = network_output.reward_hidden_state  # shape: {tuple: 2} -> (1, batch_size, 512)
    value = network_output.value  # shape: (batch_size, support_support_size)
    policy_logits = network_output.policy_logits  # shape: (batch_size, action_space_size)
    return latent_state, value_prefix, reward_hidden_state, value, policy_logits


def mz_network_output_unpack(network_output: Dict) -> Tuple:
    """
    Overview:
        unpack the network output of muzero
    Arguments:
        - network_output (:obj:`Tuple`): the network output of muzero
    """
    latent_state = network_output.latent_state  # shape:（batch_size, lstm_hidden_size, num_unroll_steps+1, num_unroll_steps+1）
    reward = network_output.reward  # shape: (batch_size, support_support_size)
    value = network_output.value  # shape: (batch_size, support_support_size)
    policy_logits = network_output.policy_logits  # shape: (batch_size, action_space_size)
    return latent_state, reward, value, policy_logits
