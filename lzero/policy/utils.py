import inspect
import logging
from typing import List, Dict, Union
from typing import Tuple

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


# modified from https://github.com/karpathy/nanoGPT/blob/master/model.py#L263
def configure_optimizers_nanogpt(model, weight_decay, learning_rate, betas, device_type):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}

    # 非常重要 对于balance pipeline ===========
    # filter out those that do not require grad
    # param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    if torch.cuda.is_available():
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    else:
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

    return optimizer


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
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)
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


def prepare_obs_stack_for_unizero(obs_batch_ori: np.ndarray, cfg: EasyDict) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Overview:
        Prepare the observation stack for UniZero model. This function processes the original batch of observations
        and prepares it for input into the network. If self-supervised learning is enabled, it also prepares the
        target batch for self-supervised learning.

    Arguments:
        - obs_batch_ori (:obj:`np.ndarray`): The original batch of observations as a Numpy array.
        - cfg (:obj:`EasyDict`): Configuration dictionary containing model parameters and other settings.

    Returns:
        - obs_batch (:obj:`torch.Tensor`): The processed batch of observations ready for network input.
        - obs_target_batch (:obj:`torch.Tensor` or None): The target batch for self-supervised learning, or None if not applicable.
    """
    assert cfg.model.model_type in ['conv', 'mlp'], f"Model type {cfg.model.model_type} not supported."
    # Convert the original observation batch to a torch tensor and move it to the specified device.
    obs_batch_ori = torch.from_numpy(obs_batch_ori).to(cfg.device).float()

    # Prepare the observation batch based on the model type (conv or other).
    if cfg.model.model_type == 'conv':
        obs_batch = obs_batch_ori[:, :cfg.model.frame_stack_num * cfg.model.image_channel, ...]
    else:
        obs_batch = obs_batch_ori[:, :cfg.model.frame_stack_num * cfg.model.observation_shape, ...]

    # Initialize the target batch for self-supervised learning if applicable.
    obs_target_batch = None
    if cfg.model.self_supervised_learning_loss:
        if cfg.model.model_type == 'conv':
            # Prepare the target batch for convolutional models.
            obs_target_batch = (
                obs_batch_ori[:, cfg.model.image_channel:, ...]
                .unfold(1, cfg.model.frame_stack_num * cfg.model.image_channel, cfg.model.image_channel)
                .reshape(obs_batch_ori.shape[0], -1, *obs_batch_ori.shape[2:])
            )
        else:
            # Prepare the target batch for non-convolutional models.
            obs_target_batch = obs_batch_ori[:, cfg.model.observation_shape:]

    return obs_batch, obs_target_batch


def prepare_obs(obs_batch_ori: np.ndarray, cfg: EasyDict, task_id = None) -> Tuple[torch.Tensor, torch.Tensor]:
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
    if task_id is None:
        stack_dim = cfg.model.frame_stack_num * (
        cfg.model.image_channel if cfg.model.model_type in ['conv', 'conv_context'] else cfg.model.observation_shape)
    else:
        stack_dim = cfg.model.frame_stack_num * (
            cfg.model.image_channel if cfg.model.model_type in ['conv', 'conv_context'] else cfg.model.observation_shape_list[task_id])
    # Slice the original observation tensor to obtain the batch for the initial inference.
    obs_batch = obs_batch_ori[:, :stack_dim]

    # Initialize the target batch for consistency loss as `None`. It will only be set if consistency loss calculation is enabled.
    obs_target_batch = None
    # If the model configuration specifies the use of self-supervised learning loss, prepare the target batch for the consistency loss.
    if cfg.model.self_supervised_learning_loss:
        # Determine the starting dimension to exclude based on the model type.
        # For 'conv', exclude the first 'image_channel' dimensions.
        # For 'mlp', exclude the first 'observation_shape' dimensions.
        if task_id is None:
            exclude_dim = cfg.model.image_channel if cfg.model.model_type in ['conv', 'conv_context'] else cfg.model.observation_shape
        else:
            exclude_dim = cfg.model.image_channel if cfg.model.model_type in ['conv', 'conv_context'] else cfg.model.observation_shape_list[task_id]

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


def get_max_entropy(action_space_size: int) -> np.float32:
    """
    Overview:
        get the max entropy of the action space.
    Arguments:
        - action_space_size (:obj:`int`): the shape of the action space
    Returns:
        - max_entropy (:obj:`float`): the max entropy of the action space
    """
    p = 1.0 / action_space_size
    return -action_space_size * p * np.log2(p)


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
        value_lst.append(output.value) # TODO:cpu

    # print(f'value_lst:{value_lst}')
    # print(f'value_lst[0]:{value_lst[0]}')
    # print(f'value_lst[0].shape:{value_lst[0].shape}')

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


def mz_rnn_fullobs_network_output_unpack(network_output: Dict) -> Tuple:
    """
    Overview:
        unpack the network output of efficientzero
    Arguments:
        - network_output (:obj:`Tuple`): the network output of efficientzero
    """
    predict_next_latent_state = network_output.predict_next_latent_state  # shape:（batch_size, lstm_hidden_size, num_unroll_steps+1, num_unroll_steps+1）
    latent_state = network_output.latent_state  # shape:（batch_size, lstm_hidden_size, num_unroll_steps+1, num_unroll_steps+1）
    value_prefix = network_output.value_prefix  # shape: (batch_size, support_support_size)
    reward_hidden_state = network_output.reward_hidden_state  # shape: {tuple: 2} -> (1, batch_size, 512)
    value = network_output.value  # shape: (batch_size, support_support_size)
    policy_logits = network_output.policy_logits  # shape: (batch_size, action_space_size)

    return predict_next_latent_state, latent_state, value_prefix, reward_hidden_state, value, policy_logits

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


# ==================== modified by tangjia=============================
import torch.distributed as dist



def example_usage():
    """
    示例用法：计算梯度冲突分析结果
    该函数生成示例梯度并计算它们之间的冲突分析结果
    结果包括平均冲突得分、最大冲突得分、冲突梯度对数量、平均冲突强度和梯度范数等信息。
    还包括余弦相似度矩阵的计算结果。
    该函数用于演示如何使用 compute_gradient_conflicts 函数进行梯度冲突分析。
    结果将打印到控制台。
    该函数不接受任何参数，直接生成示例梯度进行分析。    
    """
    # 生成示例梯度
    torch.manual_seed(42)
    gradients = [
        torch.randn(100),  # 梯度1
        torch.randn(100),  # 梯度2  
        torch.randn(100),  # 梯度3
    ]
    
    # 计算冲突
    conflicts = compute_gradient_conflicts(gradients)
    
    print("梯度冲突分析结果:")
    print(f"平均冲突得分: {conflicts['avg_conflict_score']:.4f}")
    print(f"最大冲突得分: {conflicts['max_conflict_score']:.4f}")
    print(f"冲突梯度对数量: {conflicts['num_conflicting_pairs']}")
    print(f"平均冲突强度: {conflicts['avg_conflict_intensity']:.4f}")
    print(f"梯度范数: {conflicts['gradient_norms']}")
    print("\n余弦相似度矩阵:")
    print(conflicts['cosine_similarity_matrix'])



def compute_gradient_conflicts(gradients: List[torch.Tensor]) -> dict:
    """
    计算多个梯度之间的冲突 - CUDA优化版本
    
    Args:
        gradients: 梯度列表，每个元素是一个梯度张量
    
    Returns:
        dict: 包含avg_conflict_score和cosine_similarity_matrix的字典
    """
    n_gradients = len(gradients)
    
    # 如果只有一个梯度，没有冲突
    if n_gradients <= 1:
        device = gradients[0].device if gradients else torch.device('cuda')
        return EasyDict({
            'avg_conflict_score': 0.0, 
            'max_conflict_score': 0.0, 
            'min_conflict_score': 0.0,
            'cosine_similarity_matrix': torch.zeros(1, 1, device=device)
        })
    
    # 确保所有梯度形状相同
    assert all(g.shape == gradients[0].shape for g in gradients), "梯度形状必须相同"
    
    device = gradients[0].device
    
    # 向量化计算：堆叠并normalize所有梯度
    stacked_grads = torch.stack([g.flatten() for g in gradients])
    normalized_grads = F.normalize(stacked_grads, p=2, dim=1)
    
    # 一次性计算余弦相似度矩阵
    cosine_sim_matrix = torch.mm(normalized_grads, normalized_grads.t())
    
    # 排除对角线元素
    mask = ~torch.eye(n_gradients, device=device, dtype=torch.bool)
    conflict_scores = -cosine_sim_matrix[mask]
    
    return EasyDict({
        'avg_conflict_score': conflict_scores.mean().item(),
        'max_conflict_score': conflict_scores.max().item(),
        'min_conflict_score': conflict_scores.min().item(),
        'cosine_similarity_matrix': cosine_sim_matrix
    })
 
 
def compute_gradient_conflict_distributed(local_grads, multi_gpu=True, device=0):
    """
    分布式模式下计算梯度冲突 - 分层聚合优化版本
    
    性能提升: 69.4x加速 (3.1ms vs 212.7ms)
    核心优化: 分层预处理 + NCCL直通 + 向量化计算
    
    Args:
        local_grads: 本地梯度tensor，shape: (local_task_num, encoder_grad_dim)
        multi_gpu: 是否多GPU模式
        device: 当前设备
    Returns:
        gradient_conflict: 所有rank都返回相同的梯度冲突结果
    """
    if not multi_gpu:
        # 单GPU模式：直接使用优化的单机版本
        norms = torch.norm(local_grads, dim=1)
        valid_grads = local_grads[norms > 1e-8]
        if valid_grads.shape[0] <= 1:
            device = valid_grads.device
            return EasyDict({
                'avg_conflict_score': 0.0, 
                'max_conflict_score': 0.0, 
                'min_conflict_score': 0.0,
                'cosine_similarity_matrix': torch.zeros(1, 1, device=device)
            })
        
        # 向量化计算
        device = valid_grads.device
        normalized = F.normalize(valid_grads, p=2, dim=1)
        similarity = torch.mm(normalized, normalized.t())
        mask = ~torch.eye(valid_grads.shape[0], device=device, dtype=torch.bool)
        conflicts = -similarity[mask]
        return EasyDict({
            'avg_conflict_score': conflicts.mean().item(),
            'max_conflict_score': conflicts.max().item(),
            'min_conflict_score': conflicts.min().item(),
            'cosine_similarity_matrix': similarity
        })
    
    # 多GPU分布式模式：分层聚合优化
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f'{device}')
    
    # === 第一层：本地预处理（关键优化）===
    norms = torch.norm(local_grads, dim=1)
    valid_grads = local_grads[norms > 1e-8]
    local_normalized = F.normalize(valid_grads, p=2, dim=1)  # 预归一化，避免重复计算
    
    # 收集各rank的有效梯度数量
    valid_count = torch.tensor(valid_grads.shape[0], device=device)
    valid_counts = [torch.tensor(0, device=device) for _ in range(world_size)]
    dist.all_gather(valid_counts, valid_count)
    
    total_valid = sum(v.item() for v in valid_counts)
    if total_valid <= 1:
        return EasyDict({
            'avg_conflict_score': 0.0, 
            'max_conflict_score': 0.0, 
            'min_conflict_score': 0.0,
            'cosine_similarity_matrix': torch.zeros(1, 1, device=device)
        })
    
    # 数据对齐：padding到相同大小
    max_valid = max(v.item() for v in valid_counts)
    if valid_grads.shape[0] < max_valid:
        pad_size = max_valid - valid_grads.shape[0]
        pad_tensor = torch.zeros(pad_size, valid_grads.shape[1], device=device, dtype=valid_grads.dtype)
        local_normalized = torch.cat([local_normalized, pad_tensor], dim=0)
    
    # === 第二层：高效NCCL聚合 ===
    gathered_normalized = [torch.empty_like(local_normalized) for _ in range(world_size)]
    dist.all_gather(gathered_normalized, local_normalized)  # GPU直接通信，传输预处理数据
    
    # if rank == 0:
        # === 第三层：向量化冲突计算 ===
        # 重建有效的归一化梯度
    all_valid_normalized = []
    for i, count in enumerate(valid_counts):
        if count > 0:
            all_valid_normalized.append(gathered_normalized[i][:count.item()])
    
    if len(all_valid_normalized) == 0:
        return EasyDict({
            'avg_conflict_score': 0.0, 
            'max_conflict_score': 0.0, 
            'min_conflict_score': 0.0,
            'cosine_similarity_matrix': torch.zeros(1, 1, device=device)
        })
    
    all_normalized = torch.cat(all_valid_normalized, dim=0)
    
    # 高效向量化计算（一次矩阵乘法替代O(n²)循环）
    similarity = torch.mm(all_normalized, all_normalized.t())
    mask = ~torch.eye(similarity.shape[0], device=device, dtype=torch.bool)
    conflicts = -similarity[mask]
    
    return EasyDict({
        'avg_conflict_score': conflicts.mean().item(),
        'max_conflict_score': conflicts.max().item(),
        'min_conflict_score': conflicts.min().item(),
        'cosine_similarity_matrix': similarity
    })

if __name__ == "__main__":
    example_usage()
