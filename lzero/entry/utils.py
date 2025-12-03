import os
from typing import Optional, Callable, Union, List, Tuple

import psutil
import torch
import torch.distributed as dist
from pympler.asizeof import asizeof
from tensorboardX import SummaryWriter
import torch
import torch.distributed as dist

def is_ddp_enabled():
    """
    Check if Distributed Data Parallel (DDP) is enabled by verifying if
    PyTorch's distributed package is available and initialized.
    """
    return dist.is_available() and dist.is_initialized()

def ddp_synchronize():
    """
    Perform a barrier synchronization across all processes in DDP mode.
    Ensures all processes reach this point before continuing.
    """
    if is_ddp_enabled():
        dist.barrier()

def ddp_all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """
    Perform an all-reduce operation (sum) on the given tensor across
    all processes in DDP mode. Returns the reduced tensor.

    Arguments:
        - tensor (:obj:`torch.Tensor`): The input tensor to be reduced.

    Returns:
        - torch.Tensor: The reduced tensor, summed across all processes.
    """
    if is_ddp_enabled():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor

def calculate_update_per_collect(cfg: 'EasyDict', new_data: List[List[torch.Tensor]], world_size: int = 1) -> int:
    """
    Calculate the number of updates to perform per data collection in a
    Distributed Data Parallel (DDP) setting. This ensures that all GPUs
    compute the same `update_per_collect` value, synchronized across processes.

    Arguments:
        - cfg: Configuration object containing policy settings.
        - new_data (List[List[torch.Tensor]]): The newly collected data segments.
        - world_size (int): The total number of processes.

    Returns:
        - int: The number of updates to perform per collection.
    """
    # Retrieve the update_per_collect setting from the configuration
    update_per_collect = cfg.policy.update_per_collect

    if update_per_collect is None:
        # If update_per_collect is not explicitly set, calculate it based on
        # the number of collected transitions and the replay ratio.

        # The length of game_segment (i.e., len(game_segment.action_segment)) can be smaller than cfg.policy.game_segment_length if it represents the final segment of the game.
        # On the other hand, its length will be less than cfg.policy.game_segment_length + padding_length when it is not the last game segment. Typically, padding_length is the sum of unroll_steps and td_steps.
        collected_transitions_num = sum(
            min(len(game_segment), cfg.policy.game_segment_length)
            for game_segment in new_data[0]
        )

        if torch.cuda.is_available() and world_size > 1:
            # Convert the collected transitions count to a GPU tensor for DDP operations.
            collected_transitions_tensor = torch.tensor(
                collected_transitions_num, dtype=torch.int64, device='cuda'
            )

            # Synchronize the collected transitions count across all GPUs using all-reduce.
            total_collected_transitions = ddp_all_reduce_sum(
                collected_transitions_tensor
            ).item()

            # Calculate update_per_collect based on the total synchronized transitions count.
            update_per_collect = int(total_collected_transitions * cfg.policy.replay_ratio)

            # Ensure the computed update_per_collect is positive.
            assert update_per_collect > 0, "update_per_collect must be positive"
        else:
            # If not using DDP, calculate update_per_collect directly from the local count.
            update_per_collect = int(collected_transitions_num * cfg.policy.replay_ratio)

    return update_per_collect

def initialize_zeros_batch(observation_shape: Union[int, List[int], Tuple[int]], batch_size: int, device: str) -> torch.Tensor:
    """
    Overview:
        Initialize a zeros tensor for batch observations based on the shape. This function is used to initialize the UniZero model input.
    Arguments:
        - observation_shape (:obj:`Union[int, List[int], Tuple[int]]`): The shape of the observation tensor.
        - batch_size (:obj:`int`): The batch size.
        - device (:obj:`str`): The device to store the tensor.
    Returns:
        - zeros (:obj:`torch.Tensor`): The zeros tensor.
    """
    if isinstance(observation_shape, (list, tuple)):
        shape = [batch_size, *observation_shape]
    elif isinstance(observation_shape, int):
        shape = [batch_size, observation_shape]
    else:
        raise TypeError(f"observation_shape must be either an int, a list, or a tuple, but got {type(observation_shape).__name__}")

    return torch.zeros(shape).to(device)

def initialize_pad_batch(observation_shape: Union[int, List[int], Tuple[int]], batch_size: int, device: str, pad_token_id: int = 0) -> torch.Tensor:
    """
    Overview:
        Initialize a tensor filled with `pad_token_id` for batch observations. 
        This function is designed to be flexible and can handle both textual 
        and non-textual observations:
        
        - For textual observations: it initializes `input_ids` with padding tokens, 
        ensuring consistent sequence lengths within a batch.
        - For non-textual observations: it provides a convenient way to fill 
        observation tensors with a default of 0, 
        ensuring shape compatibility and preventing uninitialized values.
    Arguments:
        - observation_shape (:obj:`Union[int, List[int], Tuple[int]]`): The shape of the observation tensor.
        - batch_size (:obj:`int`): The batch size.
        - device (:obj:`str`): The device to store the tensor.
        - pad_token_id (:obj:`int`): The token ID (or placeholder value) used for padding.
    Returns:
        - padded_tensor (:obj:`torch.Tensor`): A tensor of the given shape, 
        filled with `pad_token_id`.
    """
    if isinstance(observation_shape, (list, tuple)):
        shape = [batch_size, *observation_shape]
    elif isinstance(observation_shape, int):
        shape = [batch_size, observation_shape]
    else:
        raise TypeError(f"observation_shape must be int, list, or tuple, but got {type(observation_shape).__name__}")

    return torch.full(shape, fill_value=pad_token_id, dtype=torch.float32, device=device) if pad_token_id == -1 else torch.full(shape, fill_value=pad_token_id, dtype=torch.long, device=device)

def random_collect(
        policy_cfg: 'EasyDict',  # noqa
        policy: 'Policy',  # noqa
        RandomPolicy: 'Policy',  # noqa
        collector: 'ISerialCollector',  # noqa
        collector_env: 'BaseEnvManager',  # noqa
        replay_buffer: 'IBuffer',  # noqa
        postprocess_data_fn: Optional[Callable] = None
) -> None:  # noqa
    assert policy_cfg.random_collect_episode_num > 0

    random_policy = RandomPolicy(cfg=policy_cfg, action_space=collector_env.env_ref.action_space)
    # set the policy to random policy
    collector.reset_policy(random_policy.collect_mode)

    # set temperature for visit count distributions according to the train_iter,
    # please refer to Appendix D in MuZero paper for details.
    collect_kwargs = {'temperature': 1, 'epsilon': 0.0}

    # Collect data by default config n_sample/n_episode.
    new_data = collector.collect(n_episode=policy_cfg.random_collect_episode_num, train_iter=0,
                                 policy_kwargs=collect_kwargs)

    if postprocess_data_fn is not None:
        new_data = postprocess_data_fn(new_data)

    # save returned new_data collected by the collector
    replay_buffer.push_game_segments(new_data)
    # remove the oldest data if the replay buffer is full.
    replay_buffer.remove_oldest_data_to_fit()

    # restore the policy
    collector.reset_policy(policy.collect_mode)


def log_buffer_memory_usage(train_iter: int, buffer: "GameBuffer", writer: SummaryWriter) -> None:
    """
    Overview:
        Log the memory usage of the buffer and the current process to TensorBoard.
    Arguments:
        - train_iter (:obj:`int`): The current training iteration.
        - buffer (:obj:`GameBuffer`): The game buffer.
        - writer (:obj:`SummaryWriter`): The TensorBoard writer.
    """
    # "writer is None" means we are in a slave process in the DDP setup.
    if writer is not None:
        writer.add_scalar('Buffer/num_of_all_collected_episodes', buffer.num_of_collected_episodes, train_iter)
        writer.add_scalar('Buffer/num_of_game_segments', len(buffer.game_segment_buffer), train_iter)
        writer.add_scalar('Buffer/num_of_transitions', len(buffer.game_segment_game_pos_look_up), train_iter)

        game_segment_buffer = buffer.game_segment_buffer

        # Calculate the amount of memory occupied by self.game_segment_buffer (in bytes).
        buffer_memory_usage = asizeof(game_segment_buffer)

        # Convert buffer_memory_usage to megabytes (MB).
        buffer_memory_usage_mb = buffer_memory_usage / (1024 * 1024)

        # Record the memory usage of self.game_segment_buffer to TensorBoard.
        writer.add_scalar('Buffer/memory_usage/game_segment_buffer', buffer_memory_usage_mb, train_iter)

        # Get the amount of memory currently used by the process (in bytes).
        process = psutil.Process(os.getpid())
        process_memory_usage = process.memory_info().rss

        # Convert process_memory_usage to megabytes (MB).
        process_memory_usage_mb = process_memory_usage / (1024 * 1024)

        # Record the memory usage of the process to TensorBoard.
        writer.add_scalar('Buffer/memory_usage/process', process_memory_usage_mb, train_iter)


def log_buffer_run_time(train_iter: int, buffer: "GameBuffer", writer: SummaryWriter) -> None:
    """
    Overview:
        Log the average runtime metrics of the buffer to TensorBoard.
    Arguments:
        - train_iter (:obj:`int`): The current training iteration.
        - buffer (:obj:`GameBuffer`): The game buffer containing runtime metrics.
        - writer (:obj:`SummaryWriter`): The TensorBoard writer for logging metrics.

    .. note::
        "writer is None" indicates that the function is being called in a slave process in the DDP setup.
    """
    if writer is not None:
        sample_times = buffer.sample_times

        if sample_times == 0:
            return

        # Calculate and log average reanalyze time.
        average_reanalyze_time = buffer.compute_target_re_time / sample_times
        writer.add_scalar('Buffer/average_reanalyze_time', average_reanalyze_time, train_iter)

        # Calculate and log average origin search time.
        average_origin_search_time = buffer.origin_search_time / sample_times
        writer.add_scalar('Buffer/average_origin_search_time', average_origin_search_time, train_iter)

        # Calculate and log average reuse search time.
        average_reuse_search_time = buffer.reuse_search_time / sample_times
        writer.add_scalar('Buffer/average_reuse_search_time', average_reuse_search_time, train_iter)

        # Calculate and log average active root number.
        average_active_root_num = buffer.active_root_num / sample_times
        writer.add_scalar('Buffer/average_active_root_num', average_active_root_num, train_iter)

        # Reset the time records in the buffer.
        buffer.reset_runtime_metrics()


def convert_to_batch_for_unizero(batch_data, policy_cfg, reward_support, value_support):
    """
    Overview:
        Convert replay buffer sample data to batch_for_unizero format for world_model.compute_loss.
        This function transforms the raw data from the replay buffer into the format expected
        by the UniZero world model's compute_loss method.

    Arguments:
        - batch_data: Data sampled from replay buffer (current_batch, target_batch)
        - policy_cfg: Policy configuration object
        - reward_support: Reward support tensor for categorical distribution
        - value_support: Value support tensor for categorical distribution

    Returns:
        - batch_for_unizero (:obj:`dict`): Dictionary containing formatted data for world model
    """
    from lzero.policy.utils import to_torch_float_tensor, prepare_obs, prepare_obs_stack_for_unizero
    from lzero.policy import scalar_transform, phi_transform

    # Unpack batch data
    current_batch, target_batch = batch_data[:2]
    obs_batch_ori, action_batch, target_action_batch, mask_batch, indices, weights, make_time, timestep_batch = current_batch
    target_reward, target_value, target_policy = target_batch

    # Prepare observations
    if policy_cfg.model.frame_stack_num > 1:
        obs_batch, obs_target_batch = prepare_obs_stack_for_unizero(obs_batch_ori, policy_cfg)
    else:
        obs_batch, obs_target_batch = prepare_obs(obs_batch_ori, policy_cfg)

    # Convert to tensors
    action_batch = torch.from_numpy(action_batch).to(policy_cfg.device).unsqueeze(-1).long()
    timestep_batch = torch.from_numpy(timestep_batch).to(policy_cfg.device).unsqueeze(-1).long()
    data_list = [mask_batch, target_reward, target_value, target_policy, weights]
    mask_batch, target_reward, target_value, target_policy, weights = to_torch_float_tensor(
        data_list, policy_cfg.device
    )
    target_reward = target_reward.view(policy_cfg.batch_size, -1)
    target_value = target_value.view(policy_cfg.batch_size, -1)

    # Transform rewards and values
    transformed_target_reward = scalar_transform(target_reward)
    transformed_target_value = scalar_transform(target_value)

    # Convert to categorical distributions
    target_reward_categorical = phi_transform(reward_support, transformed_target_reward)
    target_value_categorical = phi_transform(value_support, transformed_target_value)

    # Prepare batch_for_unizero
    batch_for_unizero = {}
    if isinstance(policy_cfg.model.observation_shape, int) or len(policy_cfg.model.observation_shape) == 1:
        batch_for_unizero['observations'] = torch.cat((obs_batch, obs_target_batch), dim=1).reshape(
            policy_cfg.batch_size, -1, policy_cfg.model.observation_shape)
    elif len(policy_cfg.model.observation_shape) == 3:
        batch_for_unizero['observations'] = torch.cat((obs_batch, obs_target_batch), dim=1).reshape(
            policy_cfg.batch_size, -1, *policy_cfg.model.observation_shape)

    batch_for_unizero['actions'] = action_batch.squeeze(-1)
    batch_for_unizero['timestep'] = timestep_batch.squeeze(-1)
    batch_for_unizero['rewards'] = target_reward_categorical[:, :-1]
    batch_for_unizero['mask_padding'] = mask_batch == 1.0
    batch_for_unizero['mask_padding'] = batch_for_unizero['mask_padding'][:, :-1]
    batch_for_unizero['observations'] = batch_for_unizero['observations'][:, :-1]
    batch_for_unizero['ends'] = torch.zeros(batch_for_unizero['mask_padding'].shape, dtype=torch.long, device=policy_cfg.device)
    batch_for_unizero['target_value'] = target_value_categorical[:, :-1]
    batch_for_unizero['target_policy'] = target_policy[:, :-1]

    return batch_for_unizero


def create_unizero_loss_metrics(policy):
    """
    Overview:
        Create a metrics function for computing UniZero losses without gradient updates.
        This is used for loss landscape visualization where we need to compute losses
        at different parameter values without actually updating the model.

    Arguments:
        - policy: The policy instance containing model, configuration, and all necessary attributes

    Returns:
        - compute_metrics (:obj:`Callable`): Function that computes losses for a batch of data
    """
    import logging

    # Get reward_support and value_support from policy
    reward_support = policy.reward_support
    value_support = policy.value_support

    def compute_metrics(net, dataloader, use_cuda):
        """
        Compute losses for loss landscape visualization.

        Arguments:
            - net: The neural network model
            - dataloader: DataLoader providing batches of data
            - use_cuda: Whether to use CUDA

        Returns:
            - dict: Dictionary containing averaged losses (policy_loss, value_loss, reward_loss, total_loss)
        """
        net.eval()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_reward_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch_data in dataloader:
                try:
                    # Convert replay buffer sample to batch_for_unizero format
                    batch_for_unizero = convert_to_batch_for_unizero(
                        batch_data,
                        policy._cfg,
                        reward_support,
                        value_support
                    )

                    # Call world_model.compute_loss (no backward, no optimizer.step)
                    losses = net.world_model.compute_loss(
                        batch_for_unizero,
                        policy._target_model.world_model.tokenizer,
                        policy.value_inverse_scalar_transform_handle
                    )

                    # Extract individual losses from intermediate_losses
                    total_policy_loss += losses.intermediate_losses['loss_policy'].item()
                    total_value_loss += losses.intermediate_losses['loss_value'].item()
                    total_reward_loss += losses.intermediate_losses['loss_rewards'].item()
                    total_batches += 1
                except Exception as e:
                    logging.warning(f"Error processing batch in compute_metrics: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        if total_batches > 0:
            return {
                'policy_loss': total_policy_loss / total_batches,
                'value_loss': total_value_loss / total_batches,
                'reward_loss': total_reward_loss / total_batches,
                'total_loss': (total_policy_loss + total_value_loss + total_reward_loss) / total_batches
            }
        else:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'reward_loss': 0.0, 'total_loss': 0.0}

    return compute_metrics


class UniZeroDataLoader:
    """
    Overview:
        DataLoader wrapper for UniZero replay buffer sampling.
        This provides an iterator interface for sampling batches from the replay buffer,
        compatible with loss landscape visualization tools.

    Arguments:
        - replay_buffer: The game buffer containing collected episodes
        - policy: The policy instance for sampling
        - batch_size (:obj:`int`): Number of samples per batch
        - num_batches (:obj:`int`): Total number of batches to sample
    """
    def __init__(self, replay_buffer, policy, batch_size, num_batches):
        self.buffer = replay_buffer
        self.policy = policy
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __iter__(self):
        """Iterator that yields batches from the replay buffer"""
        for _ in range(self.num_batches):
            batch = self.buffer.sample(self.batch_size, self.policy)
            yield batch

    def __len__(self):
        """Return the total number of batches"""
        return self.num_batches
