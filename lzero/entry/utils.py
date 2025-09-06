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

    return torch.full(shape, fill_value=pad_token_id, dtype=torch.float32, device=device) if pad_token_id == 0 else torch.full(shape, fill_value=pad_token_id, dtype=torch.long, device=device)

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
