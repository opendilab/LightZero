import os

import psutil
from pympler.asizeof import asizeof
from tensorboardX import SummaryWriter
from typing import Optional, Callable


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
    new_data = collector.collect(n_episode=policy_cfg.random_collect_episode_num, train_iter=0, policy_kwargs=collect_kwargs)

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
    # "writer is None" means we are in a slave process in the DDP setup.
    if writer is not None:
        sample_times = buffer.sample_times

        if sample_times == 0:
            return

        # Record average reanalyze time.
        total_compute_target_re_time = buffer.compute_target_re_time
        average_reanalyze_time = total_compute_target_re_time / sample_times
        writer.add_scalar('Buffer/average_reanalyze_time', average_reanalyze_time, train_iter)

        # Record average origin search time.
        total_origin_search_time = buffer.origin_search_time
        average_origin_search_time = total_origin_search_time / sample_times
        writer.add_scalar('Buffer/average_origin_search_time', average_origin_search_time, train_iter)

        # Record average reuse search time.
        total_reuse_search_time = buffer.reuse_search_time
        average_reuse_search_time = total_reuse_search_time / sample_times
        writer.add_scalar('Buffer/average_reuse_search_time', average_reuse_search_time, train_iter)

        # Record average reanalyze time.
        total_active_root_num = buffer.active_root_num
        average_active_root_num = total_active_root_num / sample_times
        writer.add_scalar('Buffer/average_active_root_num', average_active_root_num, train_iter)

        # Record average reanalyze time.
        # total_infer = buffer.average_infer
        # average_average_infer = total_infer / sample_times
        # writer.add_scalar('Buffer/average_infer', average_average_infer, train_iter)


        # Reset the time records.
        buffer.sample_times = 0
        buffer.compute_target_re_time = 0
        buffer.reuse_search_time = 0
        buffer.origin_search_time = 0
        buffer.active_root_num = 0
        # buffer.average_infer = 0
