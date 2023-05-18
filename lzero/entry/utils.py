import os

import psutil
from pympler.asizeof import asizeof
from tensorboardX import SummaryWriter


def log_buffer_memory_usage(train_iter: int, buffer: "GameBuffer", writer: SummaryWriter) -> None:
    """
    Overview:
        Log the memory usage of the buffer and the current process to TensorBoard.
    Arguments:
        - train_iter (:obj:`int`): The current training iteration.
        - buffer (:obj:`GameBuffer`): The game buffer.
        - writer (:obj:`SummaryWriter`): The TensorBoard writer.
    """
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
