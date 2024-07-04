from lzero.mcts import ReZeroMZGameBuffer as GameBuffer
from zoo.atari.config.atari_rezero_mz_config import main_config, create_config
import torch
import numpy as np
from ding.config import compile_config
from ding.policy import create_policy
from tensorboardX import SummaryWriter
import psutil


def get_memory_usage():
    """
    Get the current memory usage of the process.

    Returns:
        int: Memory usage in bytes.
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss


def initialize_policy(cfg, create_cfg):
    """
    Initialize the policy based on the given configuration.

    Args:
        cfg (Config): Main configuration object.
        create_cfg (Config): Creation configuration object.

    Returns:
        Policy: Initialized policy object.
    """
    if cfg.policy.cuda and torch.cuda.is_available():
        cfg.policy.device = 'cuda'
    else:
        cfg.policy.device = 'cpu'

    cfg = compile_config(cfg, seed=0, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
    policy = create_policy(cfg.policy, model=None, enable_field=['learn', 'collect', 'eval'])

    model_path = '{template_path}/iteration_20000.pth.tar'
    if model_path is not None:
        policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))

    return policy, cfg


def run_memory_test(replay_buffer, policy, writer, sample_batch_size):
    """
    Run memory usage test for sampling from the replay buffer.

    Args:
        replay_buffer (GameBuffer): The replay buffer to sample from.
        policy (Policy): The policy object.
        writer (SummaryWriter): TensorBoard summary writer.
        sample_batch_size (int): The base batch size for sampling.
    """
    for i in range(2):
        initial_memory = get_memory_usage()
        print(f"Initial memory usage: {initial_memory} bytes")

        replay_buffer.sample(sample_batch_size * (i + 1), policy)

        final_memory = get_memory_usage()
        memory_cost = final_memory - initial_memory

        print(f"Memory usage after sampling: {final_memory} bytes")
        print(f"Memory cost of sampling: {float(memory_cost) / 1e9:.2f} GB")

        writer.add_scalar("Sampling Memory Usage (GB)", float(memory_cost) / 1e9, i + 1)

        # Reset counters
        replay_buffer.compute_target_re_time = 0
        replay_buffer.origin_search_time = 0
        replay_buffer.reuse_search_time = 0
        replay_buffer.active_root_num = 0


def main():
    """
    Main function to run the memory usage test.
    """
    cfg, create_cfg = main_config, create_config
    policy, cfg = initialize_policy(cfg, create_cfg)

    replay_buffer = GameBuffer(cfg.policy)

    # Load and push data to the replay buffer
    data = np.load('{template_path}/collected_data.npy', allow_pickle=True)
    for _ in range(50):
        replay_buffer.push_game_segments(data)

    log_dir = "logs/memory_test2"
    writer = SummaryWriter(log_dir)

    run_memory_test(replay_buffer, policy, writer, sample_batch_size=25600)

    writer.close()


if __name__ == "__main__":
    main()