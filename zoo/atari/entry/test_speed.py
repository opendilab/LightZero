from lzero.mcts import MAGameBuffer as GameBuffer
from zoo.atari.config.ma_pong1_config import main_config, create_config
import torch
import numpy as np
from ding.config import compile_config
from ding.policy import create_policy
from tensorboardX import SummaryWriter
import psutil

def get_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss

cfg, create_cfg = main_config, create_config
if cfg.policy.cuda and torch.cuda.is_available():
    cfg.policy.device = 'cuda'
else:
    cfg.policy.device = 'cpu'

cfg = compile_config(cfg, seed=0, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
policy = create_policy(cfg.policy, model=None, enable_field=['learn', 'collect', 'eval'])

model_path = '/mnt/nfs/xcy/LightZero/zoo/atari/config/data_mz_ctree/Pong/reuse1_bigbatch6_test_231215_112717/ckpt/iteration_20000.pth.tar'
if model_path is not None:
    policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))

policy_config = cfg.policy

# test ``sample``
replay_buffer1 = GameBuffer(policy_config)
data = np.load('/mnt/nfs/xcy/LightZero/zoo/atari/entry/collected_data.npy', allow_pickle=True)
for i in range(50):
    replay_buffer1.push_game_segments(data)
    # print(f"the num of transitions is {replay_buffer1.get_num_of_transitions()}")

log_dir = "logs/memory_test2"  # 指定日志目录
writer = SummaryWriter(log_dir)

sample_batch_size = 25600
for i in range(2):
    memory_usage = get_memory_usage()
    # print(f"初始内存使用量: {memory_usage} 字节")
    replay_buffer1.sample(sample_batch_size*(i+1), policy)
    temp = memory_usage
    memory_usage = get_memory_usage()
    memory_cost = memory_usage - temp
    # print(f"sample后内存使用量: {memory_usage} 字节")
    print(f"sample的内存使用量: {float(memory_cost)/1e9} G")
    writer.add_scalar("sample的内存使用量", float(memory_cost)/1e9, i+1)
    # print(f'reanalyze time is {replay_buffer1.compute_target_re_time}')
    # print(f'origin_search_time is {replay_buffer1.origin_search_time}')
    # print(f'reuse_search_time is {replay_buffer1.reuse_search_time}')
    # print(f'active_root_num is {replay_buffer1.active_root_num}')
    # writer.add_scalar("reanalyze time", replay_buffer1.compute_target_re_time, i+1)
    # writer.add_scalar("origin_search_time", replay_buffer1.origin_search_time, i+1)
    # writer.add_scalar("reuse_search_time", replay_buffer1.reuse_search_time, i+1)
    # writer.add_scalar("relative reanalyze time", replay_buffer1.compute_target_re_time/(i+1), i+1)
    # writer.add_scalar("relative origin_search_time", replay_buffer1.origin_search_time/(i+1), i+1)
    # writer.add_scalar("relative reuse_search_time", replay_buffer1.reuse_search_time/(i+1), i+1)
    # writer.add_scalar("active_root_num", replay_buffer1.active_root_num, i+1)
    replay_buffer1.compute_target_re_time = 0
    replay_buffer1.origin_search_time = 0
    replay_buffer1.reuse_search_time = 0
    replay_buffer1.active_root_num = 0

# test ``_compute_target_policy_reanalyzed``
# replay_buffer2 = GameBuffer(policy_config)
# policy_re_context = np.load('policy_re_context.npy', allow_pickle=True)
# for i in range(1):
#     replay_buffer2._compute_target_policy_reanalyzed(policy_re_context, policy._target_model)
# print(f'reanalyze time is {replay_buffer2.compute_target_re_time}')
# print(f'origin_search_time is {replay_buffer2.origin_search_time}')
# print(f'reuse_search_time is {replay_buffer2.reuse_search_time}')
# print(f'active_root_num is {replay_buffer2.active_root_num}')
    
writer.close()