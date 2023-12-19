from lzero.mcts import MAGameBuffer as GameBuffer
from zoo.atari.config.ma_pong1_config import main_config, create_config
import torch
import numpy as np
from ding.config import compile_config
from ding.policy import create_policy

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
data = np.load('collected_data.npy', allow_pickle=True)
replay_buffer1.push_game_segments(data)
sample_batch_size = 256
for i in range(20):
    replay_buffer1.sample(sample_batch_size, policy)
print(f'reanalyze time is {replay_buffer1.compute_target_re_time}')
print(f'origin_search_time is {replay_buffer1.origin_search_time}')
print(f'reuse_search_time is {replay_buffer1.reuse_search_time}')
print(f'active_root_num is {replay_buffer1.active_root_num}')

# # test ``_compute_target_policy_reanalyzed``
# replay_buffer2 = GameBuffer(policy_config)
# policy_re_context = np.load('policy_re_context.npy', allow_pickle=True)
# replay_buffer2._compute_target_policy_reanalyzed(policy_re_context, policy._target_model)