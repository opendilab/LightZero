import logging
import os
from functools import partial
from typing import Optional, Tuple

import torch
from ding.config import compile_config
from ding.envs import create_env_manager
from ding.envs import get_vec_env_setting
from ding.policy import create_policy
from ding.utils import set_pkg_seed, get_rank
from ding.rl_utils import get_epsilon_greedy_fn
from ding.worker import BaseLearner
from tensorboardX import SummaryWriter

from lzero.worker import MuZeroCollector as Collector
from lzero.worker import MuZeroEvaluator as Evaluator
from lzero.entry.utils import log_buffer_memory_usage
from lzero.policy import visit_count_temperature
from lzero.entry.utils import random_collect
from zoo.petting_zoo.model import PettingZooEncoder

def eval_muzero(main_cfg, create_cfg, seed=0):
    assert create_cfg.policy.type in ['efficientzero', 'muzero', 'sampled_efficientzero', 'gumbel_muzero', 'multi_agent_efficientzero', 'multi_agent_muzero'], \
        "train_muzero entry now only support the following algo.: 'efficientzero', 'muzero', 'sampled_efficientzero', 'gumbel_muzero', 'multi_agent_efficientzero', 'multi_agent_muzero'"

    if create_cfg.policy.type == 'muzero' or create_cfg.policy.type == 'multi_agent_muzero':
        from lzero.mcts import MuZeroGameBuffer as GameBuffer
        from lzero.model.muzero_model_mlp import MuZeroModelMLP as Encoder
    elif create_cfg.policy.type == 'efficientzero' or create_cfg.policy.type == 'multi_agent_efficientzero':
        from lzero.mcts import EfficientZeroGameBuffer as GameBuffer
        from lzero.model.efficientzero_model_mlp import EfficientZeroModelMLP as Encoder
    elif create_cfg.policy.type == 'sampled_efficientzero':
        from lzero.mcts import SampledEfficientZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'gumbel_muzero':
        from lzero.mcts import GumbelMuZeroGameBuffer as GameBuffer

    main_cfg.policy.device = 'cpu'
    main_cfg.policy.load_path = 'exp_name/ckpt/ckpt_best.pth.tar'
    main_cfg.env.replay_path = './'      # when visualize must set as  base
    create_cfg.env_manager.type = 'base' # when visualize must set as  base
    main_cfg.env.evaluator_env_num = 1   # only 1 env for save replay
    main_cfg.env.n_evaluator_episode = 1

    cfg = compile_config(main_cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
    # Create main components: env, policy
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)

    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])

    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

    model = Encoder(**cfg.policy.model, state_encoder=PettingZooEncoder())
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])
    policy.learn_mode.load_state_dict(torch.load(cfg.policy.load_path, map_location=cfg.policy.device))

    # Create worker components: learner, collector, evaluator, replay buffer, commander.
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial')) if get_rank() == 0 else None
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)

    # ==============================================================
    # MCTS+RL algorithms related core code
    # ==============================================================
    policy_config = cfg.policy
    evaluator = Evaluator(
        eval_freq=cfg.policy.eval_freq,
        n_evaluator_episode=cfg.env.n_evaluator_episode,
        stop_value=cfg.env.stop_value,
        env=evaluator_env,
        policy=policy.eval_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        policy_config=policy_config
    )
    stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter)
    return stop, reward

if __name__ == '__main__':
    from zoo.petting_zoo.config.ptz_simple_spread_ez_config import main_config, create_config
    eval_muzero(main_config, create_config, seed=0)