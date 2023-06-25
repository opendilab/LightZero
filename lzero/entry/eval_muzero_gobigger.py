import logging
import os
from functools import partial
from typing import Optional, Tuple
import numpy as np
import torch
from tensorboardX import SummaryWriter
import copy

from ding.config import compile_config
from ding.envs import create_env_manager
from ding.envs import get_vec_env_setting
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from ding.worker import BaseLearner
from lzero.worker import GoBiggerMuZeroEvaluator


def eval_muzero_gobigger(
        input_cfg: Tuple[dict, dict],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
) -> 'Policy':  # noqa
    """
    Overview:
        The train entry for MCTS+RL algorithms, including MuZero, EfficientZero, Sampled EfficientZero.
    Arguments:
        - input_cfg (:obj:`Tuple[dict, dict]`): Config in dict type.
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - model_path (:obj:`Optional[str]`): The pretrained model path, which should
            point to the ckpt file of the pretrained model, and an absolute path is recommended.
            In LightZero, the path is usually something like ``exp_name/ckpt/ckpt_best.pth.tar``.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """
    cfg, create_cfg = input_cfg
    assert create_cfg.policy.type in ['gobigger_efficientzero', 'gobigger_muzero', 'gobigger_sampled_efficientzero'], \
        "train_muzero entry now only support the following algo.: 'gobigger_efficientzero', 'gobigger_muzero', 'gobigger_sampled_efficientzero'"

    if create_cfg.policy.type == 'gobigger_efficientzero':
        from lzero.mcts import GoBiggerEfficientZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'gobigger_muzero':
        from lzero.mcts import GoBiggerMuZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'gobigger_sampled_efficientzero':
        from lzero.mcts import GoBiggerSampledEfficientZeroGameBuffer as GameBuffer

    if cfg.policy.cuda and torch.cuda.is_available():
        cfg.policy.device = 'cuda'
    else:
        cfg.policy.device = 'cpu'

    cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
    # Create main components: env, policy
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])

    env_cfg = copy.deepcopy(evaluator_env_cfg[0])
    env_cfg.contain_raw_obs = True
    vsbot_evaluator_env_cfg = [env_cfg for _ in range(len(evaluator_env_cfg))]
    vsbot_evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in vsbot_evaluator_env_cfg])

    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    vsbot_evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])

    # load pretrained model
    if model_path is not None:
        policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))

    # Create worker components: learner, collector, evaluator, replay buffer, commander.
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)

    # ==============================================================
    # MCTS+RL algorithms related core code
    # ==============================================================
    policy_config = cfg.policy
    evaluator = GoBiggerMuZeroEvaluator(
        eval_freq=cfg.policy.eval_freq,
        n_evaluator_episode=cfg.env.n_evaluator_episode,
        stop_value=cfg.env.stop_value,
        env=evaluator_env,
        policy=policy.eval_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        policy_config=policy_config
    )
    vsbot_evaluator = GoBiggerMuZeroEvaluator(
        eval_freq=cfg.policy.eval_freq,
        n_evaluator_episode=cfg.env.n_evaluator_episode,
        stop_value=cfg.env.stop_value,
        env=vsbot_evaluator_env,
        policy=policy.eval_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        policy_config=policy_config,
        instance_name='vsbot_evaluator'
    )
    # ==============================================================
    # Main loop
    # ==============================================================
    # Learner's before_run hook.
    learner.call_hook('before_run')
    # ==============================================================
    # eval trained model
    # ==============================================================
    _, reward_sp = evaluator.eval(learner.save_checkpoint, learner.train_iter)
    _, reward_vsbot = vsbot_evaluator.eval_vsbot(learner.save_checkpoint, learner.train_iter)
    return reward_sp, reward_vsbot
