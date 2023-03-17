import logging
import os
from functools import partial
from typing import Union, Optional, List, Any, Tuple

import torch
from ding.config import read_config, compile_config
from ding.envs import create_env_manager
from ding.envs import get_vec_env_setting
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from ding.worker import BaseLearner, create_buffer
from ding.worker import create_serial_collector, create_serial_evaluator
from tensorboardX import SummaryWriter
import numpy as np


def eval_alphazero(
        input_cfg: Tuple[dict, dict],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        num_episodes_each_seed: int = 1,
        print_seed_details: int = False,
) -> 'Policy':  # noqa
    """
    Overview:
        The eval entry for AlphaZero.
    Arguments:
        - input_cfg (:obj:`Tuple[dict, dict]`): Config in dict type.
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - model_path (:obj:`Optional[str]`): The pretrained model path, which should
            point to the ckpt file of the pretrained model, and an absolute path is recommended.
            In LightZero, the path is usually something like ``exp_name/ckpt/ckpt_best.pth.tar``.
        - max_train_iter (:obj:`Optional[int]`): Maximum policy update iterations in training.
        - max_env_step (:obj:`Optional[int]`): Maximum collected environment interaction steps.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """
    cfg, create_cfg = input_cfg
    create_cfg.policy.type = create_cfg.policy.type

    cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
    # Create main components: env, policy
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])

    # load pretrained model
    if model_path is not None:
        policy.learn_mode.load_state_dict(torch.load(model_path, map_location='cpu'))

    # Create worker components: learner, collector, evaluator, replay buffer, commander.
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))

    env_config = cfg.env
    evaluator = create_serial_evaluator(
        cfg.policy.eval.evaluator,
        env=evaluator_env,
        policy=policy.eval_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        env_config=env_config,
    )

    while True:
        # ==============================================================
        # eval trained model
        # ==============================================================
        returns = []
        for i in range(num_episodes_each_seed):
            stop, reward = evaluator.eval()
            returns.append(reward[0]['final_eval_reward'])

        returns = np.array(returns)

        if print_seed_details:
            print("=" * 20)
            print(f'In seed {seed}, returns: {returns}')
            print(
                f'win rate: {len(np.where(returns == 1.)[0]) / num_episodes_each_seed}, draw rate: {len(np.where(returns == 0.)[0]) / num_episodes_each_seed}, lose rate: {len(np.where(returns == -1.)[0]) / num_episodes_each_seed}')
            print("=" * 20)

        return returns.mean(), returns
