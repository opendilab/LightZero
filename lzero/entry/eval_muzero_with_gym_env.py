import os
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from tensorboardX import SummaryWriter

from ding.config import compile_config
from ding.envs import DingEnvWrapper, BaseEnvManager
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from ding.worker import BaseLearner
from lzero.envs.get_wrapped_env import get_wrappered_env
from lzero.worker import MuZeroEvaluator


def eval_muzero_with_gym_env(
        input_cfg: Tuple[dict, dict],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        num_episodes_each_seed: int = 1,
        print_seed_details: int = False,
) -> 'Policy':  # noqa
    """
    Overview:
        The eval entry for MCTS+RL algorithms, including MuZero, EfficientZero, Sampled EfficientZero.
        We create a gym environment using env_name parameter, and then convert it to the format required by LightZero using LightZeroEnvWrapper class. 
        Please refer to the get_wrappered_env method for more details.
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
    assert create_cfg.policy.type in ['efficientzero', 'muzero', 'sampled_efficientzero'], \
        "LightZero noow only support the following algo.: 'efficientzero', 'muzero', 'sampled_efficientzero'"

    cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)

    # Create main components: env, policy
    collector_env_cfg = DingEnvWrapper.create_collector_env_cfg(cfg.env)
    evaluator_env_cfg = DingEnvWrapper.create_evaluator_env_cfg(cfg.env)
    collector_env = BaseEnvManager([get_wrappered_env(c, cfg.env.env_name) for c in collector_env_cfg],
                                   cfg=BaseEnvManager.default_config())
    evaluator_env = BaseEnvManager([get_wrappered_env(c, cfg.env.env_name) for c in evaluator_env_cfg],
                                   cfg=BaseEnvManager.default_config())
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=True if cfg.policy.device == 'cuda' else False)

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
    # specific game buffer for MCTS+RL algorithms
    evaluator = MuZeroEvaluator(
        eval_freq=cfg.policy.eval_freq,
        n_evaluator_episode=cfg.env.n_evaluator_episode,
        stop_value=cfg.env.stop_value,
        env=evaluator_env,
        policy=policy.eval_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        policy_config=policy_config
    )

    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    learner.call_hook('before_run')

    while True:
        # ==============================================================
        # eval trained model
        # ==============================================================
        returns = []
        for i in range(num_episodes_each_seed):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter)
            returns.append(reward)
        returns = np.array(returns)

        if print_seed_details:
            print("=" * 20)
            print(f'In seed {seed}, returns: {returns}')
            if cfg.policy.env_type == 'board_games':
                print(
                    f'win rate: {len(np.where(returns == 1.)[0]) / num_episodes_each_seed}, draw rate: {len(np.where(returns == 0.)[0]) / num_episodes_each_seed}, lose rate: {len(np.where(returns == -1.)[0]) / num_episodes_each_seed}')
            print("=" * 20)

        return returns.mean(), returns
