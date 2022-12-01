import logging
import os
from functools import partial
from typing import Union, Optional, List, Any, Tuple

import numpy as np
import torch
from ding.config import read_config, compile_config
from ding.envs import create_env_manager
from ding.envs import get_vec_env_setting
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from ding.worker import BaseLearner
from ding.worker import create_serial_collector
from tensorboardX import SummaryWriter

from core.rl_utils import MuZeroGameBuffer, visit_count_temperature
from core.worker import MuZeroEvaluator as BaseSerialEvaluator


# @profile
def serial_pipeline_muzero_eval(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        test_episodes: int = 1,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline entry for EfficientZero and its variants, such as EfficientZero.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[List[Any]]`): A list with 3 elements: \
            ``BaseEnv`` subclass, collector env config, and evaluator env config.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - max_train_iter (:obj:`Optional[int]`): Maximum policy update iterations in training.
        - max_env_step (:obj:`Optional[int]`): Maximum collected environment interaction steps.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = input_cfg
    # create_cfg.policy.type = create_cfg.policy.type + '_command'
    create_cfg.policy.type = create_cfg.policy.type

    env_fn = None if env_setting is None else env_setting[0]
    cfg = compile_config(cfg, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg, save_cfg=True)
    # Create main components: env, policy
    if env_setting is None:
        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    else:
        env_fn, collector_env_cfg, evaluator_env_cfg = env_setting
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    # policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval', 'command'])
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])

    # load pretrained model
    if cfg.policy.model_path is not None:
        policy.learn_mode.load_state_dict(torch.load(cfg.policy.model_path, map_location='cpu'))

    # Create worker components: learner, collector, evaluator, replay buffer, commander.
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)

    # EfficientZero related code
    # specific game buffer for EfficientZero
    game_config = cfg.policy

    replay_buffer = MuZeroGameBuffer(game_config)
    collector = create_serial_collector(
        cfg.policy.collect.collector,
        env=collector_env,
        policy=policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        replay_buffer=replay_buffer,
        game_config=game_config
    )
    evaluator = BaseSerialEvaluator(
        cfg.policy.eval.evaluator,
        evaluator_env,
        policy.eval_mode,
        tb_logger,
        exp_name=cfg.exp_name,
        game_config=game_config
    )

    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    learner.call_hook('before_run')

    while True:
        collect_kwargs = {}
        # set temperature for visit count distributions according to the train_iter,
        # please refer to Appendix A.1 in EfficientZero for details
        collect_kwargs['temperature'] = np.array(
            [
                visit_count_temperature(game_config.auto_temperature, game_config.fixed_temperature_value, game_config.max_training_steps, trained_steps=learner.train_iter)
                for _ in range(game_config.collector_env_num)
            ]
        )

        # TODO(pu): eval trained model
        rewards = []
        for i in range(test_episodes):
            stop, reward = evaluator.eval(
                learner.save_checkpoint, learner.train_iter, collector.envstep, config=game_config
            )
            rewards.append(reward)
        rewards = np.array(rewards)
        # print("=" * 20)
        # print(f'In seed {seed}, we test {test_episodes} episodes.')
        # print('rewards:', rewards)
        # print('reward_mean:', rewards.mean())
        # print(f'win rate: {len(np.where(rewards == 1.)[0]) / test_episodes}, draw rate: {len(np.where(rewards == 0.)[0]) / test_episodes}, lose rate: {len(np.where(rewards == -1.)[0]) / test_episodes}')
        # print("=" * 20)
        break

    return rewards.mean(), rewards