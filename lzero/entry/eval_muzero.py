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


def train_muzero_eval(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        num_episodes_each_seed: int = 1,
        print_seed_details: int = False,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline eval entry for MCTS+RL algorithms, including MuZero, EfficientZero, Sampled EfficientZero.
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

    assert create_cfg.policy.type in ['efficientzero', 'muzero', 'sampled_efficientzero'], \
        "LightZero noow only support the following algo.: 'efficientzero', 'muzero', 'sampled_efficientzero'"

    if create_cfg.policy.type == 'muzero':
        from lzero.mcts import MuZeroGameBuffer as GameBuffer
        from lzero.worker import MuZeroEvaluator as BaseSerialEvaluator
    elif create_cfg.policy.type == 'efficientzero':
        from lzero.mcts import EfficientZeroGameBuffer as GameBuffer
        from lzero.worker import EfficientZeroEvaluator as BaseSerialEvaluator
    elif create_cfg.policy.type == 'sampled_efficientzero':
        from lzero.mcts import SampledEfficientZeroGameBuffer as GameBuffer
        from lzero.worker import SampledEfficientZeroEvaluator as BaseSerialEvaluator

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
    if cfg.policy.device == 'cuda' and torch.cuda.is_available():
        cfg.policy.cuda = True
    else:
        cfg.policy.cuda = False
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])

    # load pretrained model
    if cfg.policy.model_path is not None:
        policy.learn_mode.load_state_dict(torch.load(cfg.policy.model_path, map_location='cpu'))
        policy.collect_mode.load_state_dict(torch.load(cfg.policy.model_path, map_location='cpu'))
        policy.eval_mode.load_state_dict(torch.load(cfg.policy.model_path, map_location='cpu'))

    # Create worker components: learner, collector, evaluator, replay buffer, commander.
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)

    # ==============================================================
    # MCTS+RL algorithms related core code
    # ==============================================================
    game_config = cfg.policy
    # specific game buffer for MCTS+RL algorithms
    replay_buffer = GameBuffer(game_config)
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
        # ==============================================================
        # eval trained model
        # ==============================================================
        returns = []
        for i in range(num_episodes_each_seed):
            stop, reward = evaluator.eval(
                learner.save_checkpoint, learner.train_iter, collector.envstep, config=game_config
            )
            returns.append(reward)
        returns = np.array(returns)

        if print_seed_details:
            print("=" * 20)
            print(f'In seed {seed}, returns: {returns}')
            print(f'win rate: {len(np.where(returns == 1.)[0])/ num_episodes_each_seed}, draw rate: {len(np.where(returns == 0.)[0])/num_episodes_each_seed}, lose rate: {len(np.where(returns == -1.)[0])/ num_episodes_each_seed}')
            print("=" * 20)

        return returns.mean(), returns
