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

from core.rl_utils import SampledGameBuffer as GameBuffer, visit_count_temperature
from core.worker import SampledEfficientZeroEvaluator as BaseSerialEvaluator


# @profile
def serial_pipeline_sampled_efficientzero(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline entry for Sampled EfficientZero and its variants, such as Sampled EfficientZero.
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

    # Sampled EfficientZero related code
    # specific game buffer for Sampled EfficientZero
    game_config = cfg.policy

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

    # commander = BaseSerialCommander(
    #     cfg.policy.other.commander, learner, collector, evaluator, replay_buffer, policy.command_mode
    # )
    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    learner.call_hook('before_run')

    while True:
        # collect_kwargs = commander.step()
        collect_kwargs = {}
        # set temperature for visit count distributions according to the train_iter,
        # please refer to Appendix A.1 in Sampled EfficientZero for details
        collect_kwargs['temperature'] = np.array(
            [
                visit_count_temperature(game_config.auto_temperature, game_config.fixed_temperature_value,
                                        game_config.max_training_steps, trained_steps=learner.train_iter*cfg.policy.learn.update_per_collect)
                for _ in range(game_config.collector_env_num)
            ]
        )

        # Evaluate policy performance
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(
                learner.save_checkpoint, learner.train_iter, collector.envstep, config=game_config
            )
            if stop:
                break

        # Collect data by default config n_sample/n_episode
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)

        # import pdb; pdb.set_trace()

        # TODO(pu): collector return data
        replay_buffer.push_games(new_data[0], new_data[1])

        # remove the oldest data if the replay buffer is full.
        replay_buffer.remove_oldest_data_to_fit()

        # Learn policy from collected data
        for i in range(cfg.policy.learn.update_per_collect):
            # Learner will train ``update_per_collect`` times in one iteration.
            if replay_buffer.get_num_of_transitions() > learner.policy.get_attribute('batch_size'):
                train_data = replay_buffer.sample_train_data(learner.policy.get_attribute('batch_size'), policy)
            else:
                logging.warning(
                        f'The data in replay_buffer is not sufficient to sample a minibatch: '
                        f'batch_size: {replay_buffer.get_batch_size()},'
                        f'num_of_episodes: {replay_buffer.get_num_of_episodes()}, '
                        f'num of game historys: {replay_buffer.get_num_of_game_histories()}, '
                        f'number of transitions: {replay_buffer.get_num_of_transitions()}, '
                        f'continue to collect now ....'
                    )
                break

            learner.train(train_data, collector.envstep)

            train_steps = learner.train_iter * cfg.policy.learn.update_per_collect

            # if game_config.lr_manually:
            #     # learning rate decay manually like EfficientZero paper
            #     if train_steps  > 1e5 and train_steps  <= 2e5:
            #         policy._optimizer.lr = 0.02
            #     elif train_steps  > 2e5:
            #         policy._optimizer.lr = 0.002
            if game_config.lr_manually:
                # learning rate decay manually like MuZero paper
                if train_steps < 0.5 * game_config.max_training_steps:
                    policy._optimizer.lr = 0.2
                elif train_steps < 0.75 * game_config.max_training_steps:
                    policy._optimizer.lr = 0.02
                else:
                    policy._optimizer.lr = 0.002

        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break

    # Learner's after_run hook.
    learner.call_hook('after_run')
    return policy
