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

from lzero.entry.utils import log_buffer_memory_usage
from lzero.policy import visit_count_temperature
from lzero.policy.random_policy import LightZeroRandomPolicy
from lzero.worker import MuZeroCollector as Collector
from lzero.worker import MuZeroEvaluator as Evaluator
from .utils import random_collect


def train_muzero(
        input_cfg: Tuple[dict, dict],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    Overview:
        The train entry for MCTS+RL algorithms, including MuZero, EfficientZero, Sampled EfficientZero, Gumbel Muzero.
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
    assert create_cfg.policy.type in ['efficientzero', 'muzero', 'sampled_efficientzero', 'gumbel_muzero', 'stochastic_muzero'], \
        "train_muzero entry now only support the following algo.: 'efficientzero', 'muzero', 'sampled_efficientzero', 'gumbel_muzero', 'stochastic_muzero'"

    if create_cfg.policy.type == 'muzero':
        from lzero.mcts import MuZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'efficientzero':
        from lzero.mcts import EfficientZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'sampled_efficientzero':
        from lzero.mcts import SampledEfficientZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'gumbel_muzero':
        from lzero.mcts import GumbelMuZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'stochastic_muzero':
        from lzero.mcts import StochasticMuZeroGameBuffer as GameBuffer

    if cfg.policy.cuda and torch.cuda.is_available():
        cfg.policy.device = 'cuda'
    else:
        cfg.policy.device = 'cpu'

    cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
    # Create main components: env, policy
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)

    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])

    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

    if cfg.policy.eval_offline:
        cfg.policy.learn.learner.hook.save_ckpt_after_iter = cfg.policy.eval_freq

    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])

    # load pretrained model
    if model_path is not None:
        policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))

    # Create worker components: learner, collector, evaluator, replay buffer, commander.
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial')) if get_rank() == 0 else None
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)

    # ==============================================================
    # MCTS+RL algorithms related core code
    # ==============================================================
    policy_config = cfg.policy
    batch_size = policy_config.batch_size
    # specific game buffer for MCTS+RL algorithms
    replay_buffer = GameBuffer(policy_config)
    collector = Collector(
        env=collector_env,
        policy=policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        policy_config=policy_config
    )
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

    # ==============================================================
    # Main loop
    # ==============================================================
    # Learner's before_run hook.
    learner.call_hook('before_run')

    if cfg.policy.update_per_collect is not None:
        update_per_collect = cfg.policy.update_per_collect

    # The purpose of collecting random data before training:
    # Exploration: Collecting random data helps the agent explore the environment and avoid getting stuck in a suboptimal policy prematurely.
    # Comparison: By observing the agent's performance during random action-taking, we can establish a baseline to evaluate the effectiveness of reinforcement learning algorithms.
    if cfg.policy.random_collect_episode_num > 0:
        random_collect(cfg.policy, policy, LightZeroRandomPolicy, collector, collector_env, replay_buffer)
    if cfg.policy.eval_offline:
        eval_train_iter_list = []
        eval_train_envstep_list = []

    while True:
        log_buffer_memory_usage(learner.train_iter, replay_buffer, tb_logger)
        collect_kwargs = {}
        # set temperature for visit count distributions according to the train_iter,
        # please refer to Appendix D in MuZero paper for details.
        collect_kwargs['temperature'] = visit_count_temperature(
            policy_config.manual_temperature_decay,
            policy_config.fixed_temperature_value,
            policy_config.threshold_training_steps_for_final_temperature,
            trained_steps=learner.train_iter
        )

        if policy_config.eps.eps_greedy_exploration_in_collect:
            epsilon_greedy_fn = get_epsilon_greedy_fn(
                start=policy_config.eps.start,
                end=policy_config.eps.end,
                decay=policy_config.eps.decay,
                type_=policy_config.eps.type
            )
            collect_kwargs['epsilon'] = epsilon_greedy_fn(collector.envstep)
        else:
            collect_kwargs['epsilon'] = 0.0

        # Evaluate policy performance.
        if evaluator.should_eval(learner.train_iter):
            if cfg.policy.eval_offline:
                eval_train_iter_list.append(learner.train_iter)
                eval_train_envstep_list.append(collector.envstep)
            else:
                stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
                if stop:
                    break

        # Collect data by default config n_sample/n_episode.
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        if cfg.policy.update_per_collect is None:
            # update_per_collect is None, then update_per_collect is set to the number of collected transitions multiplied by the model_update_ratio.
            collected_transitions_num = sum([len(game_segment) for game_segment in new_data[0]])
            update_per_collect = int(collected_transitions_num * cfg.policy.model_update_ratio)
        # save returned new_data collected by the collector
        replay_buffer.push_game_segments(new_data)
        # remove the oldest data if the replay buffer is full.
        replay_buffer.remove_oldest_data_to_fit()

        # Learn policy from collected data.
        for i in range(update_per_collect):
            # Learner will train ``update_per_collect`` times in one iteration.
            if replay_buffer.get_num_of_transitions() > batch_size:
                train_data = replay_buffer.sample(batch_size, policy)
            else:
                logging.warning(
                    f'The data in replay_buffer is not sufficient to sample a mini-batch: '
                    f'batch_size: {batch_size}, '
                    f'{replay_buffer} '
                    f'continue to collect now ....'
                )
                break

            # The core train steps for MCTS+RL algorithms.
            log_vars = learner.train(train_data, collector.envstep)

            if cfg.policy.use_priority:
                replay_buffer.update_priority(train_data, log_vars[0]['value_priority_orig'])

        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            if cfg.policy.eval_offline:
                logging.info(f'eval offline beginning...')
                ckpt_dirname = './{}/ckpt'.format(learner.exp_name)
                # Evaluate the performance of the pretrained model.
                for train_iter, collector_envstep in zip(eval_train_iter_list, eval_train_envstep_list):
                    ckpt_name = 'iteration_{}.pth.tar'.format(train_iter)
                    ckpt_path = os.path.join(ckpt_dirname, ckpt_name)
                    # load the ckpt of pretrained model
                    policy.learn_mode.load_state_dict(torch.load(ckpt_path, map_location=cfg.policy.device))
                    stop, reward = evaluator.eval(learner.save_checkpoint, train_iter, collector_envstep)
                    logging.info(
                        f'eval offline at train_iter: {train_iter}, collector_envstep: {collector_envstep}, reward: {reward}')
                logging.info(f'eval offline finished!')
            break

    # Learner's after_run hook.
    learner.call_hook('after_run')
    return policy
