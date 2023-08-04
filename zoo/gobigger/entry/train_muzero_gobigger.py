import logging
import os
from functools import partial
from typing import Optional, Tuple

import torch
from ding.config import compile_config
from ding.envs import create_env_manager
from ding.envs import get_vec_env_setting
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from ding.worker import BaseLearner
from tensorboardX import SummaryWriter
import copy
from ding.rl_utils import get_epsilon_greedy_fn
from lzero.entry.utils import log_buffer_memory_usage
from lzero.policy import visit_count_temperature
from lzero.worker import GoBiggerMuZeroEvaluator
from lzero.entry.utils import random_collect
from lzero.policy.multi_agent_random_policy import MultiAgentLightZeroRandomPolicy


def train_muzero_gobigger(
        input_cfg: Tuple[dict, dict],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    Overview:
        The train entry for GoBigger MCTS+RL algorithms, including MuZero, EfficientZero, Sampled EfficientZero.
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
    assert create_cfg.policy.type in ['efficientzero', 'muzero', 'sampled_efficientzero', 'gumbel_muzero', 'multi_agent_efficientzero', 'multi_agent_muzero'], \
        "train_muzero entry now only support the following algo.: 'efficientzero', 'muzero', 'sampled_efficientzero', 'gumbel_muzero', 'multi_agent_efficientzero', 'multi_agent_muzero'"

    if create_cfg.policy.type == 'muzero':
        from lzero.mcts import MuZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'efficientzero':
        from lzero.mcts import EfficientZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'sampled_efficientzero':
        from lzero.mcts import SampledEfficientZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'gumbel_muzero':
        from lzero.mcts import GumbelMuZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'multi_agent_efficientzero':
        from lzero.mcts import MultiAgentSampledEfficientZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'multi_agent_muzero':
        from lzero.mcts import MultiAgentMuZeroGameBuffer as GameBuffer

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
    batch_size = policy_config.batch_size
    # specific game buffer for MCTS+RL algorithms
    replay_buffer = GameBuffer(policy_config)
    if policy_config.multi_agent:
        from lzero.worker import MultiAgentMuZeroCollector as Collector
        from lzero.worker import MuZeroEvaluator as Evaluator
    else:
        from lzero.worker import MuZeroCollector as Collector
        from lzero.worker import MuZeroEvaluator as Evaluator
    collector = Collector(
        env=collector_env,
        policy=policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        policy_config=policy_config
    )
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

    if cfg.policy.update_per_collect is not None:
        update_per_collect = cfg.policy.update_per_collect

    # The purpose of collecting random data before training:
    # Exploration: The collection of random data aids the agent in exploring the environment and prevents premature convergence to a suboptimal policy.
    # Comparation: The agent's performance during random action-taking can be used as a reference point to evaluate the efficacy of reinforcement learning algorithms.
    if cfg.policy.random_collect_episode_num > 0:
        random_collect(cfg.policy, policy, MultiAgentLightZeroRandomPolicy, collector, collector_env, replay_buffer)

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
            stop, reward = evaluator.eval(None, learner.train_iter, collector.envstep) # save_ckpt_fn = None 
            stop, reward = vsbot_evaluator.eval_vsbot(learner.save_checkpoint, learner.train_iter, collector.envstep)
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
            break

    # Learner's after_run hook.
    learner.call_hook('after_run')
    return policy
