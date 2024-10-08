import logging
import os
from functools import partial
from typing import Tuple, Optional

import torch
from ding.config import compile_config
from ding.envs import create_env_manager
from ding.envs import get_vec_env_setting
from ding.policy import create_policy
from ding.rl_utils import get_epsilon_greedy_fn
from ding.utils import set_pkg_seed, get_rank
from ding.worker import BaseLearner
from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from lzero.entry.utils import log_buffer_memory_usage
from lzero.policy import visit_count_temperature
from lzero.policy.random_policy import LightZeroRandomPolicy
from lzero.worker import MuZeroEvaluator as Evaluator
from lzero.worker import MuZeroCollector as Collector
from .utils import random_collect


def train_unizero(
        input_cfg: Tuple[dict, dict],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':
    """
    Overview:
        The train entry for UniZero, proposed in our paper UniZero: Generalized and Efficient Planning with Scalable Latent World Models.
        UniZero aims to enhance the planning capabilities of reinforcement learning agents by addressing the limitations found in MuZero-style algorithms,
        particularly in environments requiring the capture of long-term dependencies. More details can be found in https://arxiv.org/abs/2406.10667.
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

    # Ensure the specified policy type is supported
    assert create_cfg.policy.type in ['unizero', 'sampled_unizero'], "train_unizero entry now only supports the following algo.: 'unizero', 'sampled_unizero'"

    # Import the correct GameBuffer class based on the policy type
    game_buffer_classes = {'unizero': 'UniZeroGameBuffer', 'sampled_unizero': 'SampledUniZeroGameBuffer'}

    GameBuffer = getattr(__import__('lzero.mcts', fromlist=[game_buffer_classes[create_cfg.policy.type]]),
                         game_buffer_classes[create_cfg.policy.type])

    # Set device based on CUDA availability
    cfg.policy.device = cfg.policy.model.world_model_cfg.device if torch.cuda.is_available() else 'cpu'
    logging.info(f'cfg.policy.device: {cfg.policy.device}')

    # Compile the configuration
    cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)

    # Create main components: env, policy
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])

    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=torch.cuda.is_available())

    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])

    # Load pretrained model if specified
    if model_path is not None:
        logging.info(f'Loading model from {model_path} begin...')
        policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))
        logging.info(f'Loading model from {model_path} end!')

    # Create worker components: learner, collector, evaluator, replay buffer, commander
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial')) if get_rank() == 0 else None
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)

    # MCTS+RL algorithms related core code
    policy_config = cfg.policy
    replay_buffer = GameBuffer(policy_config)
    collector = Collector(env=collector_env, policy=policy.collect_mode, tb_logger=tb_logger, exp_name=cfg.exp_name,
                          policy_config=policy_config)
    evaluator = Evaluator(eval_freq=cfg.policy.eval_freq, n_evaluator_episode=cfg.env.n_evaluator_episode,
                          stop_value=cfg.env.stop_value, env=evaluator_env, policy=policy.eval_mode,
                          tb_logger=tb_logger, exp_name=cfg.exp_name, policy_config=policy_config)

    # Learner's before_run hook
    learner.call_hook('before_run')

    # Collect random data before training
    if cfg.policy.random_collect_episode_num > 0:
        random_collect(cfg.policy, policy, LightZeroRandomPolicy, collector, collector_env, replay_buffer)

    batch_size = policy._cfg.batch_size

    # TODO: for visualize
    # stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
    # import sys; sys.exit(0)

    while True:
        # Log buffer memory usage
        log_buffer_memory_usage(learner.train_iter, replay_buffer, tb_logger)

        # Set temperature for visit count distributions
        collect_kwargs = {
            'temperature': visit_count_temperature(
                policy_config.manual_temperature_decay,
                policy_config.fixed_temperature_value,
                policy_config.threshold_training_steps_for_final_temperature,
                trained_steps=learner.train_iter
            ),
            'epsilon': 0.0  # Default epsilon value
        }

        # Configure epsilon for epsilon-greedy exploration
        if policy_config.eps.eps_greedy_exploration_in_collect:
            epsilon_greedy_fn = get_epsilon_greedy_fn(
                start=policy_config.eps.start,
                end=policy_config.eps.end,
                decay=policy_config.eps.decay,
                type_=policy_config.eps.type
            )
            collect_kwargs['epsilon'] = epsilon_greedy_fn(collector.envstep)

        # Evaluate policy performance
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break

        # Collect new data
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)

        # Determine updates per collection
        update_per_collect = cfg.policy.update_per_collect
        if update_per_collect is None:
            # update_per_collect is None, then update_per_collect is set to the number of collected transitions multiplied by the replay_ratio.
            # The length of game_segment (i.e., len(game_segment.action_segment)) can be smaller than cfg.policy.game_segment_length if it represents the final segment of the game.
            # On the other hand, its length will be less than cfg.policy.game_segment_length + padding_length when it is not the last game segment. Typically, padding_length is the sum of unroll_steps and td_steps.
            collected_transitions_num = sum(min(len(game_segment), cfg.policy.game_segment_length) for game_segment in new_data[0])
            update_per_collect = int(collected_transitions_num * cfg.policy.replay_ratio)

        # Update replay buffer
        replay_buffer.push_game_segments(new_data)
        replay_buffer.remove_oldest_data_to_fit()

        # Train the policy if sufficient data is available
        if collector.envstep > cfg.policy.train_start_after_envsteps:
            if cfg.policy.sample_type == 'episode':
                data_sufficient = replay_buffer.get_num_of_game_segments() > batch_size
            else:
                data_sufficient = replay_buffer.get_num_of_transitions() > batch_size
            if not data_sufficient:
                logging.warning(
                    f'The data in replay_buffer is not sufficient to sample a mini-batch: '
                    f'batch_size: {batch_size}, replay_buffer: {replay_buffer}. Continue to collect now ....'
                )
                continue

            for i in range(update_per_collect):
                train_data = replay_buffer.sample(batch_size, policy)
                if cfg.policy.reanalyze_ratio > 0 and i % 20 == 0:
                    # Clear caches and precompute positional embedding matrices
                    policy.recompute_pos_emb_diff_and_clear_cache()  # TODO

                train_data.append({'train_which_component': 'transformer'})
                log_vars = learner.train(train_data, collector.envstep)

                if cfg.policy.use_priority:
                    replay_buffer.update_priority(train_data, log_vars[0]['value_priority_orig'])

        policy.recompute_pos_emb_diff_and_clear_cache()

        # Check stopping criteria
        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break

    learner.call_hook('after_run')
    return policy
