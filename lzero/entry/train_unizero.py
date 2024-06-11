import copy
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
from lzero.worker import MuZeroCollector as Collector
from lzero.worker import MuZeroEvaluator as Evaluator
from .utils import random_collect


def initialize_zeros_batch(observation_shape, batch_size, device):
    """Initialize a zeros tensor for batch observations based on the shape."""
    if isinstance(observation_shape, list):
        shape = [batch_size, *observation_shape]
    elif isinstance(observation_shape, int):
        shape = [batch_size, observation_shape]
    else:
        raise TypeError("observation_shape must be either an int or a list")
    
    return torch.zeros(shape).to(device)

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

    # Ensure the specified policy type is supported
    assert create_cfg.policy.type in [
        'efficientzero', 'unizero', 'sampled_efficientzero', 'gumbel_muzero', 'stochastic_muzero'
    ], "train_unizero entry now only supports the following algo.: 'efficientzero', 'muzero', 'sampled_efficientzero', 'gumbel_muzero'"

    # Import the correct GameBuffer class based on the policy type
    game_buffer_classes = {
        'unizero': 'UniZeroGameBuffer',
        'efficientzero': 'EfficientZeroGameBuffer',
        'sampled_efficientzero': 'SampledEfficientZeroGameBuffer',
        'gumbel_muzero': 'GumbelMuZeroGameBuffer',
        'stochastic_muzero': 'StochasticMuZeroGameBuffer'
    }

    GameBuffer = getattr(__import__('lzero.mcts', fromlist=[game_buffer_classes[create_cfg.policy.type]]),
                         game_buffer_classes[create_cfg.policy.type])

    # Set device based on CUDA availability
    cfg.policy.device = 'cuda' if cfg.policy.cuda and torch.cuda.is_available() else 'cpu'

    # Compile the configuration
    cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)

    # Create main components: env, policy
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])

    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])

    # Load pretrained model if specified
    if model_path is not None:
        policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))
        print('Loaded model from path:', model_path)

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

    if cfg.policy.update_per_collect is not None:
        update_per_collect = cfg.policy.update_per_collect

    # Collect random data before training
    if cfg.policy.random_collect_episode_num > 0:
        random_collect(cfg.policy, policy, LightZeroRandomPolicy, collector, collector_env, replay_buffer)

    num_unroll_steps = copy.deepcopy(replay_buffer._cfg.num_unroll_steps)
    collect_cnt = -1

    policy.last_batch_obs = initialize_zeros_batch(cfg.policy.model.observation_shape, len(evaluator_env_cfg),
                                                   cfg.policy.device)
    policy.last_batch_action = [-1 for _ in range(len(evaluator_env_cfg))]

    while True:
        collect_cnt += 1
        log_buffer_memory_usage(learner.train_iter, replay_buffer, tb_logger)
        collect_kwargs = {}
        # Set temperature for visit count distributions according to the train_iter
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

        # Evaluate policy performance
        if evaluator.should_eval(learner.train_iter):
            policy.last_batch_obs = initialize_zeros_batch(cfg.policy.model.observation_shape, len(evaluator_env_cfg),
                                                           cfg.policy.device)
            policy.last_batch_action = [-1 for _ in range(len(evaluator_env_cfg))]
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break

        policy.last_batch_obs = initialize_zeros_batch(cfg.policy.model.observation_shape, len(collector_env_cfg),
                                                       cfg.policy.device)
        policy.last_batch_action = [-1 for _ in range(len(collector_env_cfg))]
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)

        if cfg.policy.update_per_collect is None:
            collected_transitions_num = sum([len(game_segment) for game_segment in new_data[0]])
            update_per_collect = int(collected_transitions_num * cfg.policy.model_update_ratio)

        replay_buffer.push_game_segments(new_data)
        replay_buffer.remove_oldest_data_to_fit()

        replay_buffer._cfg.num_unroll_steps = num_unroll_steps
        batch_size = policy._cfg.batch_size
        replay_buffer._cfg.batch_size = batch_size

        if collector.envstep > cfg.policy.train_start_after_envsteps:
            for i in range(update_per_collect):
                if replay_buffer.get_num_of_game_segments() > batch_size:
                    train_data = replay_buffer.sample(batch_size, policy)
                    if cfg.policy.reanalyze_ratio > 0 and i % 20 == 0:
                        policy._target_model.world_model.past_kv_cache_init_infer.clear()
                        policy._target_model.world_model.past_kv_cache_recurrent_infer.clear()
                        policy._target_model.world_model.keys_values_wm_list.clear()
                        torch.cuda.empty_cache()
                        print('Cleared target_model past_kv_cache.')

                    train_data.append({'train_which_component': 'transformer'})
                else:
                    logging.warning(
                        f'The data in replay_buffer is not sufficient to sample a mini-batch: '
                        f'batch_size: {batch_size}, '
                        f'{replay_buffer} '
                        f'continue to collect now ....'
                    )
                    break
                log_vars = learner.train(train_data, collector.envstep)

                if cfg.policy.use_priority:
                    replay_buffer.update_priority(train_data, log_vars[0]['value_priority_orig'])

        # Precompute positional embedding matrices for inference in collect/eval stages, not for training
        policy._collect_model.world_model.precompute_pos_emb_diff_kv()
        policy._target_model.world_model.precompute_pos_emb_diff_kv()

        policy._target_model.world_model.past_kv_cache_init_infer.clear()
        for kv_cache_dict_env in policy._target_model.world_model.past_kv_cache_init_infer_envs:
            kv_cache_dict_env.clear()
        policy._target_model.world_model.past_kv_cache_recurrent_infer.clear()
        policy._target_model.world_model.keys_values_wm_list.clear()
        print('Cleared target_model past_kv_cache.')

        policy._collect_model.world_model.past_kv_cache_init_infer.clear()
        for kv_cache_dict_env in policy._collect_model.world_model.past_kv_cache_init_infer_envs:
            kv_cache_dict_env.clear()
        policy._collect_model.world_model.past_kv_cache_recurrent_infer.clear()
        policy._collect_model.world_model.keys_values_wm_list.clear()
        print('Cleared collect_model past_kv_cache.')

        torch.cuda.empty_cache()

        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break

    if cfg.policy.model.analysis_sim_norm:
        policy._collect_model.encoder_hook.remove_hooks()
        policy._target_model.encoder_hook.remove_hooks()

    learner.call_hook('after_run')
    return policy
