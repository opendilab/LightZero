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
import torch.nn as nn

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
    assert create_cfg.policy.type in ['efficientzero', 'unizero', 'sampled_efficientzero', 'gumbel_muzero', 'stochastic_muzero'], \
        "train_unizero entry now only support the following algo.: 'efficientzero', 'muzero', 'sampled_efficientzero', 'gumbel_muzero'"

    if create_cfg.policy.type == 'unizero':
        from lzero.mcts import UniZeroGameBuffer as GameBuffer
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

    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])

    # load pretrained model
    if model_path is not None:
        policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))
        print('load model from path:', model_path)

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

    import copy
    num_unroll_steps = copy.deepcopy(replay_buffer._cfg.num_unroll_steps)
    collect_cnt = -1

    # Usage
    policy.last_batch_obs = initialize_zeros_batch(
        cfg.policy.model.observation_shape,
        len(evaluator_env_cfg),
        cfg.policy.device
    )
    policy.last_batch_action = [-1 for _ in range(len(evaluator_env_cfg))]
    # TODO: comment if debugging
    # stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)

    while True:
        collect_cnt += 1
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
            policy.last_batch_obs = initialize_zeros_batch(
                cfg.policy.model.observation_shape,
                len(evaluator_env_cfg),
                cfg.policy.device
            )
            policy.last_batch_action = [-1 for _ in range(len(evaluator_env_cfg))]
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break

        policy.last_batch_obs = initialize_zeros_batch(
            cfg.policy.model.observation_shape,
            len(collector_env_cfg),
            cfg.policy.device
        )
        policy.last_batch_action = [-1 for _ in range(len(collector_env_cfg))]
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

        replay_buffer._cfg.num_unroll_steps = num_unroll_steps
        batch_size = policy._cfg.batch_size
        replay_buffer._cfg.batch_size = batch_size
        if collector.envstep > cfg.policy.transformer_start_after_envsteps:
            # TODO：transformer tokenizer交替更新
            # Learn policy from collected data.
            # for i in range(cfg.policy.update_per_collect_transformer):
            for i in range(update_per_collect):
                # Learner will train ``update_per_collect`` times in one iteration.
                # if replay_buffer.get_num_of_transitions() > batch_size:
                if replay_buffer.get_num_of_game_segments() > batch_size:  # TODO: for memory env
                    train_data = replay_buffer.sample(batch_size, policy)
                    if cfg.policy.reanalyze_ratio > 0:
                        if i % 20 == 0:
                        # if i % 2 == 0:# for reanalyze_ratio>0
                            policy._target_model.world_model.past_keys_values_cache_init_infer.clear()
                            policy._target_model.world_model.past_keys_values_cache_recurrent_infer.clear()
                            policy._target_model.world_model.keys_values_wm_list.clear() # TODO: 只适用于recurrent_inference() batch_pad
                            torch.cuda.empty_cache() # TODO: 是否需要立即释放显存
                            print('sample target_model past_keys_values_cache.clear()')

                    train_data.append({'train_which_component': 'transformer'})
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
        
        # 预先计算位置编码矩阵，只用于 collect/eval 的推理阶段，不用于训练阶段
        policy._collect_model.world_model.precompute_pos_emb_diff_kv() # 非常重要，kv更新后需要重新计算
        policy._target_model.world_model.precompute_pos_emb_diff_kv() # 非常重要，kv更新后需要重新计算

        policy._target_model.world_model.past_keys_values_cache_init_infer.clear()
        for kv_cache_dict_env in policy._target_model.world_model.past_keys_values_cache_init_infer_envs:
            kv_cache_dict_env.clear() 

        policy._target_model.world_model.past_keys_values_cache_recurrent_infer.clear()
        policy._target_model.world_model.keys_values_wm_list.clear() # TODO: 只适用于recurrent_inference() batch_pad
        print('sample target_model past_keys_values_cache.clear()')

        policy._collect_model.world_model.past_keys_values_cache_init_infer.clear() # very important
        for kv_cache_dict_env in policy._collect_model.world_model.past_keys_values_cache_init_infer_envs:
            kv_cache_dict_env.clear() 
        policy._collect_model.world_model.past_keys_values_cache_recurrent_infer.clear() # very important
        policy._collect_model.world_model.keys_values_wm_list.clear()  # TODO: 只适用于recurrent_inference() batch_pad
        torch.cuda.empty_cache() # TODO: NOTE


        # if collector.envstep > 0:
        #     # TODO: only for debug
        #     for param in policy._learn_model.world_model.tokenizer.parameters():
        #         param.requires_grad = False
        #     print("train some steps before collector.envstep > 0, then fixed")

        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break

    # Learner's after_run hook.
    learner.call_hook('after_run')
    return policy
