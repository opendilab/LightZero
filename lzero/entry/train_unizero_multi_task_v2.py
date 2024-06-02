import logging
import os
from functools import partial
from typing import Optional, Tuple, List

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
from lzero.mcts import UniZeroGameBuffer as GameBuffer


def initialize_zeros_batch(observation_shape, batch_size, device):
    """Initialize a zeros tensor for batch observations based on the shape."""
    if isinstance(observation_shape, list):
        shape = [batch_size, *observation_shape]
    elif isinstance(observation_shape, int):
        shape = [batch_size, observation_shape]
    else:
        raise TypeError("observation_shape must be either an int or a list")
    
    return torch.zeros(shape).to(device)

def train_unizero_multi_task_v2(
        input_cfg_list: List[Tuple[dict, dict]],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    (Modified documentation for multi-task learning.)
    """
    # assert all(create_cfg['policy']['type'] in ['efficientzero', 'muzero', 'sampled_efficientzero', 'gumbel_muzero', 'stochastic_muzero'] for _, create_cfg in input_cfg_list), \
    #    "train_muzero_multi_task entry now only support 'efficientzero', 'muzero', 'sampled_efficientzero', 'gumbel_muzero'"

    cfgs = []
    game_buffers = []
    collector_envs = []
    evaluator_envs = []
    collectors = []
    evaluators = []
    tb_loggers = []

    task_id, [cfg, create_cfg] = input_cfg_list[0]
    if cfg.policy.cuda and torch.cuda.is_available():
        cfg.policy.device = 'cuda'
    else:
        cfg.policy.device = 'cpu'
    cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
    # Shared policy for all tasks
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial')) if get_rank() == 0 else None
    # tb_loggers.append(tb_logger)

    if model_path is not None:
        policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))

    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    policy_config = cfg.policy

    if cfg.policy.update_per_collect is not None:
        update_per_collect = cfg.policy.update_per_collect
    batch_size = policy_config.batch_size[0]  # TODO

    # for task_id, input_cfg in enumerate(input_cfg_list):
    for task_id, input_cfg in input_cfg_list:
        if task_id > 0:
            cfg, create_cfg = input_cfg
            # Replicate the setup process for each task, creating individual components.
            # ... (same initialization code as before)
            if cfg.policy.cuda and torch.cuda.is_available():
                cfg.policy.device = 'cuda'
            else:
                cfg.policy.device = 'cpu'
            cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)

        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
        collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
        evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
        collector_env.seed(cfg.seed + task_id)  # Ensure different seeds for different tasks
        evaluator_env.seed(cfg.seed + task_id, dynamic_seed=False)
        set_pkg_seed(cfg.seed + task_id, use_cuda=cfg.policy.cuda)

        # GameBuffer = get_game_buffer_class(create_cfg['policy']['type'])  # get_game_buffer_class should return the correct GameBuffer class

        # cfg.policy.task_id = task_id
        replay_buffer = GameBuffer(cfg.policy)

        collector = Collector(
            env=collector_env,
            policy=policy.collect_mode,
            tb_logger=tb_logger,
            exp_name=cfg.exp_name,
            policy_config=policy_config,
            task_id=task_id
        )
        evaluator = Evaluator(
            eval_freq=cfg.policy.eval_freq,
            n_evaluator_episode=cfg.env.n_evaluator_episode,
            stop_value=cfg.env.stop_value,
            env=evaluator_env,
            policy=policy.eval_mode,
            tb_logger=tb_logger,
            exp_name=cfg.exp_name,
            policy_config=policy_config,
            task_id=task_id
        )
        cfgs.append(cfg)
        replay_buffer.batch_size = cfg.policy.batch_size[task_id]
        print(f'task_id:{task_id}, replay_buffer.batch_size: {replay_buffer.batch_size}, replay_buffer.task_id: {replay_buffer.task_id}')
        game_buffers.append(replay_buffer)
        collector_envs.append(collector_env)
        evaluator_envs.append(evaluator_env)
        collectors.append(collector)
        evaluators.append(evaluator)
        # if task_id>0:
        #     tb_loggers.append(tb_logger)

    # Main loop
    learner.call_hook('before_run')

    # for task_id, (cfg, evaluator, collector, replay_buffer) in enumerate(zip(cfgs, evaluators, collectors, game_buffers)):
    #     # Usage
    #     print(f'='*20)
    #     print(f'evaluate task_id: {task_id}...')
    #     policy.last_batch_obs = initialize_zeros_batch(
    #         cfg.policy.model.observation_shape,
    #         len(evaluator_env_cfg),
    #         cfg.policy.device
    #     )
    #     policy.last_batch_action = [-1 for _ in range(len(evaluator_env_cfg))]
    #     # TODO: comment if debugging
    #     stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)

    totoal_env_steps = 0
    while True:
        # 预先计算位置编码矩阵，只用于 collect/eval 的推理阶段，不用于训练阶段
        policy._collect_model.world_model.precompute_pos_emb_diff_kv() # 非常重要，kv更新后需要重新计算
        policy._target_model.world_model.precompute_pos_emb_diff_kv() # 非常重要，kv更新后需要重新计算

        # 每个环境单独收集数据，并放入各自独立的replay buffer中
        for task_id, (cfg, collector, evaluator, replay_buffer) in enumerate(zip(cfgs, collectors, evaluators, game_buffers)):

            # TODO ===========================
            policy._collect_model.world_model.past_keys_values_cache_init_infer.clear() # very important
            for kv_cache_dict_env in policy._collect_model.world_model.past_keys_values_cache_init_infer_envs:
                kv_cache_dict_env.clear() 
            policy._collect_model.world_model.past_keys_values_cache_recurrent_infer.clear() # very important
            policy._collect_model.world_model.keys_values_wm_list.clear()  # TODO: 只适用于recurrent_inference() batch_pad
            
            policy._target_model.world_model.past_keys_values_cache_init_infer.clear()
            for kv_cache_dict_env in policy._target_model.world_model.past_keys_values_cache_init_infer_envs:
                kv_cache_dict_env.clear() 
            policy._target_model.world_model.past_keys_values_cache_recurrent_infer.clear()
            policy._target_model.world_model.keys_values_wm_list.clear() # TODO: 只适用于recurrent_inference() batch_pad
            print('sample target_model past_keys_values_cache.clear()')
            torch.cuda.empty_cache() # TODO: NOTE


            # Perform task-specific collection, evaluation, and training as needed
            # ... (same collection code as before, but with task-specific components)
            # When sampling for training, sample from each task's replay buffer and combine into train_data
            # Ensure train_data includes task_id to allow policy to handle data appropriately

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
                totoal_env_steps += collector.envstep
                # collect_kwargs['epsilon'] = epsilon_greedy_fn(collector.envstep)  # TODO
                collect_kwargs['epsilon'] = epsilon_greedy_fn(totoal_env_steps)
            else:
                collect_kwargs['epsilon'] = 0.0

            # Evaluate policy performance.
            if evaluator.should_eval(learner.train_iter):
                print(f'='*20)
                print(f'evaluate task_id: {task_id}...')
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
            print(f'='*20)
            print(f'collect task_id: {task_id}...')
            new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
            
            if cfg.policy.update_per_collect is None:
                # update_per_collect is None, then update_per_collect is set to the number of collected transitions multiplied by the model_update_ratio.
                collected_transitions_num = sum([len(game_segment) for game_segment in new_data[0]])
                update_per_collect = int(collected_transitions_num * cfg.policy.model_update_ratio)
            # save returned new_data collected by the collector
            replay_buffer.push_game_segments(new_data)
            # remove the oldest data if the replay buffer is full.
            replay_buffer.remove_oldest_data_to_fit()


        not_enough_data = False
        for task_id, replay_buffer in enumerate(game_buffers):
            if replay_buffer.get_num_of_transitions() < batch_size:
                not_enough_data = True

        # Learn policy from collected data.
        if not not_enough_data:
            for i in range(update_per_collect):
                train_data_multi_task = []
                envstep_multi_task = 0
                for task_id, (cfg, collector, replay_buffer) in enumerate(zip(cfgs, collectors, game_buffers)):
                    envstep_multi_task += collector.envstep
                    # Learner will train ``update_per_collect`` times in one iteration.
                    if replay_buffer.get_num_of_transitions() > batch_size:
                        batch_size = cfg.policy.batch_size[task_id]
                        # if task_id==0: # pong
                        #     batch_size = 16*6
                        # elif task_id==1: # mspacman
                        #     batch_size = 16  
                        train_data = replay_buffer.sample(batch_size, policy)
                        if cfg.policy.reanalyze_ratio > 0:
                            if i % 20 == 0: # for reanalyze_ratio>0
                                policy._target_model.world_model.past_keys_values_cache.clear()
                                policy._target_model.world_model.keys_values_wm_list.clear() # TODO: 只适用于recurrent_inference() batch_pad
                                torch.cuda.empty_cache() # TODO: 是否需要立即释放显存
                                print('sample target_model past_keys_values_cache.clear()')

                        #  ==================== 非常重要 ====================
                        train_data.append(task_id)
                        train_data_multi_task.append(train_data)
                    else:
                        logging.warning(
                            f'The data in replay_buffer is not sufficient to sample a mini-batch: '
                            f'batch_size: {batch_size}, '
                            f'{replay_buffer} '
                            f'continue to collect now ....'
                        )
                        break

                # if len(train_data_multi_task) != 0:
                # The core train steps for MCTS+RL algorithms.
                log_vars = learner.train(train_data_multi_task, envstep_multi_task)

                # if cfg.policy.use_priority:
                #     replay_buffer.update_priority(train_data, log_vars[0]['value_priority_orig'])

        # # 预先计算位置编码矩阵，只用于 collect/eval 的推理阶段，不用于训练阶段
        # policy._collect_model.world_model.precompute_pos_emb_diff_kv() # 非常重要，kv更新后需要重新计算
        # policy._target_model.world_model.precompute_pos_emb_diff_kv() # 非常重要，kv更新后需要重新计算
        # policy._target_model.world_model.past_keys_values_cache_init_infer.clear()
        # for kv_cache_dict_env in policy._target_model.world_model.past_keys_values_cache_init_infer_envs:
        #     kv_cache_dict_env.clear() 
        # policy._target_model.world_model.past_keys_values_cache_recurrent_infer.clear()
        # policy._target_model.world_model.keys_values_wm_list.clear() # TODO: 只适用于recurrent_inference() batch_pad
        # print('sample target_model past_keys_values_cache.clear()')
        # policy._collect_model.world_model.past_keys_values_cache_init_infer.clear() # very important
        # for kv_cache_dict_env in policy._collect_model.world_model.past_keys_values_cache_init_infer_envs:
        #     kv_cache_dict_env.clear() 
        # policy._collect_model.world_model.past_keys_values_cache_recurrent_infer.clear() # very important
        # policy._collect_model.world_model.keys_values_wm_list.clear()  # TODO: 只适用于recurrent_inference() batch_pad
        # torch.cuda.empty_cache() # TODO: NOTE

        # Break condition
        if all(collector.envstep >= max_env_step for collector in collectors) or learner.train_iter >= max_train_iter:
            break

    learner.call_hook('after_run')
    return policy
