import logging
import os
from functools import partial
from typing import Tuple, Optional, List

import torch
import numpy as np
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.rl_utils import get_epsilon_greedy_fn
from ding.utils import set_pkg_seed, get_rank
from ding.worker import BaseLearner
from tensorboardX import SummaryWriter

from lzero.entry.utils import log_buffer_memory_usage, log_buffer_run_time
from lzero.policy import visit_count_temperature
from lzero.policy.random_policy import LightZeroRandomPolicy
from lzero.worker import MuZeroEvaluator as Evaluator
from lzero.worker import MuZeroSegmentCollector as Collector
from lzero.mcts import MuZeroGameBuffer as GameBuffer  # 根据不同策略选择合适的 GameBuffer
from .utils import random_collect

from ding.utils import EasyTimer
timer = EasyTimer()
from line_profiler import line_profiler


def train_muzero_multitask_segment_noddp(
        input_cfg_list: List[Tuple[int, Tuple[dict, dict]]],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    Overview:
        多任务训练入口，基于 MuZero 的多任务版本，支持多任务环境的训练。
        参考论文 UniZero: Generalized and Efficient Planning with Scalable Latent World Models。
    Arguments:
        - input_cfg_list (List[Tuple[int, Tuple[dict, dict]]]): 不同任务的配置列表。
        - seed (int): 随机种子。
        - model (Optional[torch.nn.Module]): torch.nn.Module 的实例。
        - model_path (Optional[str]): 预训练模型路径，指向预训练模型的 ckpt 文件。
        - max_train_iter (Optional[int]): 最大训练迭代次数。
        - max_env_step (Optional[int]): 最大环境交互步数。
    Returns:
        - policy (Policy): 收敛后的策略。
    """
    cfgs = []
    game_buffers = []
    collector_envs = []
    evaluator_envs = []
    collectors = []
    evaluators = []

    task_id, [cfg, create_cfg] = input_cfg_list[0]

    # Ensure the specified policy type is supported
    assert create_cfg.policy.type in ['muzero_multitask'], "train_muzero entry now only supports 'muzero'"

    # Set device based on CUDA availability
    cfg.policy.device = cfg.policy.model.world_model_cfg.device if torch.cuda.is_available() else 'cpu'
    logging.info(f'cfg.policy.device: {cfg.policy.device}')

    # Compile the configuration
    cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
    # Create shared policy for all tasks
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])

    # Load pretrained model if specified
    if model_path is not None:
        logging.info(f'Loading model from {model_path} begin...')
        policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))
        logging.info(f'Loading model from {model_path} end!')

    # Create SummaryWriter for TensorBoard logging
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial')) if get_rank() == 0 else None
    # Create shared learner for all tasks
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)

    # TODO task_id = 0:
    policy_config = cfg.policy
    batch_size = policy_config.batch_size[0]

    # 初始化多任务配置
    for task_id, input_cfg in input_cfg_list:

        if task_id > 0:
            # Get the configuration for each task
            cfg, create_cfg = input_cfg
            cfg.policy.device = 'cuda' if cfg.policy.cuda and torch.cuda.is_available() else 'cpu'
            cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
            policy_config = cfg.policy
            policy.collect_mode.get_attribute('cfg').n_episode = policy_config.n_episode
            policy.eval_mode.get_attribute('cfg').n_episode = policy_config.n_episode

        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
        collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
        evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
        collector_env.seed(cfg.seed + task_id)
        evaluator_env.seed(cfg.seed + task_id, dynamic_seed=False)
        set_pkg_seed(cfg.seed + task_id, use_cuda=cfg.policy.cuda)

        # ===== NOTE: Create different game buffer, collector, evaluator for each task ====
        # TODO: share replay buffer for all tasks
        replay_buffer = GameBuffer(policy_config)
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
        game_buffers.append(replay_buffer)
        collector_envs.append(collector_env)
        evaluator_envs.append(evaluator_env)
        collectors.append(collector)
        evaluators.append(evaluator)

    learner.call_hook('before_run')
    value_priority_tasks = {}

    buffer_reanalyze_count = 0
    train_epoch = 0
    reanalyze_batch_size = cfg.policy.reanalyze_batch_size
    update_per_collect = cfg.policy.update_per_collect

    while True:
        torch.cuda.empty_cache()

        # 遍历每个任务进行数据收集和评估
        for task_id, (cfg, collector, evaluator, replay_buffer) in enumerate(
                zip(cfgs, collectors, evaluators, game_buffers)):
            log_buffer_memory_usage(learner.train_iter, replay_buffer, tb_logger)

            collect_kwargs = {
                'temperature': visit_count_temperature(
                    policy_config.manual_temperature_decay,
                    policy_config.fixed_temperature_value,
                    policy_config.threshold_training_steps_for_final_temperature,
                    trained_steps=learner.train_iter
                ),
                'epsilon': 0.0  # 默认 epsilon 值
            }

            if policy_config.eps.eps_greedy_exploration_in_collect:
                epsilon_greedy_fn = get_epsilon_greedy_fn(
                    start=policy_config.eps.start,
                    end=policy_config.eps.end,
                    decay=policy_config.eps.decay,
                    type_=policy_config.eps.type
                )
                collect_kwargs['epsilon'] = epsilon_greedy_fn(collector.envstep)

            # 评估策略性能
            if learner.train_iter ==0 or evaluator.should_eval(learner.train_iter):
                logging.info(f'========== 评估任务 {task_id} ==========')
                stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
                if stop:
                    break

            # 收集数据
            logging.info(f'========== 收集任务 {task_id} 数据 ==========')
            # collector.reset()
            new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)

            # 确定每次收集后的更新次数
            if update_per_collect is None:
                collected_transitions_num = sum(
                    min(len(game_segment), cfg.policy.game_segment_length) for game_segment in new_data[0])
                update_per_collect = int(collected_transitions_num * cfg.policy.replay_ratio)

            # 更新回放缓冲区
            replay_buffer.push_game_segments(new_data)
            replay_buffer.remove_oldest_data_to_fit()

            # 定期重新分析缓冲区
            if cfg.policy.buffer_reanalyze_freq >= 1:
                reanalyze_interval = update_per_collect // cfg.policy.buffer_reanalyze_freq
            else:
                if train_epoch % int(1 / cfg.policy.buffer_reanalyze_freq) == 0 and replay_buffer.get_num_of_transitions() // cfg.policy.num_unroll_steps > int(reanalyze_batch_size / cfg.policy.reanalyze_partition):
                    with timer:
                        replay_buffer.reanalyze_buffer(reanalyze_batch_size, policy)
                    buffer_reanalyze_count += 1
                    logging.info(f'缓冲区重新分析次数: {buffer_reanalyze_count}')
                    logging.info(f'缓冲区重新分析时间: {timer.value}')

        # 检查是否有足够的数据进行训练
        not_enough_data = any(replay_buffer.get_num_of_transitions() < batch_size for replay_buffer in game_buffers)

        if not not_enough_data:
            # 进行训练
            for i in range(update_per_collect):
                train_data_multi_task = []
                envstep_multi_task = 0
                for task_id, (cfg, collector, replay_buffer) in enumerate(zip(cfgs, collectors, game_buffers)):
                    envstep_multi_task += collector.envstep
                    if replay_buffer.get_num_of_transitions() > batch_size:
                        batch_size = cfg.policy.batch_size[task_id]


                        if cfg.policy.buffer_reanalyze_freq >= 1:
                            if i % reanalyze_interval == 0 and replay_buffer.get_num_of_transitions() // cfg.policy.num_unroll_steps > int(
                                    reanalyze_batch_size / cfg.policy.reanalyze_partition):
                                with timer:
                                    replay_buffer.reanalyze_buffer(reanalyze_batch_size, policy)
                                buffer_reanalyze_count += 1
                                logging.info(f'缓冲区重新分析次数: {buffer_reanalyze_count}')
                                logging.info(f'缓冲区重新分析时间: {timer.value}')

                        train_data = replay_buffer.sample(batch_size, policy)
                        train_data.append(task_id)  # 添加 task_id
                        train_data_multi_task.append(train_data)
                    else:
                        logging.warning(
                            f'回放缓冲区数据不足以采样 mini-batch: '
                            f'batch_size: {batch_size}, 回放缓冲区: {replay_buffer}'
                        )
                        break

                if train_data_multi_task:
                    log_vars = learner.train(train_data_multi_task, envstep_multi_task)

                if cfg.policy.use_priority:
                    for task_id, replay_buffer in enumerate(game_buffers):
                        current_priorities = log_vars[0][f'value_priority_task{task_id}']
                        mean_priority = np.mean(current_priorities)
                        std_priority = np.std(current_priorities)
                        alpha = 0.1
                        if f'running_mean_priority_task{task_id}' not in value_priority_tasks:
                            value_priority_tasks[f'running_mean_priority_task{task_id}'] = mean_priority
                        else:
                            value_priority_tasks[f'running_mean_priority_task{task_id}'] = (
                                alpha * mean_priority + (1 - alpha) * value_priority_tasks[f'running_mean_priority_task{task_id}']
                            )
                        running_mean_priority = value_priority_tasks[f'running_mean_priority_task{task_id}']
                        normalized_priorities = (current_priorities - running_mean_priority) / (std_priority + 1e-6)

                        if cfg.policy.print_task_priority_logs:
                            print(f"任务 {task_id} - 平均优先级: {mean_priority:.8f}, "
                                  f"运行平均优先级: {running_mean_priority:.8f}, "
                                  f"标准差: {std_priority:.8f}")

            # 清除位置嵌入缓存
            train_epoch += 1

            # 检查是否达到训练结束条件
            if all(collector.envstep >= max_env_step for collector in collectors) or learner.train_iter >= max_train_iter:
                break

    # 调用学习器的 after_run 钩子
    learner.call_hook('after_run')
    return policy