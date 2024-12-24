import logging
import os
from functools import partial
from typing import List, Optional, Tuple

import numpy as np
import torch
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.rl_utils import get_epsilon_greedy_fn
from ding.utils import EasyTimer, set_pkg_seed, get_rank
from ding.worker import BaseLearner
from tensorboardX import SummaryWriter

from lzero.entry.utils import log_buffer_memory_usage
from lzero.policy import visit_count_temperature
from lzero.worker import MuZeroEvaluator as Evaluator
from lzero.worker import MuZeroSegmentCollector as Collector


timer = EasyTimer()


def train_unizero_multitask_segment_serial(
    input_cfg_list: List[Tuple[int, Tuple[dict, dict]]],
    seed: int = 0,
    model: Optional[torch.nn.Module] = None,
    model_path: Optional[str] = None,
    max_train_iter: Optional[int] = int(1e10),
    max_env_step: Optional[int] = int(1e10),
) -> 'Policy':
    """
    概述:
        UniZero的训练入口，基于论文《UniZero: Generalized and Efficient Planning with Scalable Latent World Models》提出。
        UniZero旨在通过解决MuZero风格算法在需要捕捉长期依赖的环境中的局限性，增强强化学习代理的规划能力。
        详细内容可参考 https://arxiv.org/abs/2406.10667。

    参数:
        - input_cfg_list (List[Tuple[int, Tuple[dict, dict]]]): 不同任务的配置列表。
        - seed (int): 随机种子。
        - model (Optional[torch.nn.Module]): torch.nn.Module的实例。
        - model_path (Optional[str]): 预训练模型路径，应指向预训练模型的ckpt文件。
        - max_train_iter (Optional[int]): 训练中的最大策略更新迭代次数。
        - max_env_step (Optional[int]): 收集环境交互步骤的最大数量。

    返回:
        - policy (Policy): 收敛的策略对象。
    """
    cfgs = []
    game_buffers = []
    collector_envs = []
    evaluator_envs = []
    collectors = []
    evaluators = []

    # 获取第一个任务的配置
    task_id, [cfg, create_cfg] = input_cfg_list[0]

    # 确保指定的策略类型受支持
    assert create_cfg.policy.type in ['unizero_multitask', 'sampled_unizero_multitask'], "train_unizero entry 目前仅支持 'unizero_multitask'"

    if create_cfg.policy.type == 'unizero_multitask':
        from lzero.mcts import UniZeroGameBuffer as GameBuffer
    if create_cfg.policy.type == 'sampled_unizero_multitask':
        from lzero.mcts import SampledUniZeroGameBuffer as GameBuffer

    # 根据CUDA可用性设置设备
    cfg.policy.device = cfg.policy.model.world_model_cfg.device if torch.cuda.is_available() else 'cpu'
    logging.info(f'cfg.policy.device: {cfg.policy.device}')

    # 编译配置
    cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
    # 为所有任务创建共享策略
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])

    # 如果指定了预训练模型路径，加载预训练模型
    if model_path is not None:
        logging.info(f'开始加载模型: {model_path}')
        policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))
        logging.info(f'完成加载模型: {model_path}')

    # 为TensorBoard日志创建SummaryWriter
    tb_logger = SummaryWriter(os.path.join(f'./{cfg.exp_name}/log/', 'serial')) if get_rank() == 0 else None
    # 为所有任务创建共享学习器
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)

    policy_config = cfg.policy
    batch_size = policy_config.batch_size[0]

    # 遍历所有任务的配置
    for task_id, input_cfg in input_cfg_list:
        if task_id > 0:
            cfg, create_cfg = input_cfg
            cfg.policy.device = 'cuda' if cfg.policy.cuda and torch.cuda.is_available() else 'cpu'
            cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
            policy_config = cfg.policy
            # 更新收集和评估模式的配置
            policy.collect_mode.get_attribute('cfg').n_episode = policy_config.n_episode
            policy.eval_mode.get_attribute('cfg').n_episode = policy_config.n_episode

        # 创建环境管理器
        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
        collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
        evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
        collector_env.seed(cfg.seed + task_id)
        evaluator_env.seed(cfg.seed + task_id, dynamic_seed=False)
        set_pkg_seed(cfg.seed + task_id, use_cuda=cfg.policy.cuda)

        # 创建各任务专属的游戏缓存、收集器和评估器
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
        # 预计算收集和评估时的位置嵌入矩阵（非训练阶段）
        # policy._collect_model.world_model.precompute_pos_emb_diff_kv()
        # policy._target_model.world_model.precompute_pos_emb_diff_kv()

        # 为每个任务收集数据
        for task_id, (cfg, collector, evaluator, replay_buffer) in enumerate(zip(cfgs, collectors, evaluators, game_buffers)):
            log_buffer_memory_usage(learner.train_iter, replay_buffer, tb_logger)

            collect_kwargs = {
                'temperature': visit_count_temperature(
                    policy_config.manual_temperature_decay,
                    policy_config.fixed_temperature_value,
                    policy_config.threshold_training_steps_for_final_temperature,
                    trained_steps=learner.train_iter
                ),
                'epsilon': 0.0  # 默认epsilon值
            }

            # 如果启用了epsilon-greedy探索，计算当前epsilon值
            if policy_config.eps.eps_greedy_exploration_in_collect:
                epsilon_greedy_fn = get_epsilon_greedy_fn(
                    start=policy_config.eps.start,
                    end=policy_config.eps.end,
                    decay=policy_config.eps.decay,
                    type_=policy_config.eps.type
                )
                collect_kwargs['epsilon'] = epsilon_greedy_fn(collector.envstep)

            # 评估阶段
            if evaluator.should_eval(learner.train_iter):
                print('=' * 20)
                print(f'开始评估任务 id: {task_id}...')
                stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
                if stop:
                    break

            print('=' * 20)
            print(f'开始收集任务 id: {task_id}...')

            # 在每次收集前重置初始数据，对于多任务设置非常重要
            collector._policy.reset(reset_init_data=True, task_id=task_id)
            new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)

            # 确定每次收集后的更新次数
            if update_per_collect is None:
                # 如果未设置update_per_collect，则根据收集的转换数量和重放比例计算
                collected_transitions_num = sum(
                    min(len(game_segment), cfg.policy.game_segment_length) for game_segment in new_data[0]
                )
                update_per_collect = int(collected_transitions_num * cfg.policy.replay_ratio)

            # 更新重放缓存
            replay_buffer.push_game_segments(new_data)
            replay_buffer.remove_oldest_data_to_fit()

            # 定期重新分析重放缓存
            if cfg.policy.buffer_reanalyze_freq >= 1:
                # 一个训练epoch内重新分析buffer的次数
                reanalyze_interval = update_per_collect // cfg.policy.buffer_reanalyze_freq
            else:
                # 每隔一定数量的训练epoch重新分析buffer
                if train_epoch % int(1 / cfg.policy.buffer_reanalyze_freq) == 0 and replay_buffer.get_num_of_transitions() // cfg.policy.num_unroll_steps > int(reanalyze_batch_size / cfg.policy.reanalyze_partition):
                    with timer:
                        # 每次重新分析处理reanalyze_batch_size个序列
                        replay_buffer.reanalyze_buffer(reanalyze_batch_size, policy)
                    buffer_reanalyze_count += 1
                    logging.info(f'重放缓存重新分析次数: {buffer_reanalyze_count}')
                    logging.info(f'重放缓存重新分析时间: {timer.value}')

        # 检查是否有重放缓存数据不足
        not_enough_data = any(replay_buffer.get_num_of_transitions() < batch_size for replay_buffer in game_buffers)

        # 从收集的数据中学习策略
        if not not_enough_data:
            # 学习器将在一次迭代中进行update_per_collect次训练
            for i in range(update_per_collect):
                train_data_multi_task = []
                envstep_multi_task = 0
                for task_id, (cfg, collector, replay_buffer) in enumerate(zip(cfgs, collectors, game_buffers)):
                    envstep_multi_task += collector.envstep
                    if replay_buffer.get_num_of_transitions() > batch_size:
                        batch_size = cfg.policy.batch_size[task_id]

                        if cfg.policy.buffer_reanalyze_freq >= 1:
                            # 在一个训练epoch内按照频率重新分析buffer
                            if i % reanalyze_interval == 0 and replay_buffer.get_num_of_transitions() // cfg.policy.num_unroll_steps > int(reanalyze_batch_size / cfg.policy.reanalyze_partition):
                                with timer:
                                    replay_buffer.reanalyze_buffer(reanalyze_batch_size, policy)
                                buffer_reanalyze_count += 1
                                logging.info(f'重放缓存重新分析次数: {buffer_reanalyze_count}')
                                logging.info(f'重放缓存重新分析时间: {timer.value}')

                        train_data = replay_buffer.sample(batch_size, policy)
                        # 将task_id附加到训练数据
                        train_data.append(task_id)
                        train_data_multi_task.append(train_data)
                    else:
                        logging.warning(
                            f'重放缓存中的数据不足以采样一个小批量: '
                            f'batch_size: {batch_size}, replay_buffer: {replay_buffer}'
                        )
                        break

                if train_data_multi_task:
                    log_vars = learner.train(train_data_multi_task, envstep_multi_task)

                # 如果使用优先级重放，更新各任务的优先级
                if cfg.policy.use_priority:
                    for task_id, replay_buffer in enumerate(game_buffers):
                        # 更新任务特定重放缓存的优先级
                        replay_buffer.update_priority(train_data_multi_task[task_id], log_vars[0][f'value_priority_task{task_id}'])

                        # 获取当前任务的更新后优先级
                        current_priorities = log_vars[0][f'value_priority_task{task_id}']

                        # 计算优先级的均值和标准差
                        mean_priority = np.mean(current_priorities)
                        std_priority = np.std(current_priorities)

                        # 使用指数移动平均计算运行中的均值
                        alpha = 0.1  # 平滑因子，可根据需要调整
                        if f'running_mean_priority_task{task_id}' not in value_priority_tasks:
                            # 初始化运行均值
                            value_priority_tasks[f'running_mean_priority_task{task_id}'] = mean_priority
                        else:
                            # 更新运行均值
                            value_priority_tasks[f'running_mean_priority_task{task_id}'] = (
                                alpha * mean_priority +
                                (1 - alpha) * value_priority_tasks[f'running_mean_priority_task{task_id}']
                            )

                        # 计算归一化优先级
                        running_mean_priority = value_priority_tasks[f'running_mean_priority_task{task_id}']
                        normalized_priorities = (current_priorities - running_mean_priority) / (std_priority + 1e-6)

                        # 记录统计信息
                        if cfg.policy.print_task_priority_logs:
                            print(
                                f"任务 {task_id} - 优先级均值: {mean_priority:.8f}, "
                                f"运行均值优先级: {running_mean_priority:.8f}, "
                                f"标准差: {std_priority:.8f}"
                            )

        train_epoch += 1
        policy.recompute_pos_emb_diff_and_clear_cache()

        # 检查是否达到训练或环境步数的最大限制
        if all(collector.envstep >= max_env_step for collector in collectors) or learner.train_iter >= max_train_iter:
            break

    learner.call_hook('after_run')
    return policy