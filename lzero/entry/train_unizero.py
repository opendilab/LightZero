import logging
import os
from functools import partial
from typing import Tuple, Optional

import torch
import wandb
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
import torch.distributed as dist
from ding.utils import set_pkg_seed, get_rank, get_world_size


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

    logging.info("===== 开始训练 UniZero =====")
    
    # 检查是否支持指定的 policy 类型
    assert create_cfg.policy.type in ['unizero', 'sampled_unizero'], "train_unizero 仅支持以下算法: 'unizero', 'sampled_unizero'"
    logging.info(f"使用的 policy 类型为: {create_cfg.policy.type}")

    # 根据 policy 类型导入对应的 GameBuffer 类
    game_buffer_classes = {'unizero': 'UniZeroGameBuffer', 'sampled_unizero': 'SampledUniZeroGameBuffer'}
    GameBuffer = getattr(__import__('lzero.mcts', fromlist=[game_buffer_classes[create_cfg.policy.type]]),
                         game_buffer_classes[create_cfg.policy.type])

    # 检查是否有 GPU 可用，设置设备
    cfg.policy.device = cfg.policy.model.world_model_cfg.device if torch.cuda.is_available() else 'cpu'
    logging.info(f"设备已设置为: {cfg.policy.device}")

    # 编译配置文件
    logging.info("正在编译配置文件...")
    cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
    logging.info("配置文件编译完成！")

    # 创建环境管理器
    logging.info("正在创建环境管理器...")
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    logging.info("环境管理器创建完成！")

    # 环境和随机种子初始化
    logging.info("正在初始化环境和随机种子...")
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=torch.cuda.is_available())
    logging.info("环境和随机种子初始化完成！")

    # 如果使用 wandb，初始化 wandb
    if cfg.policy.use_wandb:
        logging.info("正在初始化 wandb...")
        wandb.init(
            project="LightZero",
            config=cfg,
            sync_tensorboard=False,
            monitor_gym=False,
            save_code=True,
        )
        logging.info("wandb 初始化完成！")

    # 创建 policy
    logging.info("正在创建 policy...")
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])
    logging.info("policy 创建完成！")

    # 如果指定了模型路径，加载预训练模型
    if model_path is not None:
        logging.info(f"正在从 {model_path} 加载预训练模型...")
        policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))
        logging.info("预训练模型加载完成！")

    # 创建训练的核心组件
    logging.info("正在创建训练的核心组件...")
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial')) if get_rank() == 0 else None
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    replay_buffer = GameBuffer(cfg.policy)
    collector = Collector(env=collector_env, policy=policy.collect_mode, tb_logger=tb_logger, exp_name=cfg.exp_name,
                          policy_config=cfg.policy)
    evaluator = Evaluator(eval_freq=cfg.policy.eval_freq, n_evaluator_episode=cfg.env.n_evaluator_episode,
                          stop_value=cfg.env.stop_value, env=evaluator_env, policy=policy.eval_mode,
                          tb_logger=tb_logger, exp_name=cfg.exp_name, policy_config=cfg.policy)
    logging.info("训练核心组件创建完成！")

    # Learner 的前置 hook
    logging.info("正在执行 Learner 的 before_run hook...")
    learner.call_hook('before_run')
    logging.info("Learner 的 before_run hook 执行完成！")

    if cfg.policy.use_wandb:
        policy.set_train_iter_env_step(learner.train_iter, collector.envstep)

    # 随机收集数据
    if cfg.policy.random_collect_episode_num > 0:
        logging.info("正在进行随机数据收集...")
        random_collect(cfg.policy, policy, LightZeroRandomPolicy, collector, collector_env, replay_buffer)
        logging.info("随机数据收集完成！")

    batch_size = policy._cfg.batch_size

    if cfg.policy.multi_gpu:
        # 获取当前的 world_size 和 rank
        world_size = get_world_size()
        rank = get_rank()
    else:
        world_size = 1
        rank = 0

    while True:
        # torch.cuda.empty_cache()

        # 记录 replay buffer 的内存使用情况
        # logging.info(f"训练迭代 {learner.train_iter}: 正在记录 replay buffer 的内存使用情况...")
        log_buffer_memory_usage(learner.train_iter, replay_buffer, tb_logger)
        # logging.info(f"训练迭代 {learner.train_iter}: 内存使用记录完成！")

        # 设置温度参数
        collect_kwargs = {
            'temperature': visit_count_temperature(
                cfg.policy.manual_temperature_decay,
                cfg.policy.fixed_temperature_value,
                cfg.policy.threshold_training_steps_for_final_temperature,
                trained_steps=learner.train_iter
            ),
            'epsilon': 0.0  # 默认 epsilon 值
        }
        # logging.info(f"训练迭代 {learner.train_iter}: 温度设置完成，值为 {collect_kwargs['temperature']}")

        # 配置 epsilon-greedy 探索
        if cfg.policy.eps.eps_greedy_exploration_in_collect:
            epsilon_greedy_fn = get_epsilon_greedy_fn(
                start=cfg.policy.eps.start,
                end=cfg.policy.eps.end,
                decay=cfg.policy.eps.decay,
                type_=cfg.policy.eps.type
            )
            collect_kwargs['epsilon'] = epsilon_greedy_fn(collector.envstep)
            # logging.info(f"训练迭代 {learner.train_iter}: epsilon 设置完成，值为 {collect_kwargs['epsilon']}")

        # 评估 policy 的表现
        if evaluator.should_eval(learner.train_iter):
            logging.info(f"训练迭代 {learner.train_iter}: 开始评估...")
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            logging.info(f"训练迭代 {learner.train_iter}: 评估完成，是否停止: {stop}, 当前奖励: {reward}")
            if stop:
                logging.info("满足停止条件，训练结束！")
                break

        # 收集新数据
        # logging.info(f"Rank {rank}, 训练迭代 {learner.train_iter}: 开始收集新数据...")
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        logging.info(f"Rank {rank}, 训练迭代 {learner.train_iter}: 新数据收集完成！")

        # Determine updates per collection
        update_per_collect = cfg.policy.update_per_collect
        if update_per_collect is None:
            # update_per_collect is None, then update_per_collect is set to the number of collected transitions multiplied by the replay_ratio.
            # The length of game_segment (i.e., len(game_segment.action_segment)) can be smaller than cfg.policy.game_segment_length if it represents the final segment of the game.
            # On the other hand, its length will be less than cfg.policy.game_segment_length + padding_length when it is not the last game segment. Typically, padding_length is the sum of unroll_steps and td_steps.
            collected_transitions_num = sum(min(len(game_segment), cfg.policy.game_segment_length) for game_segment in new_data[0])
            update_per_collect = int(collected_transitions_num * cfg.policy.replay_ratio)

        # 更新 replay buffer
        # logging.info(f"训练迭代 {learner.train_iter}: 开始更新 replay buffer...")
        replay_buffer.push_game_segments(new_data)
        replay_buffer.remove_oldest_data_to_fit()
        # logging.info(f"训练迭代 {learner.train_iter}: replay buffer 更新完成！")

        if  world_size > 1:
        # 同步训练前所有rank的准备状态
            try:
                dist.barrier()
                # logging.info(f'Rank {rank}: 通过训练前的同步障碍')
            except Exception as e:
                logging.error(f'Rank {rank}: 同步障碍失败，错误: {e}')
                break

        # 检查是否有足够数据进行训练
        if collector.envstep > cfg.policy.train_start_after_envsteps:
            if cfg.policy.sample_type == 'episode':
                data_sufficient = replay_buffer.get_num_of_game_segments() > batch_size
            else:
                data_sufficient = replay_buffer.get_num_of_transitions() > batch_size
            
            if not data_sufficient:
                # NOTE: 注意ddp训练时，不同rank可能有的replay buffer 数据不足，导致有的没有进入训练阶段，从而通信超时，需要确保同时进入训练阶段
                logging.warning(f"Rank {rank}: 训练迭代 {learner.train_iter}: replay buffer 数据不足，继续收集数据...")
                continue

            logging.info(f"Rank {rank}, 训练迭代 {learner.train_iter}: 开始训练！")

            # 执行多轮训练
            for i in range(update_per_collect):
                train_data = replay_buffer.sample(batch_size, policy)
                if cfg.policy.reanalyze_ratio > 0 and i % 20 == 0:
                    policy.recompute_pos_emb_diff_and_clear_cache()
                
                if cfg.policy.use_wandb:
                    policy.set_train_iter_env_step(learner.train_iter, collector.envstep)

                # train_data.append({'train_which_component': 'transformer'})
                train_data.append(learner.train_iter)

                log_vars = learner.train(train_data, collector.envstep)
                if cfg.policy.use_priority:
                    replay_buffer.update_priority(train_data, log_vars[0]['value_priority_orig'])
            
            logging.info(f"Rank {rank}, 训练迭代 {learner.train_iter}: 训练完成！")

        policy.recompute_pos_emb_diff_and_clear_cache()

        # 检查停止条件
        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            logging.info("满足停止条件，训练结束！")
            break

    learner.call_hook('after_run')
    if cfg.policy.use_wandb:
        wandb.finish()
    logging.info("===== 训练完成 =====")
    return policy