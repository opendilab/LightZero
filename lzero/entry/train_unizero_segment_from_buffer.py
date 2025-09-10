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
from ding.utils import EasyTimer
from ding.utils import set_pkg_seed, get_rank, get_world_size
from ding.worker import BaseLearner
from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from lzero.entry.utils import log_buffer_memory_usage
from lzero.policy import visit_count_temperature
from lzero.policy.random_policy import LightZeroRandomPolicy
from lzero.worker import MuZeroEvaluator as Evaluator
from lzero.worker import MuZeroSegmentCollector as Collector
from .utils import random_collect, calculate_update_per_collect
from lzero.mcts import GameSegment
import numpy as np
timer = EasyTimer()

def train_unizero_segment_from_buffer(
        input_cfg: Tuple[dict, dict],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
        expert_buffer_path: Optional[str] = None,  # <--- 新增参数
) -> 'Policy':
    """
    Overview:
        The train entry for UniZero (with muzero_segment_collector and buffer reanalyze trick), proposed in our paper UniZero: Generalized and Efficient Planning with Scalable Latent World Models.
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
    # collector_env.seed(cfg.seed, dynamic_seed=False)
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

    # TODO(pu)
    # ==============================================================
    # 开始: 【终极修复】从纯数据加载专家Game Buffer
    # ==============================================================
    # 1. 首先，创建一个空的UniZero Buffer实例
    # 这里的 GameBuffer 是在 train_unizero_segment 开头定义的 UniZeroGameBuffer
    replay_buffer = GameBuffer(policy_config)

    if expert_buffer_path and os.path.exists(expert_buffer_path):
        logging.info(f"正在从 {expert_buffer_path} 加载专家 Game Buffer 纯数据...")
        try:
            # import ipdb;ipdb.set_trace()
            # 加载保存的字典。使用 weights_only=True 更安全，因为我们知道里面没有可执行代码。
            # 如果这导致问题，可以暂时切换回 False，但 True 是推荐的做法。
            loaded_data = torch.load(expert_buffer_path, map_location=cfg.policy.device, weights_only=False)
            
            # 2. 从NumPy字典列表重建GameSegment对象列表
            reconstructed_game_segments = []
            for numpy_dict in loaded_data['game_segment_buffer']:
                # 创建一个空的GameSegment实例
                # 注意：如果GameSegment的__init__需要参数，需要相应提供
                game_segment = GameSegment(action_space=numpy_dict['action_space_size'], game_segment_length=len(numpy_dict['action_segment']), config=policy_config)
                
                # 遍历字典，用加载的数据填充实例
                for attr, value in numpy_dict.items():
                    if isinstance(value, np.ndarray):
                        # 将NumPy数组转换回PyTorch张量，并移动到正确的设备
                        # setattr(game_segment, attr, torch.from_numpy(value).to(cfg.policy.device))
                        # 将NumPy数组转换回PyTorch张量，并将其保留在CPU上
                        setattr(game_segment, attr, torch.from_numpy(value))
                    else:
                        setattr(game_segment, attr, value)
                reconstructed_game_segments.append(game_segment)

            # 3. 将重建好的数据填充到新的replay_buffer实例中
            replay_buffer.game_segment_buffer = reconstructed_game_segments
            replay_buffer.game_pos_priorities = loaded_data.get('game_pos_priorities', [])
            replay_buffer.game_segment_game_pos_look_up = loaded_data.get('game_segment_game_pos_look_up', [])
            replay_buffer.num_of_collected_episodes = loaded_data.get('num_of_collected_episodes', 0)
            replay_buffer.base_idx = loaded_data.get('base_idx', 0)
            replay_buffer.clear_time = loaded_data.get('clear_time', 0)

            logging.info(f"专家 Game Buffer 纯数据加载并重建成功。Buffer状态: {replay_buffer}")
            
            # 可选：进入离线学习模式
            if cfg.policy.get('offline_learn', False):
                cfg.policy.random_collect_episode_num = 0
                max_env_step = 0
                logging.info("已配置为离线学习模式，将禁用数据收集。")

        except Exception as e:
            logging.error(f"从 {expert_buffer_path} 加载专家Buffer数据失败。将使用新的空Buffer。错误: {e}", exc_info=True)
    else:
        if expert_buffer_path:
            logging.warning(f"提供的专家Buffer路径不存在: {expert_buffer_path}。将使用新的空Buffer。")
    # ==============================================================
    # 结束: 加载逻辑
    # ==============================================================

    collector = Collector(env=collector_env, policy=policy.collect_mode, tb_logger=tb_logger, exp_name=cfg.exp_name,
                          policy_config=policy_config)
    evaluator = Evaluator(eval_freq=cfg.policy.eval_freq, n_evaluator_episode=cfg.env.n_evaluator_episode,
                          stop_value=cfg.env.stop_value, env=evaluator_env, policy=policy.eval_mode,
                          tb_logger=tb_logger, exp_name=cfg.exp_name, policy_config=policy_config)
    
    # Learner's before_run hook
    learner.call_hook('before_run')

    if cfg.policy.use_wandb:
        policy.set_train_iter_env_step(learner.train_iter, collector.envstep)

    # Collect random data before training
    if cfg.policy.random_collect_episode_num > 0:
        random_collect(cfg.policy, policy, LightZeroRandomPolicy, collector, collector_env, replay_buffer)

    batch_size = policy._cfg.batch_size

    # TODO: for visualize
    # stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
    
    buffer_reanalyze_count = 0
    train_epoch = 0
    reanalyze_batch_size = cfg.policy.reanalyze_batch_size

    if cfg.policy.multi_gpu:
        # Get current world size and rank
        world_size = get_world_size()
        rank = get_rank()
    else:
        world_size = 1
        rank = 0

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
        update_per_collect = calculate_update_per_collect(cfg, new_data, world_size)

        # Update replay buffer
        # replay_buffer.push_game_segments(new_data)
        # replay_buffer.remove_oldest_data_to_fit()

        # Periodically reanalyze buffer
        if cfg.policy.buffer_reanalyze_freq >= 1:
            # Reanalyze buffer <buffer_reanalyze_freq> times in one train_epoch
            reanalyze_interval = update_per_collect // cfg.policy.buffer_reanalyze_freq
        else:
            # Reanalyze buffer each <1/buffer_reanalyze_freq> train_epoch
            if train_epoch > 0 and train_epoch % int(1/cfg.policy.buffer_reanalyze_freq) == 0 and replay_buffer.get_num_of_transitions()//cfg.policy.num_unroll_steps > int(reanalyze_batch_size/cfg.policy.reanalyze_partition):
                with timer:
                    # Each reanalyze process will reanalyze <reanalyze_batch_size> sequences (<cfg.policy.num_unroll_steps> transitions per sequence)
                    replay_buffer.reanalyze_buffer(reanalyze_batch_size, policy)
                buffer_reanalyze_count += 1
                logging.info(f'Buffer reanalyze count: {buffer_reanalyze_count}')
                logging.info(f'Buffer reanalyze time: {timer.value}')

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
                if cfg.policy.buffer_reanalyze_freq >= 1:
                    # Reanalyze buffer <buffer_reanalyze_freq> times in one train_epoch
                    if i % reanalyze_interval == 0 and replay_buffer.get_num_of_transitions()//cfg.policy.num_unroll_steps > int(reanalyze_batch_size/cfg.policy.reanalyze_partition):
                        with timer:
                            # Each reanalyze process will reanalyze <reanalyze_batch_size> sequences (<cfg.policy.num_unroll_steps> transitions per sequence)
                            replay_buffer.reanalyze_buffer(reanalyze_batch_size, policy)
                        buffer_reanalyze_count += 1
                        logging.info(f'Buffer reanalyze count: {buffer_reanalyze_count}')
                        logging.info(f'Buffer reanalyze time: {timer.value}')

                # import ipdb;ipdb.set_trace()
                train_data = replay_buffer.sample(batch_size, policy)
                if cfg.policy.use_wandb:
                    policy.set_train_iter_env_step(learner.train_iter, collector.envstep)

                train_data.append(learner.train_iter)
                log_vars = learner.train(train_data, collector.envstep)

                if cfg.policy.use_priority:
                    replay_buffer.update_priority(train_data, log_vars[0]['value_priority_orig'])

        train_epoch += 1
        policy.recompute_pos_emb_diff_and_clear_cache()

        # Check stopping criteria
        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break

    learner.call_hook('after_run')
    if cfg.policy.use_wandb:
        wandb.finish()
    return policy
