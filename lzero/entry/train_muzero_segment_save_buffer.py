import logging
import os
from functools import partial
from typing import Optional, Tuple, Dict, Any

import torch
from ding.config import compile_config
from ding.envs import create_env_manager
from ding.envs import get_vec_env_setting
from ding.policy import create_policy
from ding.rl_utils import get_epsilon_greedy_fn
from ding.utils import EasyTimer
from ding.utils import set_pkg_seed, get_rank
from ding.worker import BaseLearner
from tensorboardX import SummaryWriter

from lzero.entry.utils import log_buffer_memory_usage, log_buffer_run_time
from lzero.policy import visit_count_temperature
from lzero.policy.random_policy import LightZeroRandomPolicy
from lzero.worker import MuZeroEvaluator as Evaluator
from lzero.worker import MuZeroSegmentCollector as Collector
from .utils import random_collect, calculate_update_per_collect
from easydict import EasyDict 
timer = EasyTimer()

# # ==============================================================
# # 开始: 定义将GameSegment转换为纯NumPy字典的辅助函数
# # ==============================================================
# def convert_game_segment_to_numpy(game_segment: Any) -> Dict[str, Any]:
#     """
#     将一个GameSegment对象转换为一个只包含Python基本类型和NumPy数组的字典。
#     这移除了所有PyTorch张量，使其可以被安全地、跨版本地序列化。
#     """
#     numpy_dict = {}
#     for attr, value in game_segment.__dict__.items():
#         if isinstance(value, torch.Tensor):
#             # 将Tensor转换为CPU上的NumPy数组
#             numpy_dict[attr] = value.cpu().numpy()
#         elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
#             # 处理Tensor列表
#              numpy_dict[attr] = [v.cpu().numpy() for v in value]
#         else:
#             # 其他类型直接复制
#             numpy_dict[attr] = value
#     return numpy_dict
# # ==============================================================
# # 结束: 辅助函数
# # ==============================================================

# ==============================================================
# 开始: 定义终极的、无死角的递归数据净化函数
# ==============================================================
def deep_to_serializable(data: Any) -> Any:
    """
    递归地将一个复杂的数据结构转换为完全可序列化的格式。
    - torch.Tensor -> numpy.ndarray
    - easydict.EasyDict -> dict
    - 任何带有 __dict__ 属性的自定义对象 (如 GameSegment) -> dict
    - 递归处理 list, tuple, dict 的内容。
    """
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    
    if isinstance(data, (dict, EasyDict)):
        return {k: deep_to_serializable(v) for k, v in data.items()}

    if isinstance(data, (list, tuple)):
        return type(data)(deep_to_serializable(item) for item in data)

    # ==================== 这是新增的、最关键的处理逻辑 ====================
    # 检查它是否是一个自定义类的实例 (而不是基本类型且拥有 __dict__)
    if hasattr(data, '__dict__'):
        # 将对象转换为其属性字典，并对这个字典进行递归净化
        return deep_to_serializable(data.__dict__)
    # ====================================================================

    # 对于其他基本类型 (int, float, str, bool, None, numpy.ndarray)，直接返回
    return data

def train_muzero_segment_save_buffer(
        input_cfg: Tuple[dict, dict],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    Overview:
        The train entry for MCTS+RL algorithms (with muzero_segment_collector and buffer reanalyze trick), including MuZero, EfficientZero, Sampled MuZero, Sampled EfficientZero, Gumbel MuZero, StochasticMuZero.
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
    assert create_cfg.policy.type in ['efficientzero', 'muzero', 'muzero_context', 'muzero_rnn_full_obs', 'sampled_efficientzero', 'sampled_muzero', 'gumbel_muzero', 'stochastic_muzero'], \
        "train_muzero entry now only support the following algo.: 'efficientzero', 'muzero', 'sampled_efficientzero', 'gumbel_muzero', 'stochastic_muzero'"

    GameBuffer = None
    if create_cfg.policy.type in ['muzero', 'muzero_context', 'muzero_rnn_full_obs']:
        from lzero.mcts import MuZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'efficientzero':
        from lzero.mcts import EfficientZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'sampled_efficientzero':
        from lzero.mcts import SampledEfficientZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'sampled_muzero':
        from lzero.mcts import SampledMuZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'gumbel_muzero':
        from lzero.mcts import GumbelMuZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'stochastic_muzero':
        from lzero.mcts import StochasticMuZeroGameBuffer as GameBuffer

    if cfg.policy.cuda and torch.cuda.is_available():
        cfg.policy.device = 'cuda'
    else:
        cfg.policy.device = 'cpu'

    cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])

    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

    if cfg.policy.eval_offline:
        cfg.policy.learn.learner.hook.save_ckpt_after_iter = cfg.policy.eval_freq

    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])

    if model_path is not None:
        policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))

    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial')) if get_rank() == 0 else None
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)

    policy_config = cfg.policy
    batch_size = policy_config.batch_size
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

    learner.call_hook('before_run')

    if cfg.policy.update_per_collect is not None:
        update_per_collect = cfg.policy.update_per_collect

    if cfg.policy.random_collect_episode_num > 0:
        random_collect(cfg.policy, policy, LightZeroRandomPolicy, collector, collector_env, replay_buffer)
    if cfg.policy.eval_offline:
        eval_train_iter_list = []
        eval_train_envstep_list = []

    buffer_reanalyze_count = 0
    train_epoch = 0
    reanalyze_batch_size = cfg.policy.reanalyze_batch_size

    save_buffer_interval = 100000 # TODO: 100k
    # save_buffer_interval = 2

    last_save_iter = 0

    while True:
        log_buffer_memory_usage(learner.train_iter, replay_buffer, tb_logger)
        log_buffer_run_time(learner.train_iter, replay_buffer, tb_logger)
        collect_kwargs = {}
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

        if evaluator.should_eval(learner.train_iter):
            if cfg.policy.eval_offline:
                eval_train_iter_list.append(learner.train_iter)
                eval_train_envstep_list.append(collector.envstep)
            else:
                stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
                if stop:
                    break

        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        
        update_per_collect = calculate_update_per_collect(cfg, new_data)

        replay_buffer.push_game_segments(new_data)
        replay_buffer.remove_oldest_data_to_fit()

        if cfg.policy.buffer_reanalyze_freq >= 1:
            reanalyze_interval = update_per_collect // cfg.policy.buffer_reanalyze_freq
        else:
            if train_epoch > 0 and train_epoch % (1//cfg.policy.buffer_reanalyze_freq) == 0 and replay_buffer.get_num_of_transitions()//cfg.policy.num_unroll_steps > int(reanalyze_batch_size/cfg.policy.reanalyze_partition):
                with timer:
                    replay_buffer.reanalyze_buffer(reanalyze_batch_size, policy)
                buffer_reanalyze_count += 1
                logging.info(f'Buffer reanalyze count: {buffer_reanalyze_count}')
                logging.info(f'Buffer reanalyze time: {timer.value}')

        for i in range(update_per_collect):
            if cfg.policy.buffer_reanalyze_freq >= 1:
                if i > 0 and i % reanalyze_interval == 0 and replay_buffer.get_num_of_transitions() // cfg.policy.num_unroll_steps > int(
                        reanalyze_batch_size / cfg.policy.reanalyze_partition):
                    replay_buffer.reanalyze_buffer(reanalyze_batch_size, policy)
                    buffer_reanalyze_count += 1
                    logging.info(f'Buffer reanalyze count: {buffer_reanalyze_count}')

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

            log_vars = learner.train(train_data, collector.envstep)

            if cfg.policy.use_priority:
                replay_buffer.update_priority(train_data, log_vars[0]['value_priority_orig'])

        # ==============================================================
        # 开始: 【最终修复】使用递归净化函数保存100%纯数据
        # ==============================================================
        current_iter = learner.train_iter
        if current_iter // save_buffer_interval > last_save_iter // save_buffer_interval:
            current_milestone = (current_iter // save_buffer_interval) * save_buffer_interval
            
            buffer_save_dir = os.path.join(cfg.exp_name, 'game_buffers')
            os.makedirs(buffer_save_dir, exist_ok=True)
            file_path = os.path.join(buffer_save_dir, f'muzero_game_buffer_iter_{current_milestone}.pth')
            
            logging.info(f"达到训练迭代次数 {current_milestone}，正在深度净化并保存 Game Buffer...")
            
            try:
                # 1. 创建一个包含所有核心数据的原始字典
                # 注意：这里我们不再需要手动转换任何东西
                buffer_data_to_save_raw = {
                    'cfg': replay_buffer._cfg,
                    'game_segment_buffer': replay_buffer.game_segment_buffer,
                    'game_pos_priorities': replay_buffer.game_pos_priorities,
                    'game_segment_game_pos_look_up': replay_buffer.game_segment_game_pos_look_up,
                    'num_of_collected_episodes': replay_buffer.num_of_collected_episodes,
                    'base_idx': replay_buffer.base_idx,
                    'clear_time': replay_buffer.clear_time,
                }

                # 2. 使用我们的深度净化函数处理整个数据结构
                fully_serializable_data = deep_to_serializable(buffer_data_to_save_raw)
                
                # 3. 保存这个100%纯净的字典。
                torch.save(fully_serializable_data, file_path)
                logging.info(f"Game Buffer 纯数据已成功保存至: {file_path}")
                
            except Exception as e:
                logging.error(f"在迭代次数 {current_milestone} 保存 Game Buffer 纯数据失败。错误: {e}", exc_info=True)
            
            last_save_iter = current_iter
        # ==============================================================
        # 结束: 保存逻辑
        # ==============================================================

        train_epoch += 1

        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            if cfg.policy.eval_offline:
                logging.info(f'eval offline beginning...')
                ckpt_dirname = './{}/ckpt'.format(learner.exp_name)
                for train_iter, collector_envstep in zip(eval_train_iter_list, eval_train_envstep_list):
                    ckpt_name = 'iteration_{}.pth.tar'.format(train_iter)
                    ckpt_path = os.path.join(ckpt_dirname, ckpt_name)
                    policy.learn_mode.load_state_dict(torch.load(ckpt_path, map_location=cfg.policy.device))
                    stop, reward = evaluator.eval(learner.save_checkpoint, train_iter, collector_envstep)
                    logging.info(
                        f'eval offline at train_iter: {train_iter}, collector_envstep: {collector_envstep}, reward: {reward}')
                logging.info(f'eval offline finished!')
            break

    learner.call_hook('after_run')
    return policy