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
from .utils import random_collect, calculate_update_per_collect
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
        This function serves as the training entry point for UniZero, as proposed in our paper "UniZero: Generalized and Efficient Planning with Scalable Latent World Models".
        UniZero aims to enhance the planning capabilities of reinforcement learning agents by addressing the limitations found in MuZero-style algorithms,
        particularly in environments that require capturing long-term dependencies. More details can be found in https://arxiv.org/abs/2406.10667.
    
    Arguments:
        - input_cfg (:obj:`Tuple[dict, dict]`): Configuration in dictionary format.
            ``Tuple[dict, dict]`` indicates [user_config, create_cfg].
        - seed (:obj:`int`): Random seed for reproducibility.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of a PyTorch model.
        - model_path (:obj:`Optional[str]`): Path to the pretrained model, which should
            point to the checkpoint file of the pretrained model. An absolute path is recommended.
            In LightZero, the path typically resembles ``exp_name/ckpt/ckpt_best.pth.tar``.
        - max_train_iter (:obj:`Optional[int]`): Maximum number of policy update iterations during training.
        - max_env_step (:obj:`Optional[int]`): Maximum number of environment interaction steps to collect.
    
    Returns:
        - policy (:obj:`Policy`): The converged policy after training.
    """

    cfg, create_cfg = input_cfg

    # Ensure the specified policy type is supported
    assert create_cfg.policy.type in ['unizero', 'sampled_unizero'], "train_unizero only supports the following algorithms: 'unizero', 'sampled_unizero'"
    logging.info(f"Using policy type: {create_cfg.policy.type}")

    # Import the appropriate GameBuffer class based on the policy type
    game_buffer_classes = {'unizero': 'UniZeroGameBuffer', 'sampled_unizero': 'SampledUniZeroGameBuffer'}
    GameBuffer = getattr(__import__('lzero.mcts', fromlist=[game_buffer_classes[create_cfg.policy.type]]),
                         game_buffer_classes[create_cfg.policy.type])

    # Check for GPU availability and set the device accordingly
    cfg.policy.device = cfg.policy.model.world_model_cfg.device if torch.cuda.is_available() else 'cpu'
    logging.info(f"Device set to: {cfg.policy.device}")

    # Compile the configuration file
    cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)

    # Create environment manager
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])

    # Initialize environment and random seed
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=torch.cuda.is_available())

    # Initialize wandb if specified
    if cfg.policy.use_wandb:
        logging.info("Initializing wandb...")
        wandb.init(
            project="LightZero",
            config=cfg,
            sync_tensorboard=False,
            monitor_gym=False,
            save_code=True,
        )
        logging.info("wandb initialization completed!")

    # Create policy
    logging.info("Creating policy...")
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])
    logging.info("Policy created successfully!")

    # Verify we're using the correct LightZero path
    import lzero
    print(f"\n{'='*80}")
    print(f"VERIFICATION: Using LightZero from: {lzero.__file__}")
    print(f"Expected path: /mnt/shared-storage-user/puyuan/code/LightZero")
    print(f"{'='*80}\n")

    print(f"debug 0303!!!!!!!!!"*20)

    # Load pretrained model if specified
    if model_path is not None:
        logging.info(f"Loading pretrained model from {model_path}...")
        policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))
        logging.info("Pretrained model loaded successfully!")
                
        # 1. 加载原始 state_dict
        # checkpoint = torch.load(model_path, map_location=cfg.policy.device)
        
        # import ipdb;ipdb.set_trace()

        # # 2. 定义清洗函数：专门去除 _orig_mod. 和 module. 前缀
        # def remove_compile_prefix(state_dict):
        #     new_sd = {}
        #     for k, v in state_dict.items():
        #         # 去除 torch.compile 产生的 _orig_mod. 前缀
        #         new_k = k.replace('_orig_mod.', '')
        #         # 去除 DDP 产生的 module. 前缀
        #         new_k = new_k.replace('module.', '')
        #         new_sd[new_k] = v
        #     return new_sd

        # # 3. 构建符合 _load_state_dict_learn 要求的结构
        # # 它需要 {'model': ..., 'target_model': ...}
        # new_state_dict = {}
        
        # # 处理 'model' 部分
        # if 'model' in checkpoint:
        #     logging.info("Processing 'model' keys in checkpoint...")
        #     new_state_dict['model'] = remove_compile_prefix(checkpoint['model'])
        # else:
        #     # 如果 checkpoint 本身就是扁平的 state_dict (兼容旧格式)，则视为 model
        #     # 但根据你的描述，应该是有 model key 的
        #     logging.warning("Key 'model' not found in checkpoint, assuming flat dict.")
        #     new_state_dict['model'] = remove_compile_prefix(checkpoint)

        # # 处理 'target_model' 部分
        # if 'target_model' in checkpoint:
        #     logging.info("Processing 'target_model' keys in checkpoint...")
        #     new_state_dict['target_model'] = remove_compile_prefix(checkpoint['target_model'])
        # elif 'model' in new_state_dict:
        #     # 如果没有 target_model，通常可以复用 model 的权重
        #     logging.info("Key 'target_model' not found, copying from 'model'.")
        #     new_state_dict['target_model'] = new_state_dict['model']

        # # 4. 调用 Policy 的加载方法
        # # 此时传入的 new_state_dict 结构正确，且内部 key 已清洗
        # try:
        #     # policy.learn_mode.load_state_dict(new_state_dict)
        #     policy.learn_mode.load_state_dict(checkpoint)

        #     logging.info("Pretrained model loaded successfully (Prefixes cleaned)!")
        # except RuntimeError as e:
        #     logging.error(f"Strict loading failed! Please check model definition vs checkpoint. Error: {e}")
        #     raise e

       
        # # Verify that weights were actually loaded by checking a sample weight
        # try:
        #     sample_weight = policy._learn_model._orig_mod.representation_network.pretrained_model.embeddings.word_embeddings.weight
        #     logging.info(f"Loaded weight verification - mean: {sample_weight.mean().item():.6f}, std: {sample_weight.std().item():.6f}")
        # except Exception as e:
        #     logging.warning(f"Could not verify loaded weights: {e}")

        # from .analyze_model import analyze_model_structure
        # analyze_model_structure(policy._learn_model._orig_mod)

    # Create core components for training
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial')) if get_rank() == 0 else None
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    replay_buffer = GameBuffer(cfg.policy)
    collector = Collector(env=collector_env, policy=policy.collect_mode, tb_logger=tb_logger, exp_name=cfg.exp_name,
                          policy_config=cfg.policy)
    evaluator = Evaluator(eval_freq=cfg.policy.eval_freq, n_evaluator_episode=cfg.env.n_evaluator_episode,
                          stop_value=cfg.env.stop_value, env=evaluator_env, policy=policy.eval_mode,
                          tb_logger=tb_logger, exp_name=cfg.exp_name, policy_config=cfg.policy)

    # Execute the learner's before_run hook
    learner.call_hook('before_run')

    if cfg.policy.use_wandb:
        policy.set_train_iter_env_step(learner.train_iter, collector.envstep)

    # Randomly collect data if specified
    if cfg.policy.random_collect_episode_num > 0:
        logging.info("Collecting random data...")
        random_collect(cfg.policy, policy, LightZeroRandomPolicy, collector, collector_env, replay_buffer)
        logging.info("Random data collection completed!")

    batch_size = policy._cfg.batch_size

    if cfg.policy.multi_gpu:
        # Get current world size and rank
        world_size = get_world_size()
        rank = get_rank()
    else:
        world_size = 1
        rank = 0
    # TODO: for visualize
    # stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
    # import sys; sys.exit(0)

    while True:
        # Log memory usage of the replay buffer
        log_buffer_memory_usage(learner.train_iter, replay_buffer, tb_logger)

        # Set temperature parameter for data collection
        collect_kwargs = {
            'temperature': visit_count_temperature(
                cfg.policy.manual_temperature_decay,
                cfg.policy.fixed_temperature_value,
                cfg.policy.threshold_training_steps_for_final_temperature,
                trained_steps=learner.train_iter
            ),
            'epsilon': 0.0  # Default epsilon value
        }

        # Configure epsilon-greedy exploration
        if cfg.policy.eps.eps_greedy_exploration_in_collect:
            epsilon_greedy_fn = get_epsilon_greedy_fn(
                start=cfg.policy.eps.start,
                end=cfg.policy.eps.end,
                decay=cfg.policy.eps.decay,
                type_=cfg.policy.eps.type
            )
            collect_kwargs['epsilon'] = epsilon_greedy_fn(collector.envstep)

        # Evaluate policy performance
        if learner.train_iter == 0 or evaluator.should_eval(learner.train_iter):
            logging.info(f"Training iteration {learner.train_iter}: Starting evaluation...")
            _ = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)

        # Collect new data
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        logging.info(f"Rank {rank}, Training iteration {learner.train_iter}: New data collection completed!")

        # Determine updates per collection
        update_per_collect = cfg.policy.update_per_collect
        if update_per_collect is None:
            update_per_collect = calculate_update_per_collect(cfg, new_data, world_size)

        # Update replay buffer
        replay_buffer.push_game_segments(new_data)
        replay_buffer.remove_oldest_data_to_fit()

        if world_size > 1:
            # Synchronize all ranks before training
            try:
                dist.barrier()
            except Exception as e:
                logging.error(f'Rank {rank}: Synchronization barrier failed, error: {e}')
                break

        # Check if there is sufficient data for training
        if collector.envstep > cfg.policy.train_start_after_envsteps:
            if cfg.policy.sample_type == 'episode':
                data_sufficient = replay_buffer.get_num_of_game_segments() > batch_size
            else:
                data_sufficient = replay_buffer.get_num_of_transitions() > batch_size
            
            if not data_sufficient:
                logging.warning(
                    f'Rank {rank}: The data in replay_buffer is not sufficient to sample a mini-batch: '
                    f'batch_size: {batch_size}, replay_buffer: {replay_buffer}. Continue to collect now ....'
                )
                continue

            # Execute multiple training rounds
            for i in range(update_per_collect):
                train_data = replay_buffer.sample(batch_size, policy)
                if replay_buffer._cfg.reanalyze_ratio > 0 and i % 20 == 0:
                    policy.recompute_pos_emb_diff_and_clear_cache()
                
                if cfg.policy.use_wandb:
                    policy.set_train_iter_env_step(learner.train_iter, collector.envstep)

                train_data.append(learner.train_iter)

                log_vars = learner.train(train_data, collector.envstep)
                if cfg.policy.use_priority:
                    replay_buffer.update_priority(train_data, log_vars[0]['value_priority_orig'])

        policy.recompute_pos_emb_diff_and_clear_cache()

        # Check stopping criteria
        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            logging.info("Stopping condition met, training ends!")
            break

    learner.call_hook('after_run')
    if cfg.policy.use_wandb:
        wandb.finish()
    logging.info("===== Training Completed =====")
    return policy