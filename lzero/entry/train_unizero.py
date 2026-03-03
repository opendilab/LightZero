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

        # policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))
        # policy._learn_model.world_model.precompute_pos_emb_diff_kv()
        # policy.recompute_pos_emb_diff_and_clear_cache()

        checkpoint = torch.load(model_path, map_location=cfg.policy.device)
        # ===== 添加调试代码 - 检查加载前后的状态 =====
        print("\n" + "="*80)
        print("=== 加载前检查 ===")
        print("="*80)
        ckpt_sample_keys = list(checkpoint['model'].keys())[:3]
        model_sample_keys = list(policy._learn_model.state_dict().keys())[:3]
        print(f"Checkpoint keys示例: {ckpt_sample_keys}")
        print(f"Model keys示例: {model_sample_keys}")

        # 检查前缀是否匹配
        ckpt_has_prefix = any('_orig_mod.' in k for k in checkpoint['model'].keys())
        model_has_prefix = any('_orig_mod.' in k for k in policy._learn_model.state_dict().keys())
        print(f"Checkpoint有_orig_mod.前缀: {ckpt_has_prefix}")
        print(f"Model有_orig_mod.前缀: {model_has_prefix}")

        if ckpt_has_prefix != model_has_prefix:
            print("⚠️ WARNING: 前缀不匹配！可能导致加载失败！")
        else:
            print("✓ 前缀匹配")

        # 加载前记录head权重
        head_before = policy._learn_model.world_model.head_policy.head_module[4].weight.clone()
        print(f"加载前head_policy权yiyi重统计: mean={head_before.mean():.6f}, std={head_before.std():.6f}")
        print("="*80 + "\n")
        # ===== 调试代码结束 =====

        # 🔥🔥🔥 关键修复：兼容旧格式的checkpoint
        # Jericho的checkpoint用的是'optimizer_world_model'而不是'optimizer'
        if 'optimizer' not in checkpoint and 'optimizer_world_model' in checkpoint:
            print("⚠️ 检测到旧格式checkpoint，将'optimizer_world_model'映射为'optimizer'")
            checkpoint['optimizer'] = checkpoint['optimizer_world_model']

        try:
            policy.learn_mode.load_state_dict(checkpoint)
        except Exception as e:
            print(f"\n❌ 加载checkpoint时出错: {e}")
            print(f"   异常类型: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            raise  # re-raise以便调试

        # ===== 添加加载后检查 =====
        print("\n" + "="*80)
        print("=== 加载后检查 ===")
        print("="*80)
        head_after = policy._learn_model.world_model.head_policy.head_module[4].weight
        print(f"加载后head_policy权重统计: mean={head_after.mean():.6f}, std={head_after.std():.6f}")

        if torch.equal(head_before, head_after):
            print("❌ 致命错误：加载失败！权重没有改变！")
            print("   可能原因：")
            print("   1. 前缀不匹配导致key无法匹配")
            print("   2. strict=True导致加载被拒绝")
            print("   3. 需要启用前缀清洗代码")
        else:
            print("✓ 加载成功！权重已更新")
            print(f"  权重变化: Δmean={abs(head_after.mean() - head_before.mean()):.6f}")
        print("="*80 + "\n")
        # ===== 检查结束 =====

        # ===== 🔥 关键修复：重新计算position embedding缓存 =====
        print("\n" + "="*80)
        print("=== 重新计算Position Embedding缓存 ===")
        print("="*80)
        try:
            if hasattr(policy._learn_model, 'world_model') and hasattr(policy._learn_model.world_model, 'precompute_pos_emb_diff_kv'):
                logging.info("Recomputing position embedding differences after loading checkpoint...")
                
                policy._learn_model.world_model.precompute_pos_emb_diff_kv()
                policy.recompute_pos_emb_diff_and_clear_cache()

                print("✅ Position embedding缓存已重新计算")
                print("   这确保了加载的pos_emb权重与缓存一致")
            else:
                print("⊘ 未找到precompute_pos_emb_diff_kv方法，跳过")
        except Exception as e:
            print(f"⚠️ 重新计算position embedding时出错: {e}")
            logging.warning(f"Failed to recompute position embeddings: {e}")
        print("="*80 + "\n")
        # ===== 修复结束 =====


        # ============
        # 🔥🔥🔥 新增：诊断torch.compile和对象关系
        print("\n" + "="*80)
        print("=== torch.compile和对象关系诊断 ===")
        print("="*80)
        print(f"_learn_model is _eval_model: {policy._learn_model is policy._eval_model}")
        print(f"_learn_model is _model: {policy._learn_model is policy._model}")
        print(f"_eval_model is _model: {policy._eval_model is policy._model}")
        print(f"_learn_model type: {type(policy._learn_model).__name__}")
        print(f"_eval_model type: {type(policy._eval_model).__name__}")

        # 检查world_model
        print(f"\nworld_model对象:")
        print(f"  _learn_model.world_model is _eval_model.world_model: {policy._learn_model.world_model is policy._eval_model.world_model}")

        # 检查position embedding
        print(f"\nPosition Embedding:")
        print(f"  rotary_emb配置: {cfg.policy.model.world_model_cfg.rotary_emb}")
        pos_emb_weight_learn = policy._learn_model.world_model.pos_emb.weight
        pos_emb_weight_eval = policy._eval_model.world_model.pos_emb.weight
        print(f"  _learn pos_emb.weight mean: {pos_emb_weight_learn.mean():.6f}")
        print(f"  _eval pos_emb.weight mean: {pos_emb_weight_eval.mean():.6f}")
        print(f"  是否相同对象: {pos_emb_weight_learn.data_ptr() == pos_emb_weight_eval.data_ptr()}")

        # 检查precompute cache
        print(f"\nPrecompute缓存:")
        if hasattr(policy._learn_model.world_model, 'positional_embedding_k'):
            print(f"  _learn_model有positional_embedding_k缓存")
            print(f"    cache[0] mean: {policy._learn_model.world_model.positional_embedding_k[0].mean():.6f}")
        else:
            print(f"  ❌ _learn_model没有positional_embedding_k缓存！")

        if hasattr(policy._eval_model.world_model, 'positional_embedding_k'):
            print(f"  _eval_model有positional_embedding_k缓存")
            print(f"    cache[0] mean: {policy._eval_model.world_model.positional_embedding_k[0].mean():.6f}")
        else:
            print(f"  ❌ _eval_model没有positional_embedding_k缓存！")

        # 检查last_batch_obs_eval
        print(f"\nlast_batch状态:")
        print(f"  hasattr last_batch_obs_eval: {hasattr(policy, 'last_batch_obs_eval')}")
        if hasattr(policy, 'last_batch_obs_eval'):
            print(f"    shape: {policy.last_batch_obs_eval.shape}")
            print(f"    mean: {policy.last_batch_obs_eval.float().mean():.6f}")
        else:
            print(f"  ❌ last_batch_obs_eval不存在！这是个严重问题！")

        print("="*80 + "\n")
        # 🔥 诊断结束

        logging.info("Pretrained model loaded successfully!")
                
       

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
        
        # TODO
        return None

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