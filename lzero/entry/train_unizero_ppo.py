import logging
import os
from functools import partial
from typing import Optional, Tuple, List, Dict, Any

import torch
import wandb
import numpy as np
from ding.config import compile_config
from ding.envs import create_env_manager
from ding.envs import get_vec_env_setting
from ding.policy import create_policy
from ding.utils import get_rank, get_world_size, set_pkg_seed
from torch.utils.tensorboard import SummaryWriter
from ding.worker import BaseLearner
import torch.distributed as dist

from lzero.worker.muzero_evaluator_ppo import MuZeroEvaluatorPPO as Evaluator
from lzero.worker.muzero_collector_ppo import MuZeroCollectorPPO


def train_unizero_ppo(
        input_cfg: Tuple[dict, dict],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: int = int(1e10),
) -> None:
    cfg, create_cfg = input_cfg
    assert create_cfg.policy.type == 'unizero_ppo', "train_unizero_ppo expects policy type 'unizero_ppo'"
    logging.info(f"Using policy type: {create_cfg.policy.type}")

    cfg.policy.device = cfg.policy.model.world_model_cfg.device if torch.cuda.is_available() else 'cpu'
    logging.info(f"Device set to: {cfg.policy.device}")

    cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)

    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])

    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=torch.cuda.is_available())

    rank = get_rank()

    if cfg.policy.use_wandb:
        wandb.init(
            project="LightZero",
            config=cfg,
            sync_tensorboard=False,
            monitor_gym=False,
            save_code=True,
        )

    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])
    logging.info("Policy created successfully!")

    # Load pretrained model if specified
    if model_path is not None:
        logging.info(f"Loading pretrained model from {model_path}...")
        policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))
        logging.info("Pretrained model loaded successfully!")

    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial')) if rank == 0 else None
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = MuZeroCollectorPPO(
        env=collector_env,
        policy=policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        policy_config=cfg.policy,
    )
    evaluator = Evaluator(
        eval_freq=cfg.policy.eval_freq,
        n_evaluator_episode=cfg.env.n_evaluator_episode,
        stop_value=cfg.env.stop_value,
        env=evaluator_env,
        policy=policy.eval_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        policy_config=cfg.policy,
    )

    learner.call_hook('before_run')
    if cfg.policy.use_wandb:
        policy.set_train_iter_env_step(learner.train_iter, collector.envstep)

    if cfg.policy.multi_gpu:
        world_size = get_world_size()
    else:
        world_size = 1

    transition_buffer: List[Dict[str, Any]] = []

    while True:
        # eval_stop = False
        # if (learner.train_iter == 0 or evaluator.should_eval(learner.train_iter)) and rank == 0:
        #     logging.info(f"Training iteration {learner.train_iter}: Starting evaluation...")
        #     eval_stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
        #     logging.info(f"Training iteration {learner.train_iter}: Evaluation completed, stop condition: {eval_stop}, current reward: {reward}")
        # if cfg.policy.multi_gpu and world_size > 1:
        #     stop_tensor = torch.tensor([int(eval_stop)], device=cfg.policy.device if torch.cuda.is_available() else torch.device('cpu'))
        #     dist.broadcast(stop_tensor, src=0)
        #     eval_stop = bool(stop_tensor.item())
        # if eval_stop:
        #     logging.info("Stopping condition met, training ends!")
        #     break

        collect_kwargs = dict(temperature=1.0, epsilon=0.0)
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        logging.info(f"Rank {rank}, Training iteration {learner.train_iter}: New data collection completed!")

        transitions = new_data[0]
        if transitions:
            transition_buffer.extend(transitions)

        if len(transition_buffer) < cfg.policy.ppo.mini_batch_size:
            continue

        if cfg.policy.ppo.get('advantage_normalization', True):
            advantages = np.stack([item['advantage'] for item in transition_buffer])
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            for item in transition_buffer:
                item['advantage'] = (item['advantage'] - adv_mean) / adv_std

        total_transitions = len(transition_buffer)
        mini_batch_size = cfg.policy.ppo.mini_batch_size
        for _ in range(cfg.policy.ppo.update_epochs):
            permutation = np.random.permutation(total_transitions)
            for start in range(0, total_transitions, mini_batch_size):
                batch_indices = permutation[start:start + mini_batch_size]
                if batch_indices.size == 0:
                    continue

                def stack(key: str) -> np.ndarray:
                    return np.stack([transition_buffer[i][key] for i in batch_indices])

                batch_dict = dict(
                    prev_obs=stack('prev_obs'),
                    obs=stack('obs'),
                    action_mask=stack('action_mask'),
                    action=stack('action'),
                    old_log_prob=stack('old_log_prob'),
                    advantage=stack('advantage'),
                    return_=stack('return'),
                    prev_action=stack('prev_action'),
                    timestep=stack('timestep'),
                )
                train_data = [batch_dict, None]
                train_data.append(learner.train_iter)
                learner.train(train_data, collector.envstep)

        transition_buffer.clear()

        if cfg.policy.multi_gpu and world_size > 1:
            try:
                dist.barrier()
            except Exception as e:
                logging.error(f'Rank {rank}: Synchronization barrier failed, error: {e}')
                break

        if cfg.policy.use_wandb:
            policy.set_train_iter_env_step(learner.train_iter, collector.envstep)

        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            logging.info("Reached max training condition")
            break

    learner.call_hook('after_run')
    collector.close()
    evaluator.close()
    if tb_logger is not None:
        tb_logger.close()
    if cfg.policy.use_wandb:
        wandb.finish()
