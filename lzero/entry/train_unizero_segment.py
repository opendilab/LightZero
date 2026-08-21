import os
import re
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import torch
import wandb
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.rl_utils import get_epsilon_greedy_fn
from ding.utils import EasyTimer, get_rank, get_world_size, set_pkg_seed
from ding.worker import BaseLearner
from ditk import logging
from lzero.entry.utils import log_buffer_memory_usage
from lzero.policy import visit_count_temperature
from lzero.policy.random_policy import LightZeroRandomPolicy
from lzero.worker import MuZeroEvaluator as Evaluator
from lzero.worker import MuZeroSegmentCollector as Collector
from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from .utils import calculate_update_per_collect, random_collect

timer = EasyTimer()

_PERIODIC_CHECKPOINT_PATTERN = re.compile(r'^iteration_(\d+)\.pth\.tar$')


def _restore_resume_counters(learner, collector, train_iter: int, envstep: int) -> None:
    """Restore both owners of the counters persisted in learner checkpoints.

    The collector owns the value used by the serial loop and evaluator, whereas
    ``BaseLearner`` owns the value written to checkpoints.  If only the collector is
    restored, a best checkpoint saved during the first post-resume evaluation records
    ``last_step=0`` and a second preemption silently resets envstep-based schedules.
    """
    if train_iter > 0:
        # BaseLearner.train_iter is a read-only property backed by the CountVar `_last_iter`.
        learner._last_iter.update(train_iter)
    if envstep > 0:
        collector._total_envstep_count = envstep
        learner.collector_envstep = envstep


def _required_replay_transitions(
        resume_train_iter: int, batch_size: int, resume_buffer_min_transitions: int
) -> int:
    """Return the replay population required before learning (re)starts.

    Learner checkpoints do not contain the in-memory game buffer. Updating a mature resumed model as
    soon as a single batch is available makes the first gradients come from an extremely narrow set of
    new trajectories. Fresh runs retain the historical one-batch threshold; resumed runs can request
    a short policy-collection warmup.
    """
    if batch_size <= 0:
        raise ValueError(f'batch_size must be positive, got {batch_size}')
    if resume_buffer_min_transitions < 0:
        raise ValueError(
            f'resume_buffer_min_transitions must be non-negative, got {resume_buffer_min_transitions}'
        )
    one_full_batch = batch_size + 1  # Preserve the existing strict ``> batch_size`` condition.
    if resume_train_iter <= 0:
        return one_full_batch
    return max(one_full_batch, resume_buffer_min_transitions)


def _resolve_segment_reanalyze_settings(policy_config) -> Tuple[float, int, float]:
    """Resolve segment-only reanalysis settings for minimal and legacy configs."""
    buffer_reanalyze_freq = float(
        getattr(policy_config, 'buffer_reanalyze_freq', 1 / 100000)
    )
    reanalyze_batch_size = int(getattr(policy_config, 'reanalyze_batch_size', 160))
    reanalyze_partition = float(getattr(policy_config, 'reanalyze_partition', 0.75))

    if buffer_reanalyze_freq <= 0:
        raise ValueError(
            f'buffer_reanalyze_freq must be positive, got {buffer_reanalyze_freq}'
        )
    if reanalyze_batch_size <= 0:
        raise ValueError(
            f'reanalyze_batch_size must be positive, got {reanalyze_batch_size}'
        )
    if not 0 < reanalyze_partition <= 1:
        raise ValueError(
            f'reanalyze_partition must be in (0, 1], got {reanalyze_partition}'
        )
    return buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition


def _prune_periodic_checkpoints(exp_name: str, keep_last: int) -> list:
    """Bound periodic checkpoint storage without touching evaluator best checkpoints.

    ``iteration_0`` is retained as the reproducible initialization anchor. In addition, the newest
    ``keep_last`` positive-iteration checkpoints are retained for preemption recovery. Files such as
    ``ckpt_best.pth.tar`` and names that do not exactly match ``iteration_<int>.pth.tar`` are outside
    this function's authority.
    """
    if keep_last < 0:
        raise ValueError(f'periodic_ckpt_keep_last must be non-negative, got {keep_last}')
    if keep_last == 0:
        return []

    checkpoint_dir = Path(exp_name) / 'ckpt'
    if not checkpoint_dir.is_dir():
        return []

    periodic_checkpoints = []
    for path in checkpoint_dir.iterdir():
        match = _PERIODIC_CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is not None and path.is_file():
            periodic_checkpoints.append((int(match.group(1)), path))

    positive_checkpoints = sorted(
        ((iteration, path) for iteration, path in periodic_checkpoints if iteration > 0),
        key=lambda item: item[0],
    )
    stale_checkpoints = positive_checkpoints[:-keep_last]
    removed = []
    for iteration, path in stale_checkpoints:
        try:
            path.unlink()
        except OSError as error:
            # Retention is a storage safeguard; a transient filesystem failure must not terminate
            # an otherwise healthy multi-hour training run.
            logging.warning(
                'Failed to remove stale periodic checkpoint iteration=%d (%s): %s',
                iteration,
                path,
                error,
            )
            continue
        removed.append(str(path))
        logging.info('Removed stale periodic checkpoint iteration=%d: %s', iteration, path)
    return removed


def train_unizero_segment(
        input_cfg: Tuple[dict, dict],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
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
    # The config launcher creates the per-run directory first so it can place
    # metadata and console.log there.  Keep that explicit directory name
    # instead of letting DI-engine append another timestamp merely because the
    # directory already exists.
    cfg = compile_config(
        cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True, renew_dir=False
    )

    # Create main components: env, policy
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])

    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=torch.cuda.is_available())

    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])

    # Load pretrained model if specified
    resume_train_iter, resume_envstep = 0, 0
    if model_path is not None:
        logging.info(f'Loading model from {model_path} begin...')
        checkpoint = torch.load(model_path, map_location=cfg.policy.device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            # Learner checkpoint: {'model', 'target_model', 'optimizer_world_model', 'last_iter',
            # 'last_step'}. UniZeroPolicy._load_state_dict_learn expects exactly this dict (it reads
            # the 'model' and 'target_model' keys itself).
            resume_train_iter = checkpoint.get('last_iter', 0)
            resume_envstep = checkpoint.get('last_step', 0)
            policy.learn_mode.load_state_dict(checkpoint)
        else:
            # Raw model weights file.
            policy._learn_model.load_state_dict(checkpoint)
        logging.info(f'Loading model from {model_path} end! '
                     f'(resume_train_iter={resume_train_iter}, resume_envstep={resume_envstep})')

    # Create worker components: learner, collector, evaluator, replay buffer, commander
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial')) if get_rank() == 0 else None
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)

    # MCTS+RL algorithms related core code
    policy_config = cfg.policy
    replay_buffer = GameBuffer(policy_config)
    collector = Collector(env=collector_env, policy=policy.collect_mode, tb_logger=tb_logger, exp_name=cfg.exp_name,
                          policy_config=policy_config)
    evaluator = Evaluator(eval_freq=cfg.policy.eval_freq, n_evaluator_episode=cfg.env.n_evaluator_episode,
                          stop_value=cfg.env.stop_value, env=evaluator_env, policy=policy.eval_mode,
                          tb_logger=tb_logger, exp_name=cfg.exp_name, policy_config=policy_config)

    # When resuming from a learner checkpoint, restore the training counters so
    # that schedules (encoder-clip annealing, eval cadence) and the envstep accounting continue
    # instead of restarting from zero.
    _restore_resume_counters(learner, collector, resume_train_iter, resume_envstep)

    # Learner's before_run hook
    learner.call_hook('before_run')

    if cfg.policy.use_wandb:
        policy.set_train_iter_env_step(learner.train_iter, collector.envstep)

    # Collect random data before training
    if cfg.policy.random_collect_episode_num > 0:
        random_collect(cfg.policy, policy, LightZeroRandomPolicy, collector, collector_env, replay_buffer)

    batch_size = policy._cfg.batch_size
    required_replay_transitions = _required_replay_transitions(
        resume_train_iter,
        batch_size,
        int(getattr(cfg.policy, 'resume_buffer_min_transitions', 0)),
    )
    if resume_train_iter > 0 and required_replay_transitions > batch_size + 1:
        logging.info(
            'Resume replay warmup: collect at least %d transitions before learner updates.',
            required_replay_transitions,
        )

    # TODO: for visualize
    # stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
    
    buffer_reanalyze_count = 0
    train_epoch = 0
    buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition = (
        _resolve_segment_reanalyze_settings(cfg.policy)
    )
    periodic_ckpt_keep_last = int(getattr(cfg.policy, 'periodic_ckpt_keep_last', 0))
    if periodic_ckpt_keep_last < 0:
        raise ValueError(
            f'periodic_ckpt_keep_last must be non-negative, got {periodic_ckpt_keep_last}'
        )

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
        if learner.train_iter == 0 or evaluator.should_eval(learner.train_iter):
            save_ckpt_fn = learner.save_checkpoint if getattr(cfg.policy, 'save_ckpt_in_eval', True) else None
            stop, reward = evaluator.eval(save_ckpt_fn, learner.train_iter, collector.envstep)
            if stop:
                break

        # Collect new data
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)

        # Determine updates per collection
        update_per_collect = calculate_update_per_collect(cfg, new_data, world_size)

        # Update replay buffer
        replay_buffer.push_game_segments(new_data)
        replay_buffer.remove_oldest_data_to_fit()

        # Periodically reanalyze buffer
        if buffer_reanalyze_freq >= 1:
            # Reanalyze buffer <buffer_reanalyze_freq> times in one train_epoch
            reanalyze_interval = update_per_collect // buffer_reanalyze_freq
        else:
            # Reanalyze buffer each <1/buffer_reanalyze_freq> train_epoch
            should_reanalyze = (
                train_epoch > 0
                and train_epoch % int(1 / buffer_reanalyze_freq) == 0
                and replay_buffer.get_num_of_transitions() // cfg.policy.num_unroll_steps
                > int(reanalyze_batch_size / reanalyze_partition)
            )
            if should_reanalyze:
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
            data_sufficient = data_sufficient and (
                replay_buffer.get_num_of_transitions() >= required_replay_transitions
            )
            if not data_sufficient:
                logging.warning(
                    f'The data in replay_buffer is not sufficient to sample a mini-batch: '
                    f'batch_size: {batch_size}, required_transitions: {required_replay_transitions}, '
                    f'replay_buffer: {replay_buffer}. Continue to collect now ....'
                )
                continue

            for i in range(update_per_collect):
                if buffer_reanalyze_freq >= 1:
                    # Reanalyze buffer <buffer_reanalyze_freq> times in one train_epoch
                    should_reanalyze = (
                        i % reanalyze_interval == 0
                        and replay_buffer.get_num_of_transitions() // cfg.policy.num_unroll_steps
                        > int(reanalyze_batch_size / reanalyze_partition)
                    )
                    if should_reanalyze:
                        with timer:
                            # Each reanalyze process will reanalyze <reanalyze_batch_size> sequences (<cfg.policy.num_unroll_steps> transitions per sequence)
                            replay_buffer.reanalyze_buffer(reanalyze_batch_size, policy)
                        buffer_reanalyze_count += 1
                        logging.info(f'Buffer reanalyze count: {buffer_reanalyze_count}')
                        logging.info(f'Buffer reanalyze time: {timer.value}')

                train_data = replay_buffer.sample(batch_size, policy)
                policy.set_replay_diagnostics(
                    train_data[0][4],
                    replay_buffer.get_num_of_transitions(),
                    cfg.policy.replay_buffer_size,
                )
                if cfg.policy.use_wandb:
                    policy.set_train_iter_env_step(learner.train_iter, collector.envstep)

                train_data.append(learner.train_iter)
                log_vars = learner.train(train_data, collector.envstep)

                if cfg.policy.use_priority:
                    replay_buffer.update_priority(train_data, log_vars[0]['value_priority_orig'])

        train_epoch += 1
        policy.recompute_pos_emb_diff_and_clear_cache()
        if periodic_ckpt_keep_last > 0 and rank == 0:
            _prune_periodic_checkpoints(cfg.exp_name, periodic_ckpt_keep_last)

        # Check stopping criteria
        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break

    learner.call_hook('after_run')
    if cfg.policy.use_wandb:
        wandb.finish()
    return policy
