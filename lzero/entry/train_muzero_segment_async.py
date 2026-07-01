import asyncio
import copy
import logging
import os
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple

from easydict import EasyDict


@dataclass
class _TrainEpochBudget:
    update_per_collect: int
    progress: int = 0
    reanalyze_interval: Optional[float] = None

    @property
    def done(self) -> bool:
        return self.progress >= self.update_per_collect


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    return cfg.get(key, default) if hasattr(cfg, 'get') else getattr(cfg, key, default)


def _safe_log_buffer_stats(
        train_iter: int,
        replay_buffer: Any,
        tb_logger: Any,
        log_buffer_memory_usage_fn: Any,
        log_buffer_run_time_fn: Any,
        warned_keys: set,
) -> None:
    try:
        log_buffer_memory_usage_fn(train_iter, replay_buffer, tb_logger)
    except Exception as exc:
        warning_key = ('memory', type(exc).__name__, str(exc))
        if warning_key not in warned_keys:
            warned_keys.add(warning_key)
            logging.warning("[AsyncMuZero] skip buffer memory logging at iter=%s: %r", train_iter, exc)
    try:
        log_buffer_run_time_fn(train_iter, replay_buffer, tb_logger)
    except Exception as exc:
        warning_key = ('runtime', type(exc).__name__, str(exc))
        if warning_key not in warned_keys:
            warned_keys.add(warning_key)
            logging.warning("[AsyncMuZero] skip buffer runtime logging at iter=%s: %r", train_iter, exc)


def _select_game_buffer(policy_type: str):
    if policy_type in ['muzero', 'muzero_context', 'muzero_rnn_full_obs']:
        from lzero.mcts import MuZeroGameBuffer as GameBuffer
    elif policy_type == 'efficientzero':
        from lzero.mcts import EfficientZeroGameBuffer as GameBuffer
    elif policy_type == 'sampled_efficientzero':
        from lzero.mcts import SampledEfficientZeroGameBuffer as GameBuffer
    elif policy_type == 'sampled_muzero':
        from lzero.mcts import SampledMuZeroGameBuffer as GameBuffer
    elif policy_type == 'gumbel_muzero':
        from lzero.mcts import GumbelMuZeroGameBuffer as GameBuffer
    elif policy_type == 'stochastic_muzero':
        from lzero.mcts import StochasticMuZeroGameBuffer as GameBuffer
    else:
        raise AssertionError(
            "train_muzero_segment_async only supports muzero-family segment policies, "
            f"got {policy_type!r}"
        )
    return GameBuffer


def _to_cpu_tree(obj: Any) -> Any:
    import torch

    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _to_cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu_tree(v) for v in obj)
    return copy.deepcopy(obj)


def _model_state_for_remote(policy: Any) -> Dict[str, Any]:
    return _to_cpu_tree(policy.learn_mode.state_dict()['model'])


def _full_policy_state_for_checkpoint(policy: Any) -> Dict[str, Any]:
    return _to_cpu_tree(policy.learn_mode.state_dict())


def _collect_kwargs(policy_config: EasyDict, train_iter: int, envstep: int) -> Dict[str, Any]:
    from ding.rl_utils import get_epsilon_greedy_fn
    from lzero.policy import visit_count_temperature

    kwargs: Dict[str, Any] = {}
    kwargs['temperature'] = visit_count_temperature(
        policy_config.manual_temperature_decay,
        policy_config.fixed_temperature_value,
        policy_config.threshold_training_steps_for_final_temperature,
        trained_steps=train_iter,
    )

    if policy_config.eps.eps_greedy_exploration_in_collect:
        epsilon_greedy_fn = get_epsilon_greedy_fn(
            start=policy_config.eps.start,
            end=policy_config.eps.end,
            decay=policy_config.eps.decay,
            type_=policy_config.eps.type,
        )
        kwargs['epsilon'] = epsilon_greedy_fn(envstep)
    else:
        kwargs['epsilon'] = 0.0
    return kwargs


def _should_eval(train_iter: int, last_eval_iter: int, eval_freq: int) -> bool:
    if train_iter == last_eval_iter:
        return False
    if (train_iter - last_eval_iter) < eval_freq and train_iter != 0:
        return False
    return True


def _save_checkpoint_snapshot(
        exp_name: str,
        state_dict: Dict[str, Any],
        ckpt_name: str,
        train_iter: int,
        envstep: int,
) -> str:
    from ding.utils import save_file

    dirname = './{}/ckpt'.format(exp_name)
    os.makedirs(dirname, exist_ok=True)
    path = os.path.join(dirname, ckpt_name)
    state = copy.deepcopy(state_dict)
    state.update({'last_iter': train_iter, 'last_step': envstep})
    save_file(path, state)
    logging.info(f'[AsyncMuZero] save evaluated checkpoint version={train_iter} envstep={envstep} path={path}')
    return path


async def _ray_get_async(ray_module: Any, obj_ref: Any) -> Any:
    return await asyncio.to_thread(ray_module.get, obj_ref)


def _get_async_cfg(cfg: EasyDict) -> EasyDict:
    async_cfg = EasyDict(copy.deepcopy(_cfg_get(cfg.policy, 'async_pipeline', {})))

    def set_default(key: str, value: Any) -> None:
        if key not in async_cfg:
            async_cfg[key] = value

    set_default('enabled', True)
    set_default('num_collector_actors', 1)
    set_default('num_evaluator_actors', 1)
    set_default('max_collect_inflight', async_cfg.num_collector_actors)
    set_default('max_eval_inflight', 1)
    set_default('max_train_chunk_steps', 4)
    set_default('weight_sync_interval', 1)
    set_default('max_policy_lag', 0)
    set_default('collector_num_cpus', 1)
    set_default('evaluator_num_cpus', 1)
    set_default('collector_num_gpus', 0)
    set_default('evaluator_num_gpus', 0)
    set_default('ray_local_mode', False)
    set_default('ray_ignore_reinit_error', True)
    set_default('buffer_stats_interval', 100)
    set_default('max_train_budget_queue_size', max(1, async_cfg.num_collector_actors * 2))
    set_default('poll_interval_s', 0.1)
    set_default('shutdown_timeout_s', 30)
    return async_cfg


async def _run_async_driver(
        ray_module: Any,
        cfg: EasyDict,
        create_cfg: EasyDict,
        seed: int,
        policy: Any,
        learner: Any,
        replay_buffer: Any,
        max_train_iter: int,
        max_env_step: int,
) -> None:
    from lzero.entry.utils import calculate_update_per_collect, log_buffer_memory_usage, log_buffer_run_time
    from lzero.entry.async_muzero.actors import MuZeroSegmentCollectorActor, MuZeroSegmentEvaluatorActor

    async_cfg = _get_async_cfg(cfg)
    collector_cls = MuZeroSegmentCollectorActor.options(
        num_cpus=async_cfg.collector_num_cpus,
        num_gpus=async_cfg.collector_num_gpus,
    )
    evaluator_cls = MuZeroSegmentEvaluatorActor.options(
        num_cpus=async_cfg.evaluator_num_cpus,
        num_gpus=async_cfg.evaluator_num_gpus,
    )
    collector_actors = [
        collector_cls.remote(cfg, create_cfg, seed, actor_id) for actor_id in range(async_cfg.num_collector_actors)
    ]
    evaluator_actors = [
        evaluator_cls.remote(cfg, create_cfg, seed, actor_id) for actor_id in range(async_cfg.num_evaluator_actors)
    ]

    pending_collect: Dict[asyncio.Task, Dict[str, Any]] = {}
    pending_eval: Dict[asyncio.Task, Dict[str, Any]] = {}
    train_budgets: Deque[_TrainEpochBudget] = deque()

    collector_envstep = 0
    train_epoch = 0
    buffer_reanalyze_count = 0
    last_eval_iter = 0
    best_reward = float('-inf')
    stop_requested = False
    last_published_version = -1
    last_published_model_state: Optional[Dict[str, Any]] = None
    last_buffer_stats_iter: Optional[int] = None
    buffer_stats_warning_keys: set = set()

    def get_published_model_state(force: bool = False) -> Tuple[int, Dict[str, Any]]:
        nonlocal last_published_version, last_published_model_state
        current_version = learner.train_iter
        should_publish = (
            force or last_published_model_state is None or
            (current_version - last_published_version) >= async_cfg.weight_sync_interval
        )
        if should_publish:
            last_published_model_state = _model_state_for_remote(policy)
            last_published_version = current_version
        return last_published_version, last_published_model_state

    def launch_collect(actor: Any, actor_id: int) -> None:
        version, model_state = get_published_model_state()
        model_state_ref = ray_module.put(model_state)
        kwargs = _collect_kwargs(cfg.policy, learner.train_iter, collector_envstep)
        ref = actor.collect.remote(
            model_state_ref,
            version,
            learner.train_iter,
            collector_envstep,
            kwargs,
        )
        task = asyncio.create_task(_ray_get_async(ray_module, ref))
        pending_collect[task] = {'actor': actor, 'actor_id': actor_id, 'ref': ref}

    def launch_eval(actor: Any, actor_id: int) -> None:
        nonlocal last_eval_iter
        full_state = _full_policy_state_for_checkpoint(policy)
        model_state_ref = ray_module.put(full_state['model'])
        train_iter = learner.train_iter
        envstep = collector_envstep
        ref = actor.eval.remote(model_state_ref, train_iter, train_iter, envstep, None)
        task = asyncio.create_task(_ray_get_async(ray_module, ref))
        pending_eval[task] = {
            'actor': actor,
            'actor_id': actor_id,
            'ref': ref,
            'state_dict': full_state,
            'train_iter': train_iter,
            'envstep': envstep,
        }
        last_eval_iter = train_iter

    def maybe_launch_collects() -> None:
        if stop_requested or collector_envstep >= max_env_step:
            return
        if len(train_budgets) >= async_cfg.max_train_budget_queue_size:
            return
        max_inflight = min(async_cfg.max_collect_inflight, len(collector_actors))
        busy_actor_ids = {meta['actor_id'] for meta in pending_collect.values()}
        for actor_id, actor in enumerate(collector_actors):
            if len(pending_collect) >= max_inflight:
                break
            if actor_id in busy_actor_ids:
                continue
            launch_collect(actor, actor_id)

    def maybe_launch_eval() -> None:
        if stop_requested or not evaluator_actors:
            return
        if len(pending_eval) >= async_cfg.max_eval_inflight:
            return
        if not _should_eval(learner.train_iter, last_eval_iter, cfg.policy.eval_freq):
            return
        actor_id = len(pending_eval) % len(evaluator_actors)
        launch_eval(evaluator_actors[actor_id], actor_id)

    def process_collect_result(result: Dict[str, Any]) -> None:
        nonlocal collector_envstep, buffer_reanalyze_count
        new_data = result['new_data']
        collector_envstep += int(result['envstep_delta'])
        update_per_collect = calculate_update_per_collect(cfg, new_data)
        replay_buffer.push_game_segments(new_data)
        replay_buffer.remove_oldest_data_to_fit()

        reanalyze_batch_size = cfg.policy.reanalyze_batch_size
        if cfg.policy.buffer_reanalyze_freq >= 1:
            reanalyze_interval = update_per_collect // cfg.policy.buffer_reanalyze_freq
        else:
            reanalyze_interval = None
            reanalyze_period = int(1 // cfg.policy.buffer_reanalyze_freq)
            enough_for_reanalyze = (
                replay_buffer.get_num_of_transitions() // cfg.policy.num_unroll_steps >
                int(reanalyze_batch_size / cfg.policy.reanalyze_partition)
            )
            if train_epoch % reanalyze_period == 0 and enough_for_reanalyze:
                replay_buffer.reanalyze_buffer(reanalyze_batch_size, policy)
                buffer_reanalyze_count += 1
                logging.info(f'[AsyncMuZero] Buffer reanalyze count: {buffer_reanalyze_count}')

        train_budgets.append(
            _TrainEpochBudget(update_per_collect=update_per_collect, reanalyze_interval=reanalyze_interval)
        )
        logging.info(
            '[AsyncMuZero] collect done actor=%s version=%s envstep_delta=%s global_envstep=%s '
            'update_budget=%s duration=%.3fs',
            result['actor_id'],
            result['policy_version'],
            result['envstep_delta'],
            collector_envstep,
            update_per_collect,
            result['duration'],
        )

    def train_one_step() -> bool:
        nonlocal train_epoch, buffer_reanalyze_count
        if not train_budgets:
            return False
        if learner.train_iter >= max_train_iter:
            return False
        if replay_buffer.get_num_of_transitions() <= cfg.policy.batch_size:
            return False

        budget = train_budgets[0]
        reanalyze_batch_size = cfg.policy.reanalyze_batch_size
        if cfg.policy.buffer_reanalyze_freq >= 1:
            enough_for_reanalyze = (
                replay_buffer.get_num_of_transitions() // cfg.policy.num_unroll_steps >
                int(reanalyze_batch_size / cfg.policy.reanalyze_partition)
            )
            if budget.reanalyze_interval is not None and budget.progress % budget.reanalyze_interval == 0 and enough_for_reanalyze:
                replay_buffer.reanalyze_buffer(reanalyze_batch_size, policy)
                buffer_reanalyze_count += 1
                logging.info(f'[AsyncMuZero] Buffer reanalyze count: {buffer_reanalyze_count}')

        train_data = replay_buffer.sample(cfg.policy.batch_size, policy)
        log_vars = learner.train(train_data, collector_envstep)
        if cfg.policy.use_priority:
            replay_buffer.update_priority(train_data, log_vars[0]['value_priority_orig'])

        budget.progress += 1
        if budget.done:
            train_budgets.popleft()
            train_epoch += 1
        return True

    async def process_completed(timeout: float = 0.0) -> int:
        nonlocal best_reward, stop_requested
        tasks = set(pending_collect.keys()) | set(pending_eval.keys())
        if not tasks:
            if timeout > 0:
                await asyncio.sleep(timeout)
            return 0
        done, _ = await asyncio.wait(tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if task in pending_collect:
                pending_collect.pop(task)
                result = task.result()
                process_collect_result(result)
            elif task in pending_eval:
                meta = pending_eval.pop(task)
                result = task.result()
                reward_mean = result.get('reward_mean')
                if reward_mean is not None:
                    reward_mean = float(reward_mean)
                    if reward_mean > best_reward:
                        best_reward = reward_mean
                        _save_checkpoint_snapshot(
                            cfg.exp_name,
                            meta['state_dict'],
                            'ckpt_best.pth.tar',
                            meta['train_iter'],
                            meta['envstep'],
                        )
                if result.get('stop_flag', False):
                    stop_requested = True
                logging.info(
                    '[AsyncMuZero] eval done actor=%s version=%s train_iter=%s envstep=%s '
                    'reward_mean=%s stop=%s duration=%.3fs',
                    result['actor_id'],
                    result['policy_version'],
                    result['train_iter'],
                    result['envstep'],
                    result.get('reward_mean'),
                    result.get('stop_flag'),
                    result['duration'],
                )
        return len(done)

    try:
        maybe_launch_collects()
        while True:
            if (
                    learner.train_iter != last_buffer_stats_iter and
                    learner.train_iter % max(int(async_cfg.buffer_stats_interval), 1) == 0
            ):
                _safe_log_buffer_stats(
                    learner.train_iter,
                    replay_buffer,
                    learner.tb_logger,
                    log_buffer_memory_usage,
                    log_buffer_run_time,
                    buffer_stats_warning_keys,
                )
                last_buffer_stats_iter = learner.train_iter

            await process_completed(timeout=0.0)
            maybe_launch_eval()
            maybe_launch_collects()

            trained_steps = 0
            while trained_steps < async_cfg.max_train_chunk_steps and train_one_step():
                trained_steps += 1
                if learner.train_iter >= max_train_iter:
                    stop_requested = True
                    break
                await process_completed(timeout=0.0)
                maybe_launch_eval()

            if learner.train_iter >= max_train_iter or stop_requested:
                break
            if collector_envstep >= max_env_step and not pending_collect and not train_budgets:
                break

            if trained_steps == 0:
                await process_completed(timeout=async_cfg.poll_interval_s)

        logging.info(
            '[AsyncMuZero] stopping: envstep=%s/%s train_iter=%s/%s stop_requested=%s pending_collect=%s pending_eval=%s',
            collector_envstep,
            max_env_step,
            learner.train_iter,
            max_train_iter,
            stop_requested,
            len(pending_collect),
            len(pending_eval),
        )
    finally:
        close_refs = []
        for actor in collector_actors + evaluator_actors:
            try:
                close_refs.append(actor.close.remote())
            except Exception:
                logging.exception('[AsyncMuZero] failed to enqueue actor close')
        if close_refs:
            try:
                ray_module.get(close_refs, timeout=async_cfg.shutdown_timeout_s)
            except Exception:
                logging.warning('[AsyncMuZero] timed out while closing Ray actors; killing actors')
                for actor in collector_actors + evaluator_actors:
                    try:
                        ray_module.kill(actor, no_restart=True)
                    except Exception:
                        pass


def train_muzero_segment_async(
        input_cfg: Tuple[dict, dict],
        seed: int = 0,
        model: Optional[Any] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    Ray-based asynchronous MuZero segment training entry.

    Collector and evaluator run in Ray actors. Learner and replay buffer remain
    in the driver process as a single owner to preserve GameBuffer mutation,
    sample/reanalyze ordering and learner hook semantics.
    """

    cfg, create_cfg = input_cfg
    assert create_cfg.policy.type in [
        'efficientzero', 'muzero', 'muzero_context', 'muzero_rnn_full_obs', 'sampled_efficientzero',
        'sampled_muzero', 'gumbel_muzero', 'stochastic_muzero'
    ], (
        "train_muzero_segment_async only supports: 'efficientzero', 'muzero', "
        "'sampled_efficientzero', 'gumbel_muzero', 'stochastic_muzero'"
    )

    if cfg.policy.get('eval_offline', False) or cfg.policy.get('random_collect_episode_num', 0) > 0:
        logging.warning(
            '[AsyncMuZero] eval_offline/random_collect is not supported by the async MVP; '
            'falling back to train_muzero_segment.'
        )
        from .train_muzero_segment import train_muzero_segment
        return train_muzero_segment(input_cfg, seed, model, model_path, max_train_iter, max_env_step)

    try:
        import ray
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            'Ray is required for train_muzero_segment_async. Install ray in the runtime environment '
            'or run the config without --async-pipeline.'
        ) from exc

    from ding.config import compile_config
    from ding.policy import create_policy
    from ding.utils import get_rank, set_pkg_seed
    from ding.worker import BaseLearner
    from tensorboardX import SummaryWriter
    import torch

    if cfg.policy.cuda and torch.cuda.is_available():
        cfg.policy.device = 'cuda'
    else:
        cfg.policy.device = 'cpu'
        cfg.policy.cuda = False

    GameBuffer = _select_game_buffer(create_cfg.policy.type)
    cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])
    if model_path is not None:
        policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))

    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'async')) if get_rank() == 0 else None
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    replay_buffer = GameBuffer(cfg.policy)

    async_cfg = _get_async_cfg(cfg)
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=async_cfg.ray_ignore_reinit_error,
            include_dashboard=False,
            local_mode=async_cfg.ray_local_mode,
        )

    learner.call_hook('before_run')
    try:
        asyncio.run(
            _run_async_driver(
                ray, cfg, create_cfg, seed, policy, learner, replay_buffer, max_train_iter, max_env_step
            )
        )
    finally:
        learner.call_hook('after_run')
    return policy
