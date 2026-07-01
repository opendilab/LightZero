import copy
import time
from functools import partial
from typing import Any, Dict, Optional

import ray
import torch
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.utils import import_module, set_pkg_seed
from lzero.worker import MuZeroEvaluator as Evaluator
from lzero.worker import MuZeroSegmentCollector as Collector


def _extract_reward_mean(eval_info: Optional[Dict[str, Any]]) -> Optional[float]:
    if not eval_info:
        return None
    if eval_info.get('reward_mean') is not None:
        return float(eval_info['reward_mean'])
    if eval_info.get('eval_episode_return_mean') is not None:
        return float(eval_info['eval_episode_return_mean'])
    returns = eval_info.get('eval_episode_return')
    if returns is None:
        return None
    if isinstance(returns, (int, float)):
        return float(returns)
    returns = list(returns)
    if not returns:
        return None
    return float(sum(float(v) for v in returns) / len(returns))


def _import_create_cfg_modules(create_cfg: Any) -> None:
    for section_name in ('env', 'policy'):
        section = getattr(create_cfg, section_name, None)
        if section is None:
            continue
        import_names = section.get('import_names', [])
        if import_names:
            import_module(import_names)


def _prepare_actor_cfg(cfg: Any, seed_offset: int) -> Any:
    cfg = copy.deepcopy(cfg)
    cfg.seed = int(cfg.seed) + int(seed_offset)
    if cfg.policy.cuda and torch.cuda.is_available():
        cfg.policy.device = 'cuda'
        cfg.policy.cuda = True
    else:
        cfg.policy.device = 'cpu'
        cfg.policy.cuda = False
    return cfg


@ray.remote
class MuZeroSegmentCollectorActor:
    """
    Ray actor that owns a MuZero collect-mode policy and collector env manager.

    The learner publishes model-only CPU state dicts. This actor loads a new
    state dict only when the version changes, then runs the existing
    MuZeroSegmentCollector.collect implementation unchanged.
    """

    def __init__(self, cfg: Any, create_cfg: Any, seed: int, actor_id: int = 0) -> None:
        _import_create_cfg_modules(create_cfg)
        self._actor_id = actor_id
        self._cfg = _prepare_actor_cfg(cfg, seed_offset=actor_id * 1000)
        self._latest_policy_version: Optional[int] = None

        env_fn, collector_env_cfg, _ = get_vec_env_setting(self._cfg.env)
        self._collector_env = create_env_manager(
            self._cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg]
        )
        self._collector_env.seed(seed + actor_id * 1000)
        set_pkg_seed(self._cfg.seed, use_cuda=self._cfg.policy.cuda)

        # MuZeroPolicy initializes value/reward support transforms in learn mode
        # and reuses them during collect/eval forward. Keep the actor behavior
        # collect-only while matching the synchronous pipeline's policy setup.
        self._policy = create_policy(self._cfg.policy, enable_field=['learn', 'collect', 'eval'])
        self._collector = Collector(
            env=self._collector_env,
            policy=self._policy.collect_mode,
            tb_logger=None,
            exp_name=self._cfg.exp_name,
            instance_name=f'collector_async_{actor_id}',
            policy_config=self._cfg.policy,
        )

    def collect(
            self,
            model_state: Dict[str, Any],
            policy_version: int,
            train_iter: int,
            global_envstep: int,
            policy_kwargs: Optional[dict] = None,
    ) -> Dict[str, Any]:
        if self._latest_policy_version != policy_version:
            self._policy.collect_mode.load_state_dict({'model': model_state})
            self._latest_policy_version = policy_version

        start_envstep = self._collector.envstep
        start_time = time.time()
        new_data = self._collector.collect(train_iter=train_iter, policy_kwargs=policy_kwargs or {})
        duration = time.time() - start_time
        end_envstep = self._collector.envstep

        return {
            'actor_id': self._actor_id,
            'new_data': new_data,
            'policy_version': policy_version,
            'train_iter': train_iter,
            'global_envstep_at_launch': global_envstep,
            'collector_local_envstep': end_envstep,
            'envstep_delta': end_envstep - start_envstep,
            'duration': duration,
        }

    def close(self) -> None:
        self._collector.close()


@ray.remote
class MuZeroSegmentEvaluatorActor:
    """
    Ray actor that owns a MuZero eval-mode policy and evaluator env manager.

    Evaluation receives an immutable model snapshot and never swaps weights
    while a rollout is in progress.
    """

    def __init__(self, cfg: Any, create_cfg: Any, seed: int, actor_id: int = 0) -> None:
        _import_create_cfg_modules(create_cfg)
        self._actor_id = actor_id
        self._cfg = _prepare_actor_cfg(cfg, seed_offset=100000 + actor_id * 1000)
        self._latest_policy_version: Optional[int] = None

        env_fn, _, evaluator_env_cfg = get_vec_env_setting(self._cfg.env)
        self._evaluator_env = create_env_manager(
            self._cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg]
        )
        self._evaluator_env.seed(seed + 100000 + actor_id * 1000, dynamic_seed=False)
        set_pkg_seed(self._cfg.seed, use_cuda=self._cfg.policy.cuda)

        # See collector actor note: eval forward also depends on support
        # transforms initialized by learn mode.
        self._policy = create_policy(self._cfg.policy, enable_field=['learn', 'collect', 'eval'])
        self._evaluator = Evaluator(
            eval_freq=self._cfg.policy.eval_freq,
            n_evaluator_episode=self._cfg.env.n_evaluator_episode,
            stop_value=self._cfg.env.stop_value,
            env=self._evaluator_env,
            policy=self._policy.eval_mode,
            tb_logger=None,
            exp_name=self._cfg.exp_name,
            instance_name=f'evaluator_async_{actor_id}',
            policy_config=self._cfg.policy,
        )

    def eval(
            self,
            model_state: Dict[str, Any],
            policy_version: int,
            train_iter: int,
            envstep: int,
            n_episode: Optional[int] = None,
    ) -> Dict[str, Any]:
        if self._latest_policy_version != policy_version:
            self._policy.eval_mode.load_state_dict({'model': model_state})
            self._latest_policy_version = policy_version

        start_time = time.time()
        stop_flag, eval_info = self._evaluator.eval(
            save_ckpt_fn=None,
            train_iter=train_iter,
            envstep=envstep,
            n_episode=n_episode,
        )
        duration = time.time() - start_time

        return {
            'actor_id': self._actor_id,
            'policy_version': policy_version,
            'train_iter': train_iter,
            'envstep': envstep,
            'stop_flag': stop_flag,
            'eval_info': eval_info,
            'reward_mean': _extract_reward_mean(eval_info),
            'duration': duration,
        }

    def close(self) -> None:
        self._evaluator.close()
