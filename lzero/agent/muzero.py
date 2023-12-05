import os
from functools import partial
from typing import Optional, Union, List, Tuple
from ditk import logging
from easydict import EasyDict

import numpy as np
import torch
import treetensor.torch as ttorch

from ding.config import compile_config
from ding.envs import create_env_manager
from ding.envs import get_vec_env_setting
from ding.policy import create_policy
from ding.utils import set_pkg_seed, get_rank
from ding.rl_utils import get_epsilon_greedy_fn
from ding.worker import BaseLearner
from ding.config import save_config_py, compile_config
from ding.utils import get_env_fps, render
from ding.bonus.common import TrainingReturn, EvalReturn

from tensorboardX import SummaryWriter

from lzero.entry.utils import log_buffer_memory_usage, random_collect
from lzero.policy import visit_count_temperature
from lzero.policy.muzero import MuZeroPolicy
from lzero.policy.random_policy import LightZeroRandomPolicy
from lzero.worker import MuZeroCollector as Collector
from lzero.worker import MuZeroEvaluator as Evaluator
from lzero.agent.config.muzero import supported_env_cfg
from lzero.mcts import MuZeroGameBuffer


class MuZeroAgent:

    supported_env_list = list(supported_env_cfg.keys())

    def __init__(
            self,
            env_id: str = None,
            seed: int = 0,
            exp_name: str = None,
            model: Optional[torch.nn.Module] = None,
            cfg: Optional[Union[EasyDict, dict]] = None,
            policy_state_dict: str = None,
    ) -> None:
        assert env_id is not None or cfg["main_config"]["env_id"] is not None, "Please specify env_id or cfg."

        if cfg is not None and not isinstance(cfg, EasyDict):
            cfg = EasyDict(cfg)

        if env_id is not None:
            assert env_id in MuZeroAgent.supported_env_list, "Please use supported envs: {}".format(
                MuZeroAgent.supported_env_list
            )
            if cfg is None:
                cfg = supported_env_cfg[env_id]
            else:
                assert cfg.main_config.env.env_id == env_id, "env_id in cfg should be the same as env_id in args."
        else:
            assert hasattr(cfg.main_config.env, "env_id"), "Please specify env_id in cfg."
            assert cfg.main_config.env.env_id in MuZeroAgent.supported_env_list, "Please use supported envs: {}".format(
                MuZeroAgent.supported_env_list
            )
        default_policy_config = EasyDict({"policy": MuZeroPolicy.default_config()})
        default_policy_config.policy.update(cfg.main_config.policy)
        cfg.main_config.policy = default_policy_config.policy

        if exp_name is not None:
            cfg.main_config.exp_name = exp_name
        self.origin_cfg = cfg
        self.cfg = compile_config(
            cfg.main_config, seed=seed, env=None, auto=True, policy=MuZeroPolicy, create_cfg=cfg.create_config
        )
        self.exp_name = self.cfg.exp_name

        logging.getLogger().setLevel(logging.INFO)
        self.seed = seed
        set_pkg_seed(self.seed, use_cuda=self.cfg.policy.cuda)
        if not os.path.exists(self.exp_name):
            os.makedirs(self.exp_name)
        save_config_py(cfg, os.path.join(self.exp_name, 'policy_config.py'))
        if model is None:
            if self.cfg.policy.model.model_type == 'mlp':
                from lzero.model.muzero_model_mlp import MuZeroModelMLP
                model = MuZeroModelMLP(**self.cfg.policy.model)
            elif self.cfg.policy.model.model_type == 'conv':
                from lzero.model.muzero_model import MuZeroModel
                model = MuZeroModel(**self.cfg.policy.model)
            else:
                raise NotImplementedError
        #self.policy = MuZeroPolicy(self.cfg.policy, model=model)
        if self.cfg.policy.cuda and torch.cuda.is_available():
            self.cfg.policy.device = 'cuda'
        else:
            self.cfg.policy.device = 'cpu'
        self.policy = create_policy(self.cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])
        if policy_state_dict is not None:
            self.policy.learn_mode.load_state_dict(policy_state_dict)
        self.checkpoint_save_dir = os.path.join(self.exp_name, "ckpt")

        self.env_fn, self.collector_env_cfg, self.evaluator_env_cfg = get_vec_env_setting(self.cfg.env)

    def train(
        self,
        step: int = int(1e7),
    ) -> TrainingReturn:

        collector_env = create_env_manager(
            self.cfg.env.manager, [partial(self.env_fn, cfg=c) for c in self.collector_env_cfg]
        )
        evaluator_env = create_env_manager(
            self.cfg.env.manager, [partial(self.env_fn, cfg=c) for c in self.evaluator_env_cfg]
        )

        collector_env.seed(self.cfg.seed)
        evaluator_env.seed(self.cfg.seed, dynamic_seed=False)
        set_pkg_seed(self.cfg.seed, use_cuda=self.cfg.policy.cuda)

        # Create worker components: learner, collector, evaluator, replay buffer, commander.
        tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(self.cfg.exp_name), 'serial')
                                  ) if get_rank() == 0 else None
        learner = BaseLearner(
            self.cfg.policy.learn.learner, self.policy.learn_mode, tb_logger, exp_name=self.cfg.exp_name
        )

        # ==============================================================
        # MCTS+RL algorithms related core code
        # ==============================================================
        policy_config = self.cfg.policy
        batch_size = policy_config.batch_size
        # specific game buffer for MCTS+RL algorithms
        replay_buffer = MuZeroGameBuffer(policy_config)
        collector = Collector(
            env=collector_env,
            policy=self.policy.collect_mode,
            tb_logger=tb_logger,
            exp_name=self.cfg.exp_name,
            policy_config=policy_config
        )
        evaluator = Evaluator(
            eval_freq=self.cfg.policy.eval_freq,
            n_evaluator_episode=self.cfg.env.n_evaluator_episode,
            stop_value=self.cfg.env.stop_value,
            env=evaluator_env,
            policy=self.policy.eval_mode,
            tb_logger=tb_logger,
            exp_name=self.cfg.exp_name,
            policy_config=policy_config
        )

        # ==============================================================
        # Main loop
        # ==============================================================
        # Learner's before_run hook.
        learner.call_hook('before_run')

        if self.cfg.policy.update_per_collect is not None:
            update_per_collect = self.cfg.policy.update_per_collect

        # The purpose of collecting random data before training:
        # Exploration: Collecting random data helps the agent explore the environment and avoid getting stuck in a suboptimal policy prematurely.
        # Comparison: By observing the agent's performance during random action-taking, we can establish a baseline to evaluate the effectiveness of reinforcement learning algorithms.
        if self.cfg.policy.random_collect_episode_num > 0:
            random_collect(self.cfg.policy, self.policy, LightZeroRandomPolicy, collector, collector_env, replay_buffer)

        while True:
            log_buffer_memory_usage(learner.train_iter, replay_buffer, tb_logger)
            collect_kwargs = {}
            # set temperature for visit count distributions according to the train_iter,
            # please refer to Appendix D in MuZero paper for details.
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

            # Evaluate policy performance.
            if evaluator.should_eval(learner.train_iter):
                stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
                if stop:
                    break

            # Collect data by default config n_sample/n_episode.
            new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
            if self.cfg.policy.update_per_collect is None:
                # update_per_collect is None, then update_per_collect is set to the number of collected transitions multiplied by the model_update_ratio.
                collected_transitions_num = sum([len(game_segment) for game_segment in new_data[0]])
                update_per_collect = int(collected_transitions_num * self.cfg.policy.model_update_ratio)
            # save returned new_data collected by the collector
            replay_buffer.push_game_segments(new_data)
            # remove the oldest data if the replay buffer is full.
            replay_buffer.remove_oldest_data_to_fit()

            # Learn policy from collected data.
            for i in range(update_per_collect):
                # Learner will train ``update_per_collect`` times in one iteration.
                if replay_buffer.get_num_of_transitions() > batch_size:
                    train_data = replay_buffer.sample(batch_size, self.policy)
                else:
                    logging.warning(
                        f'The data in replay_buffer is not sufficient to sample a mini-batch: '
                        f'batch_size: {batch_size}, '
                        f'{replay_buffer} '
                        f'continue to collect now ....'
                    )
                    break

                # The core train steps for MCTS+RL algorithms.
                log_vars = learner.train(train_data, collector.envstep)

                if self.cfg.policy.use_priority:
                    replay_buffer.update_priority(train_data, log_vars[0]['value_priority_orig'])

            if collector.envstep >= step:
                break

        # Learner's after_run hook.
        learner.call_hook('after_run')

        return TrainingReturn(wandb_url=None)

    def deploy(
            self,
            enable_save_replay: bool = False,
            concatenate_all_replay: bool = False,
            replay_save_path: str = None,
            seed: Optional[Union[int, List]] = None,
            debug: bool = False
    ) -> EvalReturn:

        deply_configs = [self.evaluator_env_cfg[0]]

        if type(seed) == int:
            seed_list = [seed]
        elif seed:
            seed_list = seed
        else:
            seed_list = [0]

        reward_list = []

        if enable_save_replay:
            replay_save_path = replay_save_path if replay_save_path is not None else os.path.join(
                self.exp_name, 'videos'
            )
            deply_configs[0]['replay_path'] = replay_save_path

        for seed in seed_list:

            evaluator_env = create_env_manager(self.cfg.env.manager, [partial(self.env_fn, cfg=deply_configs[0])])

            evaluator_env.seed(seed if seed is not None else self.cfg.seed, dynamic_seed=False)
            set_pkg_seed(seed if seed is not None else self.cfg.seed, use_cuda=self.cfg.policy.cuda)

            # ==============================================================
            # MCTS+RL algorithms related core code
            # ==============================================================
            policy_config = self.cfg.policy

            evaluator = Evaluator(
                eval_freq=self.cfg.policy.eval_freq,
                n_evaluator_episode=1,
                stop_value=self.cfg.env.stop_value,
                env=evaluator_env,
                policy=self.policy.eval_mode,
                exp_name=self.cfg.exp_name,
                policy_config=policy_config
            )

            # ==============================================================
            # Main loop
            # ==============================================================

            stop, reward = evaluator.eval()
            reward_list.extend(reward['eval_episode_return'])

        if enable_save_replay:
            files = os.listdir(replay_save_path)
            files = [file for file in files if file.endswith('0.mp4')]
            files.sort()
            if concatenate_all_replay:
                # create a file named 'files.txt' to store the names of all mp4 files
                with open(os.path.join(replay_save_path, 'files.txt'), 'w') as f:
                    for file in files:
                        f.write("file '{}'\n".format(file))

                # combine all the mp4 files into one mp4 file, rename it as 'deploy.mp4'
                os.system(
                    'ffmpeg -f concat -safe 0 -i {} -c copy {}/deploy.mp4'.format(
                        os.path.join(replay_save_path, 'files.txt'), replay_save_path
                    )
                )

        return EvalReturn(eval_value=np.mean(reward_list), eval_value_std=np.std(reward_list))

    def batch_evaluate(
        self,
        n_evaluator_episode: int = None,
    ) -> EvalReturn:

        evaluator_env = create_env_manager(
            self.cfg.env.manager, [partial(self.env_fn, cfg=c) for c in self.evaluator_env_cfg]
        )

        evaluator_env.seed(self.cfg.seed, dynamic_seed=False)
        set_pkg_seed(self.cfg.seed, use_cuda=self.cfg.policy.cuda)

        # ==============================================================
        # MCTS+RL algorithms related core code
        # ==============================================================
        policy_config = self.cfg.policy

        evaluator = Evaluator(
            eval_freq=self.cfg.policy.eval_freq,
            n_evaluator_episode=self.cfg.env.n_evaluator_episode
            if n_evaluator_episode is None else n_evaluator_episode,
            stop_value=self.cfg.env.stop_value,
            env=evaluator_env,
            policy=self.policy.eval_mode,
            exp_name=self.cfg.exp_name,
            policy_config=policy_config
        )

        # ==============================================================
        # Main loop
        # ==============================================================

        stop, reward = evaluator.eval()

        return EvalReturn(
            eval_value=np.mean(reward['eval_episode_return']), eval_value_std=np.std(reward['eval_episode_return'])
        )

    @property
    def best(self):
        best_model_file_path = os.path.join(self.checkpoint_save_dir, "ckpt_best.pth.tar")
        # Load best model if it exists
        if os.path.exists(best_model_file_path):
            policy_state_dict = torch.load(best_model_file_path, map_location=torch.device("cpu"))
            self.policy.learn_mode.load_state_dict(policy_state_dict)
        return self
