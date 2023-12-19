import os
from functools import partial
from typing import Optional, Union, List

import numpy as np
import torch
from ding.bonus.common import TrainingReturn, EvalReturn
from ding.config import save_config_py, compile_config
from ding.envs import create_env_manager
from ding.envs import get_vec_env_setting
from ding.policy import create_policy
from ding.rl_utils import get_epsilon_greedy_fn
from ding.utils import set_pkg_seed, get_rank
from ding.worker import BaseLearner
from ditk import logging
from easydict import EasyDict
from tensorboardX import SummaryWriter

from lzero.agent.config.sampled_efficientzero import supported_env_cfg
from lzero.entry.utils import log_buffer_memory_usage, random_collect
from lzero.mcts import SampledEfficientZeroGameBuffer
from lzero.policy import visit_count_temperature
from lzero.policy.sampled_efficientzero import SampledEfficientZeroPolicy
from lzero.policy.random_policy import LightZeroRandomPolicy
from lzero.worker import MuZeroCollector as Collector
from lzero.worker import MuZeroEvaluator as Evaluator


class SampledEfficientZeroAgent:
    """
    Overview:
        Agent class for executing Sampled EfficientZero algorithms which include methods for training, deployment, and batch evaluation.
    Interfaces:
        __init__, train, deploy, batch_evaluate
    Properties:
        best

    .. note::
        This agent class is tailored for use with the HuggingFace Model Zoo for LightZero
        (e.g. https://huggingface.co/OpenDILabCommunity/CartPole-v0-SampledEfficientZero),
         and provides methods such as "train" and "deploy".
    """

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
        """
        Overview:
            Initialize the SampledEfficientZeroAgent instance with environment parameters, model, and configuration.
        Arguments:
            - env_id (:obj:`str`): Identifier for the environment to be used, registered in gym.
            - seed (:obj:`int`): Random seed for reproducibility. Defaults to 0.
            - exp_name (:obj:`Optional[str]`): Name for the experiment. Defaults to None.
            - model (:obj:`Optional[torch.nn.Module]`): PyTorch module to be used as the model. If None, a default model is created. Defaults to None.
            - cfg (:obj:`Optional[Union[EasyDict, dict]]`): Configuration for the agent. If None, default configuration will be used. Defaults to None.
            - policy_state_dict (:obj:`Optional[str]`): Path to a pre-trained model state dictionary. If provided, state dict will be loaded. Defaults to None.

        .. note::
            - If `env_id` is not specified, it must be included in `cfg`.
            - The `supported_env_list` contains all the environment IDs that are supported by this agent.
        """
        assert env_id is not None or cfg["main_config"]["env_id"] is not None, "Please specify env_id or cfg."

        if cfg is not None and not isinstance(cfg, EasyDict):
            cfg = EasyDict(cfg)

        if env_id is not None:
            assert env_id in SampledEfficientZeroAgent.supported_env_list, "Please use supported envs: {}".format(
                SampledEfficientZeroAgent.supported_env_list
            )
            if cfg is None:
                cfg = supported_env_cfg[env_id]
            else:
                assert cfg.main_config.env.env_id == env_id, "env_id in cfg should be the same as env_id in args."
        else:
            assert hasattr(cfg.main_config.env, "env_id"), "Please specify env_id in cfg."
            assert cfg.main_config.env.env_id in SampledEfficientZeroAgent.supported_env_list, "Please use supported envs: {}".format(
                SampledEfficientZeroAgent.supported_env_list
            )
        default_policy_config = EasyDict({"policy": SampledEfficientZeroPolicy.default_config()})
        default_policy_config.policy.update(cfg.main_config.policy)
        cfg.main_config.policy = default_policy_config.policy

        if exp_name is not None:
            cfg.main_config.exp_name = exp_name
        self.origin_cfg = cfg
        self.cfg = compile_config(
            cfg.main_config, seed=seed, env=None, auto=True, policy=SampledEfficientZeroPolicy, create_cfg=cfg.create_config
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
                from lzero.model.sampled_efficientzero_model_mlp import SampledEfficientZeroModelMLP
                model = SampledEfficientZeroModelMLP(**self.cfg.policy.model)
            elif self.cfg.policy.model.model_type == 'conv':
                from lzero.model.sampled_efficientzero_model import SampledEfficientZeroModel
                model = SampledEfficientZeroModel(**self.cfg.policy.model)
            else:
                raise NotImplementedError
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
        """
        Overview:
            Train the agent through interactions with the environment.
        Arguments:
            - step (:obj:`int`): Total number of environment steps to train for. Defaults to 10 million (1e7).
        Returns:
            - A `TrainingReturn` object containing training information, such as logs and potentially a URL to a training dashboard.
        .. note::
            The method involves interacting with the environment, collecting experience, and optimizing the model.
        """

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
        replay_buffer = SampledEfficientZeroGameBuffer(policy_config)
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
        """
        Overview:
            Deploy the agent for evaluation in the environment, with optional replay saving. The performance of the
            agent will be evaluated. Average return and standard deviation of the return will be returned. 
            If `enable_save_replay` is True, replay videos are saved in the specified `replay_save_path`.
        Arguments:
            - enable_save_replay (:obj:`bool`): Flag to enable saving of replay footage. Defaults to False.
            - concatenate_all_replay (:obj:`bool`): Whether to concatenate all replay videos into one file. Defaults to False.
            - replay_save_path (:obj:`Optional[str]`): Directory path to save replay videos. Defaults to None, which sets a default path.
            - seed (:obj:`Optional[Union[int, List[int]]]`): Seed or list of seeds for environment reproducibility. Defaults to None.
            - debug (:obj:`bool`): Whether to enable the debug mode. Default to False.
        Returns:
            - An `EvalReturn` object containing evaluation metrics such as mean and standard deviation of returns.
        """

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
            if not os.path.exists(replay_save_path):
                os.makedirs(replay_save_path)
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
        """
        Overview:
            Perform a batch evaluation of the agent over a specified number of episodes: ``n_evaluator_episode``.
        Arguments:
            - n_evaluator_episode (:obj:`Optional[int]`): Number of episodes to run the evaluation.
                If None, uses default value from configuration. Defaults to None.
        Returns:
            - An `EvalReturn` object with evaluation results such as mean and standard deviation of returns.

        .. note::
            This method evaluates the agent's performance across multiple episodes to gauge its effectiveness.
        """
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
        """
        Overview:
            Provides access to the best model according to evaluation metrics.
        Returns:
            - The agent with the best model loaded.

        .. note::
            The best model is saved in the path `./exp_name/ckpt/ckpt_best.pth.tar`.
            When this property is accessed, the agent instance will load the best model state.
        """

        best_model_file_path = os.path.join(self.checkpoint_save_dir, "ckpt_best.pth.tar")
        # Load best model if it exists
        if os.path.exists(best_model_file_path):
            policy_state_dict = torch.load(best_model_file_path, map_location=torch.device("cpu"))
            self.policy.learn_mode.load_state_dict(policy_state_dict)
        return self
