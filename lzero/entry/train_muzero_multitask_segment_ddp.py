import concurrent.futures
from ditk import logging
import os
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import Policy, create_policy
from ding.rl_utils import get_epsilon_greedy_fn
from ding.utils import EasyTimer, set_pkg_seed, get_rank, get_world_size
from ding.worker import BaseLearner
from tensorboardX import SummaryWriter

from lzero.entry.utils import log_buffer_memory_usage, allocate_batch_size, EVALUATION_TIMEOUT, safe_eval
from lzero.mcts import MuZeroGameBuffer as GameBuffer
from lzero.policy import visit_count_temperature
from lzero.worker import MuZeroCollector as Collector
from lzero.worker import MuZeroEvaluator as Evaluator

# ==========================
# Global Constants
# ==========================
# Note: This file uses a shorter timeout (1 hour) compared to other multitask files (200 minutes)
# You can adjust this value or use EVALUATION_TIMEOUT from utils.py instead
EVALUATION_TIMEOUT_SECONDS: int = 3600  # 1 hour
MAX_TRAIN_ITER_INF: int = int(1e10)
MAX_ENV_STEP_INF: int = int(1e10)


class MuZeroMultiTaskTrainer:
    """
    Overview:
        A trainer class to manage the multi-task training loop for MuZero.
        It encapsulates the state and logic for initialization, data collection,
        evaluation, training, and termination.
    """

    def __init__(
            self,
            input_cfg_list: List[Tuple[int, Tuple[dict, dict]]],
            seed: int,
            model: Optional[torch.nn.Module],
            model_path: Optional[str],
            max_train_iter: int,
            max_env_step: int,
    ) -> None:
        """
        Overview:
            Initializes the multi-task trainer.
        Arguments:
            - input_cfg_list (:obj:`List[Tuple[int, Tuple[dict, dict]]]`): Configs for all tasks.
            - seed (:obj:`int`): The base random seed.
            - model (:obj:`Optional[torch.nn.Module]`): An optional pre-defined model.
            - model_path (:obj:`Optional[str]`): Path to a pre-trained model checkpoint.
            - max_train_iter (:obj:`int`): Maximum training iterations.
            - max_env_step (:obj:`int`): Maximum environment steps.
        """
        self.max_train_iter = max_train_iter
        self.max_env_step = max_env_step
        self.seed = seed
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.timer = EasyTimer()

        # State variables
        self.train_epoch = 0
        self.buffer_reanalyze_count = 0
        self.value_priority_tasks = {}

        # Task partitioning
        self.tasks_for_this_rank = self._partition_tasks(input_cfg_list)
        if not self.tasks_for_this_rank:
            logging.warning(f"Rank {self.rank}: No tasks assigned. Process will run without tasks.")
            self.is_active = False
            return
        self.is_active = True

        # Initialize shared components (Policy, Learner)
        self.policy, self.learner, self.tb_logger = self._initialize_shared_components(model, model_path)

        # Initialize task-specific components
        (
            self.cfgs, self.game_buffers, self.collectors, self.evaluators
        ) = self._initialize_task_specific_components()

        self.update_per_collect = self.cfgs[0].policy.update_per_collect

    def _partition_tasks(self, input_cfg_list: List[Tuple[int, Tuple[dict, dict]]]) -> List[
        Tuple[int, Tuple[dict, dict]]]:
        """Partitions tasks among distributed processes."""
        total_tasks = len(input_cfg_list)
        tasks_per_rank = total_tasks // self.world_size
        remainder = total_tasks % self.world_size

        if self.rank < remainder:
            start_idx = self.rank * (tasks_per_rank + 1)
            end_idx = start_idx + tasks_per_rank + 1
        else:
            start_idx = self.rank * tasks_per_rank + remainder
            end_idx = start_idx + tasks_per_rank

        logging.info(f"Rank {self.rank}/{self.world_size} is assigned tasks from index {start_idx} to {end_idx - 1}.")
        return input_cfg_list[start_idx:end_idx]

    def _initialize_shared_components(self, model: Optional[torch.nn.Module], model_path: Optional[str]) -> Tuple[
        Policy, BaseLearner, SummaryWriter]:
        """Initializes components shared across all tasks on this rank."""
        _, [cfg, create_cfg] = self.tasks_for_this_rank[0]

        # Set task_num for the shared policy
        for task_config in self.tasks_for_this_rank:
            task_config[1][0].policy.task_num = len(self.tasks_for_this_rank)

        cfg.policy.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        compiled_cfg = compile_config(cfg, seed=self.seed, auto=True, create_cfg=create_cfg, save_cfg=True)

        policy = create_policy(compiled_cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])

        if model_path:
            logging.info(f'Loading model from {model_path}...')
            policy.learn_mode.load_state_dict(torch.load(model_path, map_location=compiled_cfg.policy.device))
            logging.info(f'Model loaded successfully from {model_path}.')

        log_dir = os.path.join(f'./{compiled_cfg.exp_name}/log', f'serial_rank_{self.rank}')
        tb_logger = SummaryWriter(log_dir)
        learner = BaseLearner(compiled_cfg.policy.learn.learner, policy.learn_mode, tb_logger,
                              exp_name=compiled_cfg.exp_name)
        return policy, learner, tb_logger

    def _initialize_task_specific_components(self) -> Tuple[List, List, List, List]:
        """Initializes components for each task assigned to this rank."""
        cfgs, game_buffers, collectors, evaluators = [], [], [], []

        for local_task_id, (task_id, [cfg, create_cfg]) in enumerate(self.tasks_for_this_rank):
            task_seed = self.seed + task_id
            cfg.policy.device = 'cuda' if cfg.policy.cuda and torch.cuda.is_available() else 'cpu'
            compiled_cfg = compile_config(cfg, seed=task_seed, auto=True, create_cfg=create_cfg, save_cfg=True)

            # Create environments
            env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(compiled_cfg.env)
            collector_env = create_env_manager(compiled_cfg.env.manager,
                                               [partial(env_fn, cfg=c) for c in collector_env_cfg])
            evaluator_env = create_env_manager(compiled_cfg.env.manager,
                                               [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
            collector_env.seed(task_seed)
            evaluator_env.seed(task_seed, dynamic_seed=False)
            set_pkg_seed(task_seed, use_cuda=compiled_cfg.policy.cuda)

            # Create buffer, collector, and evaluator
            replay_buffer = GameBuffer(compiled_cfg.policy)
            # Set initial batch size from config
            replay_buffer.batch_size = compiled_cfg.policy.batch_size[task_id]

            collector = Collector(
                env=collector_env,
                policy=self.policy.collect_mode,
                tb_logger=self.tb_logger,
                exp_name=compiled_cfg.exp_name,
                policy_config=compiled_cfg.policy,
                task_id=task_id
            )
            evaluator = Evaluator(
                eval_freq=compiled_cfg.policy.eval_freq,
                n_evaluator_episode=compiled_cfg.env.n_evaluator_episode,
                stop_value=compiled_cfg.env.stop_value,
                env=evaluator_env,
                policy=self.policy.eval_mode,
                tb_logger=self.tb_logger,
                exp_name=compiled_cfg.exp_name,
                policy_config=compiled_cfg.policy,
                task_id=task_id
            )

            cfgs.append(compiled_cfg)
            game_buffers.append(replay_buffer)
            collectors.append(collector)
            evaluators.append(evaluator)

        return cfgs, game_buffers, collectors, evaluators

    def run(self) -> Policy:
        """
        Overview:
            The main training loop. Executes collection, evaluation, and training steps
            until a termination condition is met.
        Returns:
            - (:obj:`Policy`): The trained policy.
        """
        if not self.is_active:
            # This rank has no tasks, so it should wait for others to finish.
            self._wait_for_termination()
            return self.policy

        self.learner.call_hook('before_run')

        while True:
            torch.cuda.empty_cache()

            self._update_dynamic_batch_sizes()
            self._collect_and_evaluate()

            if self._is_training_ready():
                dist.barrier()
                self._train_iteration()
                dist.barrier()
            else:
                logging.warning(f"Rank {self.rank}: Not enough data for training, skipping training step.")

            if self._check_termination_conditions():
                dist.barrier()  # Final barrier to ensure all processes stop together.
                break

        self.learner.call_hook('after_run')
        return self.policy

    def _update_dynamic_batch_sizes(self) -> None:
        """Dynamically allocates batch sizes if enabled in the config."""
        if not self.cfgs[0].policy.get('allocated_batch_sizes', False):
            return

        # Linearly increase clip_scale from 1 to 4 as train_epoch goes from 0 to 1000.
        clip_scale = np.clip(1 + (3 * self.train_epoch / 1000), 1, 4)
        allocated_sizes = allocate_batch_size(self.cfgs, self.game_buffers, alpha=1.0, clip_scale=clip_scale)

        # Distribute the allocated sizes to the tasks on the current rank.
        # This requires knowing the global task distribution.
        total_tasks = self.world_size * len(self.tasks_for_this_rank) # Approximation, needs exact count
        # This part is tricky in a distributed setting without global knowledge of task indices.
        # Assuming the allocation order matches the task_id order.
        for i, cfg in enumerate(self.cfgs):
            task_id = cfg.policy.task_id
            if task_id < len(allocated_sizes):
                batch_size = allocated_sizes[task_id]
                cfg.policy.batch_size = batch_size
                # Also update the batch size in the shared policy config if necessary
                self.policy._cfg.batch_size[task_id] = batch_size


    def _collect_and_evaluate(self) -> None:
        """Runs the data collection and evaluation loop for each assigned task."""
        for i, (cfg, collector, evaluator, replay_buffer) in enumerate(
                zip(self.cfgs, self.collectors, self.evaluators, self.game_buffers)):
            log_buffer_memory_usage(self.learner.train_iter, replay_buffer, self.tb_logger, cfg.policy.task_id)

            # Evaluation step
            if evaluator.should_eval(self.learner.train_iter):
                safe_eval(evaluator, self.learner, collector, self.rank, self.world_size,
                         timeout=EVALUATION_TIMEOUT_SECONDS)

            # Collection step
            self._collect_data_for_task(cfg, collector, replay_buffer)

    def _collect_data_for_task(self, cfg: Any, collector: Collector, replay_buffer: GameBuffer) -> None:
        """Collects data for a single task and pushes it to the replay buffer."""
        policy_config = cfg.policy
        collect_kwargs = {
            'temperature': visit_count_temperature(
                policy_config.manual_temperature_decay,
                policy_config.fixed_temperature_value,
                policy_config.threshold_training_steps_for_final_temperature,
                trained_steps=self.learner.train_iter
            ),
            'epsilon': 0.0
        }
        if policy_config.eps.eps_greedy_exploration_in_collect:
            epsilon_fn = get_epsilon_greedy_fn(
                start=policy_config.eps.start, end=policy_config.eps.end,
                decay=policy_config.eps.decay, type_=policy_config.eps.type
            )
            collect_kwargs['epsilon'] = epsilon_fn(collector.envstep)

        logging.info(f'Rank {self.rank}: Collecting data for task {cfg.policy.task_id}...')
        new_data = collector.collect(train_iter=self.learner.train_iter, policy_kwargs=collect_kwargs)
        replay_buffer.push_game_segments(new_data)
        replay_buffer.remove_oldest_data_to_fit()
        logging.info(f'Rank {self.rank}: Finished data collection for task {cfg.policy.task_id}.')

        # Periodic reanalysis of the buffer
        self._reanalyze_buffer_if_needed(cfg, replay_buffer, is_during_training=False)

    def _reanalyze_buffer_if_needed(self, cfg: Any, replay_buffer: GameBuffer, is_during_training: bool,
                                    train_loop_idx: int = 0) -> None:
        """Handles the logic for reanalyzing the game buffer."""
        policy_config = cfg.policy
        reanalyze_freq = policy_config.buffer_reanalyze_freq
        reanalyze_batch_size = policy_config.reanalyze_batch_size
        reanalyze_partition = policy_config.reanalyze_partition
        update_per_collect = policy_config.update_per_collect

        should_reanalyze = False
        if reanalyze_freq >= 1:
            reanalyze_interval = update_per_collect // reanalyze_freq
            if is_during_training and train_loop_idx % reanalyze_interval == 0:
                should_reanalyze = True
        else: # reanalyze_freq is a fraction, e.g., 0.1
            if not is_during_training and self.train_epoch % int(1 / reanalyze_freq) == 0:
                should_reanalyze = True

        if should_reanalyze and replay_buffer.get_num_of_transitions() // policy_config.num_unroll_steps > int(reanalyze_batch_size / reanalyze_partition):
            with self.timer:
                replay_buffer.reanalyze_buffer(reanalyze_batch_size, self.policy)
            self.buffer_reanalyze_count += 1
            logging.info(f'Buffer reanalyze count: {self.buffer_reanalyze_count}, Time: {self.timer.value:.2f}s')

    def _is_training_ready(self) -> bool:
        """Checks if there is enough data in all buffers to start training."""
        for cfg, buffer in zip(self.cfgs, self.game_buffers):
            if buffer.get_num_of_transitions() < cfg.policy.batch_size[cfg.policy.task_id]:
                logging.warning(f"Rank {self.rank}, Task {cfg.policy.task_id}: Not enough data. "
                                f"Required: {cfg.policy.batch_size[cfg.policy.task_id]}, "
                                f"Available: {buffer.get_num_of_transitions()}")
                return False
        return True

    def _train_iteration(self) -> None:
        """Performs one full training iteration, consisting of multiple updates."""
        for i in range(self.update_per_collect):
            train_data_multi_task = []
            envstep_multi_task = 0

            for idx, (cfg, collector, replay_buffer) in enumerate(
                    zip(self.cfgs, self.collectors, self.game_buffers)):
                envstep_multi_task += collector.envstep
                batch_size = cfg.policy.batch_size[cfg.policy.task_id]

                if replay_buffer.get_num_of_transitions() > batch_size:
                    self._reanalyze_buffer_if_needed(cfg, replay_buffer, is_during_training=True, train_loop_idx=i)
                    train_data = replay_buffer.sample(batch_size, self.policy)
                    train_data.append(cfg.policy.task_id)  # Append task_id for multi-task loss
                    train_data_multi_task.append(train_data)
                else:
                    # This case should ideally be prevented by _is_training_ready
                    logging.warning(f"Skipping sample for task {cfg.policy.task_id} due to insufficient data.")
                    train_data_multi_task.clear() # Invalidate the whole batch if one task fails
                    break
            
            if train_data_multi_task:
                log_vars = self.learner.train(train_data_multi_task, envstep_multi_task)
                if self.cfgs[0].policy.use_priority:
                    self._update_priorities(train_data_multi_task, log_vars)
        
        self.train_epoch += 1

    def _update_priorities(self, train_data_multi_task: List, log_vars: List[Dict]) -> None:
        """Updates the priorities in the replay buffers after a training step."""
        for idx, (cfg, replay_buffer) in enumerate(zip(self.cfgs, self.game_buffers)):
            task_id = cfg.policy.task_id
            priority_key = f'value_priority_task{task_id}'
            
            if priority_key in log_vars[0]:
                priorities = log_vars[0][priority_key]
                replay_buffer.update_priority(train_data_multi_task[idx], priorities)

                # Log priority statistics
                if cfg.policy.get('print_task_priority_logs', False):
                    mean_priority = np.mean(priorities)
                    std_priority = np.std(priorities)
                    
                    # Update running mean of priority
                    running_mean_key = f'running_mean_priority_task{task_id}'
                    alpha = 0.1  # Smoothing factor for running average
                    if running_mean_key not in self.value_priority_tasks:
                        self.value_priority_tasks[running_mean_key] = mean_priority
                    else:
                        self.value_priority_tasks[running_mean_key] = \
                            alpha * mean_priority + (1 - alpha) * self.value_priority_tasks[running_mean_key]
                    
                    running_mean_priority = self.value_priority_tasks[running_mean_key]
                    logging.info(
                        f"Task {task_id} - Priority Stats: Mean={mean_priority:.6f}, "
                        f"Running Mean={running_mean_priority:.6f}, Std={std_priority:.6f}"
                    )

    def _check_termination_conditions(self) -> bool:
        """Checks if the training should be terminated based on env steps or train iterations."""
        try:
            # Check max_env_step
            local_envsteps = [collector.envstep for collector in self.collectors]
            all_ranks_envsteps = [None for _ in range(self.world_size)]
            dist.all_gather_object(all_ranks_envsteps, local_envsteps)
            
            # Flatten and check if all tasks have reached the step limit
            all_envsteps = [step for rank_steps in all_ranks_envsteps for step in rank_steps]
            if all(step >= self.max_env_step for step in all_envsteps):
                logging.info(f"Rank {self.rank}: All tasks reached max_env_step ({self.max_env_step}). Terminating.")
                return True

            # Check max_train_iter
            local_train_iter = torch.tensor([self.learner.train_iter], device=self.policy.device)
            all_train_iters = [torch.zeros_like(local_train_iter) for _ in range(self.world_size)]
            dist.all_gather(all_train_iters, local_train_iter)
            
            if any(it.item() >= self.max_train_iter for it in all_train_iters):
                logging.info(f"Rank {self.rank}: A process reached max_train_iter ({self.max_train_iter}). Terminating.")
                return True

        except Exception as e:
            logging.error(f'Rank {self.rank}: Failed during termination check. Error: {e}', exc_info=True)
            return True # Terminate on error to prevent hanging

        return False

    def _wait_for_termination(self) -> None:
        """
        For inactive ranks, this method blocks and waits for a termination signal
        (e.g., another rank finishing) by participating in barriers and termination checks.
        """
        while True:
            # Participate in barriers to stay in sync
            dist.barrier() # Pre-train barrier
            dist.barrier() # Post-train barrier

            if self._check_termination_conditions():
                dist.barrier() # Final barrier
                break

def train_muzero_multitask_segment_ddp(
        input_cfg_list: List[Tuple[int, Tuple[dict, dict]]],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = MAX_TRAIN_ITER_INF,
        max_env_step: Optional[int] = MAX_ENV_STEP_INF,
) -> Policy:
    """
    Overview:
        The main entry point for multi-task MuZero training using Distributed Data Parallel (DDP).
        This function sets up the distributed environment, partitions tasks, and launches the training process,
        which is managed by the MuZeroMultiTaskTrainer class.
    Arguments:
        - input_cfg_list (:obj:`List[Tuple[int, Tuple[dict, dict]]]`): A list of tuples, where each tuple contains
          a task ID and its corresponding configuration dictionaries (main_config, create_config).
        - seed (:obj:`int`): The base random seed for reproducibility. Defaults to 0.
        - model (:obj:`Optional[torch.nn.Module]`): An optional pre-defined model instance. If provided,
          it will be used instead of creating a new one from the config. Defaults to None.
        - model_path (:obj:`Optional[str]`): Path to a pre-trained model checkpoint file. If provided,
          the model weights will be loaded before training starts. Defaults to None.
        - max_train_iter (:obj:`Optional[int]`): The maximum number of training iterations.
          Training will stop if any process reaches this limit. Defaults to a very large number.
        - max_env_step (:obj:`Optional[int]`): The maximum number of environment steps for each task.
          Training will stop when all tasks have reached this limit. Defaults to a very large number.
    Returns:
        - (:obj:`Policy`): The final trained policy instance from the primary rank.
    """
    # Initialize the trainer, which handles all the complex setup and logic internally.
    trainer = MuZeroMultiTaskTrainer(
        input_cfg_list=input_cfg_list,
        seed=seed,
        model=model,
        model_path=model_path,
        max_train_iter=max_train_iter,
        max_env_step=max_env_step,
    )

    # Run the training loop and return the trained policy.
    return trainer.run()
