import logging
import os
import threading
import queue
import time
from functools import partial
from typing import Tuple, Optional

import torch
import wandb
from ding.config import compile_config
from ding.envs import create_env_manager
from ding.envs import get_vec_env_setting
from ding.policy import create_policy
from ding.rl_utils import get_epsilon_greedy_fn
from ding.utils import EasyTimer
from ding.utils import set_pkg_seed, get_rank, get_world_size
from ding.worker import BaseLearner
from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from lzero.entry.utils import log_buffer_memory_usage
from lzero.policy import visit_count_temperature
from lzero.policy.random_policy import LightZeroRandomPolicy
from lzero.worker import MuZeroEvaluator as Evaluator
from lzero.worker import MuZeroSegmentCollector as Collector
from .utils import random_collect, calculate_update_per_collect

timer = EasyTimer()


class AsyncTrainer:
    """
    异步训练器，实现collector、learner、evaluator的并行执行
    """
    
    def __init__(self, cfg, create_cfg, model=None, model_path=None, max_train_iter=int(1e10), max_env_step=int(1e10)):
        self.cfg = cfg
        self.create_cfg = create_cfg
        self.model = model
        self.model_path = model_path
        self.max_train_iter = max_train_iter
        self.max_env_step = max_env_step
        
        # --- 优化: 使用配置中的队列大小 ---
        queue_size = getattr(self.cfg.policy, 'data_queue_size', 10)
        self.data_queue = queue.Queue(maxsize=queue_size)
        
        # 异步组件
        self.policy_lock = threading.Lock()
        self.stop_event = threading.Event()
        
        # --- 优化: 添加用于评估的事件通知机制 ---
        self.eval_event = threading.Event()
        
        # 训练状态
        self.train_iter = 0
        self.env_step = 0
        self.best_reward = float('-inf')
        
        # Buffer reanalyze相关状态
        self.buffer_reanalyze_count = 0
        self.train_epoch = 0
        self.reanalyze_batch_size = getattr(self.cfg.policy, 'reanalyze_batch_size', 2000)
        
        # 初始化组件
        self._init_components()
        
    def _init_components(self):
        """初始化训练组件"""
        try:
            self.cfg = compile_config(self.cfg, seed=0, env=None, auto=True, create_cfg=self.create_cfg, save_cfg=True)
            
            if self.cfg.policy.enable_async_debug_log:
                logging.info("[ASYNC_DEBUG] Configuration compiled successfully")
            
            env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(self.cfg.env)
            self.collector_env = create_env_manager(
                self.cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg]
            )
            self.evaluator_env = create_env_manager(
                self.cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg]
            )
            
            self.collector_env.seed(self.cfg.seed)
            self.evaluator_env.seed(self.cfg.seed, dynamic_seed=False)
            set_pkg_seed(self.cfg.seed, use_cuda=torch.cuda.is_available())
            
            if self.cfg.policy.enable_async_debug_log:
                logging.info("[ASYNC_DEBUG] Environments created and seeded successfully")
            
            self.policy = create_policy(self.cfg.policy, model=self.model, enable_field=['learn', 'collect', 'eval'])
            
            if not hasattr(self.policy, 'collect_mode') or self.policy.collect_mode is None:
                raise RuntimeError("Policy collect_mode is None after creation")
            if not hasattr(self.policy, 'eval_mode') or self.policy.eval_mode is None:
                raise RuntimeError("Policy eval_mode is None after creation")
            if not hasattr(self.policy, 'learn_mode') or self.policy.learn_mode is None:
                raise RuntimeError("Policy learn_mode is None after creation")
            
            if self.model_path is not None:
                self.policy.learn_mode.load_state_dict(torch.load(self.model_path, map_location=self.cfg.policy.device))
                # --- 优化: 初始化时就同步策略到collector和evaluator ---
                self.policy.collect_mode.load_state_dict(self.policy.learn_mode.state_dict())
                self.policy.eval_mode.load_state_dict(self.policy.learn_mode.state_dict())

            if self.cfg.policy.enable_async_debug_log:
                logging.info("[ASYNC_DEBUG] Policy created and validated successfully")
            
            self.tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(self.cfg.exp_name), 'async')) if get_rank() == 0 else None
            self.learner = BaseLearner(self.cfg.policy.learn.learner, self.policy.learn_mode, self.tb_logger, exp_name=self.cfg.exp_name)
            
            GameBuffer = getattr(__import__('lzero.mcts', fromlist=['UniZeroGameBuffer']), 'UniZeroGameBuffer')
            self.replay_buffer = GameBuffer(self.cfg.policy)
            
            if self.cfg.policy.enable_async_debug_log:
                logging.info("[ASYNC_DEBUG] Learner and replay buffer created successfully")
            
            self.collector = Collector(
                env=self.collector_env, 
                policy=self.policy.collect_mode, 
                tb_logger=self.tb_logger, 
                exp_name=self.cfg.exp_name,
                policy_config=self.cfg.policy
            )
            self.evaluator = Evaluator(
                eval_freq=self.cfg.policy.eval_freq,
                n_evaluator_episode=self.cfg.env.n_evaluator_episode,
                stop_value=self.cfg.env.stop_value,
                env=self.evaluator_env,
                policy=self.policy.eval_mode,
                tb_logger=self.tb_logger,
                exp_name=self.cfg.exp_name,
                policy_config=self.cfg.policy
            )
            
            if self.cfg.policy.enable_async_debug_log:
                logging.info("[ASYNC_DEBUG] Collector and evaluator created successfully")
            
            self.learner.call_hook('before_run')
            
            if self.cfg.policy.enable_async_debug_log:
                logging.info("[ASYNC_DEBUG] All components initialized successfully")
                
        except Exception as e:
            logging.error(f"[ASYNC_DEBUG] Component initialization error: {e}")
            import traceback
            logging.error(f"[ASYNC_DEBUG] Initialization error traceback: {traceback.format_exc()}")
            raise

    def _should_reanalyze_buffer(self, update_per_collect: int, training_iteration: int) -> bool:
        if not hasattr(self.cfg.policy, 'buffer_reanalyze_freq') or self.cfg.policy.buffer_reanalyze_freq <= 0:
            return False
        if self.cfg.policy.buffer_reanalyze_freq >= 1:
            reanalyze_interval = update_per_collect // self.cfg.policy.buffer_reanalyze_freq
            return training_iteration > 0 and training_iteration % reanalyze_interval == 0
        else:
            if self.train_epoch > 0 and self.train_epoch % int(1/self.cfg.policy.buffer_reanalyze_freq) == 0:
                min_transitions = int(self.reanalyze_batch_size / getattr(self.cfg.policy, 'reanalyze_partition', 0.75))
                return self.replay_buffer.get_num_of_transitions() // self.cfg.policy.num_unroll_steps > min_transitions
            return False

    def _perform_buffer_reanalyze(self):
        try:
            with timer:
                self.replay_buffer.reanalyze_buffer(self.reanalyze_batch_size, self.policy)
            self.buffer_reanalyze_count += 1
            if self.cfg.policy.enable_async_debug_log:
                logging.info(f"[ASYNC_DEBUG] Buffer reanalyze #{self.buffer_reanalyze_count} completed, "
                           f"time={timer.value:.3f}s, buffer_transitions={self.replay_buffer.get_num_of_transitions()}")
            else:
                logging.info(f'Buffer reanalyze count: {self.buffer_reanalyze_count}, time: {timer.value:.3f}s')
        except Exception as e:
            logging.error(f"[ASYNC_DEBUG] Buffer reanalyze error: {e}")
            import traceback
            logging.error(f"[ASYNC_DEBUG] Buffer reanalyze error traceback: {traceback.format_exc()}")
            
    def _collector_worker(self):
        if self.cfg.policy.enable_async_debug_log:
            logging.info("[ASYNC_DEBUG] Collector worker started")
        collection_count = 0
        while not self.stop_event.is_set():
            try:
                collection_start_time = time.time()
                
                collect_kwargs = {
                    'temperature': visit_count_temperature(
                        self.cfg.policy.manual_temperature_decay,
                        self.cfg.policy.fixed_temperature_value,
                        self.cfg.policy.threshold_training_steps_for_final_temperature,
                        trained_steps=self.train_iter
                    ),
                    'epsilon': 0.0
                }
                if self.cfg.policy.eps.eps_greedy_exploration_in_collect:
                    epsilon_greedy_fn = get_epsilon_greedy_fn(
                        start=self.cfg.policy.eps.start,
                        end=self.cfg.policy.eps.end,
                        decay=self.cfg.policy.eps.decay,
                        type_=self.cfg.policy.eps.type
                    )
                    collect_kwargs['epsilon'] = epsilon_greedy_fn(self.env_step)

                try:
                    if hasattr(self.policy._learn_model, 'world_model'):
                        
                        logging.info("[ASYNC_DEBUG] Evaluator clearing shared world model caches via self.policy.learn_mode...")
                        self.policy._learn_model.world_model.clear_caches()
                    else:
                        logging.warning("[ASYNC_DEBUG] Could not find clear_caches method on self.policy.learn_mode._model.world_model.")
                except Exception as e:
                    logging.error(f"[ASYNC_DEBUG] Error clearing evaluator caches: {e}", exc_info=True)
                
                new_data = self.collector.collect(train_iter=self.train_iter, policy_kwargs=collect_kwargs)
                
                if new_data is None or len(new_data) == 0:
                    if self.cfg.policy.enable_async_debug_log:
                        logging.warning("[ASYNC_DEBUG] Collector: collected data is None or empty, retrying...")
                    time.sleep(0.5) # 短暂等待以避免空转
                    continue
                        
                collection_time = time.time() - collection_start_time
                collection_count += 1
                
                try:
                    self.data_queue.put((new_data, self.env_step), timeout=5.0)
                except queue.Full:
                    if self.cfg.policy.enable_async_debug_log:
                        logging.warning("[ASYNC_DEBUG] Collector: data queue is full, dropping data")
                    continue
                
                self.env_step = self.collector.envstep
                
                if self.env_step >= self.max_env_step:
                    if self.cfg.policy.enable_async_debug_log:
                        logging.info(f"[ASYNC_DEBUG] Collector reached max_env_step: {self.env_step}")
                    self.stop_event.set()
                    break
                    
            except Exception as e:
                logging.error(f"[ASYNC_DEBUG] Collector worker error: {e}")
                import traceback
                logging.error(f"[ASYNC_DEBUG] Collector error traceback: {traceback.format_exc()}")
                time.sleep(1.0)
                continue
                
        if self.cfg.policy.enable_async_debug_log:
            logging.info(f"[ASYNC_DEBUG] Collector worker stopped, total collections: {collection_count}")
    
    def _learner_worker(self):
        """学习器工作线程"""
        if self.cfg.policy.enable_async_debug_log:
            logging.info("[ASYNC_DEBUG] Learner worker started")
        
        training_count = 0
        while not self.stop_event.is_set():
            try:
                try:
                    new_data, data_env_step = self.data_queue.get(timeout=1.0)
                except queue.Empty:
                    if self.stop_event.is_set():
                        break
                    continue

                self.replay_buffer.push_game_segments(new_data)
                self.replay_buffer.remove_oldest_data_to_fit()
                
                update_per_collect = calculate_update_per_collect(self.cfg, new_data, 1)
                
                batch_size = self.cfg.policy.batch_size
                for i in range(update_per_collect):
                    if self.stop_event.is_set():
                        break

                    if self._should_reanalyze_buffer(update_per_collect, i):
                        self._perform_buffer_reanalyze()
                    
                    if self.replay_buffer.get_num_of_transitions() > batch_size:
                        try:
                            train_data = self.replay_buffer.sample(batch_size, self.policy)
                            if train_data is None:
                                logging.warning("[ASYNC_DEBUG] Learner: sampled train_data is None")
                                break
                            train_data.append(self.train_iter)
                        except Exception as sample_error:
                            logging.error(f"[ASYNC_DEBUG] Learner sampling error: {sample_error}")
                            break

                        log_vars = self.learner.train(train_data, data_env_step)
                        self.train_iter += 1
                        training_count += 1
                        
                        if self.cfg.policy.use_priority:
                            self.replay_buffer.update_priority(train_data, log_vars[0]['value_priority_orig'])
                            
                        # --- 优化: 触发评估事件 ---
                        if self.evaluator.should_eval(self.train_iter):
                            if not self.eval_event.is_set():
                                if self.cfg.policy.enable_async_debug_log:
                                    logging.info(f"[ASYNC_DEBUG] Learner is signaling evaluator at train_iter: {self.train_iter}")
                                self.eval_event.set()

                    else:
                        break
                
                self.train_epoch += 1
                
                self.policy.recompute_pos_emb_diff_and_clear_cache()

                if self.train_iter >= self.max_train_iter:
                    self.stop_event.set()
                    break
                    
            except Exception as e:
                logging.error(f"[ASYNC_DEBUG] Learner worker error: {e}")
                import traceback
                logging.error(f"[ASYNC_DEBUG] Learner error traceback: {traceback.format_exc()}")
                time.sleep(2.0)
                continue
                
        if self.cfg.policy.enable_async_debug_log:
            logging.info(f"[ASYNC_DEBUG] Learner worker stopped, total training iterations: {training_count}")
    
    def _evaluator_worker(self):
        """评估器工作线程"""
        if self.cfg.policy.enable_async_debug_log:
            logging.info("[ASYNC_DEBUG] Evaluator worker started")
        
        while not self.stop_event.is_set():
            try:
                # --- 优化: 使用事件等待，而不是固定时间休眠 ---
                # 等待学习器发出评估信号，设置超时以防万一
                eval_triggered = self.eval_event.wait(timeout=self.cfg.policy.evaluator_check_interval)
                
                if eval_triggered:
                    self.eval_event.clear()  # 清除事件，等待下一次信号
                    
                    if self.cfg.policy.enable_async_debug_log:
                        logging.info(f"[ASYNC_DEBUG] Evaluator received signal, starting evaluation at train_iter: {self.train_iter}")
                    
                    try:
                        # 所有模式（learn, collect, eval）共享同一个模型实例，因此清除任何一个模式的缓存都会影响所有模式。
                        # import ipdb; ipdb.set_trace()
                        if hasattr(self.policy._learn_model, 'world_model'):
                            
                            logging.info("[ASYNC_DEBUG] Evaluator clearing shared world model caches via self.policy.learn_mode...")
                            self.policy._learn_model.world_model.clear_caches()
                        else:
                            logging.warning("[ASYNC_DEBUG] Could not find clear_caches method on self.policy.learn_mode._model.world_model.")
                    except Exception as e:
                        logging.error(f"[ASYNC_DEBUG] Error clearing evaluator caches: {e}", exc_info=True)

                    stop, episode_info = self.evaluator.eval(
                        self.learner.save_checkpoint, 
                        self.train_iter, 
                        self.env_step
                    )
                    
                    if episode_info is not None:
                        if isinstance(episode_info, dict) and 'eval_episode_return_mean' in episode_info:
                            reward = episode_info['eval_episode_return_mean']
                            if reward > self.best_reward:
                                self.best_reward = reward
                        else:
                            logging.warning(f"[ASYNC_DEBUG] Evaluator: unexpected episode_info format: {type(episode_info)}")
                            logging.warning(f"[ASYNC_DEBUG] Evaluator: episode_info: {episode_info}")
                    
                    if stop:
                        self.stop_event.set()
                        break
                else:
                    # 超时后，检查stop_event是否被设置
                    if self.stop_event.is_set():
                        break
            
            except Exception as e:
                logging.error(f"[ASYNC_DEBUG] Evaluator worker error: {e}")
                import traceback
                logging.error(f"[ASYNC_DEBUG] Evaluator error traceback: {traceback.format_exc()}")
                time.sleep(1.0)
                continue
                
        if self.cfg.policy.enable_async_debug_log:
            logging.info(f"[ASYNC_DEBUG] Evaluator worker stopped.")
    
    def train(self):
        """开始异步训练"""
        if self.cfg.policy.enable_async_debug_log:
            logging.info("[ASYNC_DEBUG] Starting async training...")
        
        collector_thread = threading.Thread(target=self._collector_worker, name="Collector")
        learner_thread = threading.Thread(target=self._learner_worker, name="Learner")
        evaluator_thread = threading.Thread(target=self._evaluator_worker, name="Evaluator")
        
        threads = [collector_thread, learner_thread, evaluator_thread]

        try:
            for t in threads:
                t.start()
            
            # 主线程可以监控线程状态或等待中断
            for t in threads:
                # 使用 join 来等待线程结束，可以设置一个很长的超时
                # 无限期等待，直到线程自己结束或被中断
                t.join()

        except KeyboardInterrupt:
            logging.info("[ASYNC_DEBUG] KeyboardInterrupt received, shutting down workers...")
        except Exception as e:
            logging.error(f"[ASYNC_DEBUG] Main thread error: {e}")
            import traceback
            logging.error(f"[ASYNC_DEBUG] Main thread error traceback: {traceback.format_exc()}")
        finally:
            # 确保设置停止事件，通知所有线程退出
            self.stop_event.set()
            # 唤醒可能在等待的评估器线程，使其能检查到stop_event
            self.eval_event.set()

            logging.info("[ASYNC_DEBUG] Waiting for worker threads to terminate...")
            for t in threads:
                if t.is_alive():
                    t.join(timeout=5.0) # 给5秒钟时间正常退出
                    if t.is_alive():
                        logging.warning(f"[ASYNC_DEBUG] Thread {t.name} did not terminate gracefully.")

            # --- 优化: 显式关闭资源 ---
            if self.tb_logger:
                self.tb_logger.close()
            
            try:
                self.learner.call_hook('after_run')
            except Exception as e:
                logging.error(f"[ASYNC_DEBUG] After run hook error: {e}")
            
            if self.cfg.policy.use_wandb:
                try:
                    wandb.finish()
                except Exception as e:
                    logging.error(f"[ASYNC_DEBUG] Wandb finish error: {e}")
            
            logging.info(f"[ASYNC_DEBUG] Async training finished. Final stats: "
                        f"train_iter={self.train_iter}, env_step={self.env_step}, best_reward={self.best_reward}")
        
        return self.policy

# train_unizero_segment_async 函数保持不变，因为它只是入口点
def train_unizero_segment_async(
        input_cfg: Tuple[dict, dict],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':
    cfg, create_cfg = input_cfg
    assert create_cfg.policy.type in ['unizero', 'sampled_unizero'], "train_unizero entry now only supports the following algo.: 'unizero', 'sampled_unizero'"
    trainer = AsyncTrainer(cfg, create_cfg, model, model_path, max_train_iter, max_env_step)
    return trainer.train()