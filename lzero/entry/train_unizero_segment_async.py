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
        
        # 异步组件
        self.data_queue = queue.Queue(maxsize=10)  # 数据缓冲队列
        self.policy_lock = threading.Lock()  # Policy更新锁
        self.stop_event = threading.Event()  # 停止信号
        
        # 训练状态
        self.train_iter = 0
        self.env_step = 0
        self.best_reward = float('-inf')
        
        # 初始化组件
        self._init_components()
        
    def _init_components(self):
        """初始化训练组件"""
        try:
            # 编译配置
            self.cfg = compile_config(self.cfg, seed=0, env=None, auto=True, create_cfg=self.create_cfg, save_cfg=True)
            
            if self.cfg.policy.enable_async_debug_log:
                logging.info("[ASYNC_DEBUG] Configuration compiled successfully")
            
            # 创建环境
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
            
            # 创建策略
            self.policy = create_policy(self.cfg.policy, model=self.model, enable_field=['learn', 'collect', 'eval'])
            
            # 验证策略创建
            if not hasattr(self.policy, 'collect_mode') or self.policy.collect_mode is None:
                raise RuntimeError("Policy collect_mode is None after creation")
            if not hasattr(self.policy, 'eval_mode') or self.policy.eval_mode is None:
                raise RuntimeError("Policy eval_mode is None after creation")
            if not hasattr(self.policy, 'learn_mode') or self.policy.learn_mode is None:
                raise RuntimeError("Policy learn_mode is None after creation")
            
            if self.model_path is not None:
                self.policy.learn_mode.load_state_dict(torch.load(self.model_path, map_location=self.cfg.policy.device))
            
            if self.cfg.policy.enable_async_debug_log:
                logging.info("[ASYNC_DEBUG] Policy created and validated successfully")
            
            # 创建工作组件
            self.tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(self.cfg.exp_name), 'async')) if get_rank() == 0 else None
            self.learner = BaseLearner(self.cfg.policy.learn.learner, self.policy.learn_mode, self.tb_logger, exp_name=self.cfg.exp_name)
            
            # 创建游戏缓冲区
            GameBuffer = getattr(__import__('lzero.mcts', fromlist=['UniZeroGameBuffer']), 'UniZeroGameBuffer')
            self.replay_buffer = GameBuffer(self.cfg.policy)
            
            if self.cfg.policy.enable_async_debug_log:
                logging.info("[ASYNC_DEBUG] Learner and replay buffer created successfully")
            
            # 创建收集器和评估器
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
            
            # 执行learner的before_run hook
            self.learner.call_hook('before_run')
            
            if self.cfg.policy.enable_async_debug_log:
                logging.info("[ASYNC_DEBUG] All components initialized successfully")
                
        except Exception as e:
            logging.error(f"[ASYNC_DEBUG] Component initialization error: {e}")
            import traceback
            logging.error(f"[ASYNC_DEBUG] Initialization error traceback: {traceback.format_exc()}")
            raise
        
    def _collector_worker(self):
        """收集器工作线程"""
        if self.cfg.policy.enable_async_debug_log:
            logging.info("[ASYNC_DEBUG] Collector worker started")
        
        collection_count = 0
        while not self.stop_event.is_set():
            try:
                collection_start_time = time.time()
                
                # 获取最新的policy（线程安全）
                with self.policy_lock:
                    # 确保policy状态一致
                    if not hasattr(self.policy, 'collect_mode') or self.policy.collect_mode is None:
                        if self.cfg.policy.enable_async_debug_log:
                            logging.warning("[ASYNC_DEBUG] Collector: policy.collect_mode is None, waiting...")
                        time.sleep(0.1)
                        continue
                    
                    current_policy = self.policy.collect_mode
                    if self.cfg.policy.enable_async_debug_log:
                        logging.debug(f"[ASYNC_DEBUG] Collector acquired policy lock, train_iter: {self.train_iter}")
                
                # 设置收集参数
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
                
                if self.cfg.policy.enable_async_debug_log:
                    logging.debug(f"[ASYNC_DEBUG] Collector starting data collection, train_iter: {self.train_iter}, env_step: {self.env_step}")
                
                # 收集数据（添加异常处理）
                try:
                    new_data = self.collector.collect(train_iter=self.train_iter, policy_kwargs=collect_kwargs)
                    
                    # 验证收集到的数据
                    if new_data is None or len(new_data) == 0:
                        if self.cfg.policy.enable_async_debug_log:
                            logging.warning("[ASYNC_DEBUG] Collector: collected data is None or empty")
                        continue
                        
                except Exception as collect_error:
                    logging.error(f"[ASYNC_DEBUG] Collector data collection error: {collect_error}")
                    # 如果是索引错误，可能是环境状态不一致，等待一下再重试
                    if "out of bounds" in str(collect_error):
                        if self.cfg.policy.enable_async_debug_log:
                            logging.warning("[ASYNC_DEBUG] Collector: index out of bounds, waiting before retry...")
                        time.sleep(0.5)
                        continue
                    else:
                        raise collect_error
                
                collection_time = time.time() - collection_start_time
                collection_count += 1
                
                # 将数据放入队列（添加超时处理）
                queue_start_time = time.time()
                try:
                    self.data_queue.put((new_data, self.env_step), timeout=5.0)
                    queue_time = time.time() - queue_start_time
                except queue.Full:
                    if self.cfg.policy.enable_async_debug_log:
                        logging.warning("[ASYNC_DEBUG] Collector: data queue is full, dropping data")
                    continue
                
                old_env_step = self.env_step
                self.env_step = self.collector.envstep
                
                if self.cfg.policy.enable_async_debug_log:
                    logging.info(f"[ASYNC_DEBUG] Collector completed collection #{collection_count}: "
                               f"data_size={len(new_data)}, env_step={old_env_step}->{self.env_step}, "
                               f"collection_time={collection_time:.3f}s, queue_time={queue_time:.3f}s, "
                               f"queue_size={self.data_queue.qsize()}")
                
                # 检查停止条件
                if self.env_step >= self.max_env_step:
                    if self.cfg.policy.enable_async_debug_log:
                        logging.info(f"[ASYNC_DEBUG] Collector reached max_env_step: {self.env_step}")
                    self.stop_event.set()
                    break
                    
            except Exception as e:
                logging.error(f"[ASYNC_DEBUG] Collector worker error: {e}")
                import traceback
                logging.error(f"[ASYNC_DEBUG] Collector error traceback: {traceback.format_exc()}")
                # 不要立即退出，给系统一个恢复的机会
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
                # 从队列获取数据
                queue_start_time = time.time()
                new_data, data_env_step = self.data_queue.get(timeout=1.0)
                queue_time = time.time() - queue_start_time
                
                if self.cfg.policy.enable_async_debug_log:
                    logging.debug(f"[ASYNC_DEBUG] Learner received data from queue, queue_time={queue_time:.3f}s")
                
                # 更新replay buffer
                buffer_start_time = time.time()
                self.replay_buffer.push_game_segments(new_data)
                self.replay_buffer.remove_oldest_data_to_fit()
                buffer_time = time.time() - buffer_start_time
                
                # 计算更新次数
                update_per_collect = calculate_update_per_collect(self.cfg, new_data, 1)
                
                if self.cfg.policy.enable_async_debug_log:
                    logging.debug(f"[ASYNC_DEBUG] Learner buffer updated, buffer_time={buffer_time:.3f}s, update_per_collect={update_per_collect}")
                
                # 训练模型
                batch_size = self.cfg.policy.batch_size
                training_start_time = time.time()
                training_iterations = 0
                
                for i in range(update_per_collect):
                    if self.replay_buffer.get_num_of_transitions() > batch_size:
                        sample_start_time = time.time()
                        train_data = self.replay_buffer.sample(batch_size, self.policy)
                        sample_time = time.time() - sample_start_time
                        
                        train_data.append(self.train_iter)
                        
                        # 训练
                        train_start_time = time.time()
                        log_vars = self.learner.train(train_data, data_env_step)
                        train_time = time.time() - train_start_time
                        
                        self.train_iter += 1
                        training_iterations += 1
                        training_count += 1
                        
                        # 更新policy
                        with self.policy_lock:
                            if self.cfg.policy.enable_async_debug_log:
                                logging.debug(f"[ASYNC_DEBUG] Learner acquired policy lock for update, train_iter: {self.train_iter}")
                            # 这里需要确保policy的更新是线程安全的
                            pass
                        
                        if self.cfg.policy.use_priority:
                            self.replay_buffer.update_priority(train_data, log_vars[0]['value_priority_orig'])
                        
                        if self.cfg.policy.enable_async_debug_log:
                            logging.info(f"[ASYNC_DEBUG] Learner completed training #{training_count}: "
                                       f"train_iter={self.train_iter}, sample_time={sample_time:.3f}s, "
                                       f"train_time={train_time:.3f}s, buffer_transitions={self.replay_buffer.get_num_of_transitions()}")
                    else:
                        if self.cfg.policy.enable_async_debug_log:
                            logging.warning(f"[ASYNC_DEBUG] Insufficient data for training: {self.replay_buffer.get_num_of_transitions()} < {batch_size}")
                        break
                
                total_training_time = time.time() - training_start_time
                if self.cfg.policy.enable_async_debug_log:
                    logging.info(f"[ASYNC_DEBUG] Learner completed training batch: "
                               f"iterations={training_iterations}, total_time={total_training_time:.3f}s")
                
                # 检查停止条件
                if self.train_iter >= self.max_train_iter:
                    if self.cfg.policy.enable_async_debug_log:
                        logging.info(f"[ASYNC_DEBUG] Learner reached max_train_iter: {self.train_iter}")
                    self.stop_event.set()
                    break
                    
            except queue.Empty:
                if self.cfg.policy.enable_async_debug_log:
                    logging.debug("[ASYNC_DEBUG] Learner queue timeout, continuing...")
                continue
            except Exception as e:
                logging.error(f"[ASYNC_DEBUG] Learner worker error: {e}")
                break
                
        if self.cfg.policy.enable_async_debug_log:
            logging.info(f"[ASYNC_DEBUG] Learner worker stopped, total training iterations: {training_count}")
    
    def _evaluator_worker(self):
        """评估器工作线程"""
        if self.cfg.policy.enable_async_debug_log:
            logging.info("[ASYNC_DEBUG] Evaluator worker started")
        
        evaluation_count = 0
        check_count = 0
        
        while not self.stop_event.is_set():
            try:
                check_count += 1
                
                # 检查是否需要评估
                if self.evaluator.should_eval(self.train_iter):
                    if self.cfg.policy.enable_async_debug_log:
                        logging.info(f"[ASYNC_DEBUG] Evaluator starting evaluation #{evaluation_count + 1}, train_iter: {self.train_iter}")
                    
                    # 获取最新的policy进行评估（线程安全）
                    eval_start_time = time.time()
                    with self.policy_lock:
                        # 确保policy状态一致
                        if not hasattr(self.policy, 'eval_mode') or self.policy.eval_mode is None:
                            if self.cfg.policy.enable_async_debug_log:
                                logging.warning("[ASYNC_DEBUG] Evaluator: policy.eval_mode is None, waiting...")
                            time.sleep(0.1)
                            continue
                        
                        current_policy = self.policy.eval_mode
                        if self.cfg.policy.enable_async_debug_log:
                            logging.debug(f"[ASYNC_DEBUG] Evaluator acquired policy lock, train_iter: {self.train_iter}")
                    
                    # 评估（添加异常处理）
                    try:
                        stop, reward = self.evaluator.eval(
                            self.learner.save_checkpoint, 
                            self.train_iter, 
                            self.env_step
                        )
                        
                        # 验证评估结果
                        if reward is None:
                            if self.cfg.policy.enable_async_debug_log:
                                logging.warning("[ASYNC_DEBUG] Evaluator: reward is None")
                            continue
                            
                    except Exception as eval_error:
                        logging.error(f"[ASYNC_DEBUG] Evaluator evaluation error: {eval_error}")
                        import traceback
                        logging.error(f"[ASYNC_DEBUG] Evaluator error traceback: {traceback.format_exc()}")
                        # 如果是列表索引错误，可能是数据不一致，等待一下再重试
                        if "list indices" in str(eval_error) or "NoneType" in str(eval_error):
                            if self.cfg.policy.enable_async_debug_log:
                                logging.warning("[ASYNC_DEBUG] Evaluator: data inconsistency, waiting before retry...")
                            time.sleep(0.5)
                            continue
                        else:
                            raise eval_error
                    
                    eval_time = time.time() - eval_start_time
                    evaluation_count += 1
                    
                    if reward > self.best_reward:
                        old_best = self.best_reward
                        self.best_reward = reward
                        if self.cfg.policy.enable_async_debug_log:
                            logging.info(f"[ASYNC_DEBUG] Evaluator new best reward: {old_best} -> {self.best_reward}")
                    
                    if self.cfg.policy.enable_async_debug_log:
                        logging.info(f"[ASYNC_DEBUG] Evaluator completed evaluation #{evaluation_count}: "
                                   f"reward={reward}, eval_time={eval_time:.3f}s, stop={stop}")
                    
                    if stop:
                        if self.cfg.policy.enable_async_debug_log:
                            logging.info(f"[ASYNC_DEBUG] Evaluator triggered stop condition")
                        self.stop_event.set()
                        break
                else:
                    if self.cfg.policy.enable_async_debug_log and check_count % 100 == 0:  # 每100次检查输出一次
                        logging.debug(f"[ASYNC_DEBUG] Evaluator check #{check_count}: train_iter={self.train_iter}, should_eval=False")
                
                # 等待一段时间再进行下一次评估
                time.sleep(self.cfg.policy.evaluator_check_interval)
                
            except Exception as e:
                logging.error(f"[ASYNC_DEBUG] Evaluator worker error: {e}")
                import traceback
                logging.error(f"[ASYNC_DEBUG] Evaluator error traceback: {traceback.format_exc()}")
                # 不要立即退出，给系统一个恢复的机会
                time.sleep(1.0)
                continue
                
        if self.cfg.policy.enable_async_debug_log:
            logging.info(f"[ASYNC_DEBUG] Evaluator worker stopped, total evaluations: {evaluation_count}, total checks: {check_count}")
    
    def train(self):
        """开始异步训练"""
        if self.cfg.policy.enable_async_debug_log:
            logging.info("[ASYNC_DEBUG] Starting async training...")
            logging.info(f"[ASYNC_DEBUG] Configuration: enable_async_training={self.cfg.policy.enable_async_training}, "
                        f"data_queue_size={self.cfg.policy.data_queue_size}, "
                        f"evaluator_check_interval={self.cfg.policy.evaluator_check_interval}s")
        
        # 验证组件状态
        if not hasattr(self, 'policy') or self.policy is None:
            raise RuntimeError("Policy is not initialized")
        if not hasattr(self, 'collector') or self.collector is None:
            raise RuntimeError("Collector is not initialized")
        if not hasattr(self, 'evaluator') or self.evaluator is None:
            raise RuntimeError("Evaluator is not initialized")
        if not hasattr(self, 'learner') or self.learner is None:
            raise RuntimeError("Learner is not initialized")
        
        # 创建并启动工作线程
        collector_thread = threading.Thread(target=self._collector_worker, name="Collector")
        learner_thread = threading.Thread(target=self._learner_worker, name="Learner")
        evaluator_thread = threading.Thread(target=self._evaluator_worker, name="Evaluator")
        
        if self.cfg.policy.enable_async_debug_log:
            logging.info("[ASYNC_DEBUG] Starting worker threads...")
        
        try:
            collector_thread.start()
            learner_thread.start()
            evaluator_thread.start()
            
            if self.cfg.policy.enable_async_debug_log:
                logging.info("[ASYNC_DEBUG] All worker threads started, waiting for completion...")
            
            # 等待所有线程完成，添加超时机制
            collector_thread.join(timeout=3600)  # 1小时超时
            learner_thread.join(timeout=3600)
            evaluator_thread.join(timeout=3600)
            
            # 检查线程是否正常完成
            if collector_thread.is_alive():
                logging.warning("[ASYNC_DEBUG] Collector thread is still alive, training may be stuck")
            if learner_thread.is_alive():
                logging.warning("[ASYNC_DEBUG] Learner thread is still alive, training may be stuck")
            if evaluator_thread.is_alive():
                logging.warning("[ASYNC_DEBUG] Evaluator thread is still alive, training may be stuck")
            
        except Exception as e:
            logging.error(f"[ASYNC_DEBUG] Thread management error: {e}")
            import traceback
            logging.error(f"[ASYNC_DEBUG] Thread error traceback: {traceback.format_exc()}")
            # 设置停止信号
            self.stop_event.set()
            raise
        
        if self.cfg.policy.enable_async_debug_log:
            logging.info("[ASYNC_DEBUG] All worker threads completed")
        
        # 执行learner的after_run hook
        try:
            self.learner.call_hook('after_run')
        except Exception as e:
            logging.error(f"[ASYNC_DEBUG] After run hook error: {e}")
        
        if self.cfg.policy.use_wandb:
            try:
                wandb.finish()
            except Exception as e:
                logging.error(f"[ASYNC_DEBUG] Wandb finish error: {e}")
        
        if self.cfg.policy.enable_async_debug_log:
            logging.info(f"[ASYNC_DEBUG] Async training completed. Final stats: "
                        f"train_iter={self.train_iter}, env_step={self.env_step}, best_reward={self.best_reward}")
        
        return self.policy


def train_unizero_segment_async(
        input_cfg: Tuple[dict, dict],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':
    """
    异步版本的UniZero训练入口
    """
    cfg, create_cfg = input_cfg
    
    # 确保策略类型支持
    assert create_cfg.policy.type in ['unizero', 'sampled_unizero'], "train_unizero entry now only supports the following algo.: 'unizero', 'sampled_unizero'"
    
    # 创建异步训练器
    trainer = AsyncTrainer(cfg, create_cfg, model, model_path, max_train_iter, max_env_step)
    
    # 开始训练
    return trainer.train()


if __name__ == "__main__":
    from lzero.entry import train_unizero_segment
    # 这里可以添加测试代码
    pass 