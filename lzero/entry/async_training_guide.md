# LightZero 异步训练改造指南

## 概述

本文档详细说明了如何将LightZero的collector、learner、evaluator从同步串行架构改造为异步并行架构，以提高训练效率。

## 当前架构分析

### 同步架构特点
- **串行执行**：collector → learner → evaluator 按顺序执行
- **强耦合**：各组件之间存在强依赖关系
- **阻塞等待**：每个组件必须等待前一个组件完成

### 性能瓶颈
1. **CPU利用率低**：GPU训练时CPU空闲，CPU收集时GPU空闲
2. **资源浪费**：无法充分利用多核CPU和多GPU
3. **训练效率低**：总训练时间 = collector时间 + learner时间 + evaluator时间

## 异步改造方案

### 核心思想
将三个组件解耦，通过线程池和消息队列实现并行执行：

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Collector  │    │   Learner   │    │  Evaluator  │
│   Thread    │    │   Thread    │    │   Thread    │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────┐
│              Data Queue & Policy Lock               │
└─────────────────────────────────────────────────────┘
```

### 关键技术点

#### 1. 线程安全的数据共享
```python
# 数据缓冲队列
self.data_queue = queue.Queue(maxsize=10)

# Policy更新锁
self.policy_lock = threading.Lock()

# 停止信号
self.stop_event = threading.Event()
```

#### 2. 异步数据收集
```python
def _collector_worker(self):
    while not self.stop_event.is_set():
        # 获取最新policy（线程安全）
        with self.policy_lock:
            current_policy = self.policy.collect_mode
        
        # 收集数据
        new_data = self.collector.collect(...)
        
        # 放入队列供learner使用
        self.data_queue.put((new_data, self.env_step))
```

#### 3. 异步模型训练
```python
def _learner_worker(self):
    while not self.stop_event.is_set():
        # 从队列获取数据
        new_data, data_env_step = self.data_queue.get(timeout=1.0)
        
        # 训练模型
        log_vars = self.learner.train(train_data, data_env_step)
        
        # 更新policy（线程安全）
        with self.policy_lock:
            # 确保policy更新是线程安全的
            pass
```

#### 4. 异步评估
```python
def _evaluator_worker(self):
    while not self.stop_event.is_set():
        if self.evaluator.should_eval(self.train_iter):
            # 获取最新policy进行评估
            with self.policy_lock:
                current_policy = self.policy.eval_mode
            
            stop, reward = self.evaluator.eval(...)
        
        # 定期检查，不阻塞主流程
        time.sleep(1.0)
```

## 最小化改动实现

### 1. 新增异步训练入口
- 文件：`lzero/entry/train_unizero_segment_async.py`
- 功能：提供异步训练的主要逻辑

### 2. 配置文件支持
- 文件：`zoo/classic_control/cartpole/config/cartpole_unizero_segment_async_config.py`
- 新增配置项：
  ```python
  enable_async_training = True
  data_queue_size = 10
  enable_async_debug_log = True  # 控制详细调试信息输出
  ```

### 3. 使用方式
```python
from lzero.entry import train_unizero_segment_async
from zoo.classic_control.cartpole.config.cartpole_unizero_segment_async_config import main_config, create_config

# 启动异步训练
policy = train_unizero_segment_async(
    [main_config, create_config], 
    seed=0, 
    max_env_step=int(2e5)
)

# 控制调试信息输出
# 在配置文件中设置 enable_async_debug_log = True/False
# True: 输出详细的异步训练调试信息
# False: 只输出基本的训练信息
```