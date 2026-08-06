# MuZero / UniZero 异步训练流水线（Async Segment Pipeline）

本文档介绍 LightZero 中 MuZero 系列与 UniZero 的异步（async）segment 训练流水线：架构设计、与同步（sync）管线的区别，以及配置与使用方法。

## 1. 概述

- **同步管线**（`train_muzero_segment` / `train_unizero_segment`）：单进程串行执行「采集 → 训练 → 评估」，任一阶段运行时其余阶段空闲。
- **异步管线**（`train_muzero_segment_async` / `train_unizero_segment_async`）：基于 Ray，将 collector / evaluator 放入独立 actor 进程，与 learner 重叠执行，提升 GPU 利用率与 wall-clock 吞吐。
- **训练语义不变**：replay buffer、采样、reanalyze、优先级更新仍由 driver 单进程持有；async 只改变「谁在何时执行」，不改变数据流与更新规则。
- **吞吐参考**（Atari Pong 实测）：async 约 37 envstep/s，sync 约 23 envstep/s（约 1.6×）。收益来自采集/评估与学习的流水线重叠，而非单次采集更快。

支持的 policy 类型：`muzero`、`muzero_context`、`muzero_rnn_full_obs`、`efficientzero`、`sampled_efficientzero`、`sampled_muzero`、`gumbel_muzero`、`stochastic_muzero`、`unizero`、`sampled_unizero`。

## 2. 架构设计

### 2.1 角色与数据流

```text
driver（主进程）:
  创建 learner（learn_mode policy）+ GameBuffer —— 均为唯一 owner
  发布 CPU 模型快照（version = train_iter），同版本复用同一 Ray ObjectRef
  启动 collector / evaluator actor
  主循环：
    收取已完成的 collect 结果（segments + envstep_delta + policy_version）
    将 segments 写入 GameBuffer
    按 collect 批次 enqueue 训练预算（update_per_collect）
    从本地 GameBuffer 采样训练短 chunk（不超过 max_train_chunk_steps 后让出事件循环）
    按权重发布策略（weight_sync_interval / max_policy_lag）推送新快照
    处理已完成的 eval 结果（reward_mean / stop），按需保存被评估的精确 checkpoint 快照

collector actor（每个 actor 独立进程）:
  持有自己的 env manager + collect policy
  仅在 collect 边界加载模型快照（一次 collect 内策略版本固定）
  执行 MuZeroSegmentCollector.collect()，返回采样结果

evaluator actor:
  持有自己的 env manager + eval policy
  评估不可变模型快照，返回 reward_mean / stop 标志
```

### 2.2 正确性边界（单 owner 原则）

Replay buffer 的写入、采样、reanalyze、优先级更新全部发生在 driver 单进程内；actor 只接收不可变快照、返回采样/评估数据，不共享任何可变状态。因此 `game_segment_buffer`、`game_pos_priorities`、`game_segment_game_pos_look_up` 以及 UniZero 依赖 target model 的 target inference 都不存在并发读写问题。这是本设计刻意不做的事：没有把 replay buffer / reanalyze 拆成 Ray actor——吞吐上限低于完全分布式方案，但正确性边界清晰。

### 2.3 权重发布与 policy lag

- `weight_sync_interval`：learner 版本（train_iter）至少前进该步数才发布新快照。
- `max_policy_lag`：collector 允许的最大版本滞后，超过即强制发布。
- 冷启动由 policy 级参数 `train_start_after_envsteps` 控制：buffer 数据不足时只采集不训练，且此阶段不对训练预算队列限流，避免 collector 饿死。

## 3. 与 sync 管线的区别

| 维度 | sync | async |
| --- | --- | --- |
| 执行方式 | 单进程串行：采集→训练→评估 | collector/evaluator 为 Ray actor，与 learner 重叠 |
| 吞吐 / GPU 利用 | 采集/评估期间 GPU 空闲 | 各阶段流水重叠，wall-clock 吞吐约 1.6×（Pong 实测） |
| 数据新鲜度 | 采集策略恒为最新权重 | collector 策略允许滞后（上界 `max_policy_lag`），每次 collect 内部版本固定 |
| 训练语义 | 基准语义 | 完全一致：buffer/采样/优先级更新仍在 driver 单进程，仅执行位置变化 |
| 额外依赖 | 无 | 需要 Ray |
| 适用场景 | 调试、小规模实验、正确性验证 | 大规模/长跑训练、追求吞吐的实验 |

## 4. 配置与使用

### 4.1 开启 async

Atari 示例配置同时支持两种模式，默认 sync，`--async-pipeline` 切换为 async：

```bash
# MuZero
python zoo/atari/config/atari_muzero_segment_async_config.py --env ALE/Pong-v5 --seed 0 --async-pipeline
# UniZero
python zoo/atari/config/atari_unizero_segment_async_config.py --env ALE/Pong-v5 --seed 0 --async-pipeline
```

等价的配置项写法（在 policy config 中）：

```python
async_pipeline=dict(enabled=True, num_collector_actors=2, ...)
```

### 4.2 关键参数（`policy.async_pipeline`）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `enabled` | `True`（async 入口内） | 是否启用 async 流水线 |
| `num_collector_actors` | `1` | collector actor 数；增大可并行采集 |
| `num_evaluator_actors` | `1` | evaluator actor 数 |
| `max_collect_inflight` | `num_collector_actors` | 同时在飞的 collect 任务上限 |
| `max_eval_inflight` | `1` | 同时在飞的 eval 任务上限 |
| `max_train_chunk_steps` | `4` | learner 每次连续训练的最大步数，到达后让出事件循环处理 actor 消息 |
| `weight_sync_interval` | `1` | 发布新权重快照的最小 train_iter 间隔 |
| `max_policy_lag` | `0` | collector 策略允许的最大版本滞后，超过强制发布 |
| `max_train_budget_queue_size` | `2 * num_collector_actors` | 训练预算队列上限（背压）；buffer 可训练前不生效 |
| `eval_at_start` | `False` | 是否在训练开始前先评估一次 |
| `collector_num_cpus` / `evaluator_num_cpus` | `1` | 每个 actor 的 Ray CPU 资源 |
| `collector_num_gpus` / `evaluator_num_gpus` | `0` | 每个 actor 的 Ray GPU 资源（支持小数）；默认 actor 仅用 CPU |
| `buffer_stats_interval` | `100` | buffer 统计日志间隔（train_iter） |
| `poll_interval_s` | `0.1` | driver 事件循环轮询间隔（秒） |
| `shutdown_timeout_s` | `30` | 结束时等待 actor 退出的超时（秒） |
| `ray_local_mode` | `False` | Ray local mode（调试用） |

其余训练参数（batch size、replay_ratio、reanalyze 等）与 sync 完全一致。

### 4.3 本地验证

```bash
python -m pytest -q tests/test_train_muzero_segment_async.py
```

## 5. UniZero 特有约束

UniZero 的 world model 与 KV cache 使其在 async 下比 MuZero 多三条约束：

- **KV cache 生命周期**：collector/evaluator 的 `initial_inference` 依赖 per-env KV cache。actor 加载新权重后必须清空 collect/eval/target world-model 的 cache，否则 cache 内容来自旧权重；driver 侧 learner 在每个 collect 预算完成后调用 `recompute_pos_emb_diff_and_clear_cache()`；运行期按 `kv_cache_clear_interval`（单位 env step，默认 2000，0 表示禁用）定期清理。
- **learner 输入含 `train_iter`**：UniZero 的 `_forward_learn` 需要 `[current_batch, target_batch, train_iter]` 三元组（`train_iter` 驱动 label smoothing、loss schedule、encoder clip 与监控）。async driver 在每次 learn 前自动追加，无需用户处理。
- **buffer 必须单 owner**：UniZero `GameBuffer.sample()` 会使用 target model 做 target inference 与 MCTS reanalyze，无法安全地拆到多个 actor 并发执行。

## 6. 注意事项与已知边界

- 仅 async 入口依赖 Ray；未安装时入口会给出明确报错。
- 优先级回放（PER）依赖采样时记录的 `make_time` 时间戳：buffer 被清空（`clear_time` 更新）前采出的样本不会再被写回优先级，防止陈旧索引误写。
- evaluator 保存的 checkpoint 是「被评估的那一版」精确快照，保存在 `exp_name/ckpt/`。
- replay buffer 与 reanalyze 未分布式化（见 2.2），多 collector 场景下 driver 的采样/训练串行是吞吐瓶颈上限。

## 7. 验证状态

- `tests/test_train_muzero_segment_async.py`：6 passed, 1 skipped。
- 相关回归测试：`lzero/mcts/tests/test_game_buffer_index_alignment.py`（采样索引/权重对齐与优先级写回）、`lzero/model/unizero_world_models/tests/test_per_sample_is_weights.py`（逐样本 IS 权重）。
- 集群侧 1M 量级 reward 曲线与多 collector 吞吐对比以最新实验记录为准。
