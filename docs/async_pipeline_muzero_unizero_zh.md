# MuZero / UniZero Async Segment Pipeline 说明与 Code Review

生成时间：2026-07-02

## 1. 结论

本次改动将现有 MuZero segment async pipeline 泛化到 UniZero，同时保持同步入口不变。新的 async pipeline 仍采用单 owner 设计：learner 和 replay buffer 留在 driver 进程，collector/evaluator 作为 Ray actor 运行，只通过不可变模型快照和采样结果交换数据。

MuZero 最新日志显示 async pipeline 已经带来实际吞吐收益：

- async：`test-po-a5-260702_1457`，约 597,710 envstep / 138,500 train iter，wall-clock 约 37.04 envstep/s。
- 同步 baseline：`test-po-b5-260702_1457`，约 375,111 envstep / 86,400 train iter，wall-clock 约 23.24 envstep/s。
- GPU 监控：async 平均 GPU util 约 33.06%，同步约 30.17%；async 主要收益来自 collector/evaluator 与 learner 的流水线重叠，而不是单 actor 采样速度提升。

UniZero async 已完成代码适配、rjob dry-run 和单元测试验证。正式性能仍需在 narmodel_gpu 上用新增 uz-1m rjob 跑 Pong / MsPacman 曲线确认。

## 2. 当前 Async 机制

流程如下：

```text
driver:
  create learner + policy.learn_mode + GameBuffer
  publish CPU model state with version=train_iter
  launch collector actor collect.remote(...)
  launch evaluator actor eval.remote(...)
  while running:
    consume completed collect results
    push segments into GameBuffer
    enqueue train budget per collect batch
    train small chunks from the local GameBuffer
    publish new weights according to sync interval / max policy lag

collector actor:
  owns env manager + collect policy
  loads model snapshot only at collect boundary
  runs MuZeroSegmentCollector.collect()
  returns new_data + envstep_delta + policy_version

evaluator actor:
  owns env manager + eval policy
  evaluates an immutable model snapshot
  returns reward_mean / stop flag
  driver saves the exact evaluated checkpoint snapshot when enabled
```

Replay buffer, sample, reanalyze, priority update and learner hook remain single-process operations. This avoids data races in `game_segment_buffer`, `game_pos_priorities`, `game_segment_game_pos_look_up`, and UniZero world-model target inference.

## 3. Code Review 与已实施优化

| 模块 | 位置 | 问题 | 修改 | 预期收益 |
| --- | --- | --- | --- | --- |
| MuZero async driver | `lzero/entry/train_muzero_segment_async.py:313` | 相同权重版本会为每个 collect 重新 `ray.put`，重复序列化模型 state。 | 缓存 `last_published_model_ref`，同版本复用 Ray ObjectRef。 | 降低 driver CPU/序列化开销，collector 多 actor 时更明显。 |
| MuZero async driver | `lzero/entry/train_muzero_segment_async.py:361` | 冷启动阶段如果 train budget 队列满，而 `train_start_after_envsteps` 尚未达到，可能停止继续发 collect。 | 队列限流只在 replay 已满足训练条件时生效。 | 避免 cold-start collector starvation。 |
| MuZero async driver | `lzero/entry/train_muzero_segment_async.py:108`, `:398`, `:441` | `buffer_reanalyze_freq >= 1` 且 `update_per_collect < freq` 时 interval 可能为 0。 | `_reanalyze_interval()` 使用 `max(1, ...)`，并处理 `freq <= 0`。 | 避免 modulo-by-zero，兼容更高 reanalyze 频率配置。 |
| MuZero async driver | `lzero/entry/train_muzero_segment_async.py:94`, `:438` | 原 async 只按 transition 数判断数据充足，不支持 `sample_type='episode'`。 | 新增 `_has_enough_replay_data()`，按 sample type 选择 game segment 或 transition。 | 兼容 episode replay 配置。 |
| UniZero async driver | `lzero/entry/train_muzero_segment_async.py:122`, `:458` | UniZero learner 需要 `[current_batch, target_batch, train_iter]`，MuZero async 只传两项。 | 对 `unizero/sampled_unizero` 在 train 前追加 `learner.train_iter`。 | 修复 UniZero async learner 输入形态错误。 |
| UniZero async driver | `lzero/entry/train_muzero_segment_async.py:128`, `:464` | UniZero world-model / KV cache 需要在 train epoch 后清理。 | 每个 collect budget 完成后调用 `recompute_pos_emb_diff_and_clear_cache()`。 | 避免 stale KV/pos-emb cache 影响下一轮 target inference。 |
| Async actor | `lzero/entry/async_muzero/actors.py:43`, `:58`, `:127`, `:198` | UniZero actor 加载新权重后旧 KV cache 仍可能保留；CPU fallback 时 world_model device 未同步。 | actor cfg 同步 `world_model_cfg.device`；加载新权重后清理 collect/eval/target world-model cache。 | 避免旧权重 cache 污染 rollout/eval，提升 CPU/GPU fallback 稳定性。 |
| UniZero GameBuffer | `lzero/mcts/buffer/game_buffer_unizero.py:675` | `update_priority()` 把 `timestep_batch` 当成 `make_time_list`，priority 可能错误更新。 | 显式使用 `current_batch[6]` 作为 `make_time_list`。 | 修复 priority replay 长跑稳定性问题。 |
| UniZero policy | `lzero/policy/unizero.py:2079` | cache 清理只覆盖 collect/target，eval actor 及部分模型字段未覆盖。 | 按 `_learn_model/_collect_model/_eval_model/_target_model` 去重清理。 | 同步和 async cache 生命周期一致。 |
| UniZero config | `zoo/atari/config/atari_unizero_segment_config.py:49`, `:212`, `:255` | UniZero 配置无 async 参数和 async 入口选择。 | 增加 async CLI/config，并在 `--async-pipeline` 时导入 `train_unizero_segment_async`。 | 支持与 MuZero 等价的 async 流程，不影响默认同步。 |
| narmodel_gpu rjob | `zoo/atari/runs/rjob/rjob_atari_unizero_segment_async_uz1m_narmodel_gpu.sh:18` | 缺少可直接提交 Pong/MsPacman uz-1m async 作业脚本。 | 新增专用提交脚本，默认 `narmodel_gpu`、2 GPU、1M env steps、fast_noaux。 | 降低提交错误率，便于复现实验。 |

## 4. UniZero 与 MuZero Async 差异

UniZero 的主要额外约束来自 world-model 和 KV cache：

- UniZero collector/evaluator 的 `initial_inference` 依赖上一帧 observation/action 与 per-env KV cache。actor 加载新权重后必须清 cache，否则 cache 内容来自旧权重。
- UniZero learner 输入包含 `train_iter`，用于 label smoothing、loss schedule、clip/monitor 等训练逻辑。
- UniZero `GameBuffer.sample()` 会用 target model 做 world-model target inference 和 MCTS reanalyze，不能放到多个 actor 中并发写同一个 replay buffer。
- UniZero priority update 使用 `make_time_list` 判断样本是否仍属于当前 buffer；该字段必须与 timestep 区分。

因此本次迁移没有把 replay buffer 做成 Ray actor，也没有把 sample/reanalyze 拆出去。这样吞吐提升不如完全分布式 replay，但正确性边界更清晰。

## 5. rjob 用法

提交 Pong / MsPacman UniZero async uz-1m：

```bash
bash zoo/atari/runs/rjob/rjob_atari_unizero_segment_async_uz1m_narmodel_gpu.sh
```

常用覆盖参数：

```bash
RJOB_DRY_RUN=1 bash zoo/atari/runs/rjob/rjob_atari_unizero_segment_async_uz1m_narmodel_gpu.sh

SEEDS=0,1 RJOB_GPU=4 MAX_PARALLEL=4 \
bash zoo/atari/runs/rjob/rjob_atari_unizero_segment_async_uz1m_narmodel_gpu.sh

NUM_COLLECTOR_ACTORS=2 MAX_TRAIN_CHUNK_STEPS=2 \
bash zoo/atari/runs/rjob/rjob_atari_unizero_segment_async_uz1m_narmodel_gpu.sh
```

本地 dry-run：

```bash
SUBMIT_RJOB=0 DRY_RUN=1 ASYNC_PIPELINE=1 SMOKE_TEST=1 \
ATARI_ENVS='ALE/Pong-v5,ALE/MsPacman-v5' BASELINE_VARIANTS=fast_noaux \
MAX_ENV_STEP=100 bash zoo/atari/runs/rjob/run_atari_unizero_segment_rjob.sh
```

## 6. 验证

已执行：

```bash
python -m py_compile \
  lzero/entry/train_muzero_segment_async.py \
  lzero/entry/train_unizero_segment_async.py \
  lzero/entry/async_muzero/actors.py \
  zoo/atari/config/atari_unizero_segment_config.py \
  lzero/mcts/buffer/game_buffer_unizero.py \
  lzero/policy/unizero.py

/mnt/shared-storage-user/puyuan/conda_envs/lz/bin/python -m pytest -q \
  tests/test_train_muzero_segment_async.py

RJOB_DRY_RUN=1 \
bash zoo/atari/runs/rjob/rjob_atari_unizero_segment_async_uz1m_narmodel_gpu.sh
```

结果：

- `tests/test_train_muzero_segment_async.py`: 7 passed。
- rjob dry-run 正确生成 `narmodel_gpu` 提交命令，包含 Pong/MsPacman、`MAX_ENV_STEP=1000000`、`ASYNC_PIPELINE=1`。

仍需在集群验证：

- UniZero Pong / MsPacman 1M 正式训练是否稳定。
- `NUM_COLLECTOR_ACTORS=1` 与 `2` 的吞吐、GPU util、reward 曲线对比。
- async eval checkpoint 开关对 I/O 与训练吞吐的影响。
