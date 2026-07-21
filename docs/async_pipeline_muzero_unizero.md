# MuZero / UniZero Async Segment Pipeline Review

Generated: 2026-07-02

## 1. Summary

This change generalizes the existing MuZero segment async pipeline to UniZero while keeping the default synchronous paths unchanged. The async pipeline keeps learner and replay buffer ownership in the driver process. Collector and evaluator run as Ray actors and exchange only immutable model snapshots and rollout/evaluation results with the driver.

Recent MuZero logs show that the async path is already useful:

- Async run `test-po-a5-260702_1457`: about 597,710 env steps / 138,500 train iters, about 37.04 env steps/s wall-clock.
- Sync baseline `test-po-b5-260702_1457`: about 375,111 env steps / 86,400 train iters, about 23.24 env steps/s wall-clock.
- GPU monitor: async average GPU util was about 33.06%, sync was about 30.17%. The gain mainly comes from overlapping collector/evaluator work with learner work, not from making a single collector faster.

UniZero async support is code-complete and validated with static checks, unit tests, and rjob dry-runs. Full reward and memory validation should still be run on narmodel_gpu with the new uz-1m job.

## 2. Async Mechanism

```text
driver:
  create learner + policy.learn_mode + GameBuffer
  publish CPU model state with version=train_iter
  launch collector actor collect.remote(...)
  launch evaluator actor eval.remote(...)
  while running:
    consume completed collect results
    push segments into GameBuffer
    enqueue one train budget per collect batch
    train short chunks from the local GameBuffer
    publish new weights according to sync interval / max policy lag

collector actor:
  owns env manager + collect policy
  loads a model snapshot only at collect boundaries
  runs MuZeroSegmentCollector.collect()
  returns new_data + envstep_delta + policy_version

evaluator actor:
  owns env manager + eval policy
  evaluates an immutable model snapshot
  returns reward_mean / stop flag
  driver saves the exact evaluated checkpoint snapshot when enabled
```

Replay buffer mutation, sampling, reanalyze, priority update, and learner hooks remain single-process operations. This avoids data races in `game_segment_buffer`, `game_pos_priorities`, `game_segment_game_pos_look_up`, and UniZero world-model target inference.

## 3. Review Findings And Implemented Fixes

| Module | Location | Issue | Fix | Expected Benefit |
| --- | --- | --- | --- | --- |
| MuZero async driver | `lzero/entry/train_muzero_segment_async.py:313` | The same model version was repeatedly `ray.put` for every collect launch. | Cache `last_published_model_ref` and reuse the Ray ObjectRef for the same version. | Lower driver CPU and serialization overhead, especially with multiple collectors. |
| MuZero async driver | `lzero/entry/train_muzero_segment_async.py:361` | During cold start, a full train-budget queue could block new collects before `train_start_after_envsteps` was reached. | Apply budget queue throttling only after replay data is trainable. | Prevent cold-start collector starvation. |
| MuZero async driver | `lzero/entry/train_muzero_segment_async.py:108`, `:398`, `:441` | `buffer_reanalyze_freq >= 1` could produce interval `0` when `update_per_collect < freq`. | Add `_reanalyze_interval()` with `max(1, ...)` and guard `freq <= 0`. | Avoid modulo-by-zero and support aggressive reanalyze settings. |
| MuZero async driver | `lzero/entry/train_muzero_segment_async.py:94`, `:438` | Replay sufficiency was transition-only and did not respect `sample_type='episode'`. | Add `_has_enough_replay_data()` for transition vs episode replay. | Keep episode replay configs compatible. |
| UniZero async driver | `lzero/entry/train_muzero_segment_async.py:122`, `:458` | UniZero learner expects `[current_batch, target_batch, train_iter]`; MuZero async passed only two items. | Append `learner.train_iter` for `unizero/sampled_unizero`. | Fix UniZero async learner input shape. |
| UniZero async driver | `lzero/entry/train_muzero_segment_async.py:128`, `:464` | UniZero world-model / KV cache needs lifecycle cleanup after train epochs. | Call `recompute_pos_emb_diff_and_clear_cache()` after each completed collect budget. | Avoid stale KV/pos-emb cache across target inference epochs. |
| Async actor | `lzero/entry/async_muzero/actors.py:43`, `:58`, `:127`, `:198` | UniZero actors could keep KV cache from older weights; CPU fallback did not update `world_model_cfg.device`. | Sync actor world-model device and clear collect/eval/target world-model caches after loading new weights. | Prevent old-weight cache contamination in rollout/eval. |
| UniZero GameBuffer | `lzero/mcts/buffer/game_buffer_unizero.py:675` | `update_priority()` used `timestep_batch` as `make_time_list`, so priority updates could hit stale samples incorrectly. | Read `current_batch[6]` explicitly as `make_time_list`. | Improve long-run priority replay correctness. |
| UniZero policy | `lzero/policy/unizero.py:2079` | Cache cleanup covered collect/target models only. | Clear unique `_learn_model/_collect_model/_eval_model/_target_model` world models. | Make sync and async cache lifecycle consistent. |
| UniZero config | `zoo/atari/config/atari_unizero_segment_config.py:49`, `:212`, `:255` | UniZero Atari config had no async CLI/config path. | Add async options and import `train_unizero_segment_async` only when `--async-pipeline` is set. | Enable async without changing the default sync path. |
| narmodel_gpu rjob | `zoo/atari/runs/rjob/rjob_atari_unizero_segment_async_uz1m_narmodel_gpu.sh:18` | There was no ready-to-submit Pong/MsPacman uz-1m async job. | Add a dedicated narmodel_gpu submit script. | Reduce submission mistakes and standardize reproduction. |

## 4. UniZero-Specific Constraints

UniZero differs from MuZero in the async path because of the world model and KV cache:

- UniZero collect/eval `initial_inference` depends on previous observation/action and per-env KV cache. Actors must clear cache after loading new weights.
- UniZero learner input includes `train_iter`, which drives label smoothing, loss schedules, clipping, and monitoring.
- UniZero `GameBuffer.sample()` uses the target model for world-model target inference and MCTS reanalyze. The replay buffer must remain single-owner in this implementation.
- UniZero priority update depends on `make_time_list` to avoid updating samples that were removed from the buffer.

For this reason, the implementation does not make replay buffer, sample, or reanalyze into Ray actors yet. That is less distributed, but the correctness boundary is much cleaner.

## 5. rjob Usage

Submit Pong / MsPacman UniZero async uz-1m:

```bash
bash zoo/atari/runs/rjob/rjob_atari_unizero_segment_async_uz1m_narmodel_gpu.sh
```

Common overrides:

```bash
RJOB_DRY_RUN=1 bash zoo/atari/runs/rjob/rjob_atari_unizero_segment_async_uz1m_narmodel_gpu.sh

SEEDS=0,1 RJOB_GPU=4 MAX_PARALLEL=4 \
bash zoo/atari/runs/rjob/rjob_atari_unizero_segment_async_uz1m_narmodel_gpu.sh

NUM_COLLECTOR_ACTORS=2 MAX_TRAIN_CHUNK_STEPS=2 \
bash zoo/atari/runs/rjob/rjob_atari_unizero_segment_async_uz1m_narmodel_gpu.sh
```

Local dry-run:

```bash
SUBMIT_RJOB=0 DRY_RUN=1 ASYNC_PIPELINE=1 SMOKE_TEST=1 \
ATARI_ENVS='ALE/Pong-v5,ALE/MsPacman-v5' BASELINE_VARIANTS=fast_noaux \
MAX_ENV_STEP=100 bash zoo/atari/runs/rjob/run_atari_unizero_segment_rjob.sh
```

## 6. Validation

Executed:

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

Results:

- `tests/test_train_muzero_segment_async.py`: 7 passed.
- rjob dry-run produced the expected `narmodel_gpu` command with Pong/MsPacman, `MAX_ENV_STEP=1000000`, and `ASYNC_PIPELINE=1`.

Still required on cluster:

- Full UniZero Pong / MsPacman 1M stability and reward curves.
- Throughput, GPU util, and reward comparison for `NUM_COLLECTOR_ACTORS=1` vs `2`.
- Checkpoint I/O impact when async eval checkpoint saving is enabled.
