# MuZero / UniZero Async Segment Pipeline

This document describes the async segment training pipeline for the MuZero family and UniZero in LightZero: architecture, differences from the synchronous (sync) pipeline, and usage.

## 1. Overview

- **Sync pipeline** (`train_muzero_segment` / `train_unizero_segment`): a single process runs collect → train → evaluate sequentially; whenever one stage runs, the others idle.
- **Async pipeline** (`train_muzero_segment_async` / `train_unizero_segment_async`): based on Ray, collector and evaluator run as separate actor processes and overlap with the learner, improving GPU utilization and wall-clock throughput.
- **Training semantics are unchanged**: the replay buffer, sampling, reanalyze, and priority updates are still owned by the driver process. Async only changes *who runs what and when*, not the data flow or update rules.
- **Throughput reference** (Atari Pong, same ~6.2 h wall-clock window, 2026-07-02, 2 collector actors): async ≈ 36.3 env steps/s (809.5k env steps) vs sync ≈ 22.4 env steps/s (499.8k env steps) — ~1.6×. The gain comes from overlapping collection/evaluation with learning, not from faster collection itself.

Supported policy types: `muzero`, `muzero_context`, `muzero_rnn_full_obs`, `efficientzero`, `sampled_efficientzero`, `sampled_muzero`, `gumbel_muzero`, `stochastic_muzero`, `unizero`, `sampled_unizero`.

## 2. Architecture

### 2.1 Roles and data flow

```text
driver (main process):
  creates the learner (learn_mode policy) + GameBuffer — the single owner of both
  publishes CPU model snapshots (version = train_iter); the same version reuses one Ray ObjectRef
  launches collector / evaluator actors
  main loop:
    consume finished collect results (segments + envstep_delta + policy_version)
    push segments into the GameBuffer
    enqueue one train budget (update_per_collect) per collect batch
    train short chunks from the local GameBuffer (yield after max_train_chunk_steps)
    publish new snapshots per weight-sync policy (weight_sync_interval / max_policy_lag)
    consume finished eval results (reward_mean / stop); optionally save the exact evaluated checkpoint

collector actor (one process per actor):
  owns its env manager + collect policy
  loads a model snapshot only at collect boundaries (policy version is fixed within one collect)
  runs MuZeroSegmentCollector.collect() and returns the rollout data

evaluator actor:
  owns its env manager + eval policy
  evaluates an immutable model snapshot and returns reward_mean / stop flag
```

### 2.2 Correctness boundary (single-owner principle)

All replay-buffer mutation, sampling, reanalyze, and priority updates happen inside the single driver process. Actors receive immutable snapshots and return data; no mutable state is shared. Hence `game_segment_buffer`, `game_pos_priorities`, `game_segment_game_pos_look_up`, and UniZero's target-model-based target inference are free of data races. This is a deliberate trade-off: the buffer/reanalyze are not Ray actors — the throughput ceiling is lower than a fully distributed design, but the correctness boundary is much cleaner.

### 2.3 Weight publishing and policy lag

- `weight_sync_interval`: a new snapshot is published only after the learner version (train_iter) has advanced by at least this many steps.
- `max_policy_lag`: the maximum version lag tolerated by collectors; exceeding it forces a publish.
- Cold start is controlled by the policy-level `train_start_after_envsteps`: before the buffer is trainable the driver only collects, and budget-queue throttling is disabled to avoid collector starvation.

## 3. Differences from the sync pipeline

| Aspect | sync | async |
| --- | --- | --- |
| Execution | Single process, sequential collect→train→eval | Collector/evaluator are Ray actors overlapping the learner |
| Throughput / GPU util | GPU idles during collect/eval | Stages overlap; ~1.6× wall-clock throughput (measured on Pong) |
| Data freshness | Collect policy is always the latest weights | Collector policy may lag (bounded by `max_policy_lag`); version fixed within one collect |
| Training semantics | Baseline | Identical: buffer/sampling/priority updates stay in the single driver process |
| Extra dependency | None | Ray |
| Best for | Debugging, small-scale runs, correctness checks | Large-scale / long runs where throughput matters |

## 4. Configuration and usage

### 4.1 Enabling async

The Atari example configs support both modes; sync is the default, `--async-pipeline` switches to async:

```bash
# MuZero
python zoo/atari/config/atari_muzero_segment_async_config.py --env ALE/Pong-v5 --seed 0 --async-pipeline
# UniZero
python zoo/atari/config/atari_unizero_segment_async_config.py --env ALE/Pong-v5 --seed 0 --async-pipeline
```

Equivalent config-level form (in the policy config):

```python
async_pipeline=dict(enabled=True, num_collector_actors=2, ...)
```

### 4.2 Key parameters (`policy.async_pipeline`)

| Parameter | Default | Description |
| --- | --- | --- |
| `enabled` | `True` (in async entry) | Enable the async pipeline |
| `num_collector_actors` | `1` | Number of collector actors; increase for parallel collection |
| `num_evaluator_actors` | `1` | Number of evaluator actors |
| `max_collect_inflight` | `num_collector_actors` | Max concurrently running collect tasks |
| `max_eval_inflight` | `1` | Max concurrently running eval tasks |
| `max_train_chunk_steps` | `4` | Max consecutive learner updates before yielding to actor messages |
| `weight_sync_interval` | `1` | Min train_iter gap between weight snapshot publishes |
| `max_policy_lag` | `0` | Max tolerated collector policy version lag; exceeding it forces a publish |
| `max_train_budget_queue_size` | `2 * num_collector_actors` | Train-budget queue cap (backpressure); inactive until the buffer is trainable |
| `eval_at_start` | `False` | Run one evaluation before training starts |
| `collector_num_cpus` / `evaluator_num_cpus` | `1` | Ray CPU resources per actor |
| `collector_num_gpus` / `evaluator_num_gpus` | `0` | Ray GPU resources per actor (fractional allowed); actors are CPU-only by default |
| `buffer_stats_interval` | `100` | Buffer statistics logging interval (train_iter) |
| `poll_interval_s` | `0.1` | Driver event-loop poll interval (seconds) |
| `shutdown_timeout_s` | `30` | Timeout for actor shutdown at exit (seconds) |
| `ray_local_mode` | `False` | Ray local mode (for debugging) |

All other training parameters (batch size, replay_ratio, reanalyze, etc.) are identical to sync.

### 4.3 Local validation

```bash
python -m pytest -q tests/test_train_muzero_segment_async.py
```

## 5. UniZero-specific constraints

UniZero's world model and KV cache add three constraints compared with MuZero:

- **KV-cache lifecycle**: collector/evaluator `initial_inference` depends on per-env KV caches. After loading new weights, an actor must clear its collect/eval/target world-model caches (otherwise they hold old-weight content). On the driver side, the learner calls `recompute_pos_emb_diff_and_clear_cache()` after each completed collect budget, and caches are cleared periodically every `kv_cache_clear_interval` env steps (default 2000; 0 disables periodic clearing).
- **Learner input includes `train_iter`**: UniZero `_forward_learn` expects `[current_batch, target_batch, train_iter]` (`train_iter` drives label smoothing, loss schedules, encoder clipping, and monitoring). The async driver appends it automatically before each learn call.
- **The buffer must stay single-owner**: UniZero `GameBuffer.sample()` uses the target model for target inference and MCTS reanalyze, which cannot be safely distributed across actors.

## 6. Notes and known limits

- Ray is required only by the async entry; a clear error is raised if it is missing.
- Prioritized replay relies on the `make_time` timestamp recorded at sampling time: samples drawn before a buffer clear (which bumps `clear_time`) are never written back, preventing stale-index corruption.
- Evaluator checkpoints are exact snapshots of the evaluated version, saved under `exp_name/ckpt/`.
- The replay buffer and reanalyze are not distributed (see 2.2); with multiple collectors, the driver's sequential sampling/training is the throughput ceiling.

## 7. Validation status

- `tests/test_train_muzero_segment_async.py`: 6 passed, 1 skipped.
- Related regression tests: `lzero/mcts/tests/test_game_buffer_index_alignment.py` (sampling index/weight alignment and priority write-back), `lzero/model/unizero_world_models/tests/test_per_sample_is_weights.py` (per-sample IS weighting).
- For cluster-scale 1M reward curves and multi-collector throughput comparisons, refer to the latest experiment records.
