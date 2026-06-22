# UniZero KV-Cache 审查与修复报告

日期：2026-06-22

参考：UniZero 论文，重点参考 transformer world model 的 recurrent inference / KV cache 机制：https://arxiv.org/abs/2406.10667

## 1. 调用链梳理

入口链路：

1. `zoo/atari/runs/rjob/rjob_atari_unizero_segment_1node_8gpu.sh`
2. `zoo/atari/runs/rjob/run_atari_unizero_segment_rjob.sh`
3. `zoo/atari/config/atari_unizero_segment_config.py`
4. `lzero/entry/train_unizero_segment.py`
5. `lzero/policy/unizero.py`
6. `lzero/worker/muzero_segment_collector.py`
7. `lzero/worker/muzero_evaluator.py`
8. `lzero/model/unizero_model.py`
9. `lzero/model/unizero_world_models/world_model.py`
10. `lzero/model/unizero_world_models/kv_cache_manager.py`

训练 buffer 相关链路：

- `lzero/mcts/buffer/game_buffer_unizero.py`
- `lzero/mcts/buffer/game_segment.py`

buffer 审查结论：buffer 不直接持有在线推理阶段的 KV cache。它的 padding mask 会在 loss 前屏蔽 padding transition，reanalyze 路径也会把 game-batch 的 action/timestep context reshape 到 world-model 训练路径期望的形状。因此本轮没有对 buffer 做 KV-cache 专项改动。

## 2. 问题清单

| 文件 | 类型 | 问题 | 偏差 / 风险 | 修复 |
|---|---|---|---|---|
| `lzero/model/unizero_world_models/world_model.py:1342` | 正确性 | root inference 没有把真实 `env_id` 传入 cache lookup/update，partial ready env batch 会把本地 batch index 当成 env id。 | recurrent world model 的上下文必须按环境隔离；env id 混用会污染最近历史。 | 传入 `ready_env_id`，校验长度，并使用真实 env id 读写 init cache。 |
| `lzero/policy/unizero.py:1398` | 正确性 | last observation/action buffer 按 active batch 顺序切片，而不是按真实 env id 取值。 | collector/evaluator 中异步完成的环境会把错误的上一帧 action/obs 配给当前 obs。 | 增加 per-env selection 和 per-env update helper。 |
| `lzero/model/unizero_world_models/world_model.py:1361` | 正确性 | 混合 first-step 和 continuing env 的 batch 之前会被整体当成一种路径处理。旧逻辑中尤其容易把 `-1` action 送入 action embedding。 | 论文中的 recurrent inference 要对每个环境维护独立历史；混合 batch 必须按环境状态分流，否则 continuing env 丢失 action token，或 first-step env 触发非法 embedding index。 | 将 mixed batch 拆成 first-step / continuing 两个子 batch，分别前向后按原 batch 顺序 merge。 |
| `lzero/model/unizero_world_models/world_model.py:1458` 和 `:2015` | 正确性 | cache miss 后，full latent observation 前向得到的 KV 长度被记录为 `1`。 | cache hit 用真实 `KeysValues.size`，cache miss 用 `1`，两者语义不一致，会破坏 padding 与 head slicing。 | init 与 recurrent miss 路径统一记录 `self.keys_values_wm_single_env.size`。 |
| `lzero/worker/muzero_evaluator.py:295` | 正确性 / 稳定性 | evaluator 遍历 `set` 形式的 env id。 | batch/env 顺序不确定，可能导致 policy 输出与 cache slot 错位。 | 将 ready env ids 转成 sorted list，并在 batch 构造与 remap 中复用同一顺序。 |
| `lzero/model/unizero_world_models/kv_cache_manager.py:143` | 效率 | 新 manager 每次写 cache 都 clone 整个 cache。 | recurrent MCTS 写入频繁，重复分配会增加延迟和显存压力。 | pool slot 首次分配，后续在 layer 数、shape、device、dtype 兼容时用 `copy_` 原地刷新。 |
| `lzero/model/unizero_world_models/lpips.py:102` | 稳定性 | LPIPS VGG16 backbone 使用 torchvision pretrained loading，会在 rjob 上尝试联网下载并超时。 | `recon_lpips` 任务训练前失败。 | 只从本地 `TORCH_HOME` 加载 `vgg.pth` 与 `vgg16-397923af.pth`。 |
| `zoo/atari/config/atari_unizero_segment_config.py:155` | 存储 / 调试效率 | 调试实验仍会在 eval 时保存 best ckpt。 | 对 GPFS 造成不必要写入压力。 | 增加 `--save-ckpt`，rjob 默认 `SAVE_CKPT=0`。 |
| `zoo/atari/runs/rjob/run_atari_unizero_segment_rjob.sh:181` | 稳定性 | LPIPS 权重路径和 torch cache 路径不够显式。 | worker 可能因环境差异触发下载或找不到权重。 | 显式导出 `TORCH_HOME=tokenizer_pretrained_vgg`，并在启动前检查必需文件。 |

## 3. 新旧 KV-Cache 机制对比

| 维度 | 旧 legacy dict/pool | 修复后的新 `KVCacheManager` |
|---|---|---|
| init cache 归属 | `past_kv_cache_init_infer_envs[env]` 加 per-env shared pools | `init_pools[env]`，由明确的 pool 对象管理 |
| recurrent cache 归属 | 全局 dict 加 shared recurrent pool | 全局 recurrent `KVCachePool` |
| env id 对齐 | 容易出错，调用方常把本地 batch index 当 env id | collector/evaluator 到 world model 全链路传递显式 `ready_env_id` |
| cache miss 长度 | 之前记录为 `1` | 记录真实 `KeysValues.size` |
| mixed root batch | 之前容易把整个 batch 塌缩到单一路径 | 按 first-step / continuing env 拆分并 merge 输出 |
| mutation 安全性 | 依赖手写 clone/copy helper | pool 存储副本，取出后 clone 给可变 forward 使用 |
| 淘汰策略 | reverse map 分散在 `world_model.py`，容易不同步 | 集中在 `KVCachePool` |
| 分配开销 | 多处手动 clone 与新对象分配 | shape/device/dtype 匹配时复用固定 slot 存储 |

结论：旧机制在 env-id 对齐和 cache-length 修复后，可以作为参考路径继续使用，但仍然比较脆弱，因为淘汰、存储、索引职责分散在 `world_model.py` 内。新 manager 是最终落地路径，因为它把 cache 所有权、淘汰和存储复用集中到了单一结构中，更容易审计和测试。

## 4. 修复与重构明细

- `lzero/model/unizero_world_models/world_model.py:1282`
  增加 batch slicing、start-pos 标量提取、mixed root inference 输出 merge helper。

- `lzero/model/unizero_world_models/world_model.py:1342`
  initial inference 中传递并校验 `ready_env_id`。

- `lzero/model/unizero_world_models/world_model.py:1361`
  将 mixed first-step/continuing batch 拆分处理，避免把整批都当成 first-step 或 continuing。

- `lzero/model/unizero_world_models/world_model.py:1450`
  per-env init cache lookup 使用真实 `cache_env_id`，cache miss 后记录真实 KV size。

- `lzero/model/unizero_world_models/world_model.py:1717`
  `update_cache_context` 区分本地 batch index 与真实 cache env id。

- `lzero/model/unizero_world_models/world_model.py:1912`
  recurrent inference 中把本地 MCTS root index 映射回当前 env id，用于 hierarchical init-cache fallback。

- `lzero/model/unizero_world_models/kv_cache_manager.py:187`
  增加固定 slot 复用逻辑，检查 layer 数、shape、device、dtype 后用原地 copy 刷新。

- `lzero/policy/unizero.py:1391`
  增加 env-id normalization、per-env last input selection、per-env last input update。

- `lzero/policy/unizero.py:1478` 和 `:1632`
  collector 与 evaluator 均向 `initial_inference` 传递真实 env ids。

- `lzero/worker/muzero_evaluator.py:295`
  evaluator 统一使用排序后的 ready env ids。

- `lzero/model/unizero_world_models/lpips.py:15`
  LPIPS / VGG16 权重改为通过本地 `TORCH_HOME` 加载，不再触发网络下载。

- `zoo/atari/runs/rjob/*.sh`
  rjob 默认使用 `narmodel_gpu`、结构化日志、本地 VGG 权重、`USE_NEW_CACHE_MANAGER=1`、`SAVE_CKPT=0`。

## 5. 本地验证

已通过的本地检查：

- Python compile：
  对 `world_model.py`、`kv_cache_manager.py`、`unizero_model.py`、`unizero.py`、`muzero_evaluator.py`、`train_unizero_segment.py`、Atari config 执行 `python -m py_compile`。

- Shell syntax：
  对 `zoo/atari/runs/rjob/run_atari_unizero_segment_rjob.sh` 和 `zoo/atari/runs/rjob/rjob_atari_unizero_segment_1node_8gpu.sh` 执行 `bash -n`。

- KV pool 存储复用 smoke：
  验证固定 slot data pointer 复用，以及 retrieve 后的 private copy 不污染 pool 存储。

- mixed world-model smoke：
  `mixed_world_model_smoke_ok (3, 2, 8) (3, 1, 4) (3, 1, 3)`

- 更贴近 Atari token 形状的 mixed action smoke：
  使用 `batch_action=[-1, 3, -1]`、`ready_env_id=[0, 1, 2]`、16 个 observation tokens，验证 `-1` 不再进入 action embedding：
  `real_token_mixed_smoke_ok (3, 16, 32) (3, 1, 9)`

- rjob dry-run：
  确认 `narmodel_gpu`、Pong/MsPacman seed0、四个 variants、`USE_NEW_CACHE_MANAGER=1`、`SAVE_CKPT=0`、`TORCH_HOME=/mnt/shared-storage-user/puyuan/code/LightZero/tokenizer_pretrained_vgg`。

## 6. 基线来源与旧日志报错分析

历史单 Pong 运行：

- 路径：`/mnt/shared-storage-user/puyuan/code/LightZero/data_unizero/Pong/Pong_uz_nlayer2_numsegments-8_gsl20_rr0.25_Htrain10-Hinfer4_bs128_seed0_260615_134309`
- 初始 eval reward：`-21`
- collector throughput：约 `48-49 envstep/s`

更有参考价值的历史 rjob baseline：

- 路径：`/mnt/shared-storage-user/puyuan/code/LightZero/data_unizero/rjob/baseline-pong-mspacman-seed0-260622_121700`
- Pong `fast_noaux`：best eval reward `-17.33`，envstep `42249`；collector throughput 约 `74-76 envstep/s`；最大显存约 `8.878 GB`
- MsPacman `fast_noaux`：best eval reward `926.7`，envstep `50390`；collector throughput 约 `73-75 envstep/s`；最大显存约 `8.878 GB`

旧 rjob 最终报错：

- 路径：`/mnt/shared-storage-user/puyuan/code/LightZero/rjob_logs/atari_unizero_segment/uz-atari-segment-baseline-260622_130012`
- 覆盖任务：Pong/MsPacman、seed0、`fast_noaux`、`compile_noaux`、`recon_only`、`recon_lpips`
- 所有 8 个任务均出现大量 CUDA device-side assert：
  `indexSelectSmallIndex Assertion srcIndex < srcSelectDimSize failed`
- Python stack 最后多落在：
  `world_model.py -> head_observations -> slicer.py`
- 结论：该 stack 受 CUDA 异步报错影响，不代表 `slicer.py` 一定是根因。更合理的根因是旧 root inference 对 mixed ready env batch 的 action 处理错误：batch 中同时存在 first-step 的 `-1` action 和 continuing env 的合法 action 时，旧逻辑会把 `-1` 作为普通 action token 进入 `act_embedding_table`，从而触发 embedding index 越界，并最终表现为 CUDA device-side assert。
- 当前最新代码状态：该问题目前不存在。证据包括：
  1. 最新代码已没有旧式 `max(batch_action) == -1` 的整批判断。
  2. `world_model.py` 已按 per-action flag 拆分 first-step / continuing 子 batch。
  3. 本地 mixed action smoke 使用 `[-1, 3, -1]` 已通过。
  4. 当前运行中的 1GPU fallback rjob 日志中未发现 `Traceback`、`device-side assert`、`srcIndex < srcSelectDimSize`、`CUDA error`、`URLError` 或 `OLD system`。

之前 LPIPS 下载失败的 rjob：

- 路径：`/mnt/shared-storage-user/puyuan/code/LightZero/rjob_logs/atari_unizero_segment/uz-baseline-pong-mspacman-seed0-260622_122346`
- 报错：`urllib.error.URLError: <urlopen error [Errno 110] Connection timed out>`
- 原因：torchvision 在 rjob worker 上尝试下载 VGG16 权重。
- 当前状态：已修复为本地加载 `/mnt/shared-storage-user/puyuan/code/LightZero/tokenizer_pretrained_vgg` 下的权重。

## 7. rjob 实验状态

初始提交命令：

```bash
RUN_TAG=uz-kvfix-pong-mspacman-seed0-260622_130613 \
RJOB_CHARGED_GROUP=narmodel_gpu SAVE_CKPT=0 USE_NEW_CACHE_MANAGER=1 MAX_ENV_STEP=60000 \
MODE=multitask ATARI_ENVS=ALE/Pong-v5,ALE/MsPacman-v5 SEEDS=0 \
BASELINE_VARIANTS=fast_noaux,compile_noaux,recon_only,recon_lpips MAX_PARALLEL=8 \
bash zoo/atari/runs/rjob/rjob_atari_unizero_segment_1node_8gpu.sh
```

初始 rjob：

- `uz-atari-unizero-segment-1n8g-99630310`
- 状态：因 gang scheduling 资源不足，一直 pending，随后停止。

fallback 尝试：

1. `uz-atari-unizero-segment-1n8g-77761305`
   - 2 GPU、64 CPU、600000 memory、`MAX_PARALLEL=2`
   - Pong/MsPacman seed0，四个 variants
   - 调度事件：`Insufficient cpu` 和 `Insufficient nvidia.com/gpu`
   - 已停止，避免重复排队。

2. `uz-atari-unizero-segment-1n8g-88031626`
   - 1 GPU、16 CPU、200000 memory、`MAX_PARALLEL=1`
   - Pong/MsPacman seed0，四个 variants 串行执行
   - 截至 `2026-06-22 14:30:27` Asia/Hong_Kong：`Running`
   - 运行节点：`gpu-lg-cmc-h-h200-1442.host.h.pjlab.org.cn`
   - 日志路径：`/mnt/shared-storage-user/puyuan/code/LightZero/rjob_logs/atari_unizero_segment/uz-kvfix-pong-mspacman-seed0-260622_132610_1gpu`
   - TensorBoard 路径：`/mnt/shared-storage-user/puyuan/code/LightZero/data_unizero/rjob/uz-kvfix-pong-mspacman-seed0-260622_132610_1gpu`
   - 当前日志扫描未发现 `Traceback`、`device-side assert`、`srcIndex < srcSelectDimSize`、`CUDA error`、`URLError` 或 `OLD system`。

## 8. 可切换分区的启动命令

脚本名虽然保留 `1node_8gpu`，但实际资源由环境变量控制。考虑到 `narmodel_gpu` 当前约 5 GPU 且可能存在资源不连续，建议优先用 4 GPU；如果仍排队，再降到 1 GPU 串行。

在 `narmodel_gpu` 上启动 4 GPU：

```bash
cd /mnt/shared-storage-user/puyuan/code/LightZero

RUN_TAG=uz-kvfix-pong-mspacman-seed0-$(date +%y%m%d_%H%M%S) \
RJOB_CHARGED_GROUP=narmodel_gpu \
RJOB_GPU=4 RJOB_CPU=48 RJOB_MEMORY=500000 MAX_PARALLEL=4 \
SAVE_CKPT=0 USE_NEW_CACHE_MANAGER=1 MAX_ENV_STEP=60000 \
MODE=multitask ATARI_ENVS=ALE/Pong-v5,ALE/MsPacman-v5 SEEDS=0 \
BASELINE_VARIANTS=fast_noaux,compile_noaux,recon_only,recon_lpips \
bash zoo/atari/runs/rjob/rjob_atari_unizero_segment_1node_8gpu.sh
```

在 `rlinfra_gpu` 上启动 8 GPU：

```bash
cd /mnt/shared-storage-user/puyuan/code/LightZero

RUN_TAG=uz-kvfix-pong-mspacman-seed0-$(date +%y%m%d_%H%M%S) \
RJOB_CHARGED_GROUP=rlinfra_gpu \
RJOB_GPU=8 RJOB_CPU=150 RJOB_MEMORY=1500000 MAX_PARALLEL=8 \
SAVE_CKPT=0 USE_NEW_CACHE_MANAGER=1 MAX_ENV_STEP=60000 \
MODE=multitask ATARI_ENVS=ALE/Pong-v5,ALE/MsPacman-v5 SEEDS=0 \
BASELINE_VARIANTS=fast_noaux,compile_noaux,recon_only,recon_lpips \
bash zoo/atari/runs/rjob/rjob_atari_unizero_segment_1node_8gpu.sh
```

在资源紧张时的 1 GPU 串行版本：

```bash
cd /mnt/shared-storage-user/puyuan/code/LightZero

RUN_TAG=uz-kvfix-pong-mspacman-seed0-$(date +%y%m%d_%H%M%S)_1gpu \
RJOB_CHARGED_GROUP=narmodel_gpu \
RJOB_GPU=1 RJOB_CPU=16 RJOB_MEMORY=200000 MAX_PARALLEL=1 \
SAVE_CKPT=0 USE_NEW_CACHE_MANAGER=1 MAX_ENV_STEP=60000 \
MODE=multitask ATARI_ENVS=ALE/Pong-v5,ALE/MsPacman-v5 SEEDS=0 \
BASELINE_VARIANTS=fast_noaux,compile_noaux,recon_only,recon_lpips \
bash zoo/atari/runs/rjob/rjob_atari_unizero_segment_1node_8gpu.sh
```

## 9. 验收状态

- [x] 已审查新旧 KV-cache 路径。
- [x] 已定位旧路径中会影响两套实现的 bug：env-id 对齐、cache miss 长度记录、mixed first-step/continuing batch 处理。
- [x] 新 manager 已通过固定 slot copy reuse 降低重复分配。
- [x] collector/evaluator root inference 已统一使用真实 env-id 顺序。
- [x] LPIPS 已改为本地权重加载，不再依赖网络下载。
- [x] 调试期 rjob 默认关闭 checkpoint 保存。
- [x] 已向 `narmodel_gpu` 提交 Pong/MsPacman seed0 四 variants 实验。
- [x] 旧 `260622_130012` 的最终 CUDA assert 当前代码已规避，并通过本地 smoke 与当前 rjob 日志初步验证。
- [ ] 稳定 rjob 完整跑完。当前 1GPU fallback 正在运行中。
- [ ] 新实现超过 Pong/MsPacman 既有 baseline。需要等完整 TensorBoard/eval 指标产生后判断。

## 10. 遗留工作

当前实现不能在完整实验完成前声明“超过基线”。后续需要继续解析：

- 每个 task 的 `train.log`
- run tag 下的 TensorBoard event metrics
- reward 曲线、loss 曲线、collector/evaluator throughput、最大显存
- 是否完全没有 `Traceback`、`OLD system`、VGG 下载、KV shape/assert 相关错误

只有当 Pong 与 MsPacman 在可比 envstep budget 下的 eval reward 均超过第 6 节记录的历史 baseline，且吞吐和显存没有退化，才能认为最终验收全部达成。
