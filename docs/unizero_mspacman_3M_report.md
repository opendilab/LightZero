# UniZero MsPacman 3M 历史复现与单因子矩阵报告

生成时间：2026-08-24 HKT。代码分支：`polish-uz-mspacman`。历史目录
`data_unizero` 在本次工作中只读，未创建、覆盖或删除任何内容。

## 结论摘要

baseline 选用
`data_unizero/experiments/encoder_clip_ablation_0810/no_encoder_clip_from_iter85000_seed0_10m`：
它是正确 frame-skip 协议下末段均值最高、且峰值到末点评估只回落 5.8% 的候选。
TensorBoard 的最后 10% eval-step 窗口含 8 点，均值 2714.69；末点
3038.75@4,895,351，训练实际到 4,954,244 env steps。原 evaluator 文本中这 8 个
reward mean 可在
`data_unizero/experiments/encoder_clip_ablation_0810/no_encoder_clip_from_iter85000_seed0_10m/log/evaluator/evaluator_logger.txt:1283`
至 `:1493` 逐点核对，停止快照也记录为 3038.8@4.895M、最近 6 点均值 2652.7
（`data_unizero/iteration/ITERATION_LOG.md:408`）。

结论不是“已获得多 seed 最优”：扫描到的协议有效实验全部只有 seed0，而且最佳 run 是从
iteration 85,000 / envstep 1,459,525 分叉出来的 continuation
（`data_unizero/iteration/ITERATION_LOG.md:399`）。本轮严格复现它的**声明 config**并从
fresh seed0 开始；不加载历史 checkpoint，防止 baseline 偷带训练历史。seed1/2 config 已预置，
待 seed0 在 300k 门槛后追加。

## 历史实验考古

扫描命令为：

```bash
python scripts/analyze_unizero_mspacman_history.py data_unizero --output /tmp/mspacman_history.json
```

脚本按 config 内的 `ALE/MsPacman-v5` 发现 run，而非依赖目录名；按
`evaluator_step/eval_episode_return_mean` 合并 resume event，并在相同 env step 冲突时保留
wall-time 更新的记录。共解析 36 个 run、错误 0。末段定义为
`eval env_step >= 0.9 * max_eval_env_step`。以下是排除 retired frame-skip 目录后的 Top-5：

| 排名 | 实验路径 | config | commit / 分支 | peak return | 最后 10% 均值 | max env step | seeds | 状态 |
|---:|---|---|---|---:|---:|---:|---:|---|
| 1 | `experiments/encoder_clip_ablation_0810/no_encoder_clip_from_iter85000_seed0_10m` | `formatted_total_config.py` | `47d3269e`; `polish-uz-mz` | 3225@3,848,164 | 2714.69 (8点) | 4,954,244 | 1 | 用户止损于 4.95M；无 crash |
| 2 | `experiments/mspacman/baselines/mspacman_frameskipfix_peroff_kvrolling_ctx5_olc1_prefix3_bootvalctx_resume20k_seed0_0809r2` | `formatted_total_config.py` | `polish-uz-mz`; SHA 未嵌入日志 | 2177.5@413,486 | 2177.5 (1点) | 420,506 | 1 | 早停，证据很短；无 crash |
| 3 | `experiments/mspacman/ablations/mspacman_frameskipfix_peroff_kvrolling_ctx10_olc1_prefix3_eval25k_seed0_0809` | `formatted_total_config.py` | `polish-uz-mz`; SHA 未嵌入日志 | 1910@413,872 | 1910 (1点) | 417,067 | 1 | 评估探针/早停；无 crash |
| 4 | `experiments/mspacman/ablations/mspacman_frameskipfix_peroff_kvrolling_ctx5_olr01_prefix3_h4_resume_seed0_0809r2` | `formatted_total_config.py` | `polish-uz-mz`; SHA 未嵌入日志 | 1558.75@78,805 | 1415 (1点) | 178,621 | 1 | 短探针/早停；无 crash |
| 5 | `experiments/core_suite_0810_resume/mspacman_full_ctx5_bootctx_olc_prefix3_seed0_10m_r2` | `formatted_total_config.py` | `polish-uz-mz`; 0810 resume 系列 | 1956.25@222,962 | 1389.06 (4点) | 2,517,195 | 1 | 明确止损；无 crash |

表中的 config 均位于 `data_unizero/` 下对应 run。第 5 项停止原因和末段范围见
`data_unizero/iteration/ITERATION_LOG.md:403`。0809 三个短实验在生成 config/console 中未嵌入
SHA，因此只报告可追溯的分支，不猜测 commit。

另外，未进入表格的 apparent Top-1 为
`data_unizero/archive/atari_frameskip16_retired/mspacman_sync_peron_stabfix_seed0_0808`
（末段均值 2928.89、峰值 3920），但项目索引明确说明该目录使用错误 frame-skip、MDP 不可比
（`data_unizero/INDEX.md:60-64`），因此按最保守决策剔除。Top-5 也没有任何 ≥2 seed
候选，baseline 依据规则 2 选取单 seed 中末段最高且没有 >20% 峰值回落者。

## baseline 的可复现来源

历史生成 config 的关键字段在
`data_unizero/experiments/encoder_clip_ablation_0810/no_encoder_clip_from_iter85000_seed0_10m/formatted_total_config.py`：
frame-skip=4 / RGB stack1 在 `:37-47`，world model（H10、ctx5、raw-token rebuild、
OLC1/prefix3）在 `:70`，replay/optimizer/loss/temperature/PER 在 `:159-202`，
bootstrap 与 KV interval 在 `:217-254`。分叉时的代码锚为
`47d3269ed141157ef10b97558da84f5679329429`，由
`data_unizero/iteration/ITERATION_LOG.md:399` 和 `git show 47d3269e` 交叉确认。

新的声明式 config 位于
`zoo/atari/config/unizero_mspacman_3m/common.py`，12 个 seed/group 入口位于同目录。
baseline 相对历史生成 config 只做：

- `max_env_step: 10M -> 3M`、fresh seed、唯一新输出名；
- checkpoint I/O 由每 1k iter 改为每 5k iter、只保留最近 2 个、关闭 eval-best 保存并允许
  checkpoint 写失败后继续。这是 GPFS 上的硬件/存储适配，不改变采样、target 或 loss；
- 使用当前 HEAD 的 correctness 修复。特别是历史 config 声明 `value_loss_weight=0.25`，但
  commit `47d3269e` 的 `LossWithIntermediateLosses` 实际硬编码 0.5（可用
  `git show 47d3269e:lzero/model/unizero_world_models/utils.py` 的 283-314 行核对）。当前代码会真正
  执行 0.25（`lzero/policy/unizero.py:1323-1329`、
  `lzero/model/unizero_world_models/utils.py:331-365`）。这是代码语义修复造成的不可避免差异，
  没有在 baseline 中偷偷改回；v3 单独测试有效权重 0.5。

## 已试过的方向

| 方向 | 历史结果 | 证据与本轮裁决 |
|---|---|---|
| contextual bootstrap | matched 25k 从 1137.5 提至 2177.5，但 bootstrap-only 到 0.83M 最近 6 点 601、负斜率 | `data_unizero/INDEX.md:72-77`；`ITERATION_LOG.md:407`。组合中保留，不再作为单独变体。 |
| ctx5 单独 | 最终到约 4.17M，峰值后明显回落；不是充分条件 | TB 扫描结果；`ITERATION_LOG.md:403-404` 给出单因子设计。baseline 仍保留历史组合的 ctx5。 |
| OLC1/prefix3 单独 | 1.94M 最近 6 点均值 425，已止损 | `ITERATION_LOG.md:407`。不重复 OLC weight 变体。 |
| contextual reanalysis 0.02 | reanalysis-only 到约 2.73M，末段均值约 595 | TB 扫描；启动机制见 `ITERATION_LOG.md:405`。不重复。 |
| encoder latent-norm clip | fresh on/off 到约 5M 都弱；clip 不是唯一瓶颈 | `ITERATION_LOG.md:408-409`。baseline 采用历史 best 的 clip-off，但不再做 clip 变体。 |
| adaptive entropy + PER + grad20 + replay1M | 3.464M 最近 6 点均值 953.9，未超过旧 continuation | `ITERATION_LOG.md:406,408`。不重复该打包 recipe。 |
| temp1 + encoder grad cap + OLC0.1 | 0.48M 峰值 2123 后回落；后续 TB 扫描到 5.86M 末段均值 1199 | `ITERATION_LOG.md:408,410`。不重复 temperature/grad-cap 打包。 |
| H5 vs H10 | 早期 collect H10 均值 13.50 vs H5 11.72，且 TD priority 更平滑；当时尚无充分固定 eval | `20260822_...frontier_runs.md:471-510`。本轮不再改 train H；所有组固定历史 H10，避免把监督 token 数和机制混淆。 |
| coherent augmentation | 旧 sequence 内逐帧随机增强会破坏运动一致性，已修成 sequence 共用变换 | `20260822_...frontier_runs.md:341-356`。历史 best 为 augmentation-off，本轮 baseline 维持 off，不拿已修 bug 当变体。 |

有效趋势是：ctx5 + bootstrap-context + OLC/prefix 的**组合**曾显著高于 ordinary bootstrap，
而这些特征单独都不足；H10 的每 update target token 更多、早期 TD 更平滑；长期最佳仍使用
PER-off、augmentation-off、temperature0.25 和 value 实际权重0.5。由于组合和 continuation
初始化互相纠缠，本轮只做 exact-config fresh control 与三个单机制差分。

## 性能回退根因

“当前实验”按任务规则取 `data_unizero` 中 mtime 最新的 MsPacman config：
`data_unizero/experiments/mspacman_frontier_0822_v2/uz_h10_value05-seed0-3m/formatted_total_config.py`
（mtime 2026-08-23 02:52:25 HKT）。它只到约 241k envstep，末点 456.67；其 evaluator 原始
3 局值见 `.../log/evaluator/evaluator_logger.txt:166-174`。

### Config diff：historical best vs 当前实验

下表是递归展开后的全部差异；`missing` 表示字段不在旧生成 config，需由当时 policy 默认补齐。

| 字段 | historical best | 当前 | 可能影响 |
|---|---:|---:|---|
| `env.n_evaluator_episode` / policy evaluator episodes | 8 | 3 | 当前单点评估方差更大 |
| `policy.evaluator_env_num` | 8 | 3 | 同上，且改变 cache 容量 |
| `world_model.context_length` | 10 tokens (ctx5) | 8 (ctx4) | 少一组历史 observation/action |
| `world_model.env_num` | 8 | 11 | isolation 的派生容量，正确性改善 |
| `use_new_cache_manager` | false | true | cache 实现变化 |
| `root_cache_key_round_decimals` | missing/legacy 0 | 4 | 当前可容忍 batch 浮点噪声 |
| `open_loop_consistency_loss_weight` | 1.0 | 0 | 移除最佳配置的 rollout 约束 |
| `open_loop_prefix_transitions` | 3 | 0 | 移除真实历史 prefix |
| `open_loop_consistency_batch_size/horizon` | 8 / 4 | missing | OLC 关闭后的派生缺失 |
| `open_loop_diagnostic_freq` | 1000 | 0 | 只影响可观测性 |
| `policy.value_loss_weight` | 0.25（当时代码实际0.5） | 0.5 | 当前显式0.5；与旧有效目标一致但声明不同 |
| `policy.obs_loss_weight` | missing（实际10） | 10 | 数值相同，仅配置显式化 |
| `policy.use_priority` | false | true | replay 分布改变 |
| `policy.use_augmentation` | false | true | 数据分布改变；当前实现已保证 sequence coherent |
| `policy.isolate_eval_cache` | missing/false | true | 正确性改善，防止 eval/collect 逐出 |
| `empty_cuda_cache_on_cache_reset` | true | false | 主要影响吞吐/allocator，不应改回报语义 |
| `resume_buffer_min_transitions` | 100000 | 10000 | 只影响 resume 后的 replay warmup |
| `gradient_diagnostic_freq` | missing | 0 | 只影响可观测性 |
| periodic checkpoint interval | 1000 | 20000 | 只影响恢复粒度/存储 |
| `use_max_priority_for_new_data` | missing | false | PER 打开时改变新样本初始 priority |

关键当前值可直接核对当前生成 config `:25-28,70,154-200,237-257`；历史值核对上述
best config `:25-28,70,153-254`。`exp_name` 也是 diff，但只影响输出路径，未列为学习机制。

### Code diff：`47d3269e..HEAD`

按可能改变数值行为排序：

1. `ab747bc4` 接通 config loss 权重并固定 target model eval。当前 target 在初始化和每次 learn
   都强制 eval（`lzero/policy/unizero.py:1063-1066,1223-1226`），而历史 commit 的 target 是
   `train()`；这是明确 correctness 修复，但同时意味着本轮 fresh baseline 不会复现历史 dropout
   噪声和硬编码 value=0.5。
2. `ab747bc4` + `84c56aa7` 修复 root-key 容差、per-env interval clear、eval namespace。
   当前 per-env clear 位于 `lzero/policy/unizero.py:2345-2365,2430-2439,2527-2539`，root
   rounding 位于 `lzero/model/unizero_world_models/world_model.py:955-968`。这些应消除污染，
   没有证据表明它们导致回退。
3. `4eff7252` 修复 segment in-flight/replay index、resume 与 config 分层；`515570fd` 修复
   coherent augmentation；`f797c265` 去除 iteration-0 重复 eval。这些属于数据完整性/吞吐修复。
4. `659a377d` 将 world-model context 路径模块化，是最大结构性 refactor；现有 KV focused tests
   约束等价，但不能由单测证明训练曲线完全不变，列为残余 code-risk。

判定：**主要是 config 漂移与历史 continuation 初始化优势叠加；不是已有证据支持的“代码修复导致
回退”**。当前 config 同时移除了 ctx5/OLC-prefix、打开 PER/augmentation、缩小 eval 样本；
而历史最佳从成熟 checkpoint 分叉，fresh 严格 on/off 对照本来就弱
（`ITERATION_LOG.md:408`）。代码改动确实令 effective value weight 和 target dropout 与历史不同，
因此属于次要交互项，v3 专门隔离前者。

## 三个单因子变体

所有未列字段与 baseline 完全一致；自动化测试
`zoo/atari/tests/test_unizero_mspacman_3m_config.py` 会逐键断言只有预注册 diff。

### v1：eval/collect cache namespace 隔离（cache 生命周期）

- 瓶颈假设：8 个长 episode evaluator 会复用 collector env-id，回绕 init pool 并破坏暂停中的
  collector root history；这对 MsPacman 的历史状态尤其有害。机制证据见
  `20260822_...frontier_runs.md:145-147,189-197`。
- 唯一机制：`isolate_eval_cache false -> true`；派生的 `world_model.env_num 8 -> 16` 只是容量要求。
- 可证伪预测：首个 eval 后 console 的 root hit 统计不出现 namespace 竞争/越界；到 300k 时最近
  3 次 eval 均值至少比 baseline 高 15%。若 cache 健康指标相同且回报无提升，则“隔离是主要
  性能瓶颈”被证伪，但它仍是 correctness 改善。

### v2：在线 inference context 5 -> 10 blocks（信息状态）

- 瓶颈假设：ctx5 只保留约 5 个 observation/action blocks，无法稳定辨别幽灵方向、路线阶段等
  部分可观测状态；H10 learner 已有容纳 ctx10 的容量。
- 唯一机制：`infer_context_length 5 -> 10`，即 `world_model.context_length 10 -> 20 tokens`；
  H10、batch256、RR0.1 不变。
- 可证伪预测：在 100k 前 `analysis/open_loop_exposure_ratio` 或
  `analysis/rolling_teacher_latent_mse_mean` 相对 baseline 至少改善 10%，且 300k 最近 3 次 eval
  均值高 15%。若模型误差/return 都不改善或显存失败，则长历史假设被证伪。
- 与旧 ctx10 的差异：旧记录只是同 checkpoint 单次 1910 分评估、最多 417k 总步，未形成 fresh
  长训结论（`data_unizero/INDEX.md:79-83`）；本轮是历史最佳完整训练配方上的唯一 ctx 差分。

### v3：value loss 声明权重 0.25 -> 0.5（target/优化尺度）

- 瓶颈假设：历史最佳虽然 config 写 0.25，实际代码硬编码执行 0.5；修复 config plumbing 后按
  0.25 执行，可能把长回报 value 信号减半，造成行为改善滞后。
- 唯一机制：`value_loss_weight 0.25 -> 0.5`。
- 可证伪预测：100k 内 `value_loss` / `replay/value_priority_mean` 的下降斜率和
  `value_calibration/bias` 优于 baseline，同时 `grad/world_model_total_norm` 有限；300k 最近 3 次
  eval 均值至少高 15%。若 TD/calibration 不改善或 obs/total loss 恶化并且 return 更低，则否定。
- 与旧 value0.5 的差异：0822 H5/H10 组同时使用 PER、augmentation、ctx4、cache isolation，
  不是对历史最佳的单因素复现；本轮除 value 权重外逐键一致。

完整 config diff：

| group | 相对 baseline 唯一显式 diff | 派生 diff |
|---|---|---|
| baseline | 无 | shared cache `env_num=8` |
| v1 | `isolate_eval_cache: false -> true` | `env_num: 8 -> 16` |
| v2 | `infer_context_length: 5 -> 10` | `context_length: 10 -> 20 tokens` |
| v3 | `value_loss_weight: 0.25 -> 0.5` | 无 |

## 提交、监控与 kill 判据

- 资源：启动器同时支持单 worker 4 GPU、一个 4-task RJob 和四个独立 1-GPU shard；seed0
  每组一 GPU，默认不跑 Pong 门禁。seed1/2 入口全部已生成，可通过
  `UZ_SEED_QUEUE=0,1,2` 排队。
- 输出：`/mnt/shared-storage-gpfs2/trustcyberdata/private/docker-infra/tmp/puyuan/rl/lightzero/unizero_mspacman_3m_20260824`，不写 `data_unizero`。
- 幂等恢复：已有 `.completed` 则跳过；否则选择最新完整 zip checkpoint resume；存在无完整
  checkpoint 的失败目录会改名保留，绝不覆盖；每组最多 3 次启动（初次 + 2 次重试）。
- OOM 自愈：仅在日志明确匹配 CUDA OOM 时，重试显式 batch256 -> 128，并在 controller log
  记录语义变化；其他组不受影响。
- 启动健康守卫：每组 Python 启动后每 60 秒记录 envstep、最新 eval 和明确的
  NaN/OOM/Traceback 告警到 `controller/health_*.log`；达到 50k 后生成
  `.startup_health_50k`，因此即使 GPU 在排队后才释放，前 50k 巡检仍会自动执行。
- kill 建议：四组均达到 300k 后，取各自截至 300k 的最近 3 个 eval；若变体均值低于 baseline
  35%，标记 `KILL_RECOMMENDED`。35% 大于 MsPacman 3/8-episode 单点常见噪声，避免因一次坏局
  误杀；默认只建议、不自动杀，命令是
  `bash scripts/launch_unizero_mspacman_3M.sh check-kill`。

## 假设与不确定性

1. 所有 protocol-valid 历史 run 都是 seed0；没有可满足“≥2 seed 一致”的候选。
2. 最佳 run 是 1.459M checkpoint continuation，而不是 fresh；本轮 fresh 结果可能显著低于其
   历史后半段，这不等价于 config 复现错误。
3. 0809 三个短 run 未在 config/console 嵌入 SHA；报告只归属 `polish-uz-mz`，不猜 commit。
4. 历史声明 value=0.25、有效代码 value=0.5；baseline 复现声明 config，v3 复现有效 loss 权重。
5. 当前代码有 target-eval/cache/replay correctness 修复，故无法同时做到“当前 HEAD”与
   “47d3269e 逐 bit 训练行为”一致；本轮选择 current HEAD，所有差异在上文列明。
6. 当前实验按 mtime 规则是 data_unizero 内 0822 H10，而不是 GPFS 上后来启动、但不属于
   `data_unizero` 的 0823 worker；这是严格执行任务给定的自动裁决规则。

## 决策裁决记录

- 剔除 retired frame-skip16 高分，因为是不同 MDP；不以峰值跨协议排名。
- 单 seed 条件下以末段均值、峰值回落和 eval 点数选 no-clip continuation。
- 不加载历史 checkpoint：任务要求 seed0/1/2 可复制，warm-start 会让 seed 和 baseline 定义失真。
- 三个机制分别选 cache 生命周期、信息状态、value 尺度；不重复已证伪的 bootstrap-only、
  OLC-only、reanalysis-only、encoder clip 或 temp1 打包方向。
- baseline 保留 shared eval cache 以做到 config 精确复现；v1 单独量化 correctness 修复收益。
- 使用 GPFS 输出和稀疏 checkpoint，避免共享用户盘 I/O 成为吞吐瓶颈。

## 启动状态

- 14:12 HKT 提交 `uz-mspacman-repro-3m-0824-mt-r2`（四个 1-GPU task），调度事件持续为
  `4/4 tasks in gang unschedulable`；14:41 停止，未创建实验目录。
- 14:41 HKT 用 `--gang-start=false` 验证 `uz-mspacman-repro-3m-0824-mt-r3`，平台仍生成
  `4 minAvailable`，且 quota webhook 返回 `gpu: 57/56`；14:45 停止，未创建实验目录。
- 自动裁决：平台的 `--gang-start` 控制 IP 注入而不是取消多任务的 gang `minAvailable`。为了在
  共享 quota 每次只释放一张 GPU 时也能推进，退化为四个独立 1-GPU RJob；这只改变调度拓扑，
  不改变 config、seed 或输出。
- 14:45 HKT 提交 best-effort shards `uz-mspacman-repro-3m-0824-s1-{baseline,v1,v2,v3}`；
  四组持续被 admission webhook 以 `insufficient project quota: gpu : 57/56` 拒绝，14:54 停止，
  未创建实验目录。14:55 HKT 用 guaranteed QoS 提交最终任务
  `uz-mspacman-repro-3m-0824-s2-{baseline,v1,v2,v3}`；同样的 quota 拒绝证明瓶颈是项目级硬配额，
  不是 QoS。截至 15:30 四个最终 replica 连续 35 分钟保持 `PENDING`、failed count 为 0，最新
  webhook 仍为 `gpu: 57/56`。任务保留在队列中，会在 quota 释放时逐组自动准入。
- controller 日志固定为输出根下
  `controller/gpu0_unizero_mspacman_{group}_seed0_3M.log`；TensorBoard 固定为对应 run 目录的
  `log/{serial,evaluator,collector}`。实际 replica/node、初始 eval 和前 50k 健康状态待启动后补齐。

### 2026-08-25：复用 8-GPU SETA worker

- 12:19 HKT 确认四个 `s2-{baseline,v1,v2,v3}` shard 仍为 `PENDING`、未创建 run 目录，随后按
  用户指令全部停止。
- `setaevalbox-0823-2300` 为当前用户的 8×H200 RJob，原 SETA evaluation 已结束、PID 1 为
  `sleep infinity`，节点为 `gpu-lg-cmc-h-h200-0666`。Pod exec 受平台 RBAC 禁止，节点 SSH 也不
  接受当前密钥；RJob task 的 ConfigMap 是不可写 source-of-truth，command-only CRD patch 会被
  controller 回滚。
- 回收旧 sleep replica 后，`restartPolicy=Never` 使原 RJob 转为 `Stopped`。为避免重新执行旧
  SETA 命令，12:34 HKT 以相同 8 GPU/64 CPU/800 GB、narmodel private group、镜像和 GPFS
  挂载提交 `setaevalbox-0823-2300-unizero`。调度事件将它 pipelined 回原 H200-0666 节点，
  12:37 转为 `Running`。
- 12:36:54–12:36:57 四组正式进程启动：baseline PID 166、v1 PID 168、v2 PID 170、v3
  PID 164，分别由 `CUDA_VISIBLE_DEVICES=0,1,2,3` 隔离，进程内均只看见 `cuda:0`。
- 四份 formatted config 已核验：baseline shared-cache/ctx5/value0.25；v1 仅 isolate cache
  （派生 env capacity 16）；v2 仅 ctx10（20 tokens）；v3 仅 value0.5。12:37/12:38 的四份
  health log 均为 `HEALTH_OK`，无 Traceback/OOM/NaN。
- 12:39 四组均完成初始 8-episode eval，八局均为 60、mean=60；随后均完成首批 8 segments、
  bootstrap-context diagnostics 和 learner iteration 0。首次训练的 latent L2 norm 约 27.7128，
  losses 为有限值，证明四组已越过“只初始化未训练”的门槛。
