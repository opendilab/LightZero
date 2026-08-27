# UniZero MsPacman 5M：双实验根因分析、修复与四臂长跑记录

- 记录时间：2026-08-22 17:38:50 HKT
- 工作分支：`polish-uz-mspacman`
- 基线代码锚点：`4eff7252`
- 本轮修复提交：`ab747bc4`、`84c56aa7`
- 新 RJob：`uz-mspacman-frontier-0822`
- 目标：在 5M envsteps 内使 UniZero MsPacman 的评估中枢形成可持续上升趋势，并定位此前“早期冲高、随后回落/横盘”的结构性原因。

## 1. 执行摘要

两条历史实验都没有表现出可外推到 5M 的持续上升斜率，但失败机制并不完全相同：

1. 旧 baseline 的首要问题是训练样本有效率和信息状态不足。`game_segment_length=20` 与 `num_unroll_steps=10`、`td_steps=5` 组合后，每段大约只有前 5 个 root 能形成完整目标，约 75% 的 segment 位置不能成为等价的完整训练 root；同时 PER、augmentation、bootstrap value context、raw-token KV 重建均关闭。
2. experimental run 用 GSL200、bootstrap context、raw-token KV 重建和 temperature 0.25 改善了数据覆盖与行为策略，但当时的训练代码仍存在 target dropout、loss 权重配置静默失效、root hash 脆弱、interval 全局清 cache、eval/collect cache 竞争等问题，因此出现 `893 -> 1643 -> 730 -> 810 -> 507 -> 697` 的早期冲高后回落。
3. MsPacman 的长 episode、部分可观测性、稀疏且长程的回报，把“上下文状态损坏”和“value target 移动/带噪”同时放大；Pong 的短 episode、低回报范围和近似单帧 Markov 性使其对这些缺陷相对不敏感。
4. 本轮已经修复训练 target、loss 配置和 KV 生命周期，并以统一 correctness 配方启动四个 5M 变体，分别检验 value 权重、reanalyze 和 open-loop consistency。

## 2. 两条实验及证据

### 2.1 旧 baseline

实验目录：

```text
/mnt/shared-storage-user/puyuan/code/LightZero/data_lz/data_unizero_segment/MsPacman/MsPacman_uz_nlayer2_gsl20_rr0.25_Htrain10-Hinfer4_bs64_seed0
```

用户提供的运行片段已精确定位到：

```text
log/collector/collector_logger.txt:62057-62070
2026-08-22 07:37:56
total_envstep_count: 1354038
avg_envstep_per_sec: 37.256047556793355
reward_mean: 31.0
```

主要配置：

| 维度 | baseline |
|---|---:|
| game segment length | 20 |
| train unroll / TD steps | 10 / 5 |
| replay ratio | 0.25 |
| batch size | 64 |
| norm | BN |
| PER | off |
| augmentation | off |
| bootstrap value context | off |
| raw-token KV rebuild | off |
| value/reward support | `[-50, 50]` |

后期评估样本：

| train iteration | eval return mean |
|---:|---:|
| 281340 | 573.33 |
| 286409 | 596.67 |
| 291413 | 360.00 |
| 296433 | 426.67 |
| 301448 | 356.67 |
| 306487 | 270.00 |
| 311517 | 393.33 |
| 316575 | 226.67 |
| 321604 | 786.67 |
| 326625 | 363.33 |

结论：到约 1.35M envsteps 后，曲线仍主要在数百到约 1k 的区间高方差震荡，没有形成随训练预算增长而提高的中枢。

### 2.2 experimental run

实验目录：

```text
/mnt/shared-storage-user/puyuan/code/LightZero/data_unizero/mspacman-uz-experimental-seed0-5m
```

主要配置：

| 维度 | experimental |
|---|---:|
| game segment length | 200 |
| replay ratio | 0.1 |
| batch size | 256 |
| norm | LN |
| collect temperature | 0.25 |
| PER | off（该旧 run） |
| augmentation | off |
| bootstrap value context | on |
| raw-token KV rebuild | on |
| root cache clear interval | 2000 |
| value/reward support | `[-300, 300]` |

完整评估轨迹：

| train iteration | eval return mean |
|---:|---:|
| 0 | 60.00 |
| 5059 | 893.33 |
| 10102 | 1643.33 |
| 15187 | 730.00 |
| 20245 | 810.00 |
| 25319 | 506.67 |
| 30337 | 696.67 |

结论：更好的上下文和采样配置提高了早期峰值，但没有消除 value/representation 动靶及 cache 生命周期问题，10k iter 的 1643 峰值没有维持。

## 3. 第一性原理因果链

UniZero 的一次有效学习闭环可简化为：

```text
真实观测历史
  -> root representation / KV context
  -> MCTS policy 与 bootstrap value
  -> segment 写入 replay
  -> 构造 reward/value/policy/observation targets
  -> 更新 encoder + transformer + heads
  -> 新模型继续读取旧权重时期产生的历史上下文
```

任何一环发生系统性信息损失，都会在长 episode 中累积。

### 3.1 数据有效率不足

baseline 的 GSL20 必须同时容纳 10 步 unroll 和 5 步 TD bootstrap，靠近 segment 尾部的大多数 root 无法拥有完整 horizon。名义 replay ratio 为 0.25，但可形成完整目标的 root 约只有 25%。在 MsPacman 中，高价值事件本来就稀疏；再关闭 PER，会使这些事件更难被反复学习。

GSL200 能显著提高 root 覆盖率，因此新实验统一采用 GSL200，同时开启 PER，而不是继续把大量 MCTS envstep 消耗在低复用 segment 上。

### 3.2 行为状态与 bootstrap 状态不一致

当 `bootstrap_value_context=False` 时，TD target 的 bootstrap value 可以由一个与行为策略实际历史不同的信息状态计算。对于需要利用速度、方向、幽灵运动历史的 MsPacman，这不是普通观测噪声，而是 value target 的条件变量发生变化。

因此新实验统一启用：

- `bootstrap_value_context=True`
- `rebuild_kv_window_from_tokens=True`
- new cache manager

### 3.3 root KV 生命周期损坏

已确认的旧路径问题：

1. root cache 使用重编码 observation 的 float32 原始字节作为 hash。batch 宽度从 `3 -> 2 -> 1` 或 first-step/continuing 混批时，约 `1e-7` 的数值差异即可造成完全不同的 hash，使已有上下文退化为单 token。
2. `kv_cache_clear_interval=2000` 原本由每个 env 的步数计数触发，但执行的是模型级全局 `clear_caches()`；任意一个 env 到达阈值都会清掉所有 env 的 root 历史。
3. evaluator 和 collector 使用同一模型及 cache pool。长评估 episode 会回绕并逐出暂停中的 collector 上下文。
4. learner 更新后，已有 KV/embedding 来自旧 encoder 权重；这属于跨模型版本的上下文陈旧问题。当前 raw-token rebuild 能降低影响，但还需要后续 model-version 监控验证。

### 3.4 target 与 loss 动力学不稳定

旧实现中 target model 被置于 `train()`，使 dropout 进入 observation target 和其他 target 前向。target 本应是相对稳定的监督信号；随机 dropout 会把 weight-10 的 observation MSE 变成随 batch 抖动的移动目标。

此外，loss 权重曾直接硬编码为：

```text
observation = 10
value       = 0.5
reward      = 1
policy      = 1
```

配置中的 `value_loss_weight` 因此静默失效。结果不仅是“value 权重不对”，更严重的是实验记录声称改变了一个变量，而实际优化目标没有改变，导致 A/B 结论不可追溯。

## 4. 已实施修复

### 4.1 提交 `ab747bc4`

提交信息：

```text
fix(pu): stabilize UniZero MsPacman targets and KV lifecycle
```

修改内容：

1. `LossWithIntermediateLosses.set_loss_weights(...)` 在保留 per-sample/PER 路径的同时，根据 policy config 重算总 loss。
2. target model 初始化和每次 target 前向均强制 `eval()`。
3. root cache key 新增 `root_cache_key_round_decimals`；新实验使用 4 位小数量化，吸收无语义的 float32 batch 噪声。
4. interval 清理改为只清触发的 env 对应的 init/raw-token pools，不再全局清理。
5. experimental config 接通 augmentation、replay ratio、batch size、observation/value loss weights、root key round decimals、KV clear interval 等 CLI/config。
6. 新增四臂启动与自动重启脚本。

### 4.2 提交 `84c56aa7`

提交信息：

```text
fix(pu): isolate UniZero evaluation root caches
```

修改内容：

1. 增加 `isolate_eval_cache` 配置。
2. evaluator 的内部 cache env ID 使用 collector env 数量作为 offset。
3. world model pool 容量覆盖 collector + evaluator 两套 namespace。
4. eval 的 episode-end 和 interval clear 只清 evaluator namespace。

### 4.3 修改文件

```text
lzero/model/unizero_world_models/utils.py
lzero/model/unizero_world_models/world_model.py
lzero/policy/unizero.py
lzero/policy/tests/test_unizero_eval_cache_isolation.py
zoo/atari/config/atari_unizero_segment_experimental_config.py
data_unizero/rjob_controllers/mspacman_frontier_0822/run_variant.sh
data_unizero/rjob_controllers/mspacman_frontier_0822/run_all.sh
data_unizero/rjob_controllers/mspacman_frontier_0822/submit_narmodel.sh
```

静态验证：

- `python -m py_compile`：通过。
- 三个 shell launcher 的 `bash -n`：通过。
- `git diff --check`：通过。
- 当前环境中的完整 pytest 调用没有稳定返回结果，因此未将其记为“通过”；后续自然维护窗口应补跑 focused suite。

## 5. 新四臂实验设计

所有变体共享：

| 配置 | 值 |
|---|---:|
| env | `ALE/MsPacman-v5` |
| seed | 0 |
| max envsteps | 5,000,000 |
| collector envs | 8 |
| evaluator episodes | 3 |
| game segment length | 200 |
| collect simulations | 25 |
| collect temperature | 0.25 |
| replay ratio | 0.1 |
| batch size | 256 |
| PER | on |
| augmentation | on |
| bootstrap value context | on |
| raw-token KV rebuild | on |
| new cache manager | on |
| root cache key decimals | 4 |
| observation loss weight | 10 |
| periodic checkpoint | every 10k iteration, keep last 2 |

四个变量臂：

| GPU | run | 变量及目的 |
|---:|---|---|
| 0 | `uz_core_value025-seed0-5m` | value weight 0.25，降低漂移表征上 value 梯度的主导程度 |
| 1 | `uz_core_value05-seed0-5m` | value weight 0.5，作为旧硬编码语义的 matched control |
| 2 | `uz_reanalysis02-seed0-5m` | value 0.25 + contextual reanalysis 0.02，刷新旧 replay 的 target |
| 3 | `uz_openloop_prefix3-seed0-5m` | value 0.25 + recurrent consistency 0.1 + prefix 3，加强真实历史和 imagined rollout 的一致性 |

实验根目录：

```text
/mnt/shared-storage-user/puyuan/code/LightZero/data_unizero/experiments/mspacman_frontier_0822
```

控制器日志：

```text
data_unizero/experiments/mspacman_frontier_0822/controller/gpu0_uz_core_value025.log
data_unizero/experiments/mspacman_frontier_0822/controller/gpu1_uz_core_value05.log
data_unizero/experiments/mspacman_frontier_0822/controller/gpu2_uz_reanalysis02.log
data_unizero/experiments/mspacman_frontier_0822/controller/gpu3_uz_openloop_prefix3.log
```

## 6. 2026-08-22 17:39 HKT 运行快照

RJob `uz-mspacman-frontier-0822` 已创建并运行。四个变体均完成 iteration-0 评估和第一批 collection，且持续写入日志；未发现 `Traceback`、`AssertionError`、NaN 或 OOM。

| run | 最新可见 total envstep（约） | 状态 |
|---|---:|---|
| `uz_core_value025` | 44,142 | collecting/training |
| `uz_core_value05` | 42,721 | collecting/training |
| `uz_reanalysis02` | 41,670 | collecting/training |
| `uz_openloop_prefix3` | 41,462 | collecting/training |

iteration-0 的三局回报均为 60，属于随机初始化共同起点，不作为算法优劣证据。

### 6.1 重要运行时版本注记

四个 Python 进程在 `84c56aa7` 提交完成前已经启动。对各 run 的 `formatted_total_config.py` 复核显示：

```text
isolate_eval_cache=False
world_model.env_num=8
```

因此，本批当前进程已经加载 `ab747bc4` 的 target/loss/root-key/per-env-clear 修复和全部新 recipe，但**没有加载 `84c56aa7` 的 evaluator namespace 隔离**。源码分支现在已经是正确实现；只有重启后新建的进程才会得到：

```text
isolate_eval_cache=True
world_model.env_num=11  # collector 8 + evaluator 3
```

后续比较结果时必须保留此版本事实，不能把当前四条曲线误记为已经验证 eval/collect cache isolation。由于快照时仅约 42k/5M envsteps，若以完整 correctness 为首要原则，应在正式门控前重新启动这四臂或将本批标记为 pre-isolation pilot。

## 7. 判断标准与后续门控

建议按固定 envstep，而不是 train iteration 对齐：

| 门控 | 关注点 |
|---:|---|
| 0.1M | 配置执行、吞吐、loss/grad、PER IS 权重、root cache 命中与清理事件 |
| 0.5M | 是否超过随机早期峰值并形成至少三个 eval 点的上升中枢 |
| 1M | 最近 6 次 eval 均值是否高于前 6 次；若否，停止无信息空跑并进入专项探针 |
| 2M | 是否显著超过历史 UniZero 平台，目标 `>2.5k` |
| 5M | 是否达到约 5k 且仍有正斜率，并与同 envstep MuZero 对照比较 |

主要观测指标：

- eval return 的滑动中枢、斜率和逐局分布；不能只看 3-episode 单点峰值。
- `value_priority` 的均值/方差及随 envstep 的趋势。
- observation embedding norm、feature std、dormant ratio。
- target/policy entropy 与 collect visit entropy。
- PER IS weight mean/ESS，区分采样收益与有效梯度缩放。
- root cache hit ratio、per-env clear 次数、全局 clear 次数（目标为 0）。
- reanalysis 臂的 target age 和实际 realized reanalysis ratio。

## 8. 当前结论边界

已经可以确认：

- 两条旧实验都没有持续上升；吞吐不是主要矛盾。
- baseline 的 GSL20/root 覆盖率、上下文缺失和弱 replay recipe 是实质性限制。
- target dropout、loss 权重静默失效、全局 cache clear 和 hash 脆弱是代码级 correctness 问题，修复不依赖实验结果。
- experimental 的较高早期峰值说明 bootstrap context、GSL200 和 temperature 0.25 方向有价值，但不足以单独解决长期平台。

尚不能确认：

- value weight 0.25 是否优于 0.5；必须由当前 matched pair 给出。
- reanalysis 与 open-loop consistency 是否有独立正收益。
- 四臂是否能在 5M 达到持续上升；当前约 42k envsteps 仅证明成功启动。
- eval/collect cache isolation 的性能收益；当前运行进程尚未加载该提交。

因此，本轮工作的价值首先是把错误的实验语义和不可追溯变量修正为可比较实验；最终性能结论必须等待固定门控数据，而不能从 iteration-0 或单个早期峰值外推。

## 9. 2026-08-22 18:14 HKT 二次审查、吞吐优化与 v2 实验

### 9.1 对 eval/collect cache 隔离重要性的修正判断

隔离仍属于必须保留的 correctness 修复，但在当前 serial segment entry 中不是性能平台的首要解释。训练循环每个 learner epoch 结束都会调用
`recompute_pos_emb_diff_and_clear_cache()`，把旧权重生成的 collect/eval root KV 全部失效；下一次 eval 开始时 collect 历史通常已经为空。因此隔离的主要作用是：

- 保证 evaluator 和 collector 永远不共享 env-id root namespace；
- 防止今后异步/并行 entry、异常恢复或调用次序变化重新引入互相逐出；
- 使 eval 指标语义可审计，而不是直接宣称会带来大幅 reward 增益。

新增 fail-fast 容量检查：启用 `isolate_eval_cache` 时，若 `world_model.env_num < collector_env_num + evaluator_env_num`，初始化立即报错，而不是运行到 eval 时才越界或静默污染。

### 9.2 新发现的直接训练信号 Bug：时序不一致的数据增强

旧 `_forward_learn` 对 root observation 和未来 H 帧 target observation 分别调用随机 shift/intensity。两次调用会独立采样 crop 与强度，从而让第一步 transition 同时包含环境变化和人为变换变化。对 weight-10 的 observation latent MSE，这等价于要求 dynamics 拟合不存在于环境中的随机跳变，能够直接加剧 encoder drift 和 value moving-target。

修复后先沿 channel 维拼接 root 与所有 target 帧，只调用一次增强，再按原 channel 边界拆分。这样一个 replay sequence 内共享相同 crop/intensity，batch 内不同样本仍保留随机性。新增测试断言 transform 只调用一次且拆分位置准确。

### 9.3 运行效率调整

MuZero 参考 run 的命名和图表显示其 learner horizon 为 H5；当前 UniZero H10 在 observation/transformer/target 计算上更重。v2 将三臂切换为 H5，并保留一条 H10 matched control，以直接判断样本效率和墙钟效率的权衡：

| GPU | v2 run | 关键变量 |
|---:|---|---|
| 0 | `uz_h5_value05-seed0-3m` | H5, value weight 0.5，主快速臂 |
| 1 | `uz_h10_value05-seed0-3m` | H10 matched control |
| 2 | `uz_h5_value025-seed0-3m` | H5, value weight 0.25 |
| 3 | `uz_h5_reanalysis02_rbs32-seed0-3m` | H5, value 0.5, contextual reanalysis 0.02, batch 32 |

同时：

- 关闭每个 train epoch cache reset 后的 `torch.cuda.empty_cache()`；root/KV 对象仍会正确清除，只保留 PyTorch allocator 的可复用显存块，避免高频同步和重新分配。
- contextual reanalysis batch 从 160 降至 32，避免一次刷新展开 `160×(H+1)` roots 的长尾停顿。
- 周期 checkpoint 从每 10k 调整为每 20k learner iteration，继续 keep-last-2，降低约 530MB checkpoint 的 I/O 干扰。
- controller 的失败重启已修复：`set -e` 不再跳过退出码捕获；已有 run 会自动选择最新 periodic/best checkpoint 并显式 `--resume-in-place`。
- v2 预算设为 3M；2M 做选择门控，目标是保留最有希望曲线继续验证到 3M，而不是默认空跑至 5M。

### 9.4 Pong 250k 并行回归门禁

采用仓库内 seed0 known-good 配方，其历史固定评估为 `19.375@约202k`、`19.25@约253k`。关键语义为 GSL20、H10、ctx4、PER-on、adaptive entropy、encoder clip、temperature 0.25。

为兼顾 GPU 利用率，Pong 与四条 MsPacman 同时启动，Pong 共享 GPU0 的 collect/CPU 空窗。门禁读取 evaluator 原始表格，要求最近两个 eval 中至少一个 `>=15`：

- 通过：写入 `PONG_GATE_PASSED`，四条 MsPacman 继续运行；
- 未通过、崩溃或日志不足：controller 终止全部四条 MsPacman，RJob 失败退出，进入重新 code review，而不继续消耗错误版本的百万步预算。

### 9.5 验证状态

- 新增及相关 focused unit suite：`11 passed`。
- pytest 自动插件加载在此环境会使导入超过 180 秒；设置 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 后 52.87 秒完成。
- Pong 门禁 parser 对 known-good 日志正确读取 `[-3.75, 15.125, 19.375, 19.25]` 并通过。
- 同一 parser 对失败 run 的 `[-21, -21, -21, -21]` 正确拒绝。
- Python compile、三个 shell launcher `bash -n`、`git diff --check` 全部通过。

## 10. 2026-08-22 18:19 HKT v2 实际启动记录

- 分支：`polish-uz-mspacman`
- 代码提交：`515570fd fix(pu): accelerate and gate UniZero MsPacman frontier runs`
- RJob：`uz-mspacman-frontier-0822`
- 新 replica：`uz-mspacman-frontier-0822-qr8w7`
- 分区/节点：`narmodel` / `gpu-lg-cmc-h-h200-0668`
- RJob 自动恢复：Pong 门禁期间关闭；门禁通过后再恢复

旧 replica `...-qfzpx` 已先缩容至 0 并确认 active=0，再将同一 task `generated-task-0` 扩容至 1；不存在新旧进程重叠写同一目录。18:19:23 五个 fresh run 同时创建：四条 MsPacman 分别使用 GPU0–3，Pong 与 MsPacman H5 主臂共享 GPU0。

生成后的 `formatted_total_config.py` 已逐条核验：

- MsPacman：`isolate_eval_cache=True`、`world_model.env_num=11`、new cache manager、raw-token rebuild、augmentation、PER、bootstrap context、temperature 0.25 均生效。
- 三条 H5 的 `max_blocks/max_tokens=5/10`，H10 control 为 `10/20`。
- value 权重分别为计划的 `0.5/0.5/0.25/0.5`；reanalysis 臂 batch 为 32。
- `empty_cuda_cache_on_cache_reset=False` 全部生效。
- Pong：8 collect + 8 eval，`world_model.env_num=16`，GSL20/H10/ctx4/PER/adaptive entropy/encoder clip 与 known-good 语义一致。

启动健康检查：GPU1–3 MsPacman 初评约 119 秒完成，GPU0 共享卡 MsPacman 初评约 196 秒完成；四臂共同初值均为 60，并已进入 collection/training。18:24 时 H5 value0.25 和 H5 reanalysis 已完成第二轮 collection，分别到约 2.9k/2.8k envsteps；H10 control 和 GPU0 共享臂较慢，符合计算量/共享预期。当前没有 Traceback、OOM、NaN 或 cache namespace 越界。Pong 的随机策略 8 局初评 episode 更长，仍在运行，250k 门禁未提前作性能结论。

控制层补充：Pong 未通过时会写 `PONG_GATE_FAILED`、终止四个训练子进程并让 controller 正常退出，避免 RJob auto-restart 把明确判为失败的代码再次拉起。四条 variant 自身仍保留 checkpoint-aware 进程级重启；RJob 级 auto-restart 在 Pong 通过后才能重新启用。

### 10.1 启动 warmup 重复评估修复

Pong GSL20 的首轮 8 segments 只有约 160 transitions，低于 batch256 所需的 257；serial loop 因 replay 不足直接 `continue`，`learner.train_iter` 保持 0。旧条件 `train_iter == 0 or evaluator.should_eval(...)` 因短路不会更新 evaluator 内部状态，下一轮因此重复执行完整 8 局初评，实测额外消耗约 316 秒。

现增加 loop-level `last_evaluated_train_iter` 不变量，保证同一 learner iteration（包括 iteration 0）最多评估一次。该改动不改变训练数据或算法语义，只消除 replay warmup 期间的重复评估；定向 entry suite `13 passed`。当前 run 已越过 iteration-0 warmup，继续使用已加载进程；修复用于后续 fresh/restart。

## 11. 2026-08-22 18:39 HKT 早期运行快照与监控节奏

### 11.1 代码与验证更新

- `f797c265 fix(pu): avoid duplicate UniZero warmup evaluation`：修复同一 iteration-0 因 replay 不足而重复全量 eval；定向 entry suite `13 passed`。
- 本轮累计核心提交：`515570fd`（coherent augmentation、H5/H10/门禁控制器）、`f51e6511`（Pong 失败终止语义）、`f797c265`（warmup eval 去重）。
- 当前所有训练进程均为 18:19 启动时加载的 `515570fd` 语义；后续源码提交不热更新已运行 Python 进程。warmup 去重只影响未来 fresh/resume。

### 11.2 最新实验进度

| run | 最新可确认进度 | 早期墙钟特征 |
|---|---:|---|
| Pong known-good gate | train iter ~900 | GPU0 共享；约每 8.5–9 秒完成一轮 8×GSL20 collection |
| `uz_h5_value05` | ~5.3k envsteps / iter ~300 | 与 Pong 共享 GPU0，collector 约 30 envstep/s，learner 竞争明显 |
| `uz_h10_value05` | ~7.9k envsteps / iter ~700 | 独占 GPU，实际墙钟约 10 envstep/s |
| `uz_h5_value025` | ~15.7k envsteps / iter ~1.5k | 独占 GPU，实际墙钟约 16 envstep/s |
| `uz_h5_reanalysis02_rbs32` | ~14.4k envsteps / iter ~1.4k | 独占 GPU，实际墙钟约 16 envstep/s |

目前所有 controller log 的 Traceback/OOM 计数均为 0；无 `PONG_GATE_PASSED/FAILED` 标记。Pong 的两次 `-21` 是相同 train iteration 0 的重复初评，不能作为两个独立训练门控点；最终 gate parser 使用进程结束时最近两个有效日志点，且 250k 未达到 15 会终止四条 MsPacman。

早期 learner 中间量：H10 value0.5 的 value priority 约 0.64，H5 value0.25 约 0.88，H5 reanalysis 约 0.78；尚未形成长期趋势。obs norm 当前恒为 27.7128 是 `monitor_norm_freq=10000` 前重复展示首次缓存值，不能解释为表征无漂移。H5 root raw hit 约 99.15–99.28%，其中混有 episode 首根的预期 miss，后续需按 episode 数校正后再判定异常 miss。

### 11.3 监控频率

按用户要求，停止每分钟轮询，常规监控改为每 2 小时一次。下一次计划检查为约 `2026-08-22 20:40 HKT`，之后按两小时间隔检查：RJob 状态、Pong gate、各臂 envstep/eval 序列、墙钟吞吐、value priority、obs norm、root hit 与异常计数。Pong gate 的停止逻辑由 worker 内 controller 持续执行，不依赖人工两小时轮询，因此低于阈值时仍会即时停止四臂。

## 12. MuZero 参考曲线与 2M/3M 预注册判据

`data_unizero/docs/mz_mspacman_refer` 当前只包含四张 TensorBoard 截图，没有 event/scalar 原始文件；因此以下读数只用于判断数量级和曲线形态，不能冒充精确逐点标量。

从 `mz_mspacman_eval_reward.png` 可直接确认：MuZero 的 eval reward 在约 0–2M 快速抬升至数千分区间；2–5M 仍有很大的逐次评估波动，但中心位置和上包络继续上移，而不是在早期峰值后持续回落；蓝色长跑在约 6–8M 再进入约 8k–10k 区间。`mz_mspacman_collect_reward.png` 同样显示 2–5M 的 collect reward 分布逐步扩张、上包络持续抬升。故本项目所称“波动上升”应是窗口级统计趋势，不要求每次 eval 单调增加。

为避免看到结果后修改标准，后续统一使用以下判据：

1. 每个检查点保留原始 eval 序列，不用单个 best score 排名；至少报告最近 6 次与此前 6 次 eval 的中位数、均值、最大值和最小值。
2. 2M 选臂以样本效率为主、墙钟效率为辅：最近 6 次中位数高于此前 6 次，且对最近 12 次 eval 的 envstep–reward Theil–Sen 斜率为正；若趋势相近，再按实际 envstep/s 和 value-priority 是否恶化排序。
3. 单个高峰后连续回落不算通过。若最近 6 次中位数不高于此前 6 次、Theil–Sen 斜率非正且后 3 次均低于窗口中位数，则判为平台/回落，需要 code review 或配方调整。
4. 3M 最终趋势审查比较 2.0–2.5M 与 2.5–3.0M 两个窗口：后窗中位数或上四分位数至少一项上升、另一项不得显著下降，并要求最近 12 点稳健斜率为正；同时与 MuZero 截图中 2–3M 的“高方差但中心/包络上移”形态对照。
5. 若某臂到检查时不足 12 个 eval 点，不强行套用统计门槛；保留运行直至样本充足，或只做健康/吞吐判断并明确标记证据不足。

这些判据只决定“是否仍在学”和变体排序，不把截图估读的某个绝对分数硬设为 correctness gate。Pong `>=15@250k` 仍是代码回归的独立硬门禁。

### 12.1 Pong 最终点评估门禁语义修复

等待窗口中的静态复核发现，`check_pong_gate.py` 原来使用 `max(scores[-2:]) >= 15`。这会让约 200k 的较高分掩盖约 250k 的最终回落，与“Pong 在 250k envsteps 的 eval 达到约 15”不一致。现改为严格要求最后一次完整 eval `scores[-1] >= 15`，同时仍要求至少存在两个评估点。

新增 3 个回归测试覆盖最终点通过、早期峰值不能掩盖最终回落、评估点不足；全部通过。历史 known-good 序列 `[-3.75, 15.125, 19.375, 19.25]` 在新门禁下仍通过。运行中的 controller 仅在 Pong 训练进程结束后启动该独立 Python checker，因此会读取修复后的脚本并对本次约 250k 最终点执行严格判定。

## 13. 2026-08-22 19:18 HKT：H5/H10 早期差异的第一性原理复核

### 13.1 当前证据边界

四条 MsPacman 截至本次读取都只有共同冷启 eval `60@0`，尚无训练后 eval；因此当前所谓“H10 明显更好”来自 collect clipped episode return，不能等同于固定 eval raw score。按相同 `<=16.5k envsteps` 聚合：

| 变体 | collect 局数 | pooled mean ± SE | Theil–Sen 斜率（reward / 1M envsteps） |
|---|---:|---:|---:|
| H5 / value0.5（matched control） | 83 | `11.72 ± 0.64` | `+36` |
| H10 / value0.5 | 62 | `13.50 ± 0.67` | `+412` |
| H5 / value0.25 | 82 | `10.88 ± 0.43` | `+154` |
| H5 / reanalysis0.02 | 80 | `11.11 ± 0.43` | `+102` |

H10 与 matched H5 的均值差约为 1.9 个合并标准误，方向和斜率均支持用户观察，但证据强度尚不足以把 collect 指标当成性能定论。按 envstep 对齐后，GPU0/Pong 共享只影响墙钟速度，不解释上述样本效率差异；不过它使 matched H5 的进度更慢。独占 GPU 上 H5 当前实际约 `13–14 envstep/s`，H10 约 `8 envstep/s`。

同一 envstep 区间中，H10 的 scalar TD priority 约稳定在 `0.64–0.68`，H5/value0.5 约升至 `0.82–0.89`；这与 H10 更平滑的多位置 value 监督一致。但 priority 本身也是跨 H 个位置取均值，H 不同时绝对量不可直接作公平 loss 排名。

### 13.2 H10 为什么可能优于 H5

1. `replay_ratio=0.1` 决定每个 collect 周期的 optimizer update 数，与 H 无关；每次 update 的 batch 仍为 256 条序列。因此 H10 每个 update 使用约两倍的 observation/action/target positions，单位 envstep 的监督 target token 数约为 H5 的两倍。
2. reward/value/policy/obs loss 均按有效 timestep 数归一化，H10 不会简单把 loss 或梯度幅值乘二；它主要降低序列位置平均的方差，并让共享 encoder/transformer/head 在每次更新看到更多真实 transition labels。
3. 当前主 loss 是 teacher-forced：每个位置都输入真实 observation embedding。把 H 从 5 增至 10 不等于训练十步 open-loop imagination，也不直接扩大 MCTS 深度；其主要收益是训练 token 预算、较长真实上下文和位置覆盖。
4. 在线 `infer_context_length=4` 对应固定 `context_length=8 tokens`，H5 与 H10 完全相同。H10 后半序列使用的 5–10 block context 并不会在在线 root 决策中直接出现；故再增大 H 存在训练/在线上下文分布错配和边际收益递减。

### 13.3 H20 的收益、成本与正确测试方式

H20/batch256/RR0.1 会再次把单位 envstep 的 target token 预算翻倍。其 encoder/target-encoder 工作近似为 H10 的 2 倍，Transformer attention 的单 batch 理论项近似 4 倍；按当前 H5/H10 墙钟比外推，它不适合作为未经门控的 3M 主长跑。更重要的是，它仍是 teacher-forced 20-step sequence，并不能单独修复 deep open-loop rollout。

若测试 H20，应使用 `batch_size=128`：`128×20 = 256×10`，保持每次 optimizer update 的 target-position 数与 H10/batch256 近似一致，同时保留相同 RR 和 update 次数。这样虽然 attention 计算仍更高、独立 sequence 数减半，但至少不会把“更长 horizon”与“监督 token 翻倍”完全混为一谈。先跑 100k–250k 探针，按固定 eval 和墙钟吞吐门控，胜出才延长。

### 13.4 预注册变体调整决策

当前不在只有初评的时点停臂。`eval_freq=5000 learner iter` 且 RR≈0.1，首个训练后 eval 约在 50k envsteps；先取得 H10 以及专用 GPU 两条 H5 的首个固定 eval。若 H10 的固定 eval 与 collect 优势同向，则下一矩阵调整为：

1. 保留 `H5/value0.5`，作为唯一 matched horizon control；
2. 保留 `H10/value0.5`，作为当前主臂；
3. 将 reanalysis 变体改为 `H10/value0.5/reanalysis0.02/rbs32`，验证最有希望的 H10 上的增量；
4. 停止早期最弱的 `H5/value0.25`，改为 `H20/value0.5/batch128` 的 100k–250k token-matched 探针。

若 H10 首个固定 eval 不优于 H5，则不因 collect 曲线追涨：保留原矩阵至至少第二个 eval，再决定 H20；H20 不会在缺少 H10 固定评估支持时直接启动。下一次完整决策点仍按 2 小时节奏设在约 `20:40 HKT`。
