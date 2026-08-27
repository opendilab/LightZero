# UniZero MsPacman 3M 执行清单

## 提交前

- [x] 当前分支为 `polish-uz-mspacman`，工作树只包含本任务文件和用户既有未跟踪文件。
- [x] `data_unizero` 无写入、删除或覆盖。
- [x] 12 个 seed/group config 均可 `--dry-run` 加载。
- [x] baseline 与 v1/v2/v3 的递归 diff 分别为 0/1/1/1 个显式机制。
- [x] `pytest zoo/atari/tests/test_atari_unizero_segment_config.py` 通过。
- [x] `bash -n scripts/launch_unizero_mspacman_3M.sh`、`py_compile`、`git diff --check` 通过。
- [x] GPFS 输出根不存在历史碰撞，文件系统至少有 10 GiB 可用。
- [x] H10/ctx10 最大臂的模型结构不超过历史已运行 H200 配置；batch256 dry-run 生效。

## 起跑成功判定（每组）

- [x] RJob Running；四个 Python 进程分别由 `CUDA_VISIBLE_DEVICES=0,1,2,3` 隔离。
- [x] `console.log` 已写 run directory、seed 和 device `cuda:0`（单卡可见性重映射正常）。
- [x] `formatted_total_config.py` 与预注册 diff 一致。
- [x] 初始 eval 已输出，随后至少完成 collect、replay sample 和一次 learner update。
- [x] TensorBoard 同时有 evaluator、collector、learner 标量。
- [x] 无 Traceback、CUDA OOM、NaN/non-finite、cache namespace 越界。
- [ ] 前 50k envstep 持续写入；loss、value priority、latent norm、grad norm 均有限。

## 失败与自愈

- CUDA OOM：controller 自动保留失败目录，batch256 -> 128 重启；在报告中登记语义变化。
- 有完整 checkpoint：选择最新通过 zip 完整性检查的 `iteration_*.pth.tar` 原目录 resume。
- 无完整 checkpoint：失败目录改名为 `.failed-attemptN-TIMESTAMP`，fresh 重启，不覆盖证据。
- 每组最多初次 + 2 次重试；第 3 次失败标记 `TERMINAL_FAILED`，其余组继续。
- 明确代码 bug：修复、回归、commit 后重启；三个独立修复仍失败才终止任务并报告。

## 300k kill 条件

运行：

```bash
bash scripts/launch_unizero_mspacman_3M.sh check-kill
```

- 等四组均有 `>=300k` 的 eval。
- 对每组取截至 300k 最近 3 个 `eval_episode_return_mean`。
- 变体均值 `< 0.65 * baseline` 时输出 `KILL_RECOMMENDED`（35% margin）。
- 默认只建议，不自动停止；3-episode/8-episode MsPacman 方差很大，单点不触发 kill。
- 若指标 NaN/OOM/cache 越界，属于 correctness 失败，可不等 300k 即修复重启。

## 3M 成功/失败判定

- 成功：完整到 3M；最近 6 次 eval 中位数高于此前 6 次，最近 12 次 envstep-return
  Theil-Sen 斜率为正；无 NaN/持续 cache corruption。
- 部分成功：回报优于 baseline 但趋势条件不满足，追加 seed 前先复核机制 metric。
- 失败：3M 最近 6 次均值/中位数不高于前 6 次，或持续低于 baseline 35%。
- 报告 peak、末段均值和完整序列，禁止只报最高点。

## 追加 seed1/2

默认只跑 seed0。seed0 通过 300k 健康/kill 检查后，可新提交一轮：

```bash
UZ_JOB_NAME=uz-mspacman-repro-3m-seeds12-0824 \
UZ_SEED_QUEUE=1,2 \
bash scripts/launch_unizero_mspacman_3M.sh submit
```

同一个 4-GPU worker 会先并行跑完四组 seed1，再并行跑四组 seed2；已完成目录由 `.completed`
幂等跳过，中断目录自动 resume。
