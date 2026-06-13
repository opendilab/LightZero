# PriorZero说明

本文档面向 `zoo/jericho/priorzero` 分支代码，重点补充：
1. 主要文件说明；
2. 主实验启动命令与修改点；
3. 当前实验结论；
4. 后续可能改进方向。

---

## 1. 主要文件说明

### 1.1 `src/priorzero_config.py`
该文件负责**统一管理实验配置**，是 PriorZero 训练流程的入口配置源。主要职责：
- 定义可选 LLM 模型预设（`MODEL_CONFIGS`）；
- 定义 `PriorZeroLLMConfig`（LLM/RFT 训练相关核心参数）；
- 通过 `get_priorzero_config(...)` 组装环境、策略、采集、评估的总配置。

#### `llm_config`（`PriorZeroLLMConfig`）参数逐项说明
> 下述参数是当前 PriorZero LLM/WM 联合训练最关键的调参入口。建议每次实验先固定大结构（训练模式、模型规模），再小步调整损失与采样超参数。

##### A. 基础开关与模型路径
- `model_name_or_path`：LLM 的 HuggingFace/本地模型路径。
- `enable_rft`：是否开启 LLM 的 RFT（强化微调）训练。
- `enable_world_model`：是否开启 World Model（WM）训练。

##### B. LLM 训练方式（`train_mode_dict`）
- `mode`：LLM 训练模式，`full` 为全参微调，`lora` 为 LoRA 微调。
- `lora_r`：LoRA 低秩分解 rank。
- `lora_alpha`：LoRA 缩放系数。
- `lora_dropout`：LoRA 路径 dropout 比例。
- `lora_bias`：LoRA 中 bias 训练策略（`none/all/lora_only`）。
- `lora_target_modules`：应用 LoRA 的模块列表（如 `q_proj/k_proj/...`）。

##### C. 交替训练调度（`train_schedule`）
- `alternate`：是否采用 WM/LLM 严格交替训练。
- `wm_update_iters`：在交替模式下，每轮 WM 连续更新步数。
- `llm_update_iters`：在交替模式下，每轮 LLM 连续更新步数。
- `start_phase`：交替训练起始阶段（`wm` 或 `llm`）。
- `llm_collect_mode`：LLM 训练阶段的数据采集策略（`wm_collect/wm_llm_collect/no_collect`）。

##### D. MCTS 根节点先验融合
- `llm_prior_temperature`：LLM 先验分布温度（温度越高越平滑）。
- `mcts_root_logits_dict.mode`：根节点 logits 融合模式（仅 LLM、仅 WM、或二者融合）。
- `mcts_root_logits_dict.plus_method`：融合权重策略（`fixed` 或 `adaptive`）。
- `mcts_root_logits_dict.wm_weight`：`fixed` 时 WM 的固定权重。
- `mcts_root_logits_dict.llm_max_weight`：`adaptive` 时 LLM 最大权重。
- `mcts_root_logits_dict.llm_min_weight`：`adaptive` 时 LLM 最小权重。
- `mcts_root_logits_dict.max_envsteps`：`adaptive` 权重衰减参考的总环境步数。

##### E. 评估策略（`eval_dict`）
- `eval_dict.world_model`：启用“仅 WM”评估。
- `eval_dict.world_model_llm_prior`：启用“WM + LLM 先验”评估。
- `eval_dict.llm_prior`：启用“仅 LLM 先验”评估。
- `eval_dict.wm_eval_freq`：WM 评估频率。
- `eval_dict.llm_eval_freq`：LLM 评估频率。

##### F. Prompt / 序列相关
- `attn_implementation`：注意力实现方式（如 `flash_attention_2`）。
- `history_length`：输入历史轨迹长度。
- `use_cot`：是否启用 CoT 推理。
- `cot_weight`：CoT 前缀 token 在损失中的权重。
- `user_prompt_dict.history_with_reward`：prompt 中是否拼接历史 reward。
- `user_prompt_dict.observation_with_valid_actions`：prompt 中是否拼接当前合法动作。
- `prompt_max_len`：输入最大 token 长度。
- `generate_max_len`：生成最大 token 长度。
- `bf16`：是否使用 bfloat16。

##### G. vLLM 推理与采样
- `enable_vllm`：是否启用 vLLM 引擎。
- `enable_prefix_caching`：是否启用前缀缓存。
- `use_cuda_ipc`：是否使用 CUDA IPC。
- `enable_vllm_is_correction`：是否启用 vLLM 截断修正逻辑。
- `vllm_is_truncated_threshold`：vLLM 截断判定阈值区间。
- `use_mispo`：是否启用 MISPO 相关策略。
- `mispo_token_truncated_threshold`：MISPO token 级截断阈值。
- `mispo_traj_truncated_threshold`：MISPO 轨迹级截断阈值。
- `vllm_sync_backend`：vLLM 参数同步后端（如 `nccl`）。
- `vllm_tensor_parallel_size`：单个 vLLM engine 的张量并行卡数。
- `gpu_memory_utilization`：vLLM 可用显存占比。
- `vllm_enable_sleep`：空闲时是否允许 vLLM 休眠。
- `temperature`：采样温度。
- `top_p`：核采样阈值。
- `seed`：随机种子。
- `reduction`：损失聚合方式（如 `mean`）。

##### H. DeepSpeed / 梯度控制
- `deepspeed_enable_sleep`：DeepSpeed 相关休眠优化开关。
- `zero_stage`：DeepSpeed ZeRO stage。
- `gradient_checkpointing`：是否启用梯度检查点。
- `gradient_checkpointing_use_reentrant`：梯度检查点 reentrant 配置。
- `max_norm`：梯度裁剪阈值。
- `ds_tensor_parallel_size`：DeepSpeed 张量并行规模。

##### I. 批大小与数据新鲜度
- `train_batch_size`：全局训练 batch size。
- `micro_train_batch_size`：单次前向/反向 micro batch size。
- `max_rollout_staleness`：rollout 到训练的最大“离线陈旧度”。

##### J. 优化器与学习率
- `learning_rate`：学习率。
- `adam_betas`：Adam beta 系数。
- `weight_decay`：权重衰减。
- `lr_scheduler`：学习率调度器类型。
- `lr_warmup_ratio`：warmup 占总步数比例。
- `max_steps`：LLM 训练总步数上限。

##### K. 策略优化目标
- `policy_loss_type`：策略损失类型（`ppo/gspo`）。
- `reward_func.format_reward`：是否启用格式奖励。
- `reward_func.format_param.format_weight`：格式奖励权重（adv 权重约为 `1-format_weight`）。
- `advantage_type`：advantage 定义/归一化方式。
- `eps_clip_low_high`：PPO clip 范围。
- `rft_kl_coef`：RFT KL 正则系数。
- `entropy_loss_coef`：熵奖励系数。
- `kl_estimator`：KL 估计方法。

##### L. 保存与数值稳定
- `llm_save_freq`：LLM checkpoint 保存频率。
- `save_path`：模型保存路径（通常被 `exp_name` 目录覆盖）。
- `value_norm_cfg.enable_stability_optimizer`：是否启用稳定性优化器。
- `value_norm_cfg.value_norm_init_momentum`：value norm 初期动量。
- `value_norm_cfg.value_norm_final_momentum`：value norm 后期动量。
- `value_norm_cfg.value_norm_warmup_steps`：动量从初期到后期的过渡步数。
- `value_norm_cfg.value_norm_clip_percentile`：value clipping 分位点。
- `value_norm_cfg.value_norm_clip_method`：value clipping 方法。
- `value_norm_cfg.value_norm_history_size`：value norm 历史缓存长度。

---

### 1.2 `src/priorzero_entry_sync.py`
该文件是**单进程/主控同步训练入口**，核心流程包括：
- 初始化环境、policy、collector、evaluator、replay buffer；
- 构建 vLLM、PolicyModel、ReferenceModel 与 LLM trainer；
- 执行“数据收集 → WM 训练 → LLM 训练 → 评估”的循环；
- 在交替模式下按照 `train_schedule` 在 `wm/llm` 两阶段切换。

适用场景：快速调试、单节点控制逻辑验证、定位数据流问题。

### 1.3 `src/priorzero_entry_sync_ddp.py`
该文件是**DDP 多卡同步训练入口**，在 `priorzero_entry_sync.py` 基础上增强了：
- torch distributed 初始化与 rank/world_size 协同；
- all_gather 同步控制（例如不同 rank 的 LLM 样本是否齐备）；
- 多卡下 WM/LLM 阶段一致性推进与 barrier 同步。

适用场景：正式大规模实验（推荐使用该入口）。

---

## 2. 主实验启动命令（重点：改哪两个文件）

主实验建议通过 DDP 脚本启动：

```bash
cd zoo/jericho/priorzero
bash scripts/run_priorzero_ddp.sh
```

实际跑实验前，主要改两个地方：

1) `src/priorzero_config.py`
- 修改训练/融合/损失等核心配置（例如 `train_schedule`、`mcts_root_logits_dict`、`advantage_type` 等）。
- 修改模型预设（`MODEL_CONFIGS`）或 `get_priorzero_config` 中与环境相关的设置。

2) `scripts/run_priorzero_ddp.sh`
- 修改 `CUDA_DEVICES`、`NPROC_PER_NODE`、`MASTER_PORT`。
- 修改 `ENV_ID`、`LLM_MODEL`、`USE_COT`。
- 确认日志目录 `LOG_DIR`。

建议流程：
- 先在 `priorzero_config.py` 固化实验配置模板；
- 再在 `run_priorzero_ddp.sh` 做“本次任务级”覆写（环境名、卡数、端口等）；
- 用日志文件名区分实验版本，便于后续对比。

---

## 3. 目前实验结果（阶段性结论）

当前结果可总结为：
- 在 `detective / zork1 / acorncourt / omniquest` 四个环境中，**LLM/WM 交替训练模式**下，实验均出现比 Unizero更早收敛甚至性能更好的趋势；
- 但在 **LLM 冻结**（或后期 LLM 更新不足）设置下，仍需重点讨论：
  - 如何让 PriorZero 在训练后期继续稳定收敛；


---

## 4. 后续可能的改进方向

### 4.1 融合方式：先验注入位置再设计
当前重点在根节点融合（root prior）。可探索：
- 在 MCTS 的**模拟扩展阶段**（非根节点）也注入 LLM 先验；
- 设计“深度相关衰减”策略：树越深，先验权重逐步衰减；
- 对比“仅根节点融合” vs “全树局部融合”的收益与开销。

### 4.2 LLM 训练优势函数（advantage）更精细
可探索更细粒度的 advantage 设计：
- 分阶段 advantage（前期探索导向、后期收敛导向）；
- token 级 / action 级加权 advantage；
- 结合 trajectory 置信度、模型不确定度做 adaptive reweight。

### 4.3 后期收敛稳定性
围绕“LLM 冻结后如何继续提升”可尝试：
- 周期性解冻 LLM 的轻量层（如 LoRA 层）；
- 在后期降低探索温度、提高价值约束；
- 针对高价值轨迹做重采样，提升有效监督密度。

### 4.4 跨域泛化验证：扩展到 Vision 环境
建议在 Atari 等视觉决策环境上验证 PriorZero 的可迁移性：
- 将当前文本交互任务中的先验融合思路迁移到视觉观测 + 离散动作场景；
- 对比文本环境与视觉环境下，root prior / 扩展阶段先验注入的收益差异；
- 评估在高维观测下，LLM（或多模态模型）与 WM 交替训练的稳定性与样本效率。

---