"""
监控增强功能使用示例
============================

本示例展示如何在 UniZero 配置中启用和使用新增的监控功能。
"""

# ============================================================
# 示例 1: 基本配置 - 启用监控
# ============================================================

unizero_config = dict(
    type='unizero',

    # ... 其他配置 ...

    # ==================== 监控配置 ====================
    # 设置监控频率 (每 5000 次迭代监控一次)
    monitor_norm_freq=5000,

    # 启用自适应熵权重 (已有功能)
    use_adaptive_entropy_weight=True,

    # 启用 Encoder-Clip 退火 (已有功能)
    use_encoder_clip_annealing=True,
    encoder_clip_anneal_type='cosine',
    encoder_clip_start_value=30.0,
    encoder_clip_end_value=10.0,
    encoder_clip_anneal_steps=100000,

    # ... 其他配置 ...
)


# ============================================================
# 示例 2: 快速调试配置 - 更频繁的监控
# ============================================================

debug_config = dict(
    type='unizero',

    # 快速调试模式: 每 1000 次迭代监控一次
    monitor_norm_freq=1000,

    # ... 其他配置 ...
)


# ============================================================
# 示例 3: 生产环境配置 - 减少监控开销
# ============================================================

production_config = dict(
    type='unizero',

    # 生产环境: 每 10000 次迭代监控一次
    monitor_norm_freq=10000,

    # ... 其他配置 ...
)


# ============================================================
# 示例 4: 禁用监控 - 最大化训练速度
# ============================================================

no_monitoring_config = dict(
    type='unizero',

    # 完全禁用范数监控
    monitor_norm_freq=0,

    # ... 其他配置 ...
)


# ============================================================
# 示例 5: 从训练日志中提取监控指标
# ============================================================

def analyze_monitoring_logs(log_dict):
    """
    分析监控日志,提取关键指标

    Args:
        log_dict: _forward_learn 返回的日志字典
    """

    # 1. 检查梯度健康状况
    encoder_grad_norm = log_dict.get('grad/encoder/_total_norm', 0)
    transformer_grad_norm = log_dict.get('grad/transformer/_total_norm', 0)

    if encoder_grad_norm > 100:
        print(f"⚠️  警告: Encoder 梯度过大 ({encoder_grad_norm:.2f})")

    if transformer_grad_norm > 100:
        print(f"⚠️  警告: Transformer 梯度过大 ({transformer_grad_norm:.2f})")

    # 2. 检查权重健康状况
    encoder_norm = log_dict.get('norm/encoder/_total_norm', 0)
    transformer_norm = log_dict.get('norm/transformer/_total_norm', 0)

    print(f"📊 Encoder 参数范数: {encoder_norm:.2f}")
    print(f"📊 Transformer 参数范数: {transformer_norm:.2f}")

    # 3. 检查 Logits 是否饱和
    value_logit_abs_max = log_dict.get('logits/value/abs_max', 0)
    policy_logit_abs_max = log_dict.get('logits/policy/abs_max', 0)

    if value_logit_abs_max > 50:
        print(f"⚠️  警告: Value logits 可能饱和 (abs_max={value_logit_abs_max:.2f})")

    if policy_logit_abs_max > 20:
        print(f"⚠️  警告: Policy logits 可能饱和 (abs_max={policy_logit_abs_max:.2f})")

    # 4. 检查 Embeddings 范数
    obs_emb_norm_mean = log_dict.get('embeddings/obs/norm_mean', 0)
    obs_emb_norm_max = log_dict.get('embeddings/obs/norm_max', 0)

    print(f"📊 Obs Embeddings 平均范数: {obs_emb_norm_mean:.2f}")
    print(f"📊 Obs Embeddings 最大范数: {obs_emb_norm_max:.2f}")

    # 5. 检查 Transformer 输出
    x_token_mean = log_dict.get('norm/x_token/mean', 0)
    x_token_max = log_dict.get('norm/x_token/max', 0)

    print(f"📊 Transformer Token 平均范数: {x_token_mean:.2f}")
    print(f"📊 Transformer Token 最大范数: {x_token_max:.2f}")


# ============================================================
# 示例 6: TensorBoard 可视化建议
# ============================================================

"""
在 TensorBoard 中查看监控指标:

1. 查看所有模块的总范数:
   - 使用正则表达式过滤: norm/.*/_total_norm

2. 查看所有梯度范数:
   - 使用正则表达式过滤: grad/.*/_total_norm

3. 查看所有 logits 统计:
   - 使用正则表达式过滤: logits/.*

4. 比较不同模块:
   - 在同一个图表中绘制:
     * norm/encoder/_total_norm
     * norm/transformer/_total_norm
     * norm/head_value/_total_norm
     * norm/head_policy/_total_norm
"""


# ============================================================
# 示例 7: Wandb 集成示例
# ============================================================

"""
如果使用 Wandb (use_wandb=True),所有监控指标会自动记录。

在 Wandb Dashboard 中:

1. 创建自定义 Panel - "模块范数对比":
   - norm/encoder/_total_norm
   - norm/transformer/_total_norm
   - norm/head_value/_total_norm
   - norm/head_policy/_total_norm
   - norm/head_reward/_total_norm

2. 创建自定义 Panel - "梯度健康":
   - grad/encoder/_total_norm
   - grad/transformer/_total_norm
   - total_grad_norm_before_clip_wm

3. 创建自定义 Panel - "Logits 统计":
   - logits/value/abs_max
   - logits/policy/abs_max
   - logits/reward/abs_max

4. 创建 Scatter Plot - "范数 vs 损失":
   - X 轴: norm/encoder/_total_norm
   - Y 轴: weighted_total_loss
"""


# ============================================================
# 示例 8: 自定义监控指标
# ============================================================

"""
如果需要添加自定义监控指标,可以在 world_model.compute_loss 中
将中间张量添加到 intermediate_losses 字典中:

在 world_model.py 中:
    intermediate_losses['your_custom_tensor'] = your_tensor.detach()

然后在 unizero.py 的 _forward_learn 中监控:
    custom_tensor = losses.intermediate_losses.get('your_custom_tensor')
    if custom_tensor is not None:
        norm_log_dict['custom/mean'] = custom_tensor.mean().item()
        norm_log_dict['custom/std'] = custom_tensor.std().item()
"""


# ============================================================
# 示例 9: 诊断训练问题的工作流
# ============================================================

"""
当遇到训练问题时,按以下顺序检查监控指标:

1. 梯度爆炸/消失:
   ✓ 检查 grad/encoder/_total_norm
   ✓ 检查 grad/transformer/_total_norm
   ✓ 检查 total_grad_norm_before_clip_wm
   → 如果过大 (>100): 降低学习率或增加梯度裁剪
   → 如果过小 (<0.01): 提高学习率或检查数据预处理

2. 权重膨胀:
   ✓ 检查 norm/encoder/_total_norm
   ✓ 检查 norm/transformer/_total_norm
   → 如果持续增长: 增加权重衰减或启用 encoder-clip

3. 输出饱和:
   ✓ 检查 logits/value/abs_max
   ✓ 检查 logits/policy/abs_max
   → 如果过大 (>50): 启用 temperature scaling 或降低学习率

4. 表征崩塌:
   ✓ 检查 embeddings/obs/norm_mean
   ✓ 检查 norm/x_token/mean
   → 如果接近 0: 检查数据归一化或增加 dropout
"""


if __name__ == '__main__':
    print("监控增强功能使用示例")
    print("=" * 60)
    print("请参考上述示例配置来启用和使用监控功能。")
    print("详细文档请查看: MONITORING_ENHANCEMENTS.md")
