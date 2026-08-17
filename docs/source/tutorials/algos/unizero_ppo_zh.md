# UniZero + PPO

UniZero + PPO 在 UniZero 的潜在表示和世界模型之上加入 on-policy PPO actor/critic 更新。原有的
`unizero` policy 保持 MCTS 行为不变；只有显式使用 `unizero_ppo` policy 类型时才启用 PPO。

## 当前状态

当前实现仍是离散动作空间的实验版本，验证结果如下：

| 环境 | 状态 | 证据 |
| --- | --- | --- |
| CartPole | 已在集成 smoke run 中收敛 | seed 0 约 4,015 个环境步后，连续 3 个评估 episode 均达到 `200/200`，并在 23,142 步前的后续评估中保持。 |
| LunarLander-v3 | 功能可运行，尚未证明收敛 | 已完成 collect → PPO → world-model replay smoke cycle，但还没有多 seed 收敛结果。 |
| PongNoFrameskip-v4 | 功能可运行，尚未证明收敛 | 流程 smoke validation 已完成，但现有长跑证据不足以声称 Pong 收敛。 |

目前只有 CartPole 有明确的收敛结果。长跑实验至少应提供 3 个 seed 的原始评估曲线，才能作为 benchmark
结果描述。

## 训练流程

每个采集周期分为两个相互独立的训练阶段：

1. 当前 policy 采集新 episode，保存带 action mask 的 log-probability、GAE 数据以及实际执行动作时使用的
   contextual latent feature。
2. PPO 只使用这批新数据更新 actor 和 critic，并检查第一个 minibatch 的 behavior-policy ratio 是否接近 1。
3. PPO 消费完本批数据后，再通过 replay 更新 encoder、Transformer、reward 和 dynamics loss；旧 rollout 不会
   被用于 actor ratio。

这种拆分既保持了原有 MCTS UniZero 路径兼容，也避免每个 PPO minibatch 重复运行图像 encoder 和 Transformer。

## 运行示例

先安装 LightZero 的常规依赖，然后运行示例配置：

```bash
python zoo/classic_control/cartpole/config/cartpole_unizero_ppo_config.py --seed 0
python zoo/box2d/lunarlander/config/lunarlander_disc_unizero_ppo_config.py --seed 0
python zoo/atari/config/atari_unizero_ppo_config.py --env PongNoFrameskip-v4 --seed 0
```

Atari 示例还支持 `--max-env-step`、`--stop-value`、`--n-evaluator-episode` 和 `--run-tag` 参数。

## 配置

PPO 参数位于 `policy.ppo` 下，示例默认值如下：

```python
policy=dict(
    # PPO 配置使用 create_config.policy.type = "unizero_ppo"。
    policy_improvement='ppo',
    collect_with_pure_policy=True,
    learning_rate=1e-4,
    ppo=dict(
        gamma=0.997,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_weight=0.01,
        epochs=4,
        minibatch_size=256,
        normalize_advantage=True,
        target_kl=0.03,
        fresh_ratio_tolerance=1e-5,
        world_model_update_per_collect=None,
    ),
)
```

当前 PPO 只支持离散动作空间。请保持 `accumulation_steps=1`，避免 actor/critic 阶段与 world-model 阶段之间
发生梯度串联。

## 代码结构

- `lzero/policy/unizero.py`：原有 UniZero MCTS policy。
- `lzero/policy/unizero_ppo.py`：PPO policy 子类及 PPO 学习/采集逻辑。
- `lzero/model/unizero_model.py`：共享 UniZero model 及 world-model 工厂扩展点。
- `lzero/model/unizero_ppo_model.py`：使用 PPO world model 的 model 子类。
- `lzero/model/unizero_world_models/ppo_world_model.py`：actor/critic 快速路径及 PPO loss。
- `lzero/mcts/buffer/game_buffer_unizero_ppo.py`：新 rollout 和 world-model replay 扩展。

原有 policy、world model 和 replay buffer 不暴露 PPO 专用字段或方法，因此 PR 合入后原有 UniZero 配置仍保持
兼容。

## 验证

使用项目配置的 Python 环境运行聚焦回归测试：

```bash
python -m pytest -q \
  lzero/policy/tests/test_unizero_ppo.py \
  lzero/model/unizero_world_models/tests/test_per_sample_is_weights.py \
  lzero/mcts/tests/test_unizero_reanalysis_context.py
```

当前分支聚焦回归测试为 45 项通过。正式实验建议记录环境步数对应的评估回报、PPO KL/clip 指标、advantage
统计、模型 loss、吞吐以及全部随机种子。

## 相关链接

- [UniZero 论文](https://arxiv.org/abs/2406.10667)
- [PPO policy 实现](https://github.com/opendilab/LightZero/blob/feat/uz-ppo/lzero/policy/unizero_ppo.py)
- [Pull request #498](https://github.com/opendilab/LightZero/pull/498)
