# UniZero + PPO

UniZero + PPO uses UniZero's latent representation and world model with an on-policy PPO actor/critic update.
The original `unizero` policy is unchanged and continues to use MCTS. PPO is selected explicitly with the
`unizero_ppo` policy type.

## Status

This branch is an experimental discrete-action implementation. The current validation status is:

| Environment | Status | Evidence |
| --- | --- | --- |
| CartPole | Converged in the smoke/integration run | Seed 0 reached `200/200` in all three evaluation episodes after about 4,015 environment steps and repeated it through 23,142 steps. |
| LunarLander-v3 | Functional, convergence not established | Collect → PPO → world-model replay smoke cycles complete; no multi-seed convergence result is reported. |
| PongNoFrameskip-v4 | Single-seed long run completed; benchmark validation still open | A 1,011,996-environment-step seed-0 run completed normally. Its last evaluation before termination was `19.67 ± 0.47` over three episodes (`[19, 20, 20]`), after rising from `-21` at initialization. This is encouraging single-seed evidence, not a multi-seed benchmark claim. |

CartPole has a positive convergence result, and Pong now has a completed 1M-step single-seed run. Longer validation
across more environments is still required. Report raw evaluation curves and at least three seeds before describing
the result as a benchmark.

## How it works

Each collection cycle has two independent training phases:

1. The current policy collects fresh episodes and stores masked action log-probabilities, GAE data, and the exact
   contextual latent feature used to act.
2. PPO updates the actor and critic only on that fresh rollout. The first minibatch checks that the behavior-policy
   ratio is close to one.
3. Replay updates train the encoder, Transformer, and reward/dynamics losses after PPO has consumed the rollout.
   Old rollout data is never reused for actor ratios.

This separation keeps the MCTS UniZero path compatible and avoids rerunning the image encoder and Transformer for
every PPO minibatch.

## Run an example

Install the normal LightZero dependencies first, then run one of the example configurations:

```bash
python zoo/classic_control/cartpole/config/cartpole_unizero_ppo_config.py --seed 0
python zoo/box2d/lunarlander/config/lunarlander_disc_unizero_ppo_config.py --seed 0
python zoo/atari/config/atari_unizero_ppo_config.py --env PongNoFrameskip-v4 --seed 0
```

The Atari example accepts `--max-env-step`, `--stop-value`, `--n-evaluator-episode`, and `--run-tag`.

## Configuration

PPO-specific settings live under `policy.ppo`. The example defaults are:

```python
policy=dict(
    # The PPO configs use create_config.policy.type = "unizero_ppo".
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

PPO currently supports discrete action spaces only. Keep `accumulation_steps=1`; this prevents gradients from
crossing the actor/critic and world-model phase boundary.

## Implementation layout

- `lzero/policy/unizero.py`: legacy UniZero MCTS policy.
- `lzero/policy/unizero_ppo.py`: PPO policy subclass and PPO learning/collection logic.
- `lzero/model/unizero_model.py`: shared UniZero model and world-model factory hook.
- `lzero/model/unizero_ppo_model.py`: PPO model subclass using the PPO world model.
- `lzero/model/unizero_world_models/ppo_world_model.py`: actor/critic fast path and PPO-specific loss handling.
- `lzero/mcts/buffer/game_buffer_unizero_ppo.py`: fresh-rollout and world-model replay extensions.

The legacy policy, world model, and replay buffer do not expose PPO-only fields or methods. This keeps existing
UniZero configurations source-compatible after the PR is merged.

## Validation

Run the focused regression suite with the project's configured Python environment:

```bash
python -m pytest -q \
  lzero/policy/tests/test_unizero_ppo.py \
  lzero/model/unizero_world_models/tests/test_per_sample_is_weights.py \
  lzero/mcts/tests/test_unizero_reanalysis_context.py
```

The current branch passes 45 focused tests. For a real experiment, record evaluation return versus environment
steps, PPO KL/clip metrics, rollout advantage statistics, model losses, throughput, and all random seeds.

## Related links

- [UniZero paper](https://arxiv.org/abs/2406.10667)
- [PPO policy implementation](https://github.com/opendilab/LightZero/blob/feat/uz-ppo/lzero/policy/unizero_ppo.py)
- [Pull request #498](https://github.com/opendilab/LightZero/pull/498)
