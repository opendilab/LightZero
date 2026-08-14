# UniZero + PPO

This implementation turns UniZero into a configurable **latent world model + policy-gradient improvement**
framework. The original algorithm remains the default and uses MCTS for policy improvement. Setting
`policy.policy_improvement = "ppo"` replaces search-based action improvement with an on-policy PPO actor/critic
update while retaining UniZero's tokenizer, Transformer world model, reward prediction and latent-dynamics losses.

## Design

One training cycle is deliberately ordered as follows:

1. Collect complete episodes with the current masked categorical policy, without MCTS.
2. Store the executed action's behavior log-probability, behavior value and contextual Transformer feature.
3. Compute GAE over complete chronological episodes. Compute returns from the raw advantages first.
4. Normalize advantages once over all valid transitions in the fresh rollout.
5. Select only this collection version, split it into non-overlapping UniZero sequences, shuffle it for every PPO
   epoch, and update only the actor/critic heads on the cached contextual latent features. Reconstruction and reward
   losses are zeroed in this phase so their much larger gradients cannot consume the actor's global clipping budget.
6. Run replay-based encoder/Transformer/dynamics updates only after PPO has consumed the fresh rollout. These
   updates freeze both actor and critic heads, avoiding stale behavior-value targets.
7. Release cached behavior features/log-probabilities/advantages after the PPO epochs; long-lived replay retains
   only the transition data needed by the latent world model.

The replay-only phase also skips target-network value inference and MCTS policy reanalysis: those heads are frozen,
so computing their targets would add substantial cost (especially on Atari) without contributing a gradient.
Conversely, the PPO phase consumes cached latents through a head-only fast path: it does not rerun the image encoder,
Transformer, target tokenizer or reconstruction heads for every PPO epoch.

The contextual feature is important. UniZero collection uses a rolling KV cache, so recomputing a sampled state as
the first item of an isolated replay sequence does not in general reproduce the behavior policy. PPO therefore
evaluates the new actor and critic on the exact latent feature used during collection. This makes the first policy
ratio exactly one while allowing replay updates to train the encoder and Transformer separately.

The learner enforces two freshness checks:

- every PPO minibatch must have the current collection version;
- every valid ratio in the first minibatch must be within `ppo.fresh_ratio_tolerance` of one (checked through the
  logged minimum and maximum, not merely a mean that could hide cancellation).

`ppo/approx_kl`, `ppo/clip_fraction`, and the mean/min/max ratio are logged. PPO epochs stop early when
`ppo/approx_kl > ppo.target_kl`.

## Findings from [PR #473](https://github.com/opendilab/LightZero/pull/473)

The reference branch `tAnGjIa520:unizero-ppo-updates` established the initial end-to-end direction, but several
issues explain why a small CartPole run could improve while LunarLander and Pong remained unreliable:

1. **The latest-data marker was advanced before fetching PPO data.** In split mode,
   `mark_latest_transitions_consumed()` ran before `fetch_latest_batch()`, so the computed number of new transitions
   was zero and PPO was skipped.
2. **World-model updates preceded PPO.** The shared Transformer/policy parameters changed before behavior logits
   were used, so PPO's first ratio was already stale.
3. **Context was not reproduced.** Stored behavior logits came from rolling-KV inference, whereas learner logits
   came from an isolated replay slice. Even with identical parameters, the two policies could differ.
4. **Advantages were normalized per episode.** That gives every episode equal scale irrespective of rollout
   statistics and is especially noisy for sparse-reward Atari. This implementation normalizes the full fresh rollout.
5. **Overlapping replay starts duplicated PPO transitions.** Sampling every transition as the beginning of an
   `H`-step sequence can count one action up to `H` times. PPO starts now use stride `H` and masks cover the final
   partial sequence.
6. **Full behavior logits were stored.** Only the executed action's masked log-probability is required. Storing the
   scalar is both exact and substantially cheaper for large action spaces.
7. **Environment learning rates drifted from UniZero.** The reference LunarLander and Atari PPO configs changed the
   optimizer/LR. The configs here keep the matching UniZero model, optimizer and `1e-4` learning rate so comparisons
   isolate the improvement operator.
8. **PPO returns were initially normalized indirectly.** Returns must be `V + raw_advantage`; only the advantage is
   normalized. The current data model makes this ordering explicit and tested.
9. **UniZero's per-unroll-step loss discount was also applied to PPO.** GAE has already discounted future rewards;
   discounting the surrogate again makes a transition's weight depend on its arbitrary chunk position. PPO actor and
   critic losses now weight all valid rollout transitions uniformly, while world-model losses keep UniZero's scheme.
10. **PPO and world-model objectives shared one clipped backward pass.** In the first LunarLander diagnostic batch,
    the reconstruction-dominated gradient norm was about `2535` against a global clip of `20`; the policy head
    contributed only about `0.09%` of the total norm. This can still solve CartPole but effectively suppresses the
    actor on harder tasks. PPO is now an actor/critic-only phase, followed by a separate latent/reward replay phase;
    a regression test verifies which parameter groups receive gradients.
11. **Gymnasium v3 exposed wrapper assumptions unrelated to PPO.** LunarLander used the removed `v2` id, read
    `reward_range` through `TimeLimit`, and assumed every discrete action had an ndarray `shape`. The example and
    wrapper now use `LunarLander-v3`, read the unwrapped reward range, and accept scalar or array actions.
12. **Learner iteration was not a unique rollout version during warm-up.** Atari intentionally collects `2000`
    environment steps before learning, so several rollouts can share the same `learner.train_iter`. Freshness now
    uses an independent monotonically increasing collection version; skipped warm-up rollouts immediately release
    their PPO-only tensors while retaining transitions for later world-model replay.

## PPO versus MCTS policy improvement

| Property | UniZero + MCTS | UniZero + PPO |
| --- | --- | --- |
| Improvement signal | Search visit distribution | Clipped on-policy advantage |
| Acting cost | Many recurrent model calls per action | One contextual policy forward |
| Training data | Off-policy replay with optional reanalysis | Fresh rollout for actor; replay for world model |
| Model-error exposure | Search compounds model error along simulated branches | Actor gradient uses real rewards/GAE |
| Credit assignment | Search-backed targets and TD bootstrap | GAE with learned critic |
| Parallelism | Expensive per-root tree search | Batched rollout and minibatch SGD |
| Best fit | Strong planning model, small branching factor | Fast acting, large batches, reliable on-policy collection |

MCTS can improve a weak policy immediately by spending more inference compute, but its target quality depends on
multi-step model accuracy. PPO is cheaper at action time and does not optimize imagined reward directly; it instead
uses the latent model as a representation and auxiliary dynamics learner. Its main costs are on-policy sample usage
and sensitivity to advantage/value quality.

## Relation to PWM

PWM also separates a latent world model from a learned policy, but its optimization mechanism is different. PWM
rolls a continuous actor through differentiable learned dynamics and backpropagates first-order gradients through
short imagined trajectories. The current UniZero + PPO path uses a score-function policy gradient from real
environment trajectories and cached contextual latents. Consequently:

- PWM can be extremely sample- and wall-clock-efficient with an accurate pretrained differentiable model, but the
  actor gradient is directly biased by model error.
- UniZero + PPO does not differentiate through imagined transitions, so its policy gradient is grounded in real
  rewards, while the learned model still improves representation and reward/value prediction.
- PWM is naturally aimed at continuous control. This first UniZero + PPO implementation intentionally supports
  discrete action spaces; continuous policies should store the squashed distribution's exact behavior log-prob and
  use the same freshness invariants before being enabled.

This makes the two methods complementary instances of the broader latent-world-model + gradient-based policy
improvement pattern rather than algorithmically equivalent implementations.

The local PWM implementation makes the distinction concrete: `PWM.compute_actor_loss` encodes a real starting
observation, repeatedly applies `actor(z)` and `wm.step(z, action)`, and differentiates the accumulated predicted
reward plus terminal critic through that short latent rollout. Its actor, critic ensemble and world model also have
separate optimizers. UniZero + PPO instead stores the contextual latent used for a real action and applies a clipped
likelihood-ratio objective to that fixed feature; no policy gradient passes through reward prediction or latent
dynamics.

| Axis | UniZero + MCTS | UniZero + PPO | PWM |
| --- | --- | --- | --- |
| Improvement operator | Tree search and visit targets | Clipped score-function gradient | First-order pathwise gradient |
| Policy data | Replay roots, optionally reanalyzed | Strictly fresh real rollouts | Short differentiable latent rollouts |
| World-model role | Simulator inside search and representation learner | Representation/auxiliary replay learner | Differentiable policy-training environment |
| Main model-error effect | Search ranking/backup bias over branches | Representation and value quality; actor reward remains real | Direct bias in actor gradient and imagined return |
| Variance/bias tradeoff | Search-compute dependent | Higher sampling variance, lower dynamics-gradient bias | Lower pathwise variance, stronger model-gradient bias |
| Critic | UniZero categorical value head | Same head, trained on GAE returns | Ensemble, terminal bootstrap and TD(λ) training |
| Action space in current code | Discrete and existing UniZero variants | Discrete only | Continuous, tanh-squashed actor |
| Acting compute | `num_simulations` recurrent expansions | One policy forward | One actor forward (training uses `H` model steps) |
| Safe data reuse | Replay/reanalysis | Replay only for world model; never old actor ratios | Model/data buffer plus newly imagined rollouts |

This comparison also explains why “PPO on a world model” should not be implemented by simply substituting predicted
rewards into GAE: that would inherit PWM's model-gradient/model-bias regime without its pathwise estimator, while
still paying PPO's on-policy variance. The implemented boundary is deliberate: real trajectories improve the actor;
replay improves the latent model; cached contextual features make the boundary exact.

## Configuration and commands

The PPO-specific options are under `policy.ppo`:

```python
policy=dict(
    policy_improvement='ppo',
    collect_with_pure_policy=True,
    learning_rate=0.0001,
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

`accumulation_steps` must remain `1` in PPO mode. Otherwise residual gradients could cross the actor/critic and
world-model phase boundary. The policy validates this rather than silently mixing objectives.

Run the aligned examples with:

```bash
python zoo/classic_control/cartpole/config/cartpole_unizero_ppo_config.py --seed 0
python zoo/box2d/lunarlander/config/lunarlander_disc_unizero_ppo_config.py --seed 0
python zoo/atari/config/atari_unizero_ppo_config.py --env PongNoFrameskip-v4 --seed 0
```

For controlled comparisons, change only `policy_improvement`, the MCTS simulation budget, and PPO-specific options.
Keep the encoder/Transformer size, learning rate, discount, collection budget, preprocessing and evaluation seeds
identical.

## Validation criteria

A run should not be treated as a convergence result based on one seed. At minimum, record three seeds and report:

- evaluation return versus environment steps and wall-clock time;
- actor `approx_kl`, clip fraction, ratio range and entropy;
- raw advantage mean/std before normalization;
- value loss/calibration and latent/reward-model losses;
- collection throughput and MCTS simulations avoided;
- mean and standard deviation of final return across seeds.

The unit tests cover GAE/return ordering, rollout-wide normalization, action masking, ratio freshness, interleaved
episode/segment alignment, non-overlapping fresh sampling, rollout-tensor release and actor/critic gradient isolation.

Current single-seed integration evidence (not a multi-seed benchmark):

- CartPole seed 0 reached `200/200` in all three evaluation episodes after about `4,015` collected steps and repeated
  the result at later evaluations through `23,142` steps. The first PPO ratio mean was exactly `1.0`.
- LunarLander-v3 and Pong both complete collect → PPO → replay-world-model smoke cycles. Long-running convergence
  jobs should be attached to the PR together with raw logs; cluster quota delays must not be reported as convergence.
