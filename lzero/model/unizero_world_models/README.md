# UniZero World Model

This package implements the latent Transformer world model used by UniZero. It
encodes observations into latent tokens and predicts the next latent state,
reward, policy, and value from an interleaved observation/action history.

## Architecture

For the standard Atari setup, one transition is represented by two tokens:

```text
[observation latent, action, observation latent, action, ...]
```

The main components are:

- `Tokenizer`: encodes observations into latent observation tokens;
- `Transformer`: applies causal attention to observation/action tokens;
- prediction heads: produce next-latent, reward, policy, and value outputs;
- KV caches: reuse Transformer history during collection, evaluation, and MCTS.

`world_model.py` contains the core model and inference lifecycle. Optional or
specialized behavior is separated into:

- `cache_window.py`: exact KV-window rebuilding primitives;
- `reanalysis_context.py`: replay-root context reconstruction and contextual
  policy/value evaluation;
- `open_loop.py`: open-loop diagnostics and auxiliary objectives;
- `world_model_multitask.py`: multi-task extensions.

## Training and Inference

Training uses teacher-forced observation/action sequences. The target path
for replay reanalysis evaluates all `H+1` observation roots, including the real
bootstrap state `s[t+H]`; ordinary learner training keeps the legacy `H`-step
computation because its final placeholder is discarded.

Online inference keeps a bounded context of `context_length` tokens. KV caches
are isolated per environment, including asynchronous episode resets. Batches
with different valid history lengths are left-padded and supplied with an
attention mask so padding cannot affect predictions.

## Position Encoding and KV Windows

Two position schemes are available:

- `rotary_emb=False`: learned absolute position embeddings;
- `rotary_emb=True`: RoPE applied to attention queries and keys using episode
  positions.

A learned-absolute KV window cannot be shifted exactly by adding projected
position differences: cached K/V tensors depend nonlinearly on the full hidden
context. Three window behaviors therefore exist:

| Configuration | Behavior |
| --- | --- |
| `rebuild_kv_window_from_tokens=True` | Retain bounded raw embedded tokens and replay them to rebuild an exact rolling window. |
| `exact_kv_window_reset=True` | Rebuild from the latest latent only; useful as a diagnostic but intentionally drops older history. |
| both `False` | Use the legacy position-difference path for compatibility with older experiments. |

The two rebuild options are mutually exclusive. Raw-token rebuilding is for
learned absolute positions and is rejected with RoPE. Overflowing samples are
rebuilt together in one Transformer call, then copied back to independent
caches.

RoPE caches retain the rotations already attached to their keys when old tokens
are trimmed, so they do not require learned-position rebasing. The multi-task
world model currently rejects RoPE because its cache API does not yet propagate
an episode position for every root.

## Replay Reanalysis

UniZero replay roots are not self-contained: the current latent and its
Transformer prefix jointly define the state. The implementation always:

- aligns value/policy targets with the real `H+1` observation roots;
- splits C++ replay searches into batches no wider than the online environment
  capacity, preventing the recurrent cache ring from overwriting live trees;
- preserves root order and episode positions across search chunks.

Two history-conditioned target variants are available but disabled by default:

- `contextual_reanalysis=True` reconstructs each replay root's bounded
  observation/action prefix. The same prefix forward supplies both the root
  policy prior and its recurrent KV cache.
- `bootstrap_value_context=True` evaluates TD bootstrap values from the same
  rolling replay context available to online planning.

These options support task-specific tokenizers and heads when multi-task mode
does not add task tokens. Context reconstruction with add/concat/register task
tokens is rejected explicitly because its exact raw-token semantics are not yet
implemented.

## Optional Open-loop Analysis and Training

All open-loop features are disabled by default.

`open_loop_diagnostic_freq > 0` compares three dropout-free paths:

- full teacher forcing;
- rolling teacher forcing, which uses the online window but real future latents;
- open-loop rolling, which feeds predicted latents back into the model.

The main ratios separate rolling-context error from autoregressive exposure
error:

```text
rolling_context_ratio   = rolling_teacher_mse / full_teacher_mse
open_loop_exposure_ratio = open_loop_mse / rolling_teacher_mse
open_loop_total_ratio    = open_loop_mse / full_teacher_mse
```

Two mutually exclusive auxiliary objectives can be enabled:

- `open_loop_consistency_loss_weight > 0`: supervise predicted latents over a
  short differentiable rollout;
- `open_loop_recurrent_loss_weight > 0`: additionally supervise reward after
  each action and policy/value after each predicted next state.

`open_loop_consistency_batch_size`, `open_loop_consistency_horizon`, and
`open_loop_prefix_transitions` control rollout cost and context. These paths
currently require a single-task, discrete-action, one-observation-token model
with learned absolute positions and `rebuild_kv_window_from_tokens=True`.

## Default Behavior

The UniZero policy defaults to:

```python
context_length = 8                 # four observation/action blocks
rotary_emb = False
exact_kv_window_reset = False
rebuild_kv_window_from_tokens = False
contextual_reanalysis = False
bootstrap_value_context = False
open_loop_consistency_loss_weight = 0.0
open_loop_recurrent_loss_weight = 0.0
open_loop_prefix_transitions = 0
```

Environment-specific configs may override these values for controlled
experiments. Keep algorithmic options disabled when reproducing the default
baseline, and change one mechanism at a time when evaluating them.
