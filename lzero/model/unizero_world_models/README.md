## Position Encoding in UniZero World Model

This section provides a detailed explanation of the position encoding strategies used in the UniZero world model and presents two configurable options based on the value of the configuration parameter `self.config.rotary_emb`.

> **Configuration Options:**
> - When `self.config.rotary_emb = False`, **Absolute Position Encoding** (based on `nn.Embedding`) is used.
> - When `self.config.rotary_emb = True`, **Rotary Position Encoding** (RoPE) is used.

---

### 1. Absolute Position Encoding (Based on `nn.Embedding`)

When the configuration parameter `self.config.rotary_emb` is set to **False**, the model uses `nn.Embedding` for position encoding. The implementation process involves the following steps:

#### 1.1 Embedding Layer Initialization

- **Initialization:**  
  An embedding layer is instantiated using `nn.Embedding`, which maps each position index in the sequence to a fixed-dimensional embedding vector.

#### 1.2 Context Length Restriction

- **kv_cache Management:**  
  Due to the limitation of context length (`context_length`), the model retains only the most recent `<context_length>` steps when caching key-value pairs (kv_cache) to ensure computational efficiency and manageable memory consumption.

#### 1.3 Advancing a Full KV Window

A learned absolute-position window cannot be shifted exactly by adding projected
position differences to cached keys and values. A cached key/value is computed
from a normalized, contextual hidden state, so it is not a linear function of
the position embedding alone. Repeated algebraic correction is therefore only
an approximation and can accumulate error.

For exact learned-absolute-position inference, enable
`rebuild_kv_window_from_tokens`. UniZero then retains the bounded raw embedded
observation/action tokens alongside each cache entry. When the window advances,
it keeps the newest tokens, assigns them positions starting at zero, and runs
them through the Transformer again to rebuild every layer's keys and values.
The raw-token and KV stores share the same eviction and reset lifecycle.

`exact_kv_window_reset` is a diagnostic alternative that rebuilds only from the
latest latent observation. It avoids invalid K/V algebra, but intentionally
discards older context. The legacy position-difference path remains available
for checkpoint comparisons and should not be treated as exact.

---

### 2. Rotary Position Encoding (RoPE)

When the configuration parameter `self.config.rotary_emb` is set to **True**, the model adopts ROPE (Rotary Position Embedding) for position encoding. The main features and implementation process of ROPE are as follows:

#### 2.1 ROPE Initialization

- **Precalculation of Frequency Components:**  
  Frequency components are precalculated and applied to the query and key tensors through a rotational position embedding, directly incorporating positional information into the self-attention computation.

#### 2.2 Episode Time Step-based Indexing

- **Indexing Approach:**  
  Each position index is assigned based on the episode’s time step.  
  For example, when states (`s`) and actions (`a`) alternate, each time step occupies two position indices.  
  Suppose a game consists of 50 steps with states and actions in sequence:  
  `(s₁, a₁, s₂, a₂, ..., s₅₀, a₅₀)`  
  
  The corresponding position indices would be:  
  `1, 2, 3, 4, ..., 99, 100`

#### 2.3 Principles of ROPE

- **Theoretical Basis:**  
  The design of ROPE is inspired by the paper [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864).  
  This method not only encodes absolute positions using a rotation matrix but also directly integrates relative positional information into the self-attention computation, thereby achieving:
  - Greater flexibility (adjustable sequence length);
  - A gradual decay in inter-token dependency with increasing relative distance;
  - Compatibility with relative position encoding in linear self-attention architectures.



### 3. Choosing a Mode

Learned absolute positions and RoPE have different checkpoint parameters and
must be selected before training. RoPE keys retain their original rotations
when old tokens are trimmed, so they do not require learned-absolute K/V
rebasing. Compare the modes empirically for a task; dependency length alone is
not sufficient to predict performance. The current multi-task world model does
not propagate per-root episode positions through its cache API and therefore
rejects RoPE rather than silently assigning incorrect positions.

### 4. Replay Reanalysis and KV Context

UniZero replay reanalysis is not equivalent to MuZero reanalysis. A MuZero
recurrent state is self-contained, whereas a UniZero root also depends on its
Transformer KV prefix. Before each sampled root, the buffer now recovers the
available replay observation/action prefix, encodes it with the matching target
tokenizer, and applies the same bounded raw-token rolling rule used online. The
resulting prefix forward supplies both the root KV cache and its contextual
policy prior, so the root prior and its first recurrent edge describe the same
history. Root caches are stored in isolated per-root init slots because equal
current latents can have different histories.

Both ordinary and sampled C++ replay searches are split into chunks no wider
than the online environment batch. This keeps each tree within the recurrent
cache capacity and restores KV hits among descendants created during the same
search. Episode positions and H+1 roots preserve their original order across
chunks. Current multi-task configs without task-token conditioning select the
task-specific tokenizer and prediction heads. Add/concat/register task-token
context reconstruction is rejected explicitly until its exact raw-token
semantics are implemented. Reanalysis remains configurable and default-off in
the Atari experiments because correctness does not by itself establish a
performance benefit.

### 5. Context-aligned TD Bootstrap Values

`bootstrap_value_context` (CLI: `--bootstrap-value-context`) evaluates TD
bootstrap roots from the same replay prefix and rolling window available to
online planning. Without it, the first bootstrap root has no preceding history
while later roots can see a longer training-only sequence, making value targets
depend on state information unavailable to the online planner.

The contextual path obtains root latents directly from the target tokenizer and
executes only the required context Transformer. A full legacy training-sequence
forward is retained for the first batch and every 1000th batch to log
legacy/contextual mean, standard deviation, delta RMS/max, and context lengths;
the other 999 batches skip that unused forward. This optimization changes
neither root latents nor contextual target values. Task-specific tokenizers and
value heads are selected when multi-task mode does not use extra task tokens.

### 6. Open-loop Latent Diagnostics

`open_loop_diagnostic_freq > 0` enables a detached diagnostic at the requested
learner interval. All three paths run in evaluation mode so dropout cannot
contaminate their comparison:

- full teacher forcing uses the complete training sequence;
- rolling teacher forcing uses the online KV-window rule but feeds real later
  observation embeddings, isolating window truncation and cache semantics;
- open-loop rolling feeds each predicted latent back, matching MCTS exposure.

`rolling_context_ratio` is rolling-teacher MSE divided by full-teacher MSE.
`open_loop_exposure_ratio` is open-loop MSE divided by rolling-teacher MSE, and
`open_loop_total_ratio` is open-loop MSE divided by full-teacher MSE. These are
logging-only measurements and never contribute gradients or change targets.
When `open_loop_prefix_transitions` is nonzero, all three diagnostic paths use
that same teacher prefix and post-prefix target slice, so the logged exposure
metrics describe the mechanism trained by the optional loss below.

### 7. Optional Open-loop Consistency Loss

`open_loop_consistency_loss_weight > 0` adds a short differentiable rollout
that feeds each predicted latent back into the world model and matches it to
the target encoder's later observation embeddings. This directly trains the
distribution used by recurrent MCTS instead of only the teacher-forced path.
The rollout uses evaluation mode (while retaining gradients), raw-token window
rebuilding, and a configurable sample count and horizon. Its loss is an
auxiliary batch scalar and is not rescaled by prioritized-replay importance
weights.

`open_loop_prefix_transitions` optionally prepends real replay transitions
before the differentiable rollout. For example, a prefix of three builds
`[o0,a0,o1,a1,o2,a2,o3]`, the seven-token steady history retained by a
10-token inference cache before its next action. Supervised targets still
start after the prefix. This isolates history-conditioned exposure from simply
increasing rollout horizon; its default is zero.

The default weight is zero, so existing training is unchanged. This option is
currently supported only for the single-task, discrete-action, learned
absolute-position world model with raw-token KV-window rebuilding. Enable it
only after the diagnostic ratios show that autoregressive exposure, rather
than rolling-window semantics, is the dominant error source.

### 8. Optional MuZero-style Recurrent Loss

`open_loop_recurrent_loss_weight > 0` extends the same predicted-latent rollout
with the supervision used by MuZero's recurrent learner: latent and reward are
trained after each action, then policy and value are trained after feeding the
predicted observation back as the next state. Component losses use the normal
UniZero weights (latent 10, reward 1, value 0.5, policy 1); the configured
recurrent weight scales their combined auxiliary loss. Batch size and horizon
reuse the open-loop consistency settings, as does the optional teacher prefix.

This option is mutually exclusive with latent-only open-loop consistency,
which it already contains. It has the same support restrictions and defaults
to zero, so existing configs are unchanged.
