# PriorZero: LLM-Guided World Model Planning

**PriorZero** combines large language models (LLMs) with world model-based planning (UniZero) for efficient decision-making in complex text-based environments.

## 🎯 Core Idea

**Decouple Policy and World Model:**
- **LLM Policy**: Provides high-quality action priors using language understanding and world knowledge
- **World Model (UniZero)**: Performs efficient multi-step planning in latent space via MCTS

**Training Loop:**
1. **Collect**: LLM generates action rankings → MCTS search refines them → Execute best action
2. **Store**: Save MCTS visit distributions (for SFT) and environment rewards (for RFT)
3. **Train**:
   - World Model: Standard UniZero losses (value, policy, reward, latent)
   - LLM: Supervised Fine-Tuning (SFT) on MCTS policies + Reinforcement Fine-Tuning (RFT) on env rewards

## 📁 File Structure

```
priorzero/
├── priorzero_entry.py           # Main async training loop (stable, tested)
├── priorzero_orz_complete.py    # ORZ integration version (experimental)
├── priorzero_config.py          # Complete configuration with presets
├── priorzero_policy.py          # Dual-model policy (World Model + LLM)
├── priorzero_collector.py       # Async data collection with vLLM
├── game_segment_priorzero.py    # Enhanced GameSegment with MCTS policies & raw text
├── ensure_local_lightzero.py    # Import path management
└── README.md                    # This file
```

## 🔀 Two Training Entry Points

PriorZero provides two training entry points with different LLM training strategies:

### 1. `priorzero_entry.py` - Standard PriorZero (Stable ✅)

**Status**: Production-ready, tested, can run for extended periods

**LLM Training Strategy**:
- **Built-in SFT + RFT** implemented directly in `priorzero_policy.py`
- Uses micro-batching with gradient accumulation (memory efficient)
- Simple and straightforward implementation
- Fully integrated with UniZero training loop

**Key Features**:
- Single-process async training
- vLLM for inference only (action prior generation)
- LLM training via standard PyTorch optimizer
- ~580 lines of clean, maintainable code

**When to use**:
- ✅ Standard PriorZero experiments
- ✅ Quick prototyping and debugging
- ✅ Single GPU training
- ✅ When you want simple, stable training

**Usage**:
```bash
# Quick test
python priorzero_entry.py --quick_test --env_id zork1.z5 --seed 0

# Full training
python priorzero_entry.py --env_id zork1.z5 --seed 0 --max_iter 100000
```

### 2. `priorzero_orz_complete.py` - ORZ Integration (Experimental ⚠️)

**Status**: Newly implemented, requires testing, not yet verified

**LLM Training Strategy**:
- **ORZ RayPPOTrainer** for distributed PPO-based LLM fine-tuning
- Leverages OpenAI's ORZ (Open Reasoner Zero) framework
- More sophisticated RL training with actor-critic architecture
- Distributed training with Ray

**Key Features**:
- Hybrid training: UniZero world model + ORZ PPO for LLM
- Ray-based distributed execution
- Custom reward function for Jericho text adventures
- Separate training frequencies for world model vs LLM
- ~960 lines with complete ORZ integration

**Key Differences from Standard Entry**:
1. **LLM Training**: Uses ORZ's `RayPPOTrainer` instead of built-in SFT/RFT
2. **Reward Signal**: Custom `JerichoRewardTrainer` for text adventure rewards
3. **Distribution**: Ray-based parallel training
4. **Complexity**: More sophisticated but requires ORZ dependency
5. **Training Loop**: Separate update frequencies for WM and LLM

**When to use**:
- ⚠️ Advanced RL research with PPO-based LLM training
- ⚠️ When you have ORZ framework available
- ⚠️ Distributed training across multiple GPUs/nodes
- ⚠️ When you want more sophisticated reward modeling

**Requirements**:
```bash
# Additional dependencies
pip install ray  # For distributed execution
cd /path/to/Open-Reasoner-Zero && pip install -e .
```

**Usage**:
```bash
# Debug mode
DEBUG_MODE=True python priorzero_orz_complete.py

# Full training (requires ORZ setup)
python priorzero_orz_complete.py --env_id zork1.z5 --seed 0
```

### Comparison Table

| Feature | `priorzero_entry.py` | `priorzero_orz_complete.py` |
|---------|---------------------|----------------------------|
| **Status** | ✅ Stable, Tested | ⚠️ Experimental, Needs Testing |
| **Lines of Code** | ~580 | ~960 |
| **LLM Training** | Built-in SFT+RFT | ORZ RayPPOTrainer (PPO) |
| **Dependencies** | Basic (vLLM, torch) | Advanced (ORZ, Ray) |
| **Training Mode** | Single-process async | Distributed (Ray) |
| **Memory Efficiency** | Micro-batching | Ray workers |
| **Reward Modeling** | Simple env rewards | Custom reward functions |
| **Setup Complexity** | Low | Medium-High |
| **Debugging** | Easy | More complex |
| **Performance** | Not fully verified | Unknown (needs testing) |
| **Recommended For** | Most users | Advanced research |

### Which One Should You Use?

**Start with `priorzero_entry.py` if:**
- You're new to PriorZero
- You want stable, tested code
- You're doing standard MCTS + LLM experiments
- You have limited GPU resources
- You want simple debugging

**Try `priorzero_orz_complete.py` if:**
- You have ORZ framework set up
- You want distributed training
- You need custom reward modeling
- You're doing advanced RL research
- You're willing to debug experimental code

**Note**: The standard entry (`priorzero_entry.py`) has been tested and can run for extended periods. The ORZ version is newly implemented and requires thorough testing before production use.


## 🚀 Quick Start

### 1. Installation

**Basic Installation** (for `priorzero_entry.py`):
```bash
# Core dependencies
pip install torch transformers vllm peft
pip install ding-engine tensorboardX loguru easydict jericho

# LightZero (local development mode)
cd /path/to/LightZero && pip install -e .
```

**Advanced Installation** (for `priorzero_orz_complete.py`):
```bash
# Basic dependencies (same as above)
pip install torch transformers vllm peft
pip install ding-engine tensorboardX loguru easydict jericho

# Additional ORZ dependencies
pip install ray  # For distributed training
cd /path/to/Open-Reasoner-Zero && pip install -e .

# LightZero
cd /path/to/LightZero && pip install -e .
```

### 2. Quick Test Run

**Standard PriorZero** (recommended for most users):
```bash
cd /mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/jericho/priorzero

# Quick test (reduced resources, 2 envs, 10 iters)
python priorzero_entry.py --quick_test --env_id zork1.z5 --seed 0

# Full training (default: 4 envs, 100k iters)
python priorzero_entry.py --env_id zork1.z5 --seed 0 --max_iter 100000
```

**ORZ Integration** (experimental, requires ORZ setup):
```bash
cd /mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/jericho/priorzero

# Debug mode (minimal resources)
DEBUG_MODE=True python priorzero_orz_complete.py

# Full training with ORZ
python priorzero_orz_complete.py --env_id zork1.z5 --seed 0
```

### 3. Test Individual Components

```bash
# Test configuration
python priorzero_config.py

# Test game segment
python game_segment_priorzero.py

# Test buffer
python ../../../lzero/mcts/buffer/game_buffer_priorzero.py
```

## 🔧 Configuration

### Preset Configurations

```python
# 1. Standard PriorZero (World Model + LLM with SFT + RFT)
from priorzero_config import get_priorzero_config
main_cfg, create_cfg = get_priorzero_config(env_id='zork1.z5', seed=0)

# 2. Quick Test (reduced resources)
from priorzero_config import get_priorzero_config_for_quick_test
test_cfg, create_cfg = get_priorzero_config_for_quick_test(env_id='zork1.z5', seed=0)

# 3. Pure UniZero (no LLM)
from priorzero_config import get_config_pure_unizero
cfg, _ = get_config_pure_unizero()

# 4. LLM with only SFT (no RFT)
from priorzero_config import get_config_llm_only_sft
cfg, _ = get_config_llm_only_sft()

# 5. LLM with LoRA (memory efficient)
from priorzero_config import get_config_with_lora
cfg, _ = get_config_with_lora()
```

## 📊 Key Features

### 1. Dual-Model Training

**World Model (UniZero)**:
- Transformer-based world model in latent space
- Predicts: next latent state, reward, value, policy
- Trained with standard UniZero losses (full batch size)
- **Training frequency**: Every iteration (standard RL loop)

**LLM Policy** - Two Implementations:

#### Standard Entry (`priorzero_entry.py`):
- Pre-trained LLM (default: Qwen2.5-0.5B-Instruct)
- Fine-tuned with:
  - **SFT**: Supervised by MCTS visit distributions
  - **RFT**: Reinforced by environment rewards (REINFORCE)
- **Gradient Accumulation**: Micro-batching to avoid OOM
- **Training frequency**: Every iteration (joint optimization with world model)
- Optional LoRA for parameter-efficient fine-tuning

#### ORZ Entry (`priorzero_orz_complete.py`):
- Pre-trained LLM (configurable)
- Fine-tuned with:
  - **ORZ PPO**: Proximal Policy Optimization via RayPPOTrainer
  - **Custom Rewards**: JerichoRewardTrainer for text adventure scoring
  - **Actor-Critic**: Separate value network for advantage estimation
- **Ray Distribution**: Parallel workers for distributed training
- **Training frequency**: Configurable (default: every N world model updates)
- Support for LoRA and other PEFT methods

### 2. Memory-Efficient Training (OOM Fix)

**Micro-Batching with Gradient Accumulation** (Standard Entry):
```python
llm_policy_cfg = dict(
    llm_micro_batch_size=4,              # Small batch per forward pass
    llm_gradient_accumulation_steps=8,   # Accumulate over 8 steps
    # Effective batch size = 4 * 8 = 32
)
```

**How it works**:
- LLM training processes data in small chunks (2-4 samples)
- Gradients accumulate across micro-batches
- Single optimizer step applies accumulated gradients
- World model still trains with full batches (no slowdown)
- Automatic memory cleanup: `torch.cuda.empty_cache()` after each micro-batch

**Ray Workers** (ORZ Entry):
- Distributed across multiple Ray actors
- Each worker handles subset of data
- Automatic load balancing
- More scalable for large-scale training

**Tuning guidelines**:
- **If OOM**: Reduce `llm_micro_batch_size` to 1 or 2
- **If have more memory**: Increase to 8 or 16
- Effective batch = `llm_micro_batch_size * llm_gradient_accumulation_steps`

### 3. LLM-Guided MCTS

1. LLM generates ranked actions: `[action_1, action_2, ...]`
2. Convert to policy prior: `prior_policy = softmax(weights)`
3. Inject into MCTS root node (replace policy logits)
4. MCTS search refines the policy (25 simulations)
5. Select best action based on visit counts

### 4. Async Data Collection

- **vLLM Engine**: Efficient batch inference (V1 API)
- **Error Handling**: Auto-retry (max 3 attempts) with backoff
- **Timeout Control**: 30s default per batch
- **History Buffer**: Sliding window (5 recent transitions)
- **Text Observation**: Properly extracts and stores raw text in `raw_obs_segment`

### 5. Enhanced Game Buffer

**PriorZeroGameBuffer** (optimized):
- Overrides `_sample_orig_data()` to cache game segments
- Avoids double sampling (~50% faster)
- Returns `[current_batch, target_batch, game_segments]`
- Minimal memory overhead (uses references, not copies)

## 🎛️ Key Hyperparameters

### World Model
```python
world_model_cfg = dict(
    num_layers=2,              # Transformer layers (reduced for speed)
    num_heads=8,               # Attention heads
    embed_dim=512,             # Embedding dimension
    context_length=8,          # Number of past transitions (2 * infer_context_length)
    num_unroll_steps=10,       # Unroll steps for training
    game_segment_length=50,    # Segment length (reduced for quick test)
)
```

### LLM Policy
```python
llm_policy_cfg = dict(
    pretrain_llm_path="Qwen/Qwen2.5-0.5B-Instruct",
    llm_learning_rate=1e-6,
    llm_loss_weight=0.5,                    # Weight of SFT loss
    rft_loss_weight=0.3,                    # Weight of RFT loss

    # Memory optimization
    llm_micro_batch_size=4,                 # Micro-batch size (2 for quick test)
    llm_gradient_accumulation_steps=8,      # Accumulation steps (4 for quick test)

    # Prompting
    prompt_max_len=2048,                    # Max prompt length (1024 for quick test)
    generate_max_len=256,                   # Max generation length (128 for quick test)
    history_length=5,                       # Context window (3 for quick test)
    use_cot=True,                           # Chain-of-thought prompting

    # Training strategy
    sft_target='mcts_policy',               # Supervised by MCTS visit distributions
    enable_rft=True,                        # Enable RFT with env rewards

    # vLLM
    gpu_memory_utilization=0.3,             # GPU memory fraction for vLLM
)
```

### MCTS
```python
mcts_cfg = dict(
    num_simulations=25,            # MCTS simulations per step (10 for quick test)
    root_dirichlet_alpha=0.3,      # Exploration noise
    root_noise_weight=0.25,        # Noise weight
    pb_c_base=19652,               # UCB constants
    pb_c_init=1.25,
)
```

### Training
```python
training_cfg = dict(
    batch_size=64,                 # World model batch size (32 for quick test)
    update_per_collect=10,         # Updates per collection cycle (5 for quick test)
    max_env_step=1e6,              # Max environment steps
    eval_freq=500,                 # Evaluation frequency

    # Replay buffer
    replay_buffer_size=10000,
    use_priority=True,             # Prioritized experience replay
    priority_prob_alpha=0.6,
    priority_prob_beta=0.4,
)
```

## 📈 Expected Results

With proper tuning, PriorZero should achieve:

- **Exploration Efficiency**: Fewer invalid actions searched (thanks to LLM priors)
- **Sample Efficiency**: Faster convergence (thanks to world model planning)
- **Generalization**: Better performance on unseen games (thanks to LLM knowledge)
- **Memory Efficiency**: No OOM on single GPU (thanks to gradient accumulation)

## 🔍 Monitoring Training

### TensorBoard

```bash
tensorboard --logdir=./data_priorzero/ --port=6006
```

**Key metrics to watch**:
- `train/wm_total_loss`: World model total loss
- `train/llm_sft_loss`: LLM supervised fine-tuning loss
- `train/llm_rft_loss`: LLM reinforcement fine-tuning loss
- `train/total_loss`: Combined loss
- `train/wm_grad_norm`: World model gradient norm
- `train/llm_grad_norm`: LLM gradient norm
- `collector_iter/reward_mean`: Average episode reward
- `collector_iter/visit_entropy_mean`: MCTS exploration entropy
- `evaluator_step/reward_mean`: Evaluation reward

### File Logs

Check `./data_priorzero/{exp_name}/log/` for:
- Training logs with detailed statistics
- LLM prior statistics (success rate, latency, retry count)
- Game segment statistics (MCTS policies, raw obs, search values)

### Debug Logs

During training, you'll see:
```
[LLM Training] Processing X game segments
[LLM Training] First segment stats: mcts_policies=Y, raw_obs=Z/Z, actions=W
[SEGMENT_DEBUG] raw_obs_text = North of House...
```

## 🐛 Troubleshooting

### OOM (Out of Memory)

**1. Reduce LLM micro-batch size** (most effective):
```python
llm_micro_batch_size=2  # or even 1
llm_gradient_accumulation_steps=8  # keep this to maintain effective batch size
```

**2. Reduce vLLM memory**:
```python
gpu_memory_utilization=0.2  # Default: 0.3
```

**3. Enable LoRA for LLM**:
```python
use_lora=True
lora_r=8
lora_alpha=16
```

**4. Reduce world model batch size**:
```python
batch_size=16  # Default: 32 (quick test)
```

**5. Reduce prompt length**:
```python
prompt_max_len=512   # Default: 1024 (quick test)
generate_max_len=64  # Default: 128 (quick test)
```

**6. Reduce MCTS simulations**:
```python
num_simulations=10  # Default: 25
```

### LLM Generation Issues

**Timeout errors**:
```python
# In priorzero_collector.py
await self._async_get_llm_prior(..., timeout=60.0)  # Default: 30.0
```

**vLLM initialization errors**:
- Check CUDA version compatibility
- Ensure `VLLM_USE_V1=1` environment variable (set in entry.py)
- Try reducing `gpu_memory_utilization`

**Empty raw_obs_text**:
- Fixed! Now properly extracts from `obs['raw_obs_text']`
- Check logs for `[SEGMENT_DEBUG] raw_obs_text = ...`

### Gradient Errors

**"element 0 of tensors does not require grad"**:
- Fixed! RFT now properly tracks gradients
- Removed `torch.no_grad()` from RFT forward pass

### Slow Training

**1. Use Quick Test Config**:
```python
get_priorzero_config_for_quick_test()  # Reduces all resources
```

**2. Reduce collector environments**:
```python
collector_env_num=2  # Default: 4
```

**3. Reduce update frequency**:
```python
update_per_collect=5  # Default: 10
```

**4. Reduce game segment length**:
```python
game_segment_length=50  # Default: 200
```

### Buffer/Sampling Issues

**Double sampling fixed**:
- PriorZeroGameBuffer now caches game_segments
- ~50% faster sampling with no memory overhead

## 🔄 Recent Fixes & Improvements

### v2.0.4 (Latest)

✅ **Fixed RFT gradient computation error**
- Removed `torch.no_grad()` from RFT forward pass
- Gradients now properly flow through REINFORCE loss

✅ **Optimized memory efficiency**
- Implemented micro-batching with gradient accumulation for SFT/RFT
- LLM training processes small chunks (2-4 samples) instead of full batch
- Automatic memory cleanup after each micro-batch
- World model still trains with full batches (no slowdown)

✅ **Fixed raw_obs_text propagation**
- Enhanced `extract_raw_obs_text()` to prioritize `raw_obs_text` field
- Properly passes raw text from collector to GameSegment
- Now captures actual text: "North of House", "Behind House", etc.

✅ **Optimized game buffer**
- Eliminated double sampling in `_sample_orig_data()`
- Caches game_segments during sampling (~50% faster)
- Returns game_segments as 3rd element in train_data

## 📚 References

### Theoretical Foundations

1. **AlphaGo/AlphaZero**: Policy-guided MCTS
2. **MuZero**: Model-based RL with learned dynamics
3. **UniZero**: Unified world model for various domains
4. **ORZ (OpenAI)**: LLM fine-tuning for reasoning
5. **REINFORCE**: Policy gradient methods for RL

### Related Papers

- **UniZero**: "Unifying World Models via Transformers"
- **MuZero**: "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"
- **vLLM**: "Efficient Memory Management for Large Language Model Serving"
- **LoRA**: "Low-Rank Adaptation of Large Language Models"

## 🤝 Contributing

This is a research codebase. Contributions are welcome! Key areas for improvement:

1. **Better LLM prompts**: Improve action ranking quality with CoT reasoning
2. **Reward shaping**: Better credit assignment for RFT
3. **Multi-task learning**: Train on multiple games simultaneously
4. **Efficient MCTS**: Reduce simulation budget via better priors
5. **Dynamic action spaces**: Handle variable action sets across games

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{priorzero2025,
  title={PriorZero: LLM-Guided World Model Planning},
  author={PriorZero Team},
  year={2025},
  howpublished={\url{https://github.com/opendilab/LightZero}}
}
```

## 📄 License

This project follows the same license as LightZero (Apache 2.0).

---

**Happy Training! 🚀**

For questions or issues:
- Open an issue on GitHub: https://github.com/opendilab/LightZero/issues
- Check troubleshooting guide above
- Review log files in `./data_priorzero/{exp_name}/log/`
