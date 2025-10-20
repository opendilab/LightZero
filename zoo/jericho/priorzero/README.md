# PriorZero: LLM-Guided World Model Planning

**PriorZero** combines the strengths of large language models (LLMs) and world model-based planning (UniZero) to achieve efficient and effective decision-making in complex text-based environments.

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
├── priorzero_entry.py           # Main training loop (async)
├── priorzero_config.py          # Complete configuration
├── priorzero_policy.py          # Dual-model policy (World Model + LLM)
├── priorzero_collector.py       # Async data collection with LLM priors
├── priorzero_evaluator.py       # Evaluation (inherits from MuZeroEvaluator)
├── game_segment_priorzero.py    # Enhanced game segment with MCTS policies
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Installation

Ensure you have the following dependencies:
```bash
# Core dependencies
pip install torch transformers vllm peft
pip install ding-engine tensorboardX loguru easydict

# LightZero
cd LightZero && pip install -e .
```

### 2. Quick Test Run

```bash
cd /mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/jericho/priorzero

# Quick test (reduced resources)
python priorzero_entry.py --quick_test --env_id zork1.z5 --seed 0

# Full training
python priorzero_entry.py --env_id zork1.z5 --seed 0 --max_iter 100000
```

### 3. Test Individual Components

```bash
# Test configuration
python priorzero_config.py

# Test game segment
python game_segment_priorzero.py
```

## 🔧 Configuration

### Standard Configuration

```python
from priorzero_config import get_priorzero_config

main_cfg, create_cfg = get_priorzero_config(
    env_id='zork1.z5',
    seed=0
)
```

### Quick Test Configuration (Reduced Resources)

```python
from priorzero_config import get_priorzero_config_for_quick_test

test_cfg, create_cfg = get_priorzero_config_for_quick_test(
    env_id='zork1.z5',
    seed=0
)
```

### Preset Configurations

```python
# Pure UniZero (no LLM)
from priorzero_config import get_config_pure_unizero
cfg, _ = get_config_pure_unizero()

# LLM with only SFT (no RFT)
from priorzero_config import get_config_llm_only_sft
cfg, _ = get_config_llm_only_sft()

# LLM with LoRA (memory efficient)
from priorzero_config import get_config_with_lora
cfg, _ = get_config_with_lora()
```

## 📊 Key Features

### 1. Dual-Model Training

**World Model (UniZero)**:
- Transformer-based world model in latent space
- Predicts: next latent state, reward, value, policy
- Trained with standard UniZero losses

**LLM Policy**:
- Pre-trained LLM (e.g., Qwen2.5-1.5B)
- Fine-tuned with:
  - **SFT**: Supervised by MCTS visit distributions
  - **RFT**: Reinforced by environment rewards (REINFORCE)
- Optional LoRA for parameter-efficient fine-tuning

### 2. LLM-Guided MCTS

1. LLM generates ranked actions: `[action_1, action_2, ...]`
2. Convert to policy prior: `prior_policy = softmax(weights)`
3. Inject into MCTS root node (replace policy logits)
4. MCTS search refines the policy
5. Select best action based on visit counts

### 3. Async Data Collection

- vLLM for efficient batch inference
- Error handling with retries (max 3 attempts)
- Timeout control (default 30s)
- History buffer management (sliding window)

### 4. Comprehensive Logging

- TensorBoard: Training curves, eval metrics
- File logs: Detailed statistics
- LLM prior effectiveness metrics

## 🎛️ Key Hyperparameters

### World Model
```python
world_model_cfg = dict(
    num_layers=4,           # Transformer layers
    num_heads=8,            # Attention heads
    embed_dim=768,          # Embedding dimension
    context_length=8,       # Number of past transitions
    num_unroll_steps=10,    # Unroll steps for training
)
```

### LLM Policy
```python
llm_policy_cfg = dict(
    pretrain_llm_path="Qwen/Qwen2.5-0.5B-Instruct",
    llm_learning_rate=1e-6,
    llm_loss_weight=0.5,    # Weight of SFT loss
    rft_loss_weight=0.3,    # Weight of RFT loss
    history_length=5,       # Context window
    use_cot=True,           # Chain-of-thought prompting
)
```

### MCTS
```python
mcts_cfg = dict(
    num_simulations=25,         # MCTS simulations per step
    root_dirichlet_alpha=0.3,   # Exploration noise
    root_noise_weight=0.25,     # Noise weight
)
```

## 📈 Expected Results

With proper tuning, PriorZero should:
- **Exploration Efficiency**: Fewer invalid actions searched (thanks to LLM priors)
- **Sample Efficiency**: Faster convergence (thanks to world model planning)
- **Generalization**: Better performance on unseen games (thanks to LLM knowledge)

## 🔍 Monitoring Training

### TensorBoard

```bash
tensorboard --logdir=./data_priorzero/ --port=6006
```

Key metrics to watch:
- `train/wm_total_loss`: World model loss
- `train/llm_sft_loss`: LLM supervised fine-tuning loss
- `train/llm_rft_loss`: LLM reinforcement fine-tuning loss
- `train/total_loss`: Combined loss
- `evals/reward_mean`: Average evaluation reward

### File Logs

Check `./data_priorzero/{exp_name}/log/` for:
- Training logs
- LLM prior statistics (success rate, latency)
- Game segment statistics

## 🐛 Troubleshooting

### OOM (Out of Memory)

1. **Reduce vLLM memory**:
   ```python
   gpu_memory_utilization=0.2  # Default: 0.3
   ```

2. **Enable LoRA for LLM**:
   ```python
   use_lora=True
   lora_r=8
   ```

3. **Reduce batch size**:
   ```python
   batch_size=16  # Default: 32
   ```

4. **Reduce MCTS simulations**:
   ```python
   num_simulations=10  # Default: 25
   ```

### LLM Generation Timeout

Increase timeout in collector:
```python
# In priorzero_collector.py
await self._async_get_llm_prior(..., timeout=60.0)  # Default: 30.0
```

### Slow Training

1. **Use Quick Test Config**: Reduces all resource requirements
2. **Reduce collector environments**: `collector_env_num=2`
3. **Reduce update frequency**: `update_per_collect=5`

## 📚 References

### Theoretical Foundations

1. **AlphaGo/AlphaZero**: Policy-guided MCTS
2. **MuZero**: Model-based RL with learned dynamics
3. **UniZero**: Unified world model for various domains
4. **ORZ (OpenAI)**: LLM fine-tuning for reasoning

### Related Papers

- **UniZero**: "Unifying World Models via Transformers"
- **MuZero**: "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"
- **Language Models as World Models**: Various recent works on LLM planning

## 🤝 Contributing

This is a research codebase. Contributions are welcome! Key areas for improvement:

1. **Better LLM prompts**: Improve action ranking quality
2. **Reward shaping**: Better credit assignment for RFT
3. **Multi-task learning**: Train on multiple games simultaneously
4. **Efficient MCTS**: Reduce simulation budget via better priors

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{priorzero2025,
  title={PriorZero: LLM-Guided World Model Planning},
  author={PriorZero Team},
  year={2025},
  howpublished={\url{https://github.com/your-repo/priorzero}}
}
```

## 📄 License

This project follows the same license as LightZero.

---

**Happy Training! 🚀**

For questions or issues, please open an issue on GitHub or contact the maintainers.
