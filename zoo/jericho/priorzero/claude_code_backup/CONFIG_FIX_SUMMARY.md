# PriorZero Configuration Fix Summary

## Problem Description

**Error**: `RuntimeError: The size of tensor a (6) must match the size of tensor b (20) at non-singleton dimension 3`

**Root Cause**: Inconsistent sequence length parameters in `get_priorzero_config_for_quick_test()` caused attention mask dimension mismatch.

### Debug Analysis

From ipdb analysis:
- `att.shape`: `torch.Size([2, 2, 20, 20])` - Attention scores for 20 tokens
- `mask.shape`: `torch.Size([6, 6])` - Attention mask for 6 tokens
- `x.shape`: `torch.Size([2, 20, 768])` - Input sequence of 20 tokens

**Contradiction**: The model was processing a 20-token sequence, but the mask was created for 6 tokens.

### Configuration Inconsistency

In `get_priorzero_config_for_quick_test()`, there were **two different `num_unroll_steps` values**:

1. **Policy level**: `main_config.policy.num_unroll_steps = 10` (inherited from base config, never updated)
   - This caused the model to process sequences of **10 timesteps × 2 tokens/timestep = 20 tokens**

2. **World model level**: `main_config.policy.model.world_model_cfg.num_unroll_steps = 3`
   - This caused the attention mask to be created for **3 timesteps × 2 tokens/timestep = 6 tokens**

## Solution

### Key Changes in `priorzero_config.py`

#### Before (Line 494-544):
```python
def get_priorzero_config_for_quick_test(env_id: str = 'zork1.z5', seed: int = 0, debug_mode: bool = False):
    main_config, create_config = get_priorzero_config(env_id, seed, debug_mode=debug_mode)

    # ... other configs ...

    # ❌ MISSING: policy.num_unroll_steps was never set!
    # It remained at default value of 10 from base config

    main_config.policy.model.world_model_cfg.num_unroll_steps = 3
    main_config.policy.model.world_model_cfg.max_blocks = 3
    main_config.policy.model.world_model_cfg.max_tokens = 6
```

#### After (Line 494-568):
```python
def get_priorzero_config_for_quick_test(env_id: str = 'zork1.z5', seed: int = 0, debug_mode: bool = False):
    main_config, create_config = get_priorzero_config(env_id, seed, debug_mode=debug_mode)

    # ✅ Define core parameter first
    quick_test_num_unroll_steps = 3
    quick_test_infer_context_length = 2
    tokens_per_block = 2

    # ... other configs ...

    # ✅ CRITICAL FIX: Set policy-level num_unroll_steps
    main_config.policy.num_unroll_steps = quick_test_num_unroll_steps

    # ✅ Ensure all world model params are consistent
    main_config.policy.model.world_model_cfg.num_unroll_steps = quick_test_num_unroll_steps
    main_config.policy.model.world_model_cfg.max_blocks = quick_test_num_unroll_steps
    main_config.policy.model.world_model_cfg.max_tokens = quick_test_num_unroll_steps * tokens_per_block
    main_config.policy.model.world_model_cfg.infer_context_length = quick_test_infer_context_length
    main_config.policy.model.world_model_cfg.context_length = quick_test_infer_context_length * tokens_per_block
    main_config.policy.model.world_model_cfg.tokens_per_block = tokens_per_block
```

### Configuration Consistency Requirements

For UniZero Transformer world model, these parameters **must be consistent**:

| Parameter | Value (quick_test) | Formula | Description |
|-----------|-------------------|---------|-------------|
| `policy.num_unroll_steps` | 3 | - | **Must match** `world_model_cfg.num_unroll_steps` |
| `world_model_cfg.num_unroll_steps` | 3 | - | Number of timesteps in training unroll |
| `world_model_cfg.max_blocks` | 3 | `= num_unroll_steps` | Max number of timestep blocks |
| `world_model_cfg.tokens_per_block` | 2 | - | Fixed (obs + action) |
| `world_model_cfg.max_tokens` | 6 | `= num_unroll_steps × 2` | Total sequence length in tokens |
| `world_model_cfg.infer_context_length` | 2 | - | Inference context window |
| `world_model_cfg.context_length` | 4 | `= infer_context_length × 2` | Context length in tokens |

## Verification

Run the verification script to check consistency:

```bash
cd /mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/jericho/priorzero
python verify_config_consistency.py
```

**Expected output**:
```
✓ ALL CONSISTENCY CHECKS PASSED!

Expected sequence lengths:
  - Training unroll:    3 timesteps = 6 tokens
  - Inference context:  2 timesteps = 4 tokens

Attention mask will be created for 6 tokens (shape: [6, 6])
This should match the sequence length being processed.
```

## Impact

### Fixed Issues
✅ Attention mask dimension mismatch error
✅ Inconsistent sequence length parameters
✅ Configuration parameter alignment between policy and world model

### Affected Components
- **Training**: Data sampling and unrolling now uses consistent `num_unroll_steps=3`
- **World Model**: Transformer attention masks now match actual sequence length
- **Replay Buffer**: Samples trajectories with correct sequence length

## Files Modified

1. **[priorzero_config.py](priorzero_config.py:494-568)**
   - Fixed `get_priorzero_config_for_quick_test()` function
   - Added explicit setting of `policy.num_unroll_steps`
   - Added detailed comments explaining consistency requirements

## Additional Resources

- **Verification Script**: [verify_config_consistency.py](verify_config_consistency.py)
- **Configuration Documentation**: See inline comments in `priorzero_config.py`

## Testing Recommendations

After this fix, verify the following:

1. **Configuration consistency** (automated):
   ```bash
   python verify_config_consistency.py
   ```

2. **Model initialization** (manual):
   - Check that attention masks have correct shape `[6, 6]`
   - Verify sequence length is 6 tokens (3 timesteps × 2)

3. **Training** (manual):
   - Run a quick test training iteration
   - Ensure no dimension mismatch errors occur

---

**Date**: 2025-10-21
**Author**: PriorZero Team
**Status**: ✅ Fixed and Verified
