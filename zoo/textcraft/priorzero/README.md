# PriorZero TextCraft

PriorZero (MCTS + World Model + LLM Prior) adapted for the AgentGym-RL TextCraft environment — a Minecraft-style crafting task where the agent must gather ingredients and follow recipes to craft a target item.

## Prerequisites

Start the TextCraft server:
```bash
cd /path/to/AgentGym-RL/AgentGym/agentenv-textcraft
python -m agentenv_textcraft.launch --port 36005
```

## Quick Start

```bash
# Full training (4 GPUs)
bash scripts/run_priorzero_ddp.sh

# Debug mode (single GPU)
CUDA_VISIBLE_DEVICES=0 python src/priorzero_entry_sync_ddp.py \
    --env_id textcraft --env_addr http://127.0.0.1:36005 \
    --data_idx 0 --model qwen2.5-3b --use_cot --quick_test
```

## Configuration

Key parameters in `scripts/run_priorzero_ddp.sh`:
- `DATA_IDX`: Selects goal item from crafting tree (sorted by depth)
- `LLM_MODEL`: Model size (`qwen2.5-0.5b`, `qwen2.5-1.5b`, `qwen2.5-3b`, `qwen2.5-7b`)
- `USE_COT`: Enable chain-of-thought reasoning (recommended: `true`)
- `AGENTGYM_SERVER_ADDR`: TextCraft server address (default: `http://127.0.0.1:36005`)

## Environment Details

- **Reward**: Binary (0 = not done, 1 = goal item crafted)
- **Actions**: Free-form text — `craft <item> using <ingredients>`, `get <count> <item>`, `inventory`
- **Max steps**: 30 (aligned with AgentGym-RL baseline)
