# PriorZero on BabyAI (AgentGym-RL)

BabyAI is a 2D grid world with natural language missions ("go to the red ball", "pick up the blue key") and text-based observations describing the agent's local view. The AgentGym server provides **high-level semantic actions** (e.g., "go to red ball 1", "toggle and go through green closed door 1") that abstract over low-level movement, making the dynamic action space similar to Jericho.

## Prerequisites

Start the AgentGym BabyAI server before training:

```bash
cd /path/to/AgentGym-RL/AgentGym/agentenv-babyai
pip install -e .
python3 -m uvicorn agentenv_babyai:app --host 0.0.0.0 --port 8000    
```

Verify: `curl http://127.0.0.1:8000/` should return 200.

## Key Differences from Jericho PriorZero

| Aspect | Jericho | BabyAI |
|---|---|---|
| Connection | Local Python `env.step()` | HTTP client → AgentGym server |
| Action space | Dynamic text commands (10-100+) | Dynamic high-level actions (3-15) or 7 atomic |
| Observation | Game engine text | Natural language grid description |
| Mission | Implicit in game context | Explicit "Your goal: ..." string |
| Reward | Sparse integer score | Continuous [0,1]: `1 - 0.9*(steps/max_steps)` |
| `data_idx` encoding | N/A (game file path) | `level = idx % 40 + 1`, `seed = idx // 40` |

## Quick Start

Debug mode (1 GPU, 20 steps):
```bash
cd zoo/babyai/priorzero
torchrun --nproc_per_node=1 ./src/priorzero_entry_sync_ddp.py \
    --quick_test --env_addr http://127.0.0.1:8000 --data_idx 0
```

Full training (4 GPUs):
```bash
cd zoo/babyai/priorzero
bash scripts/run_priorzero_ddp.sh
```

Use `--use_low_level_actions` to switch to 7 atomic actions (turn left/right, move forward, pickup, drop, toggle, check).

## Known Issues

1. Observation "left/right" is relative to agent heading, not map coordinates
2. `pickup`/`toggle` only affect the cell directly ahead — wrong calls waste a step
3. Compound missions (PutNext, Sequence) may need mission decomposition not covered by default prompts
4. Early episodes have low reward due to step-count decay — watch the trend, not absolute values
5. If many episodes fail at reset, check the AgentGym server first (connection issues), not the algorithm
