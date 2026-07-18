#!/usr/bin/env bash
# =============================================================================
# Ablation Study: VLM Prior on LunarLander
# Runs all parameter combinations and saves results to ablation_results.json
#
# Usage (on GPU worker):
#   cd zoo/jericho/priorzero
#   bash run_ablation.sh
# =============================================================================
set -euo pipefail

PYTHON="/mnt/shared-storage-user/puyuan/xiongjyu/envs/rft/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/scripts/eval_vl_prior.py"
OUTPUT_DIR="${SCRIPT_DIR}/ablation_output"
MERGED_JSON="${SCRIPT_DIR}/ablation_results.json"

# NUM_EPISODES=20
NUM_EPISODES=2

SEED=0
MAX_STEPS=1000

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo " Ablation Study: VLM Prior on LunarLander"
echo " Episodes per combo: ${NUM_EPISODES}"
echo " Output dir: ${OUTPUT_DIR}"
echo "=============================================="

# Counter for tracking progress
TOTAL=0
DONE=0

# ---------------------------------------------------------------------------
# Define all combinations
# ---------------------------------------------------------------------------
# Format: "tag|policy|prompt_style|vlm_image_mode|image_size"
COMBOS=(
    # "random_baseline|random|concise|current_only|64"
    "vlm_concise_current_64|vlm|concise|current_only|64"
    "vlm_concise_current_256|vlm|concise|current_only|256"
    "vlm_concise_first_and_current_64|vlm|concise|first_and_current|64"
    "vlm_concise_first_and_current_256|vlm|concise|first_and_current|256"
    "vlm_legacy_current_64|vlm|legacy|current_only|64"
    "vlm_legacy_current_256|vlm|legacy|current_only|256"
    "vlm_legacy_first_and_current_64|vlm|legacy|first_and_current|64"
    "vlm_legacy_first_and_current_256|vlm|legacy|first_and_current|256"
)

TOTAL=${#COMBOS[@]}

# ---------------------------------------------------------------------------
# Run each combination
# ---------------------------------------------------------------------------
for combo in "${COMBOS[@]}"; do
    IFS='|' read -r TAG POLICY PROMPT_STYLE IMAGE_MODE IMAGE_SIZE <<< "$combo"
    DONE=$((DONE + 1))
    OUTFILE="${OUTPUT_DIR}/${TAG}.json"

    echo ""
    echo "----------------------------------------------"
    echo " [${DONE}/${TOTAL}] Running: ${TAG}"
    echo "   policy=${POLICY}  prompt=${PROMPT_STYLE}  img_mode=${IMAGE_MODE}  res=${IMAGE_SIZE}"
    echo "----------------------------------------------"

    if [ "$POLICY" == "random" ]; then
        $PYTHON "$EVAL_SCRIPT" \
            --policies random \
            --num_episodes "$NUM_EPISODES" \
            --seed "$SEED" \
            --max_steps "$MAX_STEPS" \
            --image_size "$IMAGE_SIZE" \
            --output "$OUTFILE"
    else
        $PYTHON "$EVAL_SCRIPT" \
            --policies vlm \
            --num_episodes "$NUM_EPISODES" \
            --seed "$SEED" \
            --max_steps "$MAX_STEPS" \
            --image_size "$IMAGE_SIZE" \
            --prompt_style "$PROMPT_STYLE" \
            --vlm_image_mode "$IMAGE_MODE" \
            --output "$OUTFILE"
    fi

    echo " >> Saved to ${OUTFILE}"
done

# ---------------------------------------------------------------------------
# Merge all results into one JSON
# ---------------------------------------------------------------------------
echo ""
echo "=============================================="
echo " Merging results..."
echo "=============================================="

$PYTHON -c "
import json, glob, os

merged = {}
for f in sorted(glob.glob('${OUTPUT_DIR}/*.json')):
    with open(f) as fh:
        data = json.load(fh)
    # Each file has {tag: summary_dict}
    merged.update(data)

# Add metadata about the combination parameters for easier analysis
combo_meta = {
    'random_baseline':                    {'policy': 'random', 'prompt_style': '-',       'image_mode': '-',                 'image_size': 64},
    'vlm_concise_current_64':             {'policy': 'vlm',    'prompt_style': 'concise', 'image_mode': 'current_only',      'image_size': 64},
    'vlm_concise_current_256':            {'policy': 'vlm',    'prompt_style': 'concise', 'image_mode': 'current_only',      'image_size': 256},
    'vlm_concise_first_and_current_64':   {'policy': 'vlm',    'prompt_style': 'concise', 'image_mode': 'first_and_current', 'image_size': 64},
    'vlm_concise_first_and_current_256':  {'policy': 'vlm',    'prompt_style': 'concise', 'image_mode': 'first_and_current', 'image_size': 256},
    'vlm_legacy_current_64':              {'policy': 'vlm',    'prompt_style': 'legacy',  'image_mode': 'current_only',      'image_size': 64},
    'vlm_legacy_current_256':             {'policy': 'vlm',    'prompt_style': 'legacy',  'image_mode': 'current_only',      'image_size': 256},
    'vlm_legacy_first_and_current_64':    {'policy': 'vlm',    'prompt_style': 'legacy',  'image_mode': 'first_and_current', 'image_size': 64},
    'vlm_legacy_first_and_current_256':   {'policy': 'vlm',    'prompt_style': 'legacy',  'image_mode': 'first_and_current', 'image_size': 256},
}

# Enrich each result with combo metadata
for key in merged:
    # Match by checking if key starts with any combo tag
    for tag, meta in combo_meta.items():
        if key == tag or key.startswith(tag.replace(tag.split('_')[0] + '_', '', 1)):
            merged[key]['combo_meta'] = meta
            break
    # Fallback: try to match the 'policy' field in the result
    if 'combo_meta' not in merged[key]:
        for tag, meta in combo_meta.items():
            if merged[key].get('policy', '') == tag or tag in merged[key].get('policy', ''):
                merged[key]['combo_meta'] = meta
                break

output = {
    'experiment': 'VLM Prior Ablation on LunarLander-v2',
    'num_episodes': ${NUM_EPISODES},
    'seed': ${SEED},
    'results': merged,
}

with open('${MERGED_JSON}', 'w') as f:
    json.dump(output, f, indent=2)

print(f'Merged {len(merged)} results -> ${MERGED_JSON}')
"

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
echo ""
echo "=============================================="
echo " Printing analysis table..."
echo "=============================================="

$PYTHON -c "
import json

with open('${MERGED_JSON}') as f:
    data = json.load(f)

results = data['results']

# Print Markdown table
print()
print('| # | Configuration | Policy | Prompt | Image Mode | Resolution | Mean Reward | Std | Min | Max | Avg Steps |')
print('|---|--------------|--------|--------|------------|------------|-------------|-----|-----|-----|-----------|')

# Sort: random first, then by reward descending
items = sorted(results.items(), key=lambda x: (x[1].get('combo_meta', {}).get('policy', '') != 'random', -x[1]['reward_mean']))

for i, (tag, r) in enumerate(items, 1):
    meta = r.get('combo_meta', {})
    policy = meta.get('policy', r.get('policy', '?'))
    prompt = meta.get('prompt_style', '-')
    img_mode = meta.get('image_mode', '-')
    img_size = meta.get('image_size', '-')
    res_str = f'{img_size}x{img_size}' if img_size != '-' else '-'

    print(f'| {i} | {tag:45s} | {policy:6s} | {prompt:7s} | {img_mode:18s} | {res_str:10s} | {r[\"reward_mean\"]:11.2f} | {r[\"reward_std\"]:5.2f} | {r[\"reward_min\"]:5.0f} | {r[\"reward_max\"]:5.0f} | {r[\"steps_mean\"]:9.0f} |')

print()

# Quick analysis
random_reward = None
best_vlm_tag = None
best_vlm_reward = -1e9

for tag, r in results.items():
    meta = r.get('combo_meta', {})
    if meta.get('policy') == 'random':
        random_reward = r['reward_mean']
    elif r['reward_mean'] > best_vlm_reward:
        best_vlm_reward = r['reward_mean']
        best_vlm_tag = tag

print('=== Quick Analysis ===')
if random_reward is not None:
    print(f'Random baseline: {random_reward:.2f}')
if best_vlm_tag:
    print(f'Best VLM config: {best_vlm_tag} -> {best_vlm_reward:.2f}')
    if random_reward is not None:
        diff = best_vlm_reward - random_reward
        print(f'Improvement over random: {diff:+.2f} ({diff/abs(random_reward)*100:+.1f}%)')

# Dimension analysis
print()
print('=== Dimension-wise Analysis ===')

def avg_reward(filter_fn):
    vals = [r['reward_mean'] for t, r in results.items() if filter_fn(t, r)]
    return sum(vals)/len(vals) if vals else float('nan')

# Concise vs Legacy
concise_avg = avg_reward(lambda t, r: r.get('combo_meta', {}).get('prompt_style') == 'concise')
legacy_avg = avg_reward(lambda t, r: r.get('combo_meta', {}).get('prompt_style') == 'legacy')
print(f'Concise prompt avg reward: {concise_avg:.2f}')
print(f'Legacy  prompt avg reward: {legacy_avg:.2f}')
print(f'  -> Concise vs Legacy delta: {concise_avg - legacy_avg:+.2f}')

# Current-only vs First+Current
current_avg = avg_reward(lambda t, r: r.get('combo_meta', {}).get('image_mode') == 'current_only')
first_cur_avg = avg_reward(lambda t, r: r.get('combo_meta', {}).get('image_mode') == 'first_and_current')
print(f'Current-only avg reward:      {current_avg:.2f}')
print(f'First+Current avg reward:     {first_cur_avg:.2f}')
print(f'  -> First+Current delta:     {first_cur_avg - current_avg:+.2f}')

# 64 vs 256
res64_avg = avg_reward(lambda t, r: r.get('combo_meta', {}).get('image_size') == 64 and r.get('combo_meta', {}).get('policy') == 'vlm')
res256_avg = avg_reward(lambda t, r: r.get('combo_meta', {}).get('image_size') == 256 and r.get('combo_meta', {}).get('policy') == 'vlm')
print(f'64x64   avg reward:   {res64_avg:.2f}')
print(f'256x256 avg reward:   {res256_avg:.2f}')
print(f'  -> Upscale delta:   {res256_avg - res64_avg:+.2f}')
"

echo ""
echo "=============================================="
echo " Ablation study complete!"
echo " Full results: ${MERGED_JSON}"
echo "=============================================="
