# UniZero Loss Landscape Visualization

This tool generates 2D loss landscape visualizations for trained UniZero models.

## Installation

The `loss_landscape_core` library has been copied into the LightZero directory. No additional installation is needed.

## Usage

### Basic Usage

```bash
cd /mnt/shared-storage-user/tangjia/unizero/LightZero

python lzero/tools/visualize_loss_landscape.py \
    --checkpoint data_lz/data_unizero/Pong/xxx/ckpt/ckpt_best.pth.tar \
    --env PongNoFrameskip-v4 \
    --seed 0
```

### Advanced Usage

```bash
# Custom grid size and output directory
python lzero/tools/visualize_loss_landscape.py \
    --checkpoint data_lz/data_unizero/Pong/xxx/ckpt/ckpt_best.pth.tar \
    --env PongNoFrameskip-v4 \
    --output my_landscape_output \
    --grid-size 21 \
    --num-batches 50

# Use CPU only
python lzero/tools/visualize_loss_landscape.py \
    --checkpoint data_lz/data_unizero/Pong/xxx/ckpt/ckpt_best.pth.tar \
    --env PongNoFrameskip-v4 \
    --no-cuda
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint` | str | Required | Path to checkpoint file |
| `--env` | str | PongNoFrameskip-v4 | Environment ID |
| `--seed` | int | 0 | Random seed |
| `--output` | str | ./landscape_output | Output directory |
| `--grid-size` | int | 11 | Grid size (11 or 21) |
| `--num-batches` | int | 20 | Number of batches for landscape |
| `--batch-size` | int | 64 | Batch size for sampling |
| `--num-episodes` | int | 10 | Episodes to collect for replay buffer |
| `--no-cuda` | flag | False | Disable CUDA |

## Output

The script generates the following files in the output directory:

- `loss_landscape.h5` - Data file containing loss values at each grid point
- `loss_landscape.h5_train_loss_policy_loss_*.pdf` - Policy loss visualizations (4 files)
- `loss_landscape.h5_train_loss_value_loss_*.pdf` - Value loss visualizations (4 files)
- `loss_landscape.h5_train_loss_reward_loss_*.pdf` - Reward loss visualizations (4 files)
- `loss_landscape.h5_train_loss_consistency_loss_*.pdf` - Consistency loss visualizations (4 files)
- `loss_landscape.h5_train_loss_total_loss_*.pdf` - Total loss visualizations (4 files)

Each metric generates 4 plots:
- `2dcontour.pdf` - Contour lines
- `2dcontourf.pdf` - Filled contour
- `2dheat.pdf` - Heatmap
- `3dsurface.pdf` - 3D surface

## Grid Size Recommendations

- **11×11 grid** (default): ~2-3 minutes, good for quick exploration
- **21×21 grid**: ~10-15 minutes, more detailed visualization

## Example Checkpoint Path

```
data_lz/data_unizero/Pong/Pong_uz_brf0.02-rbs160-rp0.75_nlayer2_numsegments-8_gsl20_rr0.25_Htrain10-Hinfer4_bs64_seed0/ckpt/ckpt_best.pth.tar
```

## Important Notes

1. **Replay Buffer**: The script collects some episodes to fill the replay buffer, which is needed for computing the landscape. This may take 1-2 minutes.

2. **Data Usage**: The landscape computation uses sampled mini-batches from the collected data to estimate loss values at different weight perturbations.

3. **Metrics Computed**:
   - Policy Loss: Cross-entropy loss for policy prediction
   - Value Loss: Cross-entropy loss for value prediction
   - Reward Loss: Cross-entropy loss for reward prediction
   - Consistency Loss: Self-supervised learning loss (if available)
   - Total Loss: Sum of weighted losses

4. **GPU Memory**: Requires ~2-4 GB GPU memory for standard configurations.

## Troubleshooting

### Out of Memory
- Reduce `--batch-size`
- Reduce `--num-batches`
- Use `--no-cuda` for CPU-only mode

### Slow Computation
- Reduce `--grid-size` to 11
- Reduce `--num-batches`
- Use a smaller `--batch-size`

### Missing Data
- Increase `--num-episodes` to collect more data
- Make sure checkpoint is valid and contains model weights

## References

- Loss Landscape Visualization: [Loss Surfaces](https://github.com/zingyi-li/Loss-Surfaces)
- UniZero: [Paper](https://arxiv.org/abs/2406.10667)
