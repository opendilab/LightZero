# Loss Landscape Core

A clean, modular PyTorch library for visualizing neural network loss landscapes.

This library is abstracted and refactored from the [loss-landscape](https://github.com/tomgoldstein/loss-landscape) repository, providing a general-purpose toolkit for neural network loss landscape visualization that works with any PyTorch model. Below we demonstrate both general examples (CIFAR-10) and specific applications to UniZero models.

## Installation

```bash
pip install torch torchvision h5py matplotlib scipy seaborn numpy
```

## Usage

### Using Loss Landscape Module to Plot Loss Surface

```python
from lzero.loss_landscape import LossLandscape

# Create LossLandscape object
# net: Your PyTorch model
# dataloader: Data loader
# criterion: Loss function (e.g., nn.CrossEntropyLoss())
# use_cuda: Whether to use GPU
landscape = LossLandscape(net, dataloader, criterion, use_cuda=True)

# Compute 2D landscape
result = landscape.compute_2d(
    xrange=(-1, 1, 51),
    yrange=(-1, 1, 51)
)

# Visualize (multiple formats)
landscape.plot_2d_contour()      # Contour lines
landscape.plot_2d_surface()       # 3D surface
landscape.export_paraview()       # High-quality rendering
```

**Simple Example Visualization Results:**

| 2D Contour | 2D Contour Filled | 2D Heatmap | 3D Surface |
| --- | --- | --- | --- |
| <img src="./images/simple/simple_example_2dcontour.png" width="200" /> | <img src="./images/simple/simple_example_2dcontourf.png" width="200" /> | <img src="./images/simple/simple_example_2dheat.png" width="200" /> | <img src="./images/simple/simple_example_3dsurface.png" width="200" /> |

### More Simple Examples (CIFAR-10)

See the `examples/` directory for:
- **example_1d.py** - 1D loss curve
- **example_2d.py** - 2D loss surface with ParaView

```bash
cd lzero/loss_landscape
python examples/example_1d.py
python examples/example_2d.py
```

### Using Loss Landscape to Plot UniZero's Loss Landscape

For UniZero models, use the batch script to evaluate loss landscapes across multiple training checkpoints:

```bash
bash lzero/loss_landscape/run_loss_landscape_batch.sh
```

The script automatically:
1. Iterates through all checkpoint files (iteration_10000.pth.tar to iteration_100000.pth.tar)
2. Loads model weights for each checkpoint
3. Collects evaluation data from the game environment
4. Computes loss landscape on a 21×21 grid (441 evaluation points)
5. Generates multiple visualization images (contour, filled contour, heatmap, 3D surface)

**Configuration required before use:**
- `CKPT_BASE_DIR`: Directory containing checkpoint files
- `CONFIG_SCRIPT`: Path to loss landscape configuration script
- `BASE_LOG_DIR`: Output directory for results
- `ENV_ID`: Atari game environment ID (e.g., "PongNoFrameskip-v4")

### UniZero's Loss Landscape Visualization Results

This library has been applied to visualize loss landscapes of UniZero models in Atari games. Results are generated through the following process:

**Batch Evaluation Process:**
Running the `run_loss_landscape_batch.sh` script performs batch loss landscape evaluation on multiple training iterations of checkpoints (10K-100K). The script automatically:
1. Loads each checkpoint from the specified iteration
2. Collects a batch of data from the game environment
3. Computes the loss landscape on a 21×21 grid (441 evaluation points)
4. Generates multiple visualization images (contour, filled contour, heatmap, 3D surface)

**Results Display:**

Loss landscape visualizations for Total Loss across different training iterations for Pong and MsPacman environments. Each row shows a different iteration checkpoint, and columns show different visualization styles.

#### Pong Environment - Total Loss

| Iteration | 2D Contour | 2D Contour Filled | 2D Heatmap | 3D Surface |
| --- | --- | --- | --- | --- |
| iter10K | <img src="./images/Pong/iter10K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dcontour.png" width="200" /> | <img src="./images/Pong/iter10K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dcontourf.png" width="200" /> | <img src="./images/Pong/iter10K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dheat.png" width="200" /> | <img src="./images/Pong/iter10K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_3dsurface.png" width="200" /> |
| iter50K | <img src="./images/Pong/iter50K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dcontour.png" width="200" /> | <img src="./images/Pong/iter50K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dcontourf.png" width="200" /> | <img src="./images/Pong/iter50K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dheat.png" width="200" /> | <img src="./images/Pong/iter50K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_3dsurface.png" width="200" /> |
| iter100K | <img src="./images/Pong/iter100K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dcontour.png" width="200" /> | <img src="./images/Pong/iter100K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dcontourf.png" width="200" /> | <img src="./images/Pong/iter100K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dheat.png" width="200" /> | <img src="./images/Pong/iter100K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_3dsurface.png" width="200" /> |

#### MsPacman Environment - Total Loss

| Iteration | 2D Contour | 2D Contour Filled | 2D Heatmap | 3D Surface |
| --- | --- | --- | --- | --- |
| iter10K | <img src="./images/MsPacman/iter10K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dcontour.png" width="200" /> | <img src="./images/MsPacman/iter10K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dcontourf.png" width="200" /> | <img src="./images/MsPacman/iter10K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dheat.png" width="200" /> | <img src="./images/MsPacman/iter10K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_3dsurface.png" width="200" /> |
| iter50K | <img src="./images/MsPacman/iter50K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dcontour.png" width="200" /> | <img src="./images/MsPacman/iter50K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dcontourf.png" width="200" /> | <img src="./images/MsPacman/iter50K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dheat.png" width="200" /> | <img src="./images/MsPacman/iter50K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_3dsurface.png" width="200" /> |
| iter100K | <img src="./images/MsPacman/iter100K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dcontour.png" width="200" /> | <img src="./images/MsPacman/iter100K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dcontourf.png" width="200" /> | <img src="./images/MsPacman/iter100K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dheat.png" width="200" /> | <img src="./images/MsPacman/iter100K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_3dsurface.png" width="200" /> |

## Visualization Files

After loss landscape computation completes, the system generates the following types of output files:

- **HDF5 Data Files** (`.h5`) - Raw data storage containing coordinates and loss values
- **PDF Visualization Files** - Chart files for reports and presentations, including the following 4 views:
  - `*_2dcontour.pdf` - Contour line plot showing loss contours
  - `*_2dcontourf.pdf` - Filled contour plot with color-filled loss regions
  - `*_2dheat.pdf` - Heatmap using color intensity to represent loss values
  - `*_3dsurface.pdf` - 3D surface plot showing loss landscape in 3D perspective
- **ParaView Files** (`.vtp`) - Professional 3D rendering format

## loss landscape Module Structure

```
loss_landscape/
├── core/                       # Core functionality
│   ├── direction.py            # Direction generation and normalization
│   ├── evaluator.py            # Loss and accuracy evaluation
│   └── perturbation.py         # Weight perturbation
├── utils/                      # Utilities
│   ├── storage.py              # HDF5 file I/O
│   ├── plot_1d.py              # 1D plotting
│   ├── plot_2d.py              # 2D plotting (with multi-metric support)
│   ├── projection.py           # Direction projection
│   └── paraview.py             # ParaView export
├── loss_landscape_api.py       # High-level API
├── __init__.py                 # Package initialization
├── examples/                   # Example scripts
│   ├── example_1d.py
│   └── example_2d.py
├── images/                     # Visualization results
│   ├── simple/                 # Simple examples
│   ├── Pong/                   # Pong environment results
│   └── MsPacman/               # MsPacman environment results
└── run_loss_landscape_batch.sh # Batch processing script
```

## Performance Testing

### Computation Time

- **1D landscape**: O(num_points) - Linear in sampling points
- **2D landscape**: O(num_points_x × num_points_y) - Quadratic
- Each point requires one forward pass through all data

### Estimated Runtimes (Modern GPUs)

**For a single checkpoint with 21×21 resolution:** The evaluation process involves: (1) loading the checkpoint, (2) collecting a batch of data from the environment, (3) computing loss values at 441 points (21×21 grid) in the parameter space by perturbing model weights, and (4) generating multiple visualization plots (contour, heatmap, 3D surface). This takes approximately 30 minutes on H200 GPU.

### How to Speed Up Computation

1. **Reduce grid resolution**: Use 11×11 or 15×15 instead of 21×21 for testing
2. **Use fewer batches**: Reduce num_batches parameter (e.g., 20-50 instead of 100)
3. **Use GPU acceleration**: Enable `use_cuda=True` for ~10-100x speedup
4. **Reduce batch size**: Smaller batches fit in GPU memory
5. **Parallel evaluation**: Use multiple GPUs with data parallelism

## Tips for Better Results

1. **Use normalized data**: Ensure your dataloader uses normalized data
2. **Sufficient sampling**: Use at least 51 points per dimension
3. **Appropriate loss range**: Adjust `vmin`/`vmax` in contour plots
4. **Log scale**: Use `log=True` in ParaView export for large dynamic range
5. **Resolution**: Export to ParaView for publication-quality figures

## References

- Original paper: [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913)
- Original code: https://github.com/tomgoldstein/loss-landscape
- PyTorch documentation: https://pytorch.org/docs/
- ParaView: https://www.paraview.org/

## Citation

If you use this library in your research, please cite the original work and our work: [UniZero](https://openreview.net/pdf?id=Gl6dF9soQo) and [LightZero](https://proceedings.neurips.cc/paper_files/paper/2023/file/765043fe026f7d704c96cec027f13843-Paper-Datasets_and_Benchmarks.pdf)

```bibtex
@inproceedings{li2018visualizing,
  title={Visualizing the Loss Landscape of Neural Nets},
  author={Li, Hao and Xu, Zheng and Taylor, Gavin and Studer, Christoph and Goldstein, Tom},
  booktitle={Neural Information Processing Systems},
  year={2018}
}

@article{niu2024lightzero,
  title={LightZero: A Unified Benchmark for Monte Carlo Tree Search in General Sequential Decision Scenarios},
  author={Niu, Yazhe and Pu, Yuan and Yang, Zhenjie and Li, Xueyan and Zhou, Tong and Ren, Jiyuan and Hu, Shuai and Li, Hongsheng and Liu, Yu},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}

@article{puunizero,
  title={UniZero: Generalized and Efficient Planning with Scalable Latent World Models},
  author={Pu, Yuan and Niu, Yazhe and Yang, Zhenjie and Ren, Jiyuan and Li, Hongsheng and Liu, Yu},
  journal={Transactions on Machine Learning Research}
}
```

## License

MIT License - See original [loss-landscape](https://github.com/tomgoldstein/loss-landscape) repository
