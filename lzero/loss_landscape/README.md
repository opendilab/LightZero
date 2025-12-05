# Loss Landscape Core

**This tool is specifically designed for visualizing loss landscapes of UniZero model checkpoints.**

A clean, modular PyTorch library for visualizing neural network loss landscapes.

## Installation

```bash
pip install torch torchvision h5py matplotlib scipy seaborn numpy
```

## Usage

### 2D Loss Surface

```python
from lzero.loss_landscape import LossLandscape

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

### Running Examples

See the `examples/` directory for:
- **example_1d.py** - 1D loss curve
- **example_2d.py** - 2D loss surface with ParaView
- **example_custom_metrics.py** - Custom metrics functions

```bash
python examples/example_1d.py
python examples/example_2d.py
```

## Module Structure

```
loss_landscape/
├── core/                  # Core functionality
│   ├── direction.py       # Direction generation and normalization
│   ├── evaluator.py       # Loss and accuracy evaluation
│   └── perturbation.py    # Weight perturbation
├── utils/                 # Utilities
│   ├── storage.py         # HDF5 file I/O
│   ├── plot_1d.py         # 1D plotting
│   ├── plot_2d.py         # 2D plotting (with multi-metric support)
│   └── paraview.py        # ParaView export
├── api.py                 # High-level API
├── __init__.py            # Package initialization
└── examples/              # Example scripts
    ├── example_1d.py
    ├── example_2d.py
    └── example_custom_metrics.py
```

## API Reference

### LossLandscape Class

Main class for loss landscape computation and visualization.

**Constructor:**
```python
LossLandscape(net, dataloader, criterion=None, use_cuda=False, surf_file=None)
```

**Parameters:**
- `net`: PyTorch model
- `dataloader`: DataLoader for evaluation
- `criterion`: Loss function or metrics function
  - Can be `nn.Module` (CrossEntropyLoss, MSELoss, etc.) ← Standard mode
  - Can be `callable` returning `dict` of metrics ← Custom mode
  - Default: `nn.CrossEntropyLoss()`
- `use_cuda`: Whether to use GPU (default: False)
- `surf_file`: Path for saving HDF5 results (default: 'loss_surface.h5')

### Methods

#### `compute_1d()`

```python
result = landscape.compute_1d(
    directions='random',      # or 'target' with target_model
    xrange=(-1, 1, 51),       # (min, max, num_points)
    dir_type='weights',       # or 'states'
    normalize='filter',       # filter|layer|weight|dfilter|dlayer
    ignore='biasbn',          # Ignore bias and BN parameters
    target_model=None,        # Required if directions='target'
    save=True                 # Save to HDF5 file
)
```

**Returns:**
- Standard mode: `{'losses': np.array([...]), 'accuracies': np.array([...]), 'xcoordinates': np.array([...])}`
- Custom mode: `{'losses': {metric_name: np.array([...]), ...}, 'xcoordinates': np.array([...])}`

#### `compute_2d()`

```python
result = landscape.compute_2d(
    xrange=(-1, 1, 51),     # (min, max, num_points)
    yrange=(-1, 1, 51),     # (min, max, num_points)
    dir_type='weights',     # or 'states'
    normalize='filter',     # filter|layer|weight|dfilter|dlayer
    ignore='biasbn',        # Ignore bias and BN parameters
    x_target=None,          # Optional target model for x direction
    y_target=None,          # Optional target model for y direction
    save=True               # Save to HDF5 file
)
```

#### `plot_1d()`

```python
landscape.plot_1d(xmin=-1, xmax=1, loss_max=5, log=False, show=False)
```

#### `plot_2d_contour()`

```python
landscape.plot_2d_contour(
    surf_name='train_loss',  # Name of metric to plot, or 'auto' for all
    vmin=0.1,                # Minimum value for contour levels
    vmax=10,                 # Maximum value for contour levels
    vlevel=0.5,              # Spacing between contour levels
    show=False               # Whether to display plot
)
```

**Output files:**
- `*_2dcontour.pdf`: Contour lines
- `*_2dcontourf.pdf`: Filled contour
- `*_2dheat.pdf`: Heatmap

#### `plot_2d_surface()`

```python
landscape.plot_2d_surface(
    surf_name='train_loss',  # Name of metric to plot, or 'auto' for all
    show=False               # Whether to display plot
)
```

**Output files:**
- `*_3dsurface.pdf`: 3D surface plot

#### `export_paraview()`

```python
vtp_file = landscape.export_paraview(
    surf_name='train_loss',  # Which surface to export
    log=False,               # Use log scale
    zmax=-1,                 # Clip max z value (-1: no clipping)
    interp=-1                # Interpolate resolution (-1: no interpolation)
)
```

## Performance Considerations

### Computation Time

- **1D landscape**: O(num_points) - Linear in sampling points
- **2D landscape**: O(num_points_x × num_points_y) - Quadratic
- Each point requires one forward pass through all data

### GPU Requirements

- Supports any NVIDIA GPU with CUDA support
- Falls back to CPU automatically if CUDA unavailable

### Estimated Runtimes (Modern GPUs)

**For 21×21 loss landscape evaluation:**
- **H200 GPU**: ~30 minutes (441 evaluations)
- **A100 GPU**: ~45-60 minutes
- **H100 GPU**: ~20-25 minutes

**Factors affecting runtime:**
- Model size: Larger models require more computation
- Grid resolution: Higher resolution increases evaluation count quadratically
- Number of batches: More batches improve loss estimates but increase computation
- Data size: Larger datasets mean longer loss evaluation per point

### Tips for Faster Computation

1. **Reduce grid resolution**: Use 11×11 or 15×15 instead of 21×21 for testing
2. **Use fewer batches**: Reduce num_batches parameter (e.g., 20-50 instead of 100)
3. **Use GPU acceleration**: Enable `use_cuda=True` for ~10-100x speedup
4. **Reduce batch size**: Smaller batches fit in GPU memory
5. **Parallel evaluation**: Use multiple GPUs with data parallelism

## Output Files

### HDF5 Data Files (`.h5`)

**Standard mode:**
```
Keys:
  - xcoordinates, ycoordinates: Coordinates
  - train_loss: Training loss values
  - train_acc: Training accuracy
```

**Custom mode:**
```
Keys:
  - xcoordinates, ycoordinates: Coordinates
  - train_loss_metric1: Metric 1 values
  - train_loss_metric2: Metric 2 values
  - (more for each custom metric)
```

### PDF Visualization Files

For each metric, generates 4 files:
```
*_2dcontour.pdf    # Contour lines
*_2dcontourf.pdf   # Filled contour
*_2dheat.pdf       # Heatmap
*_3dsurface.pdf    # 3D surface
```

### ParaView Files (`.vtp`)

VTK format compatible with ParaView for professional rendering.

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

If you use this library in your research, please cite the original work:

```bibtex
@inproceedings{li2018visualizing,
  title={Visualizing the Loss Landscape of Neural Nets},
  author={Li, Hao and Xu, Zheng and Taylor, Gavin and Studer, Christoph and Goldstein, Tom},
  booktitle={Neural Information Processing Systems},
  year={2018}
}
```

## Visualization Results

Loss landscape visualizations for Total Loss across different training iterations for Pong and MsPacman environments.
Each row shows a different iteration checkpoint, and columns show different visualization styles.

### Pong Environment - Total Loss

| Iteration | 2D Contour | 2D Contour Filled | 2D Heatmap | 3D Surface |
| --- | --- | --- | --- | --- |
| iter10K | <img src="./images/Pong/iter10K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dcontour.png" width="200" /> | <img src="./images/Pong/iter10K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dcontourf.png" width="200" /> | <img src="./images/Pong/iter10K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dheat.png" width="200" /> | <img src="./images/Pong/iter10K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_3dsurface.png" width="200" /> |
| iter50K | <img src="./images/Pong/iter50K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dcontour.png" width="200" /> | <img src="./images/Pong/iter50K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dcontourf.png" width="200" /> | <img src="./images/Pong/iter50K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dheat.png" width="200" /> | <img src="./images/Pong/iter50K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_3dsurface.png" width="200" /> |
| iter100K | <img src="./images/Pong/iter100K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dcontour.png" width="200" /> | <img src="./images/Pong/iter100K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dcontourf.png" width="200" /> | <img src="./images/Pong/iter100K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_2dheat.png" width="200" /> | <img src="./images/Pong/iter100K/loss_landscape_PongNoFrameskip_21x21.h5_train_loss_total_loss_3dsurface.png" width="200" /> |

### MsPacman Environment - Total Loss

| Iteration | 2D Contour | 2D Contour Filled | 2D Heatmap | 3D Surface |
| --- | --- | --- | --- | --- |
| iter10K | <img src="./images/MsPacman/iter10K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dcontour.png" width="200" /> | <img src="./images/MsPacman/iter10K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dcontourf.png" width="200" /> | <img src="./images/MsPacman/iter10K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dheat.png" width="200" /> | <img src="./images/MsPacman/iter10K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_3dsurface.png" width="200" /> |
| iter50K | <img src="./images/MsPacman/iter50K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dcontour.png" width="200" /> | <img src="./images/MsPacman/iter50K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dcontourf.png" width="200" /> | <img src="./images/MsPacman/iter50K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dheat.png" width="200" /> | <img src="./images/MsPacman/iter50K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_3dsurface.png" width="200" /> |
| iter100K | <img src="./images/MsPacman/iter100K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dcontour.png" width="200" /> | <img src="./images/MsPacman/iter100K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dcontourf.png" width="200" /> | <img src="./images/MsPacman/iter100K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_2dheat.png" width="200" /> | <img src="./images/MsPacman/iter100K/loss_landscape_MsPacmanNoFrameskip_21x21.h5_train_loss_total_loss_3dsurface.png" width="200" /> |

## License

MIT License - See original [loss-landscape](https://github.com/tomgoldstein/loss-landscape) repository
