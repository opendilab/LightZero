# Loss Landscape Core

A clean, modular PyTorch library for visualizing neural network loss landscapes.

Extracted and refactored from the original [loss-landscape](https://github.com/zingyi-li/Loss-Surfaces) project with:
- Simplified API for single GPU usage
- Removed MPI dependencies
- Modular architecture for easy integration
- **⭐ NEW: Custom Metrics Function Support** - Compute multiple metrics in one pass
- **⭐ NEW: Auto-Detection and Multi-Metric Visualization**
- Full support for 1D curves, 2D contours, 3D surfaces, and ParaView export
- Full backward compatibility with standard PyTorch losses

## Features

✨ **Key Features**:

- ✅ **Simplified API** - Easy to use for single-GPU machines
- ✅ **1D/2D Landscapes** - Compute loss curves and 2D surfaces
- ✅ **Multiple Visualization Types**
  - Contour plots (line-based and filled)
  - Heatmaps with color bars
  - 3D surface plots
  - ParaView-compatible VTP export
- ✅ **⭐ Custom Metrics Function** - NEW!
  - Define your own metrics function
  - Compute multiple loss values and metrics simultaneously
  - Return metrics as dictionary
  - Automatic storage and visualization
- ✅ **⭐ Auto-Metric Detection** - NEW!
  - Automatically detect all computed metrics
  - Generate separate visualizations for each metric
  - Use `surf_name='auto'` to plot everything
- ✅ **Flexible Criterion Types**
  - Standard PyTorch loss modules (CrossEntropyLoss, MSELoss, etc.)
  - Custom callable functions returning metric dictionaries
- ✅ **Type-Safe Auto-Detection**
  - Automatically distinguishes between PyTorch losses and custom functions
  - No need for additional flags or configuration
- ✅ **Full Backward Compatibility**
  - All existing code continues to work unchanged
  - Existing HDF5 files are still readable
  - Plotting functions work with both modes

## Installation

```bash
pip install torch torchvision h5py matplotlib scipy seaborn numpy
```

## Quick Start

### 1D Loss Curve (Standard Loss)

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from loss_landscape_core import LossLandscape

# Setup model and data
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)

# Create landscape with standard PyTorch loss
landscape = LossLandscape(model, dataloader,
                         criterion=nn.CrossEntropyLoss(),
                         use_cuda=True)

# Compute 1D landscape
result = landscape.compute_1d(
    directions='random',
    xrange=(-1, 1, 51),
    normalize='filter',
    ignore='biasbn'
)

# Visualize
landscape.plot_1d(loss_max=5, show=True)
```

### 2D Loss Surface

```python
# Compute 2D landscape
result = landscape.compute_2d(
    xrange=(-1, 1, 51),
    yrange=(-1, 1, 51),
    normalize='filter'
)

# Plot contours and 3D surface
landscape.plot_2d_contour(vmin=0.1, vmax=10, vlevel=0.5)
landscape.plot_2d_surface(show=True)
```

### ⭐ Custom Metrics (NEW!)

Compute multiple custom metrics simultaneously:

```python
import torch
import torch.nn as nn
from loss_landscape_core import LossLandscape

def compute_custom_metrics(net, dataloader, use_cuda):
    """
    Custom metrics function that computes multiple metrics.

    Args:
        net: PyTorch model
        dataloader: Data loader for evaluation
        use_cuda: Whether to use GPU

    Returns:
        Dictionary with metric values
    """
    net.eval()
    device = 'cuda' if use_cuda else 'cpu'

    total_ce = 0.0
    total_smooth_l1 = 0.0
    total_correct = 0
    total_samples = 0

    criterion_ce = nn.CrossEntropyLoss()
    criterion_smooth_l1 = nn.SmoothL1Loss()

    with torch.no_grad():
        for inputs, targets in dataloader:
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)

            # Compute different loss functions
            ce_loss = criterion_ce(outputs, targets)
            targets_onehot = torch.nn.functional.one_hot(
                targets, num_classes=outputs.size(1)).float()
            smooth_l1_loss = criterion_smooth_l1(outputs, targets_onehot)

            # Accumulate metrics
            total_ce += ce_loss.item() * inputs.size(0)
            total_smooth_l1 += smooth_l1_loss.item() * inputs.size(0)

            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += inputs.size(0)

    # Return multiple metrics as dictionary
    return {
        'ce_loss': total_ce / total_samples,
        'smooth_l1_loss': total_smooth_l1 / total_samples,
        'accuracy': 100.0 * total_correct / total_samples
    }

# Use with LossLandscape
landscape = LossLandscape(model, dataloader,
                         criterion=compute_custom_metrics,  # Pass function
                         use_cuda=True)

# Compute 2D landscape with custom metrics
result = landscape.compute_2d(
    xrange=(-1, 1, 11),
    yrange=(-1, 1, 11)
)

# result['losses'] is now a dict:
# {
#     'ce_loss': shape (11, 11),
#     'smooth_l1_loss': shape (11, 11),
#     'accuracy': shape (11, 11)
# }

# Auto-plot all metrics
landscape.plot_2d_contour(surf_name='auto', vmin=0.1, vmax=10)
landscape.plot_2d_surface(surf_name='auto')

# Or plot specific metric
landscape.plot_2d_contour(surf_name='train_loss_ce_loss', vmin=0.1, vmax=5)
```

### Export to ParaView (High-Quality Rendering)

```python
# Export 2D surface for ParaView rendering
vtp_file = landscape.export_paraview(surf_name='train_loss_ce_loss',
                                     log=False, zmax=10, interp=-1)
```

Then open the `.vtp` file with [ParaView](https://www.paraview.org/) for professional visualization.

## Custom Metrics Guide

### What are Custom Metrics?

Custom metrics functions allow you to:
1. **Compute multiple loss values** - Compare different loss functions simultaneously
2. **Track various metrics** - Accuracy, F1-score, precision, recall, etc.
3. **Visualize jointly** - See how different metrics vary across weight space
4. **Analyze trade-offs** - Understand relationships between different objectives

### How It Works

**Type Detection Mechanism:**

The library automatically detects criterion type:

```python
# Standard PyTorch loss (nn.Module) → Standard mode
criterion = nn.CrossEntropyLoss()

# Custom metrics function (callable, not nn.Module) → Custom mode
def my_metrics(net, dataloader, use_cuda):
    return {'metric1': value1, 'metric2': value2}
```

**Storage Format:**

- **Standard mode**: Saves as `'train_loss'`, `'train_acc'` (backward compatible)
- **Custom mode**: Saves as `'train_loss_metric_name'`, `'train_loss_metric2'`, ...

**Automatic Plotting:**

```python
# Plot all detected metrics
landscape.plot_2d_contour(surf_name='auto')  # Detects all train_loss_* keys

# Plot specific metric
landscape.plot_2d_contour(surf_name='train_loss_f1_score')

# Default: plot standard loss (backward compatible)
landscape.plot_2d_contour()  # Plots 'train_loss'
```

### Function Signature

Custom metrics functions must follow this signature:

```python
def my_metrics(net: nn.Module,
               dataloader: DataLoader,
               use_cuda: bool) -> dict:
    """
    Compute metrics for the model on given dataloader.

    Args:
        net: PyTorch model (in eval mode)
        dataloader: DataLoader for evaluation
        use_cuda: Whether model is on GPU

    Returns:
        Dictionary mapping metric names to values:
        {'metric_name': float_value, ...}
    """
    pass
```

### Complete Example

See `example_custom_metrics.py` for a complete working example with:
- Standard loss mode (backward compatible)
- 1D landscape with custom metrics
- 2D landscape with custom metrics
- Auto-plotting of all metrics

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

**Methods:**

#### `compute_1d()`
Compute 1D loss landscape.

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

**Standard mode** (with nn.Module criterion):
```python
{
    'losses': np.array([...]),      # 1D array of loss values
    'accuracies': np.array([...]),  # 1D array of accuracy values
    'xcoordinates': np.array([...]) # X coordinates
}
```

**Custom mode** (with callable criterion):
```python
{
    'losses': {
        'metric1': np.array([...]),
        'metric2': np.array([...]),
        ...
    },
    'xcoordinates': np.array([...])
}
```

#### `compute_2d()`
Compute 2D loss landscape.

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

**Returns:**

**Standard mode:**
```python
{
    'losses': np.array([...]).shape(nx, ny),      # 2D array
    'accuracies': np.array([...]).shape(nx, ny),  # 2D array
    'xcoordinates': np.array([...]),
    'ycoordinates': np.array([...])
}
```

**Custom mode:**
```python
{
    'losses': {
        'metric1': np.array([...]).shape(nx, ny),
        'metric2': np.array([...]).shape(nx, ny),
        ...
    },
    'xcoordinates': np.array([...]),
    'ycoordinates': np.array([...])
}
```

#### `plot_1d()`
Visualize 1D landscape.

```python
landscape.plot_1d(xmin=-1, xmax=1, loss_max=5, log=False, show=False)
```

#### `plot_2d_contour()`
Plot 2D contour visualization.

```python
landscape.plot_2d_contour(
    surf_name='train_loss',  # Name of metric to plot
                             # Use 'auto' to plot all detected metrics
    vmin=0.1,                # Minimum value for contour levels
    vmax=10,                 # Maximum value for contour levels
    vlevel=0.5,              # Spacing between contour levels
    show=False               # Whether to display plot
)
```

**Parameters:**
- `surf_name`:
  - `'train_loss'` (default): Plot standard loss (backward compatible)
  - `'auto'`: Auto-detect and plot all metrics (new!)
  - `'train_loss_metric_name'`: Plot specific custom metric

**Output files:**
- `*_2dcontour.pdf`: Contour lines
- `*_2dcontourf.pdf`: Filled contour
- `*_2dheat.pdf`: Heatmap

#### `plot_2d_surface()`
Plot 3D surface visualization.

```python
landscape.plot_2d_surface(
    surf_name='train_loss',  # Name of metric to plot
                             # Use 'auto' to plot all detected metrics
    show=False               # Whether to display plot
)
```

**Output files:**
- `*_3dsurface.pdf`: 3D surface plot

#### `export_paraview()`
Export to ParaView VTP format.

```python
vtp_file = landscape.export_paraview(
    surf_name='train_loss',  # Which surface to export
    log=False,               # Use log scale
    zmax=-1,                 # Clip max z value (-1: no clipping)
    interp=-1                # Interpolate resolution (-1: no interpolation)
)
```

## Key Concepts

### Standard Mode vs Custom Mode

| Aspect | Standard Mode | Custom Mode |
|--------|---------------|------------|
| Criterion | `nn.Module` (loss function) | `callable` function |
| Returns | Single loss + accuracy | Dict of metrics |
| Example | `nn.CrossEntropyLoss()` | `def my_metrics(...)` |
| Output Keys | `'train_loss'`, `'train_acc'` | `'train_loss_metric1'`, `'train_loss_metric2'` |
| Plotting | Fixed to `'train_loss'` | Auto-detects all metrics |
| Backward Compatible | ✅ Yes (original behavior) | ✅ Yes (new addition) |

### Direction Types

- **`weights`**: Direction in weight space (includes all parameters)
- **`states`**: Direction in state_dict space (includes BN running statistics)

Use `weights` for general-purpose analysis, `states` when analyzing BatchNorm layers.

### Normalization Methods

- **`filter`**: Normalize at filter level (recommended)
  - Each filter has same norm as in original weights
  - Works well for convolutional networks
- **`layer`**: Normalize at layer level
- **`weight`**: Scale by weight magnitude
- **`dfilter`**: Unit norm per filter
- **`dlayer`**: Unit norm per layer

### Ignore Options

- **`biasbn`**: Ignore bias and batch normalization parameters
  - Set their direction components to zero
  - Recommended for most analyses

## Module Structure

```
loss_landscape_core/
├── core/                  # Core functionality
│   ├── direction.py       # Direction generation and normalization
│   ├── evaluator.py       # Loss and accuracy evaluation
│   └── perturbation.py    # Weight perturbation
├── utils/                 # Utilities
│   ├── storage.py         # HDF5 file I/O
│   └── projection.py      # Projection and angle calculations
├── viz/                   # Visualization
│   ├── plot_1d.py         # 1D plotting
│   ├── plot_2d.py         # 2D plotting (with multi-metric support)
│   └── paraview.py        # ParaView export
├── api.py                 # High-level API (with custom metrics support)
├── __init__.py            # Package initialization
└── README.md              # This file
```

## Advanced Usage

### Using Target Directions (Interpolation Between Models)

Visualize the loss landscape between two trained models:

```python
# Train two models with different hyperparameters
model1 = train_model(lr=0.1, batch_size=128)
model2 = train_model(lr=0.01, batch_size=256)

# Create landscape
landscape = LossLandscape(model1, dataloader)

# Compute landscape from model1 to model2
result = landscape.compute_1d(
    directions='target',
    target_model=model2,
    xrange=(0, 1, 51)  # 0 = model1, 1 = model2
)
```

### Multiple Custom Metrics Example

```python
def compute_comprehensive_metrics(net, dataloader, use_cuda):
    """Compute multiple loss functions and metrics."""

    criterion_ce = nn.CrossEntropyLoss()
    criterion_smooth_l1 = nn.SmoothL1Loss()
    criterion_mse = nn.MSELoss()

    total_ce = 0.0
    total_smooth_l1 = 0.0
    total_mse = 0.0
    total_correct = 0
    total_samples = 0

    net.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)

            # Multiple loss functions
            ce_loss = criterion_ce(outputs, targets)
            targets_onehot = torch.nn.functional.one_hot(
                targets, num_classes=outputs.size(1)).float()
            smooth_l1_loss = criterion_smooth_l1(outputs, targets_onehot)
            mse_loss = criterion_mse(outputs, targets_onehot)

            # Accumulate
            total_ce += ce_loss.item() * inputs.size(0)
            total_smooth_l1 += smooth_l1_loss.item() * inputs.size(0)
            total_mse += mse_loss.item() * inputs.size(0)

            # Accuracy
            _, pred = outputs.max(1)
            total_correct += (pred == targets).sum().item()
            total_samples += inputs.size(0)

    return {
        'ce_loss': total_ce / total_samples,
        'smooth_l1': total_smooth_l1 / total_samples,
        'mse_loss': total_mse / total_samples,
        'accuracy': 100.0 * total_correct / total_samples
    }

landscape = LossLandscape(model, dataloader,
                         criterion=compute_comprehensive_metrics)

# Compute and visualize all metrics
result = landscape.compute_2d(xrange=(-1, 1, 21), yrange=(-1, 1, 21))
landscape.plot_2d_contour(surf_name='auto')  # Generates 16 PDF files!
```

### Save and Load Surfaces

The computed loss surfaces are automatically saved as HDF5 files:

```python
# Load surface computed previously
import h5py

f = h5py.File('loss_surface.h5', 'r')
print(f.keys())

# Standard mode keys: ['xcoordinates', 'ycoordinates', 'train_loss', 'train_acc']
# Custom mode keys: ['xcoordinates', 'ycoordinates', 'train_loss_metric1', 'train_loss_metric2', ...]

losses = f['train_loss'][:]
f.close()
```

### Direct Low-Level Functions

For more control, use core modules directly:

```python
from loss_landscape_core.core import direction, evaluator, perturbation
from loss_landscape_core import utils

# Create direction
d = direction.create_random_direction(model, dir_type='weights')

# Perturb weights
original_weights = direction.get_weights(model)
perturbation.set_weights(model, original_weights, [d], step=0.5)

# Evaluate
loss, acc = evaluator.eval_loss(model, criterion, dataloader, use_cuda=True)

# Restore
perturbation.set_weights(model, original_weights)
```

## Output Files

The library generates the following output files:

### HDF5 Data Files (`.h5`)

**Standard mode** (with nn.Module criterion):
```
Keys:
  - xcoordinates, ycoordinates: Coordinates
  - train_loss: Training loss values
  - train_acc: Training accuracy
```

**Custom mode** (with callable criterion):
```
Keys:
  - xcoordinates, ycoordinates: Coordinates
  - train_loss_ce_loss: CE loss values
  - train_loss_smooth_l1: Smooth L1 loss values
  - train_loss_accuracy: Accuracy values
  - (more for each custom metric)
```

### PDF Visualization Files

For each metric, generates 4 files:

```
*_train_loss_metricname_2dcontour.pdf    # Contour lines
*_train_loss_metricname_2dcontourf.pdf   # Filled contour
*_train_loss_metricname_2dheat.pdf       # Heatmap
*_train_loss_metricname_3dsurface.pdf    # 3D surface
```

**Example with 3 metrics:**
```
8 files total:
  - 4 files for metric 1
  - 4 files for metric 2
  - 4 files for metric 3
```

### ParaView Files (`.vtp`)

VTK format compatible with ParaView for professional rendering:
```
loss_surface.h5_train_loss_ce_loss.vtp
loss_surface.h5_train_loss_accuracy.vtp
```

## Example Scripts

Two example scripts are provided:

### 1. `example_custom_metrics.py`

Demonstrates:
- Standard loss mode (backward compatible)
- 1D landscape with custom metrics
- 2D landscape with custom metrics
- Auto-plotting of all metrics

Run:
```bash
python example_custom_metrics.py
```

### 2. `test_2d_landscape_fast_multi_metrics.py`

Fast 2D landscape demo with optimization:
- ResNet56 on CIFAR-10 (1/10 data subset)
- 11×11 grid (vs 21×21 standard)
- 3 metrics: CE Loss, Smooth L1 Loss, Accuracy
- ~2-3 minutes runtime
- Auto-generates 12 visualizations

Run:
```bash
python test_2d_landscape_fast_multi_metrics.py
```

## Tips for Better Results

1. **Use normalized data**: Ensure your dataloader uses normalized data (important for meaningful loss values)

2. **Sufficient sampling**: Use at least 51 points per dimension (51×51 for 2D) to capture surface features

3. **Appropriate loss range**: Adjust `vmin`/`vmax` in contour plots to highlight interesting features

4. **Log scale**: Use `log=True` in ParaView export for landscapes with large dynamic range

5. **Resolution**: For publication-quality figures, export to ParaView and render with higher resolution

6. **Custom metrics**: For best results with custom metrics:
   - Ensure consistent metrics computation across all data points
   - Use batch normalization carefully (consider `dir_type='states'`)
   - Normalize outputs before computing metrics

## Performance Considerations

### Computation Time

- **1D landscape**: O(num_points) - Linear in sampling points
- **2D landscape**: O(num_points_x × num_points_y) - Quadratic
- Each point requires one forward pass through all data

### Memory Usage

- Storing 2D landscape: ~4 MB per metric (21×21 float32 array)
- Multiple metrics add linearly

### GPU Requirements

- Supports any NVIDIA GPU with CUDA support
- GPU memory needed: ~2× model + 1× batch size
- Falls back to CPU automatically if CUDA unavailable

## Troubleshooting

### CUDA Out of Memory

- Reduce batch size in DataLoader
- Use smaller dataset subset
- Use `use_cuda=False` for CPU mode

### Large Dynamic Range

Use `log=True` in ParaView export to visualize with logarithmic scale

### Missing Metrics

Ensure custom metrics function returns all keys consistently:

```python
def my_metrics(net, dataloader, use_cuda):
    # Must return dict with same keys for each call
    return {
        'loss': loss_val,
        'accuracy': acc_val
    }  # Same keys every time
```

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

## License

MIT License - See original [loss-landscape](https://github.com/zingyi-li/Loss-Surfaces) repository

## References

- Original paper: [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913)
- Original code: https://github.com/zingyi-li/Loss-Surfaces
- PyTorch documentation: https://pytorch.org/docs/
- ParaView: https://www.paraview.org/

## What's New

### Recent Additions

- **Custom Metrics Support**: Define and compute multiple metrics with custom functions
- **Auto-Detection**: Automatically detect and plot all computed metrics
- **Type-Safe API**: Smart criterion type detection (no flags needed)
- **Enhanced Plotting**: `plot_2d_contour()` and `plot_2d_surface()` now support `surf_name` parameter
- **Better Documentation**: Comprehensive examples and API reference

### Backward Compatibility

All changes are fully backward compatible:
- Existing code using `nn.CrossEntropyLoss()` continues to work
- Default behavior unchanged
- New features are opt-in

## Support

For issues, questions, or suggestions:
1. Check the example scripts
2. Review the API Reference
3. Consult the original paper and code
