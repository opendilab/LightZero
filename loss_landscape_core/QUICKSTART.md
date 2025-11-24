# Loss Landscape Core - Quick Start Guide

## Installation

```bash
cd /mnt/shared-storage-user/tangjia/temp/loss_landscape_core
pip install torch torchvision h5py matplotlib scipy seaborn numpy
```

## Basic Usage

### 1D Loss Curve (Fastest)

```python
from loss_landscape_core import LossLandscape
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load model and data
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                          transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)

# Create and compute landscape
landscape = LossLandscape(model, dataloader, use_cuda=True)
result = landscape.compute_1d(xrange=(-1, 1, 51))

# Visualize
landscape.plot_1d(show=True)
```

### 2D Loss Surface

```python
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

## Running Examples

### Example 1: 1D Loss Curve

```bash
cd examples
python example_1d.py
```

Generates:
- `loss_1d.h5` - computed surface
- `loss_1d.h5_1d_loss_acc.pdf` - plot

### Example 2: 2D Loss Surface with ParaView

```bash
python example_2d.py
```

Generates:
- `loss_2d.h5` - computed surface
- Multiple PDF plots
- `loss_2d.h5_train_loss.vtp` - ParaView format

## Key Features

✅ **Simple API** - Easy-to-use high-level interface
✅ **Single GPU** - No MPI setup required
✅ **1D & 2D** - Both 1D curves and 2D surfaces
✅ **Multiple visualizations** - Plots, contours, 3D surfaces
✅ **ParaView export** - Professional rendering (VTP format)
✅ **Modular design** - Clean, extensible architecture

## File Structure

```
loss_landscape_core/
├── core/              Core functionality
├── utils/             Utilities (storage, projection)
├── viz/               Visualization (1D, 2D, ParaView)
├── api.py             High-level API (LossLandscape class)
├── examples/          Example scripts
└── README.md          Full documentation
```

## Common Parameters

### `compute_1d()` / `compute_2d()`

- `directions`: 'random' or 'target' (requires target_model)
- `dir_type`: 'weights' or 'states'
- `normalize`: 'filter' (recommended), 'layer', 'weight', etc.
- `ignore`: 'biasbn' to ignore bias and batch norm

### `plot_2d_contour()`

- `vmin`, `vmax`: Value range for contours
- `vlevel`: Spacing between contour levels
- `show`: Display interactive plot

## Tips

1. **For faster computation**: Use fewer datapoints (21x21 instead of 51x51)
2. **For publication**: Export to ParaView with log scale and interpolation
3. **Data normalization**: Ensure consistent normalization between training and evaluation
4. **GPU memory**: Large models may require smaller batch sizes

## Troubleshooting

**Out of memory?**
- Reduce batch size
- Use fewer sampling points (e.g., 21x21 instead of 51x51)
- Use dataloader subset

**ParaView not installed?**
- Download from https://www.paraview.org/download/
- Or use matplotlib plots directly

**Slow computation?**
- Normal for 2D surfaces (51x51 = 2601 evaluations)
- Use GPU for faster computation
- Reduce resolution for testing

## Next Steps

- Read `README.md` for full documentation
- Check `examples/` for more detailed examples
- Explore `core/`, `utils/`, `viz/` for low-level API

## License

MIT - See original [loss-landscape](https://github.com/zingyi-li/Loss-Surfaces) project
