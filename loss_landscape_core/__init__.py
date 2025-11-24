"""
Loss Landscape - Visualize the loss landscape of neural networks.

A PyTorch library for computing and visualizing loss surfaces of neural networks.
Supports 1D and 2D loss curves, contour plots, 3D surfaces, and export to ParaView.

Basic Usage:
    from loss_landscape_core import LossLandscape

    landscape = LossLandscape(model, dataloader, criterion, use_cuda=True)

    # 1D landscape
    result_1d = landscape.compute_1d(xrange=(-1, 1, 51))
    landscape.plot_1d()

    # 2D landscape
    result_2d = landscape.compute_2d(xrange=(-1, 1, 51), yrange=(-1, 1, 51))
    landscape.plot_2d_contour()
    landscape.plot_2d_surface()

    # Export to ParaView
    landscape.export_paraview(log=True, zmax=10)
"""

from .api import LossLandscape

__version__ = '1.0.0'
__author__ = 'Loss Landscape Contributors'

__all__ = ['LossLandscape']
