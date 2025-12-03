"""
Overview:
    Comprehensive loss landscape visualization toolkit for neural networks.
    Enables exploration of high-dimensional loss surfaces through 1D and 2D sampling,
    with multiple visualization formats including contour plots, 3D surfaces, and ParaView export.

This module provides:
    - 1D and 2D loss landscape computation along random or target directions
    - Multiple visualization formats: contour plots, heatmaps, 3D surfaces
    - ParaView export for professional-quality rendering
    - Support for custom metrics functions
    - Filter-wise and layer-wise direction normalization options

Key Classes:
    - LossLandscape: Main API for computing and visualizing loss landscapes

Quick Start:
    from lzero.loss_landscape import LossLandscape

    landscape = LossLandscape(model, dataloader, criterion, use_cuda=True)

    # 1D landscape: loss along random direction
    result_1d = landscape.compute_1d(xrange=(-1, 1, 51))
    landscape.plot_1d()

    # 2D landscape: loss on plane spanned by two random directions
    result_2d = landscape.compute_2d(xrange=(-1, 1, 21), yrange=(-1, 1, 21))
    landscape.plot_2d_contour(surf_name='auto')  # Auto-detect metrics
    landscape.plot_2d_surface(surf_name='auto')

    # Export to ParaView for publication-quality visualization
    landscape.export_paraview(log=True, zmax=10)

References:
    - "Visualizing the Loss Landscape of Neural Nets" (Li et al., 2018)
    - https://arxiv.org/abs/1712.09913
    - Original implementation: https://github.com/tomgoldstein/loss-landscape
"""

from .loss_landscape_api import LossLandscape

__version__ = '1.0.0'
__author__ = 'Loss Landscape Contributors'

__all__ = ['LossLandscape']
