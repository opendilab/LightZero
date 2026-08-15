"""
Overview:
    Utility modules for storage, geometry, and visualization of loss landscapes.
    Handles HDF5 file I/O, geometric projections, and multiple visualization formats.

This module provides:
    - HDF5 storage utilities for landscapes and direction vectors
    - Vector projection and angle calculation for trajectory analysis
    - 1D visualization: dual-axis loss and accuracy plots
    - 2D visualization: contour plots, heatmaps, and 3D surfaces
    - ParaView export for professional-quality rendering

Key Functions - Storage:
    - write_list/read_list: HDF5 I/O for direction vectors

Key Functions - Geometry:
    - tensorlist_to_tensor/npvec_to_tensorlist: Tensor reshaping
    - project_1D/project_2D: Vector projection on directions
    - cal_angle: Cosine similarity between vectors

Key Functions - Visualization:
    - plot_1d_loss: 1D landscape plots with dual y-axes
    - plot_2d_contour: 2D contour/heatmap plots
    - plot_2d_surface: 3D surface visualization
    - h5_to_vtp: ParaView-compatible export

Typical Visualization Workflow:
    1. Compute 2D landscape: landscape.compute_2d()
    2. Generate contours: landscape.plot_2d_contour(surf_name='auto')
    3. Export for publication: landscape.export_paraview(log=True, zmax=10)
"""

from .storage import write_list, read_list
from .projection import (
    tensorlist_to_tensor,
    nplist_to_tensor,
    npvec_to_tensorlist,
    cal_angle,
    project_1D,
    project_2D,
)
from .plot_1d import plot_1d_loss, plot_1d
from .plot_2d import plot_2d_contour, plot_2d_surface, plot_2d
from .paraview import h5_to_vtp

__all__ = [
    'write_list',
    'read_list',
    'tensorlist_to_tensor',
    'nplist_to_tensor',
    'npvec_to_tensorlist',
    'cal_angle',
    'project_1D',
    'project_2D',
    'plot_1d_loss',
    'plot_1d',
    'plot_2d_contour',
    'plot_2d_surface',
    'plot_2d',
    'h5_to_vtp',
]
