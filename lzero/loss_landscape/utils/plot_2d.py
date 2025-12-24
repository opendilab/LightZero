"""
Overview:
    2D loss landscape visualization utilities for neural network loss surface analysis.
    This module provides comprehensive visualization tools for exploring loss landscapes
    in two-dimensional parameter space, essential for understanding optimization geometry.

This module provides:
    - Contour plots with customizable level spacing for landscape topology analysis
    - Filled contour plots with color gradients for intuitive visualization
    - Heatmaps using seaborn for high-contrast metric visualization
    - 3D surface plots with interactive viewing capabilities
    - Auto-detection of multiple metrics for comprehensive analysis
    - Automatic handling of flat or near-constant landscapes

Key Functions:
    - plot_2d_contour: Generate contour, filled contour, and heatmap from HDF5 file
    - plot_2d_surface: Create 3D surface plots from HDF5 file with ParaView-compatible output
    - plot_2d: Simple 2D plotting from numpy arrays for quick visualization
"""

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Optional, Any


def plot_2d_contour(surf_file: str, surf_name: str = 'train_loss', vmin: Optional[float] = 0.1, vmax: Optional[float] = 10, vlevel: Optional[float] = 0.5, show: bool = False) -> None:
    """
    Overview:
        Generate 2D contour plots, filled contours, and heatmaps for loss landscape analysis.
        Automatically detects and visualizes multiple metrics when surf_name='auto', producing
        three types of plots for each metric to provide comprehensive landscape understanding.

    Arguments:
        - surf_file (:obj:`str`): Path to HDF5 surface file containing computed loss surfaces
        - surf_name (:obj:`str`, optional): Name of surface to plot, e.g., 'train_loss'.
            If 'auto', detects and plots all metrics with 'train_loss_' prefix. Default is 'train_loss'
        - vmin (:obj:`float`, optional): Minimum value for contour levels. Default is 0.1
        - vmax (:obj:`float`, optional): Maximum value for contour levels. Default is 10
        - vlevel (:obj:`float`, optional): Spacing between contour levels. Default is 0.5
        - show (:obj:`bool`, optional): Whether to display plots interactively. Default is False

    Returns:
        - None: Saves three PDF files per metric:
            - {surf_file}_{metric}_2dcontour.pdf: Line contour plot with labels
            - {surf_file}_{metric}_2dcontourf.pdf: Filled contour plot with colorbar
            - {surf_file}_{metric}_2dheat.pdf: Heatmap visualization

    Notes:
        - Automatically adjusts vmin/vmax/vlevel if values are invalid for the data range
        - Detects nearly flat landscapes (relative variation < 1e-6) and generates heatmap only
        - For flat landscapes, prints warning about negligible variation
        - Removes 'train_loss_' prefix from metric names in plot titles for clarity
        - Requires at least 2x2 grid of coordinates for meaningful contour plots

    Examples::
        >>> # Plot single metric with default settings
        >>> plot_2d_contour('model_surface.h5', surf_name='train_loss')

        >>> # Auto-detect and plot all metrics with custom levels
        >>> plot_2d_contour('model_surface.h5', surf_name='auto', vmin=0, vmax=5, vlevel=0.25)
    """
    print('-' * 60)
    print('Plotting 2D contour')
    print('-' * 60)

    f = h5py.File(surf_file, 'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)

    if surf_name == 'auto':
        # Find all train_loss_* keys
        metric_keys = [k for k in f.keys() if k.startswith('train_loss_')]
        if not metric_keys:
            # Fall back to standard 'train_loss' key
            metric_keys = ['train_loss'] if 'train_loss' in f.keys() else []
    else:
        metric_keys = [surf_name]

    if not metric_keys:
        raise KeyError(f"No metrics found in {surf_file}")

    print(f"Loading: {surf_file}")
    print(f"X range: {len(x)}, Y range: {len(y)}")
    print(f"Found {len(metric_keys)} metric(s): {metric_keys}")

    if len(x) <= 1 or len(y) <= 1:
        print("Insufficient coordinates for plotting contours")
        f.close()
        return

    for key in metric_keys:
        if key not in f.keys():
            print(f"Warning: '{key}' not found, skipping")
            continue

        Z = np.array(f[key][:])
        z_min, z_max = np.min(Z), np.max(Z)
        z_range = z_max - z_min
        z_mean = np.mean(Z)
        print(f"\nPlotting {key}: min={z_min:.4f}, max={z_max:.4f}, range={z_range:.6f}")

        # Extract clean metric name by removing 'train_loss_' prefix if present
        metric_label = key.replace('train_loss_', '')

        # Check if surface has meaningful variation (not nearly flat)
        # Use relative variation threshold to handle different loss scales
        relative_variation = z_range / z_mean if z_mean != 0 else 0
        has_variation = z_range > 1e-10 and relative_variation > 1e-6

        if not has_variation:
            print(f"  Warning: {key} has negligible variation (range={z_range:.2e}, relative={relative_variation:.2e})")
            print(f"  This suggests the loss landscape is nearly flat in the explored region.")
            print(f"  Skipping contour plots (not meaningful for constant values)")

            # Still generate a simple heatmap to show the constant value
            fig = plt.figure(figsize=(12, 10))
            sns.heatmap(Z, cmap='viridis', cbar=True, annot=False,
                        xticklabels=False, yticklabels=False)
            plt.title(f'{metric_label} - Heatmap (Nearly Constant: {z_mean:.4f})', fontsize='x-large')
            save_path = f"{surf_file}_{key}_2dheat.pdf"
            fig.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
            print(f"  Saved heatmap only: {save_path}")
            plt.close(fig)
            continue

        # Auto-adjust vmin/vmax if needed
        actual_vmin = vmin if vmin is not None else z_min
        actual_vmax = vmax if vmax is not None else z_max
        actual_vlevel = vlevel if vlevel is not None else (actual_vmax - actual_vmin) / 10

        # Ensure valid range
        if actual_vmin >= actual_vmax:
            actual_vmin = z_min
            actual_vmax = z_max

        if actual_vmax - actual_vmin < actual_vlevel:
            actual_vlevel = (actual_vmax - actual_vmin) / 10

        # Plot contour lines
        fig = plt.figure(figsize=(12, 10))
        CS = plt.contour(X, Y, Z, cmap='summer',
                         levels=np.arange(actual_vmin, actual_vmax, actual_vlevel))
        plt.clabel(CS, inline=1, fontsize=8)
        plt.xlabel('X Direction', fontsize='x-large')
        plt.ylabel('Y Direction', fontsize='x-large')
        plt.title(f'{metric_label} - Contour Plot', fontsize='x-large')
        save_path = f"{surf_file}_{key}_2dcontour.pdf"
        fig.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"  Saved: {save_path}")
        plt.close(fig)

        # Plot filled contours
        fig = plt.figure(figsize=(12, 10))
        CS = plt.contourf(X, Y, Z, cmap='summer',
                          levels=np.arange(actual_vmin, actual_vmax, actual_vlevel))
        plt.colorbar(CS, label=metric_label)
        plt.xlabel('X Direction', fontsize='x-large')
        plt.ylabel('Y Direction', fontsize='x-large')
        plt.title(f'{metric_label} - Filled Contour', fontsize='x-large')
        save_path = f"{surf_file}_{key}_2dcontourf.pdf"
        fig.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"  Saved: {save_path}")
        plt.close(fig)

        # Plot heatmap
        fig = plt.figure(figsize=(12, 10))
        sns.heatmap(Z, cmap='viridis', cbar=True, vmin=actual_vmin, vmax=actual_vmax,
                    xticklabels=False, yticklabels=False)
        plt.title(f'{metric_label} - Heatmap', fontsize='x-large')
        save_path = f"{surf_file}_{key}_2dheat.pdf"
        fig.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"  Saved: {save_path}")
        plt.close(fig)

    f.close()
    if show:
        plt.show()


def plot_2d_surface(surf_file: str, surf_name: str = 'train_loss', show: bool = False) -> None:
    """
    Overview:
        Create 3D surface plots of loss landscapes for interactive visualization.
        Generates professional-quality 3D renderings that reveal the geometric structure
        of the loss surface, useful for understanding optimization challenges like saddle points.

    Arguments:
        - surf_file (:obj:`str`): Path to HDF5 surface file containing computed loss surfaces
        - surf_name (:obj:`str`, optional): Name of surface to plot, e.g., 'train_loss'.
            If 'auto', detects and plots all metrics with 'train_loss_' prefix. Default is 'train_loss'
        - show (:obj:`bool`, optional): Whether to display plots interactively. Default is False

    Returns:
        - None: Saves PDF file per metric with naming format: {surf_file}_{metric}_3dsurface.pdf

    Notes:
        - Uses coolwarm colormap to highlight high/low loss regions
        - Automatically detects nearly flat surfaces (relative range < 1e-6) and adds warning to title
        - Antialiasing enabled for smooth surface rendering
        - Alpha channel set to 0.9 for slight transparency
        - Colorbar scaled to 50% height for better proportions

    Examples::
        >>> # Generate 3D surface plot for single metric
        >>> plot_2d_surface('model_surface.h5', surf_name='train_loss')

        >>> # Auto-detect and plot all metrics with interactive display
        >>> plot_2d_surface('model_surface.h5', surf_name='auto', show=True)
    """
    print('-' * 60)
    print('Plotting 3D surface')
    print('-' * 60)

    f = h5py.File(surf_file, 'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)

    if surf_name == 'auto':
        # Find all train_loss_* keys
        metric_keys = [k for k in f.keys() if k.startswith('train_loss_')]
        if not metric_keys:
            # Fall back to standard 'train_loss' key
            metric_keys = ['train_loss'] if 'train_loss' in f.keys() else []
    else:
        metric_keys = [surf_name]

    if not metric_keys:
        raise KeyError(f"No metrics found in {surf_file}")

    print(f"Loading: {surf_file}")
    print(f"X range: {len(x)}, Y range: {len(y)}")
    print(f"Found {len(metric_keys)} metric(s): {metric_keys}")

    for key in metric_keys:
        if key not in f.keys():
            print(f"Warning: '{key}' not found, skipping")
            continue

        Z = np.array(f[key][:])
        z_min, z_max = np.min(Z), np.max(Z)
        z_range = z_max - z_min
        print(f"\nPlotting {key}: min={z_min:.4f}, max={z_max:.4f}, range={z_range:.6f}")

        # Extract clean metric name by removing 'train_loss_' prefix if present
        metric_label = key.replace('train_loss_', '')

        # Create 3D surface plot with projection
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot surface with smooth rendering
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=True, alpha=0.9)
        fig.colorbar(surf, shrink=0.5, aspect=5, label=metric_label)

        # Set axis labels
        ax.set_xlabel('X Direction', fontsize='x-large')
        ax.set_ylabel('Y Direction', fontsize='x-large')
        ax.set_zlabel(metric_label, fontsize='x-large')

        # Warn if surface is nearly flat (may indicate narrow exploration range)
        if z_range < z_min * 1e-6:
            title = f'{metric_label} - 3D Surface (Nearly Flat)'
        else:
            title = f'{metric_label} - 3D Surface'
        ax.set_title(title, fontsize='x-large')

        save_path = f"{surf_file}_{key}_3dsurface.pdf"
        fig.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"  Saved: {save_path}")
        plt.close(fig)

    f.close()
    if show:
        plt.show()


def plot_2d(losses: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, surf_name: str = 'loss', show: bool = False) -> Any:
    """
    Overview:
        Create 2D contour and 3D surface plots from numpy arrays without HDF5 dependency.
        Useful for quick visualization during development or when data is already in memory.

    Arguments:
        - losses (:obj:`numpy.ndarray`): 2D array of loss values (shape: len(y_coords) x len(x_coords))
        - x_coords (:obj:`numpy.ndarray`): 1D array of x-axis coordinates
        - y_coords (:obj:`numpy.ndarray`): 1D array of y-axis coordinates
        - surf_name (:obj:`str`, optional): Label for the surface in plots. Default is 'loss'
        - show (:obj:`bool`, optional): Whether to display plots interactively. Default is False

    Returns:
        - fig (:obj:`matplotlib.figure.Figure`): Matplotlib figure object of the 3D surface plot

    Notes:
        - Generates both contour and 3D surface plots automatically
        - Unlike plot_2d_contour/plot_2d_surface, this does not save files automatically
        - Uses 15 contour levels by default for good detail without clutter
        - Returns only the 3D figure object for backward compatibility

    Examples::
        >>> # Quick 2D landscape visualization
        >>> x = np.linspace(-1, 1, 25)
        >>> y = np.linspace(-1, 1, 25)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = X**2 + Y**2  # Simple quadratic bowl
        >>> fig = plot_2d(Z, x, y, surf_name='Quadratic Loss')
        >>> fig.savefig('quick_landscape.png')
    """
    X, Y = np.meshgrid(x_coords, y_coords)

    # Contour plot
    fig = plt.figure(figsize=(10, 8))
    CS = plt.contour(X, Y, losses, cmap='summer', levels=15)
    plt.clabel(CS, inline=1, fontsize=8)
    plt.colorbar(CS, label=surf_name)
    plt.xlabel('X Direction', fontsize='x-large')
    plt.ylabel('Y Direction', fontsize='x-large')
    plt.title(f'{surf_name} - Contour', fontsize='x-large')

    if show:
        plt.show()

    # 3D surface
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, losses, cmap=cm.coolwarm,
                           linewidth=0, antialiased=True)
    fig.colorbar(surf, shrink=0.5, aspect=5, label=surf_name)
    ax.set_xlabel('X', fontsize='x-large')
    ax.set_ylabel('Y', fontsize='x-large')
    ax.set_zlabel(surf_name, fontsize='x-large')
    ax.set_title(f'{surf_name} - 3D Surface', fontsize='x-large')

    if show:
        plt.show()

    return fig
