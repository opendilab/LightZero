"""
Overview:
    1D loss landscape visualization utilities for neural network training analysis.
    This module provides functions to plot loss and accuracy curves along single
    directions in parameter space, useful for understanding optimization trajectories.

This module provides:
    - HDF5-based loss curve plotting with dual y-axes for loss and accuracy
    - Simple plotting from numpy arrays without file dependencies
    - Logarithmic scale support for loss visualization
    - PDF export for publication-quality figures

Key Functions:
    - plot_1d_loss: Plot 1D curves from HDF5 surface file with loss and accuracy
    - plot_1d: Simple 1D plot from numpy arrays for quick visualization
"""

import h5py
import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple, Optional, Any


def plot_1d_loss(surf_file: str, xmin: float = -1.0, xmax: float = 1.0, loss_max: float = 5, log: bool = False, show: bool = False, save_dir: str = '') -> None:
    """
    Overview:
        Plot 1D loss and accuracy curves from HDF5 surface file with dual y-axes.
        Creates publication-quality visualizations showing loss (blue) and accuracy (red)
        on the same plot, with support for both training and test metrics.

    Arguments:
        - surf_file (:obj:`str`): Path to HDF5 surface file containing loss and accuracy data
        - xmin (:obj:`float`, optional): Minimum x-axis value. Default is -1.0 (auto-determined from file)
        - xmax (:obj:`float`, optional): Maximum x-axis value. Default is 1.0 (auto-determined from file)
        - loss_max (:obj:`float`, optional): Maximum y-axis value for loss. Default is 5
        - log (:obj:`bool`, optional): Use logarithmic scale for loss axis. Default is False
        - show (:obj:`bool`, optional): Whether to display plot interactively. Default is False
        - save_dir (:obj:`str`, optional): Directory to save plots. Default is '' (same as surf_file location)

    Returns:
        - None: Saves plot as PDF file with naming format: {surf_file}_1d_loss_acc[_log].pdf

    Notes:
        - Automatically detects and plots test metrics if available in HDF5 file
        - Uses solid lines for training metrics and dashed lines for test metrics
        - Left y-axis (blue) shows loss values, right y-axis (red) shows accuracy percentage
        - Output file includes '_log' suffix when logarithmic scale is used

    Examples::
        >>> # Plot loss landscape with automatic axis scaling
        >>> plot_1d_loss('model_surface.h5')

        >>> # Plot with logarithmic scale and custom range
        >>> plot_1d_loss('model_surface.h5', xmin=-0.5, xmax=0.5, log=True, loss_max=10)
    """
    print('-' * 60)
    print('Plotting 1D loss curve')
    print('-' * 60)

    f = h5py.File(surf_file, 'r')
    print(f"Available keys: {list(f.keys())}")

    x = f['xcoordinates'][:]
    assert 'train_loss' in f.keys(), "'train_loss' not found in file"

    train_loss = f['train_loss'][:]
    train_acc = f['train_acc'][:]

    print(f"X range: {x.min():.3f} to {x.max():.3f}")
    print(f"Train loss range: {train_loss.min():.3f} to {train_loss.max():.3f}")
    print(f"Train acc range: {train_acc.min():.1f} to {train_acc.max():.1f}")

    # Auto-determine axis limits from data if using defaults
    xmin = xmin if xmin != -1.0 else x.min()
    xmax = xmax if xmax != 1.0 else x.max()

    # Use surface file directory if save_dir not specified
    if not save_dir:
        save_dir = ''

    # Create dual-axis figure for loss and accuracy
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    if log:
        tr_loss, = ax1.semilogy(x, train_loss, 'b-', label='Training loss', linewidth=2)
    else:
        tr_loss, = ax1.plot(x, train_loss, 'b-', label='Training loss', linewidth=2)

    tr_acc, = ax2.plot(x, train_acc, 'r-', label='Training accuracy', linewidth=2)

    # Add test curves if available
    if 'test_loss' in f.keys():
        test_loss = f['test_loss'][:]
        test_acc = f['test_acc'][:]
        if log:
            te_loss, = ax1.semilogy(x, test_loss, 'b--', label='Test loss', linewidth=2)
        else:
            te_loss, = ax1.plot(x, test_loss, 'b--', label='Test loss', linewidth=2)
        te_acc, = ax2.plot(x, test_acc, 'r--', label='Test accuracy', linewidth=2)

    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(0, loss_max)
    ax1.set_ylabel('Loss', color='b', fontsize='x-large')
    ax1.tick_params('y', colors='b', labelsize='large')
    ax1.tick_params('x', labelsize='large')

    ax2.set_ylim(0, 100)
    ax2.set_ylabel('Accuracy (%)', color='r', fontsize='x-large')
    ax2.tick_params('y', colors='r', labelsize='large')

    ax1.set_xlabel('Direction', fontsize='x-large')

    # Combine legends
    lines = [tr_loss, tr_acc]
    labels = ['Training loss', 'Training accuracy']
    if 'test_loss' in f.keys():
        lines.extend([te_loss, te_acc])
        labels.extend(['Test loss', 'Test accuracy'])
    ax1.legend(lines, labels, loc='upper center', fontsize='large')

    suffix = '_log' if log else ''
    save_path = surf_file + f'_1d_loss_acc{suffix}.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved: {save_path}")

    if show:
        plt.show()

    f.close()


def plot_1d(losses: np.ndarray, coords: np.ndarray, loss_type: str = 'train_loss', log: bool = False, loss_max: float = 5, show: bool = False) -> Tuple[Any, Any]:
    """
    Overview:
        Create simple 1D loss plot from numpy arrays without HDF5 file dependency.
        Useful for quick visualization during development or when data is already in memory.

    Arguments:
        - losses (:obj:`numpy.ndarray`): 1D array of loss values to plot
        - coords (:obj:`numpy.ndarray`): 1D array of coordinate values (x-axis)
        - loss_type (:obj:`str`, optional): Label for the loss curve in legend. Default is 'train_loss'
        - log (:obj:`bool`, optional): Use logarithmic scale for y-axis. Default is False
        - loss_max (:obj:`float`, optional): Maximum y-axis value for loss. Default is 5
        - show (:obj:`bool`, optional): Whether to display plot interactively. Default is False

    Returns:
        - fig (:obj:`matplotlib.figure.Figure`): Matplotlib figure object
        - ax (:obj:`matplotlib.axes.Axes`): Matplotlib axes object for further customization

    Notes:
        - Unlike plot_1d_loss, this does not save files automatically
        - Returns matplotlib objects for additional customization or manual saving
        - Grid is enabled by default with 30% transparency for better readability

    Examples::
        >>> # Quick loss visualization
        >>> losses = np.array([2.5, 2.0, 1.5, 1.2, 1.0])
        >>> coords = np.linspace(-1, 1, 5)
        >>> fig, ax = plot_1d(losses, coords, loss_type='validation_loss', log=True)
        >>> fig.savefig('custom_loss_plot.png')
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if log:
        ax.semilogy(coords, losses, 'b-', linewidth=2, label=loss_type)
    else:
        ax.plot(coords, losses, 'b-', linewidth=2, label=loss_type)

    ax.set_xlabel('Direction', fontsize='x-large')
    ax.set_ylabel('Loss', fontsize='x-large')
    ax.set_ylim(0, loss_max)
    ax.tick_params(labelsize='large')
    ax.legend(fontsize='large')
    ax.grid(True, alpha=0.3)

    if show:
        plt.show()

    return fig, ax
