"""
2D loss landscape visualization.

Supports contour plots, heatmaps, and 3D surface plots.
"""

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


def plot_2d_contour(surf_file, surf_name='train_loss', vmin=0.1, vmax=10, vlevel=0.5, show=False):
    """Plot 2D contour map for loss landscape.

    ⭐ NEW: Auto-detects and plots multiple metrics if available.

    Args:
        surf_file: Path to HDF5 surface file
        surf_name: Name of surface to plot (default: 'train_loss')
                  If 'auto', will detect and plot all available metrics
        vmin, vmax: Value range for contour levels
        vlevel: Spacing between contour levels
        show: Whether to display plot
    """
    print('-' * 60)
    print('Plotting 2D contour')
    print('-' * 60)

    f = h5py.File(surf_file, 'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)

    # ⭐ NEW: Auto-detect multiple metrics
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

    # ⭐ NEW: Plot each metric
    for key in metric_keys:
        if key not in f.keys():
            print(f"Warning: '{key}' not found, skipping")
            continue

        Z = np.array(f[key][:])
        z_min, z_max = np.min(Z), np.max(Z)
        z_range = z_max - z_min
        z_mean = np.mean(Z)
        print(f"\nPlotting {key}: min={z_min:.4f}, max={z_max:.4f}, range={z_range:.6f}")

        # Extract metric name (remove 'train_loss_' prefix if present)
        metric_label = key.replace('train_loss_', '')

        # Check if Z has meaningful variation
        # Use relative variation threshold: range should be at least 0.01% of mean value
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


def plot_2d_surface(surf_file, surf_name='train_loss', show=False):
    """Plot 3D surface for loss landscape.

    ⭐ NEW: Auto-detects and plots multiple metrics if available.

    Args:
        surf_file: Path to HDF5 surface file
        surf_name: Name of surface to plot (default: 'train_loss')
                  If 'auto', will detect and plot all available metrics
        show: Whether to display plot
    """
    print('-' * 60)
    print('Plotting 3D surface')
    print('-' * 60)

    f = h5py.File(surf_file, 'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)

    # ⭐ NEW: Auto-detect multiple metrics
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

    # ⭐ NEW: Plot each metric
    for key in metric_keys:
        if key not in f.keys():
            print(f"Warning: '{key}' not found, skipping")
            continue

        Z = np.array(f[key][:])
        z_min, z_max = np.min(Z), np.max(Z)
        z_range = z_max - z_min
        print(f"\nPlotting {key}: min={z_min:.4f}, max={z_max:.4f}, range={z_range:.6f}")

        # Extract metric name (remove 'train_loss_' prefix if present)
        metric_label = key.replace('train_loss_', '')

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=True, alpha=0.9)
        fig.colorbar(surf, shrink=0.5, aspect=5, label=metric_label)

        ax.set_xlabel('X Direction', fontsize='x-large')
        ax.set_ylabel('Y Direction', fontsize='x-large')
        ax.set_zlabel(metric_label, fontsize='x-large')

        # Add warning to title if range is very small
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


def plot_2d(losses, x_coords, y_coords, surf_name='loss', show=False):
    """Simple 2D plot from numpy arrays (without HDF5).

    Args:
        losses: 2D numpy array of loss values
        x_coords: 1D array of x coordinates
        y_coords: 1D array of y coordinates
        surf_name: Label for the surface
        show: Whether to display plot
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
