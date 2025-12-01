"""
1D loss landscape visualization.
"""

import h5py
import numpy as np
from matplotlib import pyplot as plt


def plot_1d_loss(surf_file, xmin=-1.0, xmax=1.0, loss_max=5, log=False, show=False, save_dir=''):
    """Plot 1D loss curve with accuracy from HDF5 surface file.

    Args:
        surf_file: Path to HDF5 surface file
        xmin, xmax: Range of x-axis (default: from file)
        loss_max: Maximum loss value for y-axis
        log: Use logarithmic scale for loss
        show: Whether to display plot
        save_dir: Directory to save plots (default: same directory as surf_file)
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

    # Adjust axis limits
    xmin = xmin if xmin != -1.0 else x.min()
    xmax = xmax if xmax != 1.0 else x.max()

    # Save directory
    if not save_dir:
        save_dir = ''

    # Plot loss and accuracy on same figure
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


def plot_1d(losses, coords, loss_type='train_loss', log=False, loss_max=5, show=False):
    """Simple 1D plot from numpy arrays (without HDF5).

    Args:
        losses: 1D numpy array of loss values
        coords: 1D numpy array of coordinates
        loss_type: Label for the loss curve
        log: Use logarithmic scale
        loss_max: Maximum loss value for y-axis
        show: Whether to display plot
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
