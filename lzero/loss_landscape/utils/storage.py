"""
HDF5 file storage utilities for directions and loss surfaces.
"""

import torch
import h5py
import os
import numpy as np


def write_list(f, name, direction):
    """Save a list of tensors to HDF5 file.

    Args:
        f: h5py file object
        name: Group name (e.g., 'xdirection', 'ydirection')
        direction: List of tensors or numpy arrays
    """
    grp = f.create_group(name)
    for i, l in enumerate(direction):
        if isinstance(l, torch.Tensor):
            l = l.numpy()
        grp.create_dataset(str(i), data=l)


def read_list(f, name):
    """Read a list of arrays from HDF5 group.

    Args:
        f: h5py file object
        name: Group name to read from

    Returns:
        List of numpy arrays
    """
    grp = f[name]
    return [grp[str(i)][:] for i in range(len(grp))]


def setup_surface_file(surf_file, dir_file, xmin, xmax, xnum, ymin=None, ymax=None, ynum=None):
    """Initialize HDF5 file for storing loss surface.

    Args:
        surf_file: Path to surface file
        dir_file: Path to direction file
        xmin, xmax, xnum: X-axis parameters
        ymin, ymax, ynum: Y-axis parameters (for 2D)
    """
    # Skip if file already exists and has coordinates
    if os.path.exists(surf_file):
        f = h5py.File(surf_file, 'r')
        if (ymin is not None and 'ycoordinates' in f.keys()) or 'xcoordinates' in f.keys():
            f.close()
            print(f"{surf_file} is already set up")
            return
        f.close()

    # Create surface file
    f = h5py.File(surf_file, 'a')
    f['dir_file'] = dir_file

    # Create coordinates
    xcoordinates = np.linspace(xmin, xmax, num=int(xnum))
    f['xcoordinates'] = xcoordinates

    if ymin is not None:
        ycoordinates = np.linspace(ymin, ymax, num=int(ynum))
        f['ycoordinates'] = ycoordinates

    f.close()
    print(f"Surface file initialized: {surf_file}")


def save_loss_surface(surf_file, loss_key, acc_key, losses, accuracies):
    """Save computed loss and accuracy arrays to HDF5 file.

    Args:
        surf_file: Path to surface file
        loss_key: Key for loss data (e.g., 'train_loss')
        acc_key: Key for accuracy data (e.g., 'train_acc')
        losses: 2D array of loss values
        accuracies: 2D array of accuracy values
    """
    f = h5py.File(surf_file, 'a')
    if loss_key in f.keys():
        del f[loss_key]
    if acc_key in f.keys():
        del f[acc_key]

    f[loss_key] = losses
    f[acc_key] = accuracies
    f.flush()
    f.close()


def load_loss_surface(surf_file, loss_key='train_loss', acc_key='train_acc'):
    """Load loss surface from HDF5 file.

    Args:
        surf_file: Path to surface file
        loss_key: Key for loss data
        acc_key: Key for accuracy data

    Returns:
        Tuple of (losses, accuracies, xcoordinates, ycoordinates)
    """
    f = h5py.File(surf_file, 'r')

    losses = f[loss_key][:] if loss_key in f.keys() else None
    accuracies = f[acc_key][:] if acc_key in f.keys() else None
    xcoordinates = f['xcoordinates'][:]
    ycoordinates = f['ycoordinates'][:] if 'ycoordinates' in f.keys() else None

    f.close()

    return losses, accuracies, xcoordinates, ycoordinates
