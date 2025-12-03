"""
Overview:
    HDF5 file storage and retrieval utilities for loss landscape data persistence.
    This module provides efficient storage of direction vectors, loss surfaces, and
    associated metadata using the HDF5 format for large-scale landscape experiments.

This module provides:
    - Direction vector storage and loading (list of tensors/arrays to HDF5 groups)
    - Loss surface file initialization with coordinate grids for 1D/2D landscapes
    - Loss and accuracy surface saving with automatic key management
    - Loss surface loading with flexible metric key handling
    - Coordinate grid setup for reproducible landscape exploration

Key Functions:
    - write_list: Save list of tensors/arrays to HDF5 group for direction persistence
    - read_list: Load list of arrays from HDF5 group to reconstruct directions
    - setup_surface_file: Initialize HDF5 file with coordinate grids for landscape computation
    - save_loss_surface: Save computed loss/accuracy surfaces to HDF5 file
    - load_loss_surface: Load loss/accuracy surfaces and coordinates from HDF5 file

Notes:
    - HDF5 format ideal for large multi-dimensional arrays with compression support
    - Direction files store random/target directions for reproducible landscape computation
    - Surface files store both coordinates and computed metrics (loss, accuracy, etc.)
    - Coordinate grids define the exploration range in parameter space
"""

import torch
import h5py
import os
import numpy as np
from typing import List, Union, Optional, Tuple


def write_list(f: h5py.File, name: str, direction: Union[List[torch.Tensor], List[np.ndarray]]) -> None:
    """
    Overview:
        Save a list of tensors or numpy arrays to an HDF5 group for persistent direction storage.
        Each element in the list is stored as a separate dataset within the group.

    Arguments:
        - f (:obj:`h5py.File`): Open HDF5 file object in write or append mode
        - name (:obj:`str`): Group name for the direction (e.g., 'xdirection', 'ydirection')
        - direction (:obj:`List[torch.Tensor]` or :obj:`List[numpy.ndarray]`): List of parameter
            tensors/arrays representing a direction vector in structured format

    Returns:
        - None: Writes data to HDF5 file in-place

    Notes:
        - Automatically converts torch.Tensor to numpy arrays for HDF5 compatibility
        - Each tensor/array stored with integer key (0, 1, 2, ...) preserving order
        - Useful for saving random directions or target directions between model checkpoints
        - Direction can later be loaded with read_list and converted back to tensors

    Examples::
        >>> # Save random direction to HDF5
        >>> import h5py
        >>> direction = [torch.randn(10, 5), torch.randn(5)]
        >>> with h5py.File('directions.h5', 'w') as f:
        ...     write_list(f, 'random_dir_1', direction)
    """
    grp = f.create_group(name)
    for i, l in enumerate(direction):
        if isinstance(l, torch.Tensor):
            l = l.numpy()  # Convert to numpy for HDF5 storage
        grp.create_dataset(str(i), data=l)


def read_list(f: h5py.File, name: str) -> List[np.ndarray]:
    """
    Overview:
        Load a list of numpy arrays from an HDF5 group to reconstruct stored directions.
        Inverse operation of write_list, retrieves all datasets from a named group.

    Arguments:
        - f (:obj:`h5py.File`): Open HDF5 file object in read mode
        - name (:obj:`str`): Group name to read from (e.g., 'xdirection', 'ydirection')

    Returns:
        - direction (:obj:`List[numpy.ndarray]`): List of numpy arrays in original order

    Notes:
        - Arrays are loaded in order of integer keys (0, 1, 2, ...)
        - Returns numpy arrays; use npvec_to_tensorlist to convert to PyTorch tensors
        - Useful for loading saved random directions for reproducible landscape computation

    Examples::
        >>> # Load previously saved direction
        >>> import h5py
        >>> with h5py.File('directions.h5', 'r') as f:
        ...     direction = read_list(f, 'random_dir_1')
        >>> print(f"Loaded {len(direction)} parameter arrays")
    """
    grp = f[name]
    return [grp[str(i)][:] for i in range(len(grp))]


def setup_surface_file(surf_file: str, dir_file: str, xmin: float, xmax: float, xnum: int, ymin: Optional[float] = None, ymax: Optional[float] = None, ynum: Optional[int] = None) -> None:
    """
    Overview:
        Initialize HDF5 file for loss surface storage with coordinate grids.
        Creates coordinate arrays for 1D or 2D landscape exploration and links to direction file.

    Arguments:
        - surf_file (:obj:`str`): Path to output surface file (will be created or appended)
        - dir_file (:obj:`str`): Path to direction file (stored as reference in surface file)
        - xmin (:obj:`float`): Minimum value for x-axis coordinate grid
        - xmax (:obj:`float`): Maximum value for x-axis coordinate grid
        - xnum (:obj:`int`): Number of points along x-axis
        - ymin (:obj:`float`, optional): Minimum value for y-axis (None for 1D landscape). Default is None
        - ymax (:obj:`float`, optional): Maximum value for y-axis (None for 1D landscape). Default is None
        - ynum (:obj:`int`, optional): Number of points along y-axis (None for 1D landscape). Default is None

    Returns:
        - None: Creates/modifies HDF5 file with coordinate datasets

    Notes:
        - Skips setup if file already exists and contains coordinate data (avoids overwriting)
        - For 1D landscapes: only xcoordinates created (ymin, ymax, ynum should be None)
        - For 2D landscapes: both xcoordinates and ycoordinates created
        - Coordinate grids use np.linspace for uniform spacing
        - Direction file reference allows tracing which directions were used

    Examples::
        >>> # Setup 1D landscape file
        >>> setup_surface_file('landscape_1d.h5', 'directions.h5',
        ...                    xmin=-1.0, xmax=1.0, xnum=51)

        >>> # Setup 2D landscape file
        >>> setup_surface_file('landscape_2d.h5', 'directions.h5',
        ...                    xmin=-1.0, xmax=1.0, xnum=25,
        ...                    ymin=-1.0, ymax=1.0, ynum=25)
    """
    # Check if file already exists with valid coordinates (avoid overwriting)
    if os.path.exists(surf_file):
        f = h5py.File(surf_file, 'r')
        if (ymin is not None and 'ycoordinates' in f.keys()) or 'xcoordinates' in f.keys():
            f.close()
            print(f"{surf_file} is already set up")
            return
        f.close()

    # Create or append to surface file
    f = h5py.File(surf_file, 'a')
    f['dir_file'] = dir_file  # Store reference to direction file

    # Create x-axis coordinate grid
    xcoordinates = np.linspace(xmin, xmax, num=int(xnum))
    f['xcoordinates'] = xcoordinates

    # Create y-axis coordinate grid if 2D landscape
    if ymin is not None:
        ycoordinates = np.linspace(ymin, ymax, num=int(ynum))
        f['ycoordinates'] = ycoordinates

    f.close()
    print(f"Surface file initialized: {surf_file}")


def save_loss_surface(surf_file: str, loss_key: str, acc_key: str, losses: np.ndarray, accuracies: np.ndarray) -> None:
    """
    Overview:
        Save computed loss and accuracy surfaces to HDF5 file with automatic key management.
        Updates existing datasets or creates new ones for landscape metrics.

    Arguments:
        - surf_file (:obj:`str`): Path to surface file (must already be initialized)
        - loss_key (:obj:`str`): Dataset key for loss values (e.g., 'train_loss', 'test_loss')
        - acc_key (:obj:`str`): Dataset key for accuracy values (e.g., 'train_acc', 'test_acc')
        - losses (:obj:`numpy.ndarray`): 1D or 2D array of computed loss values
        - accuracies (:obj:`numpy.ndarray`): 1D or 2D array of computed accuracy values

    Returns:
        - None: Writes datasets to HDF5 file

    Notes:
        - Automatically deletes existing datasets with same keys before writing (update behavior)
        - For 1D landscapes: losses and accuracies are 1D arrays matching xcoordinates
        - For 2D landscapes: losses and accuracies are 2D arrays matching meshgrid(xcoordinates, ycoordinates)
        - flush() ensures data is written to disk immediately
        - Multiple metrics can be stored by using different keys (e.g., 'train_loss', 'val_loss')

    Examples::
        >>> # Save 1D landscape results
        >>> losses = np.array([2.5, 2.0, 1.8, 1.5, 1.3])
        >>> accs = np.array([60, 65, 70, 75, 80])
        >>> save_loss_surface('landscape_1d.h5', 'train_loss', 'train_acc', losses, accs)

        >>> # Save 2D landscape results
        >>> losses_2d = np.random.rand(25, 25)
        >>> accs_2d = np.random.rand(25, 25) * 100
        >>> save_loss_surface('landscape_2d.h5', 'train_loss', 'train_acc', losses_2d, accs_2d)
    """
    f = h5py.File(surf_file, 'a')

    # Delete existing datasets if present (enables updating results)
    if loss_key in f.keys():
        del f[loss_key]
    if acc_key in f.keys():
        del f[acc_key]

    # Write new datasets
    f[loss_key] = losses
    f[acc_key] = accuracies
    f.flush()  # Ensure data is written to disk
    f.close()


def load_loss_surface(surf_file: str, loss_key: str = 'train_loss', acc_key: str = 'train_acc') -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    """
    Overview:
        Load loss surface data and coordinate grids from HDF5 file for visualization or analysis.
        Retrieves computed metrics along with the coordinate arrays used for landscape exploration.

    Arguments:
        - surf_file (:obj:`str`): Path to surface file to load
        - loss_key (:obj:`str`, optional): Dataset key for loss values. Default is 'train_loss'
        - acc_key (:obj:`str`, optional): Dataset key for accuracy values. Default is 'train_acc'

    Returns:
        - losses (:obj:`numpy.ndarray` or :obj:`None`): Loss values array (1D or 2D), None if key not found
        - accuracies (:obj:`numpy.ndarray` or :obj:`None`): Accuracy values array (1D or 2D), None if key not found
        - xcoordinates (:obj:`numpy.ndarray`): X-axis coordinate grid
        - ycoordinates (:obj:`numpy.ndarray` or :obj:`None`): Y-axis coordinate grid (None for 1D landscapes)

    Notes:
        - Returns None for losses/accuracies if requested keys don't exist (graceful handling)
        - ycoordinates is None for 1D landscapes (only xcoordinates present)
        - Loaded arrays are numpy format; convert to torch if needed for further computation
        - Use this to reload landscapes for plotting without recomputation

    Examples::
        >>> # Load 1D landscape
        >>> losses, accs, x, y = load_loss_surface('landscape_1d.h5')
        >>> print(f"Loaded 1D landscape: {len(x)} points")
        >>> print(f"Y-coords: {y}")  # None for 1D

        >>> # Load 2D landscape with custom keys
        >>> losses, accs, x, y = load_loss_surface('landscape_2d.h5',
        ...                                         loss_key='val_loss',
        ...                                         acc_key='val_acc')
        >>> print(f"Loaded 2D landscape: {x.shape[0]}x{y.shape[0]} grid")
    """
    f = h5py.File(surf_file, 'r')

    # Load metrics (return None if key doesn't exist)
    losses = f[loss_key][:] if loss_key in f.keys() else None
    accuracies = f[acc_key][:] if acc_key in f.keys() else None

    # Load coordinate grids
    xcoordinates = f['xcoordinates'][:]
    ycoordinates = f['ycoordinates'][:] if 'ycoordinates' in f.keys() else None

    f.close()

    return losses, accuracies, xcoordinates, ycoordinates
