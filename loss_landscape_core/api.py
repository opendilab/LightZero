"""
High-level API for loss landscape visualization.

Provides a simple interface to compute and visualize loss landscapes.
"""

import torch
import torch.nn as nn
import numpy as np
import h5py
import os

from .core import direction, evaluator, perturbation
from . import utils
from . import viz


class LossLandscape:
    """Main class for loss landscape computation and visualization.

    Example:
        # 1D landscape
        landscape = LossLandscape(model, dataloader, criterion, use_cuda=True)
        losses = landscape.compute_1d(
            directions='random',
            xrange=(-1, 1, 51),
            normalize='filter'
        )
        viz.plot_1d_loss(landscape.surf_file)

        # 2D landscape
        losses = landscape.compute_2d(
            xrange=(-1, 1, 51),
            yrange=(-1, 1, 51)
        )
        viz.plot_2d_contour(landscape.surf_file)
    """

    def __init__(self, net, dataloader, criterion=None, use_cuda=False, surf_file=None):
        """Initialize loss landscape.

        Args:
            net: PyTorch neural network model
            dataloader: DataLoader for evaluation
            criterion: Loss function (default: CrossEntropyLoss)
                      Can be:
                      - nn.Module (CrossEntropyLoss, MSELoss, etc.)
                      - callable function returning dict of metrics
            use_cuda: Whether to use GPU
            surf_file: Path to save surface file (default: auto-generated)
        """
        self.net = net
        self.dataloader = dataloader
        self.use_cuda = use_cuda

        # ⭐ NEW: Detect criterion type
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # Check if it's a custom metrics function (callable but not nn.Module)
        if callable(criterion) and not isinstance(criterion, nn.Module):
            self.custom_metrics = criterion
            self.criterion = None
            self.use_custom_metrics = True
        else:
            # Standard PyTorch criterion
            self.criterion = criterion
            self.custom_metrics = None
            self.use_custom_metrics = False

        # Store original weights and states
        self.weights = direction.get_weights(net)
        self.states = direction.get_states(net)

        # Default surface file
        if surf_file is None:
            surf_file = 'loss_surface.h5'
        self.surf_file = surf_file

        self.directions = None

    def compute_1d(self, directions='random', xrange=(-1, 1, 51),
                   dir_type='weights', normalize='filter', ignore='biasbn',
                   target_model=None, save=True):
        """Compute 1D loss landscape.

        Args:
            directions: 'random' or 'target' (requires target_model)
            xrange: Tuple of (xmin, xmax, xnum)
            dir_type: 'weights' or 'states'
            normalize: Normalization method (filter|layer|weight)
            ignore: 'biasbn' to ignore bias and BN parameters
            target_model: Required if directions='target'
            save: Whether to save to HDF5 file

        Returns:
            Dict with keys 'losses', 'accuracies', 'xcoordinates'
        """
        xmin, xmax, xnum = xrange

        # Create direction
        if directions == 'random':
            direction_vec = [direction.create_random_direction(
                self.net, dir_type, ignore, normalize)]
        elif directions == 'target':
            assert target_model is not None, "target_model required for 'target' direction"
            direction_vec = [direction.create_target_direction(
                self.net, target_model, dir_type)]
        else:
            raise ValueError(f"Unknown direction type: {directions}")

        self.directions = direction_vec

        # Compute loss surface
        xcoords = np.linspace(xmin, xmax, int(xnum))

        # ⭐ NEW: Support for custom metrics
        if self.use_custom_metrics:
            # Initialize dict of lists for each metric
            metrics_dict = {}
        else:
            # Standard loss and accuracy
            losses = []
            accuracies = []

        print("Computing 1D loss landscape...")
        for i, x in enumerate(xcoords):
            if dir_type == 'weights':
                perturbation.set_weights(self.net, self.weights, direction_vec, x)
            else:  # states
                perturbation.set_states(self.net, self.states, direction_vec, x)

            # ⭐ NEW: Call custom metrics if provided
            if self.use_custom_metrics:
                metrics = self.custom_metrics(self.net, self.dataloader, self.use_cuda)
                if i == 0:
                    # Initialize dict on first iteration
                    metrics_dict = {k: [] for k in metrics.keys()}

                # Append all metrics
                for k, v in metrics.items():
                    metrics_dict[k].append(v)

                # Print first metric for progress display
                first_key = list(metrics.keys())[0]
                print(f"  [{i+1:3d}/{int(xnum)}] {first_key}: {metrics[first_key]:.4f}")
            else:
                # Standard evaluation
                loss, acc = evaluator.eval_loss(self.net, self.criterion,
                                                self.dataloader, self.use_cuda)
                losses.append(loss)
                accuracies.append(acc)
                print(f"  [{i+1:3d}/{int(xnum)}] Loss: {loss:.4f}, Acc: {acc:.1f}%")

        # Restore original weights
        if dir_type == 'weights':
            perturbation.set_weights(self.net, self.weights)
        else:
            perturbation.set_states(self.net, self.states)

        # ⭐ NEW: Construct result dict, compatible with both modes
        if self.use_custom_metrics:
            # Convert lists to numpy arrays
            losses_result = {k: np.array(v) for k, v in metrics_dict.items()}
            result = {
                'losses': losses_result,  # Dict of arrays
                'xcoordinates': xcoords
            }
        else:
            # Standard format (backward compatible)
            result = {
                'losses': np.array(losses),
                'accuracies': np.array(accuracies),
                'xcoordinates': xcoords
            }

        if save:
            self._save_1d_surface(result)

        return result

    def compute_2d(self, xrange=(-1, 1, 51), yrange=(-1, 1, 51),
                   dir_type='weights', normalize='filter', ignore='biasbn',
                   x_target=None, y_target=None, save=True):
        """Compute 2D loss landscape.

        Args:
            xrange, yrange: Tuple of (min, max, num)
            dir_type: 'weights' or 'states'
            normalize: Normalization method
            ignore: 'biasbn' to ignore bias and BN parameters
            x_target, y_target: Target models for directions (optional)
            save: Whether to save to HDF5 file

        Returns:
            Dict with keys 'losses', 'accuracies', 'xcoordinates', 'ycoordinates'
        """
        xmin, xmax, xnum = xrange
        ymin, ymax, ynum = yrange

        # Create directions
        if x_target is not None:
            x_direction = direction.create_target_direction(
                self.net, x_target, dir_type)
        else:
            x_direction = direction.create_random_direction(
                self.net, dir_type, ignore, normalize)

        if y_target is not None:
            y_direction = direction.create_target_direction(
                self.net, y_target, dir_type)
        else:
            y_direction = direction.create_random_direction(
                self.net, dir_type, ignore, normalize)

        self.directions = [x_direction, y_direction]

        # Compute loss surface
        xcoords = np.linspace(xmin, xmax, int(xnum))
        ycoords = np.linspace(ymin, ymax, int(ynum))

        # ⭐ NEW: Support for custom metrics
        if self.use_custom_metrics:
            # Initialize placeholder for metrics dict
            metrics_dict = None
        else:
            # Standard loss and accuracy arrays
            losses = np.zeros((len(xcoords), len(ycoords)))
            accuracies = np.zeros((len(xcoords), len(ycoords)))

        print("Computing 2D loss landscape...")
        for i, x in enumerate(xcoords):
            for j, y in enumerate(ycoords):
                if dir_type == 'weights':
                    perturbation.set_weights(self.net, self.weights,
                                            self.directions, [x, y])
                else:  # states
                    perturbation.set_states(self.net, self.states,
                                           self.directions, [x, y])

                # ⭐ NEW: Call custom metrics if provided
                if self.use_custom_metrics:
                    metrics = self.custom_metrics(self.net, self.dataloader, self.use_cuda)
                    if metrics_dict is None:
                        # Initialize dict on first iteration
                        metrics_dict = {
                            k: np.zeros((len(xcoords), len(ycoords)))
                            for k in metrics.keys()
                        }
                    # Store all metrics
                    for k, v in metrics.items():
                        metrics_dict[k][i, j] = v
                else:
                    # Standard evaluation
                    loss, acc = evaluator.eval_loss(self.net, self.criterion,
                                                    self.dataloader, self.use_cuda)
                    losses[i, j] = loss
                    accuracies[i, j] = acc

            print(f"  Row {i+1}/{len(xcoords)} complete")

        # Restore original weights
        if dir_type == 'weights':
            perturbation.set_weights(self.net, self.weights)
        else:
            perturbation.set_states(self.net, self.states)

        # ⭐ NEW: Construct result dict, compatible with both modes
        if self.use_custom_metrics:
            result = {
                'losses': metrics_dict,  # Dict of 2D arrays
                'xcoordinates': xcoords,
                'ycoordinates': ycoords
            }
        else:
            # Standard format (backward compatible)
            result = {
                'losses': losses,
                'accuracies': accuracies,
                'xcoordinates': xcoords,
                'ycoordinates': ycoords
            }

        if save:
            self._save_2d_surface(result)

        return result

    def _save_1d_surface(self, result):
        """Save 1D loss surface to HDF5 file.

        Compatible with both single loss and multiple metrics.
        """
        f = h5py.File(self.surf_file, 'w')
        f['xcoordinates'] = result['xcoordinates']

        # ⭐ NEW: Handle both single loss and multi metrics
        if isinstance(result['losses'], dict):
            # Multiple metrics (custom)
            for metric_name, metric_values in result['losses'].items():
                f[f'train_loss_{metric_name}'] = metric_values
        else:
            # Single loss (standard - backward compatible)
            f['train_loss'] = result['losses']
            if 'accuracies' in result:
                f['train_acc'] = result['accuracies']

        f.close()
        print(f"\nSaved to: {self.surf_file}")

    def _save_2d_surface(self, result):
        """Save 2D loss surface to HDF5 file.

        Compatible with both single loss and multiple metrics.
        """
        f = h5py.File(self.surf_file, 'w')
        f['xcoordinates'] = result['xcoordinates']
        f['ycoordinates'] = result['ycoordinates']

        # ⭐ NEW: Handle both single loss and multi metrics
        if isinstance(result['losses'], dict):
            # Multiple metrics (custom)
            for metric_name, metric_values in result['losses'].items():
                f[f'train_loss_{metric_name}'] = metric_values
        else:
            # Single loss (standard - backward compatible)
            f['train_loss'] = result['losses']
            if 'accuracies' in result:
                f['train_acc'] = result['accuracies']

        f.close()
        print(f"\nSaved to: {self.surf_file}")

    def plot_1d(self, xmin=None, xmax=None, loss_max=5, log=False, show=False):
        """Plot 1D loss curve from saved surface file."""
        viz.plot_1d_loss(self.surf_file, xmin or -1, xmax or 1,
                         loss_max, log, show)

    def plot_2d_contour(self, surf_name='train_loss', vmin=0.1, vmax=10, vlevel=0.5, show=False):
        """Plot 2D contour from saved surface file.

        ⭐ NEW: Supports multiple metrics with surf_name='auto'

        Args:
            surf_name: Name of surface to plot (default: 'train_loss')
                      Use 'auto' to detect and plot all available metrics
            vmin, vmax: Value range for contour levels
            vlevel: Spacing between contour levels
            show: Whether to display plot
        """
        viz.plot_2d_contour(self.surf_file, surf_name,
                            vmin, vmax, vlevel, show)

    def plot_2d_surface(self, surf_name='train_loss', show=False):
        """Plot 3D surface from saved surface file.

        ⭐ NEW: Supports multiple metrics with surf_name='auto'

        Args:
            surf_name: Name of surface to plot (default: 'train_loss')
                      Use 'auto' to detect and plot all available metrics
            show: Whether to display plot
        """
        viz.plot_2d_surface(self.surf_file, surf_name, show)

    def export_paraview(self, surf_name='train_loss', log=False, zmax=-1, interp=-1):
        """Export 2D landscape to ParaView VTP format.

        Args:
            surf_name: Surface name to export
            log: Use logarithmic scale
            zmax: Clip maximum z value
            interp: Interpolate to higher resolution (-1: no interpolation)

        Returns:
            Path to generated VTP file
        """
        vtp_file = self.surf_file.replace('.h5', '.vtp')
        viz.h5_to_vtp(self.surf_file, surf_name, log, zmax, interp, vtp_file)
        print(f"\nExported to: {vtp_file}")
        return vtp_file
